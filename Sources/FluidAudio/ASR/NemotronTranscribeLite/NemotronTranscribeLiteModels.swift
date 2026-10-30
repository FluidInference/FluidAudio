@preconcurrency import CoreML
import Foundation

/// Loaded Nemotron 3.5 Transcribe Lite models and host-side tables.
///
/// From `FluidInference/nemotron-3.5-transcribe-lite-coreml`:
///   - `encoder`: normalized mel (1, 128, 4800) + `enc_len` → encoder states (1, 1280, 600) fp16
///   - `ctcHead`: states (1, 1280, 600) + `enc_len` → `token_ids` (1, 1200), one per 40 ms
///   - `vocabulary`: 23,862 SentencePiece pieces (id → piece); the CTC blank is id 23862
///   - `melWindow` (400) and `melFilterbank` (128 × 257): the checkpoint's own preprocessor tables
public struct NemotronTranscribeLiteModels: Sendable {

    public let encoder: MLModel
    public let ctcHead: MLModel
    public let vocabulary: [String]
    public let melWindow: [Float]
    public let melFilterbank: [Float]

    private static let logger = AppLogger(category: "NemotronTranscribeLiteModels")

    public init(
        encoder: MLModel, ctcHead: MLModel, vocabulary: [String], melWindow: [Float], melFilterbank: [Float]
    ) {
        self.encoder = encoder
        self.ctcHead = ctcHead
        self.vocabulary = vocabulary
        self.melWindow = melWindow
        self.melFilterbank = melFilterbank
    }

    /// Download (if needed) and load the models.
    ///
    /// - Parameter computeUnits: compute units for the encoder and CTC head. The GPU is fastest on
    ///   Apple silicon Macs (`.cpuAndGPU`, default); `.cpuAndNeuralEngine` also works, ~2.5x slower.
    public static func downloadAndLoad(
        computeUnits: MLComputeUnits = .cpuAndGPU,
        progressHandler: ProgressHandler? = nil
    ) async throws -> NemotronTranscribeLiteModels {
        // Check before downloading: ModelHub.loadWithRecovery treats a load() error as a corrupt cache, so on an
        // unsupported OS it would delete the ~1.3 GB download and fetch it again before failing.
        guard #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) else {
            throw ASRError.unsupportedPlatform("Nemotron 3.5 Transcribe Lite needs macOS 15 / iOS 18 or newer")
        }
        let modelsRoot = modelsRootDirectory()
        let targetDir = modelsRoot.appendingPathComponent(
            Repo.nemotron35TranscribeLite.folderName, isDirectory: true)
        return try await ModelHub.loadWithRecovery(
            .nemotron35TranscribeLite, directory: modelsRoot,
            requiredFiles: ModelNames.NemotronTranscribeLite.requiredModels,
            progressHandler: progressHandler
        ) {
            try load(from: targetDir, computeUnits: computeUnits)
        }
    }

    /// Load models from a directory laid out like the HF repo (`coreml/`, `assets/`).
    public static func load(
        from directory: URL, computeUnits: MLComputeUnits = .cpuAndGPU
    ) throws -> NemotronTranscribeLiteModels {
        // The published models are ML Programs for macOS 15 / iOS 18.
        guard #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) else {
            throw ASRError.unsupportedPlatform("Nemotron 3.5 Transcribe Lite needs macOS 15 / iOS 18 or newer")
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let names = ModelNames.NemotronTranscribeLite.self
        let encoder = try loadModel(names.encoderFile, from: directory, configuration: config)
        let ctcHead = try loadModel(names.ctcHeadFile, from: directory, configuration: config)
        let vocabulary = try loadVocabulary(directory.appendingPathComponent(names.vocabularyFile))
        let window = try loadFloats(directory.appendingPathComponent(names.melWindowFile))
        let filterbank = try loadFloats(directory.appendingPathComponent(names.melFilterbankFile))
        let c = NemotronTranscribeLiteConfig.self
        guard window.count == c.winLength, filterbank.count == c.nMels * (c.nFFT / 2 + 1) else {
            throw ASRError.processingFailed(
                "Unexpected mel table sizes: window \(window.count), filterbank \(filterbank.count)")
        }
        guard vocabulary.count == c.blankId else {
            throw ASRError.processingFailed("Unexpected vocabulary size \(vocabulary.count), expected \(c.blankId)")
        }
        logger.info("Loaded Nemotron 3.5 Transcribe Lite (vocab \(vocabulary.count))")
        return NemotronTranscribeLiteModels(
            encoder: encoder, ctcHead: ctcHead, vocabulary: vocabulary, melWindow: window,
            melFilterbank: filterbank)
    }

    // MARK: - Private

    private static func loadModel(
        _ relativePath: String, from directory: URL, configuration: MLModelConfiguration
    ) throws -> MLModel {
        let compiled = directory.appendingPathComponent(relativePath)
        let package = compiled.deletingPathExtension().appendingPathExtension("mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            url = try MLModel.compileModel(at: package)
        } else {
            throw ASRError.processingFailed("Nemotron 3.5 Transcribe Lite model not found: \(relativePath)")
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private static func loadVocabulary(_ url: URL) throws -> [String] {
        let data = try Data(contentsOf: url)
        guard let pieces = try JSONSerialization.jsonObject(with: data) as? [String] else {
            throw ASRError.processingFailed("Failed to parse \(url.lastPathComponent) (expected a JSON array)")
        }
        return pieces
    }

    /// Raw little-endian float32.
    private static func loadFloats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count % MemoryLayout<Float>.size == 0 else {
            throw ASRError.processingFailed("\(url.lastPathComponent) is not a float32 table")
        }
        return data.withUnsafeBytes { raw in
            (0..<(data.count / 4)).map {
                Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
            }
        }
    }

    private static func modelsRootDirectory() -> URL {
        let fm = FileManager.default
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            return appSupport.appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return fm.temporaryDirectory.appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}
