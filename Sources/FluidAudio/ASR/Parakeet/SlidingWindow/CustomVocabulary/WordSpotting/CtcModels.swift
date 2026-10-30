@preconcurrency import CoreML
import Foundation

/// Available CTC model variants
///
/// Note on greedy decoding:
/// - ctc110m: Blank-dominant (CTC head is auxiliary loss), greedy produces ~113% WER
/// - ctc06b: CoreML conversion issue causes greedy to produce ~158% WER (should be ~14%)
///
/// Recommended approach: Use TDT for transcription + CTC for vocabulary scoring
/// via constrained CTC rescoring. This achieves ~15% WER with high vocab precision.
public enum CtcModelVariant: String, CaseIterable, Sendable {
    /// Hybrid TDT+CTC 110M model (blank-dominant, greedy decoding produces garbage)
    case ctc110m
    /// Pure CTC 0.6B model (CoreML greedy decoding broken, use constrained CTC instead)
    case ctc06b

    public var repo: Repo {
        switch self {
        case .ctc110m: return .parakeetCtc110m
        case .ctc06b: return .parakeetCtc06b
        }
    }

    public var displayName: String {
        switch self {
        case .ctc110m: return "Parakeet CTC 110M (hybrid)"
        case .ctc06b: return "Parakeet CTC 0.6B (pure)"
        }
    }
}

/// Container for the Parakeet CTC CoreML models used for
/// keyword spotting (Argmax-style Custom Vocabulary pipeline).
public struct CtcModels: Sendable {

    public let melSpectrogram: MLModel
    public let encoder: MLModel
    /// Configuration of the encoder. The `MelSpectrogram` loads with `melSpectrogramConfiguration()`.
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]
    public let variant: CtcModelVariant

    private static let logger = AppLogger(category: "CtcModels")

    public init(
        melSpectrogram: MLModel,
        encoder: MLModel,
        configuration: MLModelConfiguration,
        vocabulary: [Int: String],
        variant: CtcModelVariant = .ctc110m
    ) {
        self.melSpectrogram = melSpectrogram
        self.encoder = encoder
        self.configuration = configuration
        self.vocabulary = vocabulary
        self.variant = variant
    }
}

extension CtcModels {

    /// Load CTC models directly from a custom directory (e.g., canary-1b-v2).
    ///
    /// This method loads models directly without going through ModelHub,
    /// allowing support for custom CTC models like canary-1b-v2.
    ///
    /// Expected directory structure:
    /// - MelSpectrogram.mlmodelc/
    /// - AudioEncoder.mlmodelc/
    /// - vocab.json
    ///
    /// - Parameters:
    ///   - directory: Directory containing the CoreML bundles and vocab.json
    ///   - variant: Which CTC model variant this is (default: .ctc110m)
    /// - Returns: Loaded `CtcModels` instance.
    public static func loadDirect(from directory: URL, variant: CtcModelVariant = .ctc110m) async throws -> CtcModels {
        logger.info("Loading CTC models directly from: \(directory.path)")

        let config = defaultConfiguration()
        let melConfig = melSpectrogramConfiguration()

        // Load MelSpectrogram model
        let melPath = directory.appendingPathComponent("MelSpectrogram.mlmodelc")
        guard FileManager.default.fileExists(atPath: melPath.path) else {
            throw AsrModelsError.modelNotFound("MelSpectrogram", melPath)
        }

        // Try loading directly first (for pre-compiled models), then try compiling
        let melModel: MLModel
        do {
            melModel = try MLModel(contentsOf: melPath, configuration: melConfig)
            logger.info("Loaded MelSpectrogram directly from: \(melPath.path)")
        } catch {
            logger.info("Direct load failed, attempting compilation...")
            let compiledMelURL = try await MLModel.compileModel(at: melPath)
            melModel = try MLModel(contentsOf: compiledMelURL, configuration: melConfig)
        }

        // Load AudioEncoder model
        let encoderPath = directory.appendingPathComponent("AudioEncoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: encoderPath.path) else {
            throw AsrModelsError.modelNotFound("AudioEncoder", encoderPath)
        }

        let encoderModel: MLModel
        do {
            encoderModel = try MLModel(contentsOf: encoderPath, configuration: config)
            logger.info("Loaded AudioEncoder directly from: \(encoderPath.path)")
        } catch {
            logger.info("Direct load failed, attempting compilation...")
            let compiledEncoderURL = try await MLModel.compileModel(at: encoderPath)
            encoderModel = try MLModel(contentsOf: compiledEncoderURL, configuration: config)
        }

        // Log compute units configuration
        let computeUnitsStr: String
        switch config.computeUnits {
        case .cpuOnly: computeUnitsStr = "cpuOnly"
        case .cpuAndGPU: computeUnitsStr = "cpuAndGPU"
        case .cpuAndNeuralEngine: computeUnitsStr = "cpuAndNeuralEngine"
        case .all: computeUnitsStr = "all"
        @unknown default: computeUnitsStr = "unknown"
        }
        logger.info("CTC encoder loaded with computeUnits: \(computeUnitsStr) (MelSpectrogram: cpuOnly)")

        // Load vocabulary
        let vocab = try loadVocabulary(from: directory)

        logger.info("Successfully loaded CTC models directly (\(vocab.count) vocab tokens)")

        return CtcModels(
            melSpectrogram: melModel,
            encoder: encoderModel,
            configuration: config,
            vocabulary: vocab,
            variant: variant
        )
    }

    /// Load CTC models from a directory.
    ///
    /// - Parameters:
    ///   - directory: Directory containing the downloaded CoreML bundles.
    ///   - variant: Which CTC model variant to load (default: .ctc110m).
    /// - Returns: Loaded `CtcModels` instance.
    public static func load(
        from directory: URL,
        variant: CtcModelVariant = .ctc110m
    ) async throws -> CtcModels {
        logger.info("Loading CTC models (\(variant.displayName)) from: \(directory.path)")

        let parentDirectory = directory.deletingLastPathComponent()
        let config = defaultConfiguration()

        // ModelHub expects the base directory (without the repo folder) and
        // resolves the repo's folderName internally.
        let melModels = try await ModelHub.loadModels(
            variant.repo,
            modelNames: [ModelNames.CTC.melSpectrogramPath],
            directory: parentDirectory,
            computeUnits: melSpectrogramConfiguration().computeUnits
        )
        let encoderModels = try await ModelHub.loadModels(
            variant.repo,
            modelNames: [ModelNames.CTC.audioEncoderPath],
            directory: parentDirectory,
            computeUnits: config.computeUnits
        )

        guard
            let melModel = melModels[ModelNames.CTC.melSpectrogramPath],
            let encoderModel = encoderModels[ModelNames.CTC.audioEncoderPath]
        else {
            throw AsrModelsError.loadingFailed("Failed to load CTC MelSpectrogram or AudioEncoder models")
        }

        let vocab = try loadVocabulary(from: directory)

        logger.info("Successfully loaded CTC models (\(variant.displayName)) and vocabulary")

        return CtcModels(
            melSpectrogram: melModel,
            encoder: encoderModel,
            configuration: config,
            vocabulary: vocab,
            variant: variant
        )
    }

    /// Download CTC models to the default cache directory (or a custom one) if needed.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory for variant).
    ///   - variant: Which CTC model variant to download (default: .ctc110m).
    ///   - force: Whether to force re-download even if models exist.
    /// - Returns: The directory where models were downloaded.
    @discardableResult
    public static func download(
        to directory: URL? = nil,
        variant: CtcModelVariant = .ctc110m,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory(for: variant)
        logger.info("Preparing CTC models (\(variant.displayName)) at: \(targetDir.path)")

        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir) {
            logger.info("CTC models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        let specs = [
            (ModelNames.CTC.melSpectrogramPath, melSpectrogramConfiguration().computeUnits),
            (ModelNames.CTC.audioEncoderPath, defaultConfiguration().computeUnits),
        ]

        for (name, computeUnits) in specs {
            _ = try await ModelHub.loadModels(
                variant.repo,
                modelNames: [name],
                directory: parentDir,
                computeUnits: computeUnits
            )
        }

        logger.info("Successfully downloaded CTC models (\(variant.displayName))")
        return targetDir
    }

    /// Convenience helper that downloads (if needed) and loads the CTC models.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory for variant).
    ///   - variant: Which CTC model variant to use (default: .ctc110m).
    /// - Returns: Loaded `CtcModels` instance.
    public static func downloadAndLoad(
        to directory: URL? = nil,
        variant: CtcModelVariant = .ctc110m
    ) async throws -> CtcModels {
        let targetDir = try await download(to: directory, variant: variant)
        return try await load(from: targetDir, variant: variant)
    }

    /// Default CoreML configuration for CTC inference (the audio encoder).
    public static func defaultConfiguration() -> MLModelConfiguration {
        MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuAndNeuralEngine)
    }

    /// CoreML configuration for the `MelSpectrogram` front end: CPU only, like the standalone preprocessors of
    /// Parakeet TDT, Canary, SenseVoice, Paraformer, CAM++ and FSMN-VAD. With the Neural Engine allowed, macOS
    /// 27.0.1 on M1 Pro and M4 Max places part of it there, and those features make the encoder emit blank for
    /// every frame (#1074).
    public static func melSpectrogramConfiguration() -> MLModelConfiguration {
        MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuOnly)
    }

    /// Check whether required CTC model bundles and vocabulary exist at a directory.
    public static func modelsExist(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let repoPath = directory

        let required = ModelNames.CTC.requiredModels
        let modelsPresent = required.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        let vocabPath = repoPath.appendingPathComponent(ModelNames.CTC.vocabularyPath)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    /// Default cache directory for CTC models (within Application Support).
    ///
    /// - Parameter variant: Which CTC model variant (default: .ctc110m).
    /// - Returns: The default cache directory for the specified variant.
    public static func defaultCacheDirectory(for variant: CtcModelVariant = .ctc110m) -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(variant.repo.folderName, isDirectory: true)
    }

    /// Load vocabulary from vocab.json in the given directory.
    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent("vocab.json")
        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw AsrModelsError.modelNotFound("vocab.json", vocabPath)
        }

        let data = try Data(contentsOf: vocabPath)
        let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]

        var vocabulary: [Int: String] = [:]
        for (key, value) in jsonDict {
            if let tokenId = Int(key) {
                vocabulary[tokenId] = value
            }
        }

        logger.info("Loaded CTC vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
        return vocabulary
    }
}
