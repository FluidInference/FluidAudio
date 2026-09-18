import AVFoundation
@preconcurrency import CoreML
import Foundation
import OSLog

/// LocalVQE speech enhancement: neural acoustic echo cancellation, noise
/// suppression and dereverberation for 16 kHz speech.
///
/// Core ML port of [localai-org/LocalVQE](https://github.com/localai-org/LocalVQE)
/// (Apache-2.0), a streaming CPU-tuned derivative of DeepVQE. The model
/// needs two inputs: the microphone signal and a far-end **reference**
/// (a loopback of what the loudspeaker played). Without a reference it
/// still denoises and dereverberates; pass silence.
///
/// ```swift
/// let vqe = try await LocalVqeManager()
/// let clean = try await vqe.process(mic: micSamples, reference: farEndSamples)
/// ```
///
/// For live audio, open a `LocalVqeStream` and push buffers as they arrive:
///
/// ```swift
/// let stream = try await vqe.makeStream()
/// let out = try await stream.enhance(mic: micHop, reference: refHop)
/// ```
///
/// **Beta**: verified against the upstream PyTorch and GGML engines
/// (72 dB SNR, 16-bit-wav limited) on the upstream double-talk demo clip;
/// not yet exercised in production call pipelines.
public actor LocalVqeManager {

    private let logger = AppLogger(category: "LocalVqeManager")

    public static let sampleRate = 16000
    /// Analysis hop: every call consumes and produces a multiple of this.
    public static let hopSize = 256
    /// Algorithmic delay of the enhanced output relative to the input.
    public static let outputDelaySamples = hopSize

    public let config: LocalVqeConfig
    private let audioConverter = AudioConverter()
    private var model: MLModel?

    public var isAvailable: Bool { model != nil }

    /// Download (if needed) and load the configured model from HuggingFace.
    public init(
        config: LocalVqeConfig = .default,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        self.config = config
        let start = Date()
        let fileName = ModelNames.LocalVQE.modelFile(variant: config.variant, chunk: config.chunk)
        let models = try await ModelHub.loadModels(
            .localVqe,
            modelNames: [fileName],
            directory: Self.defaultBaseDirectory().appendingPathComponent("Models"),
            computeUnits: config.computeUnits,
            variant: ModelNames.LocalVQE.variantKey(variant: config.variant, chunk: config.chunk),
            progressHandler: progressHandler
        )
        guard let model = models[fileName] else {
            throw LocalVqeError.modelLoadingFailed("\(fileName) missing after download")
        }
        self.model = model
        logger.info(
            "LocalVQE \(config.variant.rawValue)/\(config.chunk.rawValue) loaded in \(String(format: "%.2f", Date().timeIntervalSince(start)))s"
        )
    }

    /// Load a compiled model bundle from a local directory (no download).
    /// `modelDirectory` must contain `<stem>-<chunk>.mlmodelc` for the
    /// configured variant/chunk (see `ModelNames.LocalVQE.modelFile`).
    public init(config: LocalVqeConfig = .default, modelDirectory: URL) throws {
        self.config = config
        let fileName = ModelNames.LocalVQE.modelFile(variant: config.variant, chunk: config.chunk)
        let url = modelDirectory.appendingPathComponent(fileName)
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits
        do {
            self.model = try MLModel(contentsOf: url, configuration: mlConfig)
        } catch {
            throw LocalVqeError.modelLoadingFailed("\(url.path): \(error.localizedDescription)")
        }
    }

    /// Wrap an already-loaded model (must match `config.chunk`).
    public init(config: LocalVqeConfig = .default, model: MLModel) {
        self.config = config
        self.model = model
    }

    // MARK: - Whole-clip processing

    /// Enhance a complete clip. `mic` and `reference` are 16 kHz mono and
    /// must have equal length; the result has the same length and is
    /// sample-aligned with `mic`.
    public func process(mic: [Float], reference: [Float]) async throws -> [Float] {
        let stream = try makeStream()
        var out = try await stream.enhance(mic: mic, reference: reference)
        out.append(contentsOf: try await stream.flush())
        return out
    }

    /// Enhance a clip whose far-end reference is silent (noise suppression +
    /// dereverberation only).
    public func process(mic: [Float]) async throws -> [Float] {
        try await process(mic: mic, reference: [Float](repeating: 0, count: mic.count))
    }

    /// Enhance a mic recording using a reference recording. Both files are
    /// converted to 16 kHz mono; the shorter one is zero-padded.
    public func process(micURL: URL, referenceURL: URL?) async throws -> [Float] {
        let mic = try audioConverter.resampleAudioFile(micURL)
        var reference = try referenceURL.map { try audioConverter.resampleAudioFile($0) } ?? []
        if reference.count < mic.count {
            reference.append(contentsOf: [Float](repeating: 0, count: mic.count - reference.count))
        } else if reference.count > mic.count {
            reference.removeLast(reference.count - mic.count)
        }
        return try await process(mic: mic, reference: reference)
    }

    // MARK: - Streaming

    /// Open an independent streaming session on the loaded model.
    public func makeStream() throws -> LocalVqeStream {
        guard let model else { throw LocalVqeError.notInitialized }
        return try LocalVqeStream(model: model, samplesPerCall: config.chunk.samplesPerCall)
    }

    private static func defaultBaseDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("FluidAudio", isDirectory: true)
    }
}
