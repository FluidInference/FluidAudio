#if TTS
@preconcurrency import CoreML
import Foundation

/// Text-to-speech with the small Pocket TTS models (18m / 7m / 2m): the FlowLM
/// on the Neural Engine through one MLState multifunction package, the Mimi
/// decoder in fp32 on the CPU by default (`mimiPrecision`, see
/// `PocketTtsSmallMimiPrecision` for the fused all-ANE alternative).
///
/// ```swift
/// let tts = PocketTtsSmallManager(model: .english7m)
/// try await tts.initialize()
/// let wav = try await tts.synthesize(text: "Hello, world!")
/// ```
///
/// Requires macOS 15 / iOS 18 (MLState + multifunction models). Shipped
/// voices are the 27 upstream presets (`alba` default); live voice cloning
/// is not wired for the small models yet.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
public actor PocketTtsSmallManager {

    private static let logger = AppLogger(category: "PocketTtsSmallManager")

    public nonisolated let model: PocketTtsSmallModel
    public nonisolated let computeUnits: MLComputeUnits
    public nonisolated let mimiPrecision: PocketTtsSmallMimiPrecision
    public private(set) var defaultVoice: String

    private var constants: PocketTtsSmallConstants?
    private var models: PocketTtsSmallModels?
    private var packDirectory: URL?
    private var voiceCache: [String: PocketTtsSmallEngine.Fp16Snapshot] = [:]

    public init(
        model: PocketTtsSmallModel = .english7m,
        defaultVoice: String? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        mimiPrecision: PocketTtsSmallMimiPrecision = .fp32
    ) {
        self.model = model
        self.defaultVoice = defaultVoice ?? model.defaultVoice
        self.computeUnits = computeUnits
        self.mimiPrecision = mimiPrecision
    }

    public var isInitialized: Bool { models != nil }
    public var arch: PocketTtsSmallArch? { constants?.arch }

    /// Download (if needed) and load the pack.
    public func initialize(directory: URL? = nil, progressHandler: ProgressHandler? = nil) async throws {
        let pack = try await PocketTtsSmallDownloader.ensurePack(
            model, directory: directory, progressHandler: progressHandler)
        try load(from: pack)
    }

    /// Load from a local pack directory (`build/<model>/` of the converter).
    public func load(from packDirectory: URL) throws {
        let constants = try PocketTtsSmallConstants.load(from: packDirectory)
        let t0 = Date()
        self.models = try PocketTtsSmallModels(
            packDirectory: packDirectory, computeUnits: computeUnits, mimiPrecision: mimiPrecision)
        self.constants = constants
        self.packDirectory = packDirectory
        self.voiceCache = [:]
        Self.logger.info(
            "Loaded PocketTTS \(model.rawValue) (\(constants.arch.dModel)-d, \(constants.arch.numLayers)L, "
                + "mimi \(mimiPrecision.rawValue)) in \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")
    }

    public func setDefaultVoice(_ voice: String) { defaultVoice = voice }

    /// Shipped voices available in the loaded pack.
    public func availableVoices() throws -> [String] {
        guard let constants else { throw PocketTTSError.modelNotFound("PocketTTS small model not initialized") }
        return constants.voiceNames()
    }

    /// Open a streaming session with a shipped voice.
    public func makeSession(
        voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws -> PocketTtsSmallSession {
        guard let models, let constants else {
            throw PocketTTSError.modelNotFound("PocketTTS small model not initialized")
        }
        // One engine (one MLState) per session: concurrent sessions must not
        // share the mutable KV / Mimi state. The MLModel handles are shared.
        let engine = try PocketTtsSmallEngine(models: models, constants: constants)
        let name = voice ?? defaultVoice
        let snapshot: PocketTtsSmallEngine.Fp16Snapshot
        if let cached = voiceCache[name] {
            snapshot = cached
        } else if constants.hasKVSnapshot(name) {
            snapshot = try await engine.fp16Snapshot(from: constants.loadVoice(name))
            voiceCache[name] = snapshot
        } else {
            // A voice shipped as conditioning rows: prefill it into an empty cache
            // once and keep the resulting KV state as this voice's snapshot.
            let voice = try constants.loadVoiceConditioning(name)
            try await engine.resetToZero()
            try await engine.prefill(flatConditioning: voice.rows, tokenCount: voice.count)
            snapshot = try await engine.captureSnapshot()
            voiceCache[name] = snapshot
        }
        var textNormalizer: (@Sendable (String) -> String)?
        if model == .phonon31m {
            textNormalizer = { @Sendable text in PhononTextNormalizer.normalize(text) }
        }
        let session = PocketTtsSmallSession(
            engine: engine, constants: constants, voiceSnapshot: snapshot,
            temperature: temperature, seed: seed ?? UInt64.random(in: 0...UInt64.max),
            textNormalizer: textNormalizer)
        await session.start()
        return session
    }

    /// Synthesize to 80 ms frames.
    public func synthesizeStreaming(
        text: String, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws -> AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error> {
        let session = try await makeSession(voice: voice, temperature: temperature, seed: seed)
        session.enqueue(text)
        session.finish()
        return session.frames
    }

    /// Synthesize to 24 kHz 16-bit WAV data.
    public func synthesize(
        text: String, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil, deEss: Bool = true
    ) async throws -> Data {
        let samples = try await synthesizeSamples(
            text: text, voice: voice, temperature: temperature, seed: seed, deEss: deEss)
        return try AudioWAV.data(from: samples, sampleRate: Double(PocketTtsConstants.audioSampleRate))
    }

    /// Synthesize to raw Float32 samples at 24 kHz.
    public func synthesizeSamples(
        text: String, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil, deEss: Bool = true
    ) async throws -> [Float] {
        let t0 = Date()
        var samples: [Float] = []
        var frames = 0
        for try await frame in try await synthesizeStreaming(
            text: text, voice: voice, temperature: temperature, seed: seed)
        {
            samples.append(contentsOf: frame.samples)
            frames += 1
        }
        if deEss {
            AudioPostProcessor.applyTtsPostProcessing(
                &samples, sampleRate: Float(PocketTtsConstants.audioSampleRate), deEssAmount: -3.0, smoothing: false)
        }
        let secs = Double(samples.count) / Double(PocketTtsConstants.audioSampleRate)
        let elapsed = Date().timeIntervalSince(t0)
        Self.logger.info(
            "\(model.rawValue): \(frames) frames, \(String(format: "%.2f", secs))s audio in "
                + "\(String(format: "%.0f", elapsed * 1000))ms (RTFx \(String(format: "%.1f", secs / max(elapsed, 1e-6))))"
        )
        return samples
    }

    public func synthesizeToFile(
        text: String, outputURL: URL, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws {
        let wav = try await synthesize(text: text, voice: voice, temperature: temperature, seed: seed)
        try wav.write(to: outputURL)
    }
}

/// Downloads a small-model pack (`v4/<model>/`) from HuggingFace: the MLState
/// package, the fp32 `mimi_decoder.mlmodelc`, the constants and the voices —
/// the other fp32 I/O models and the `.mlpackage` sources are skipped.
public enum PocketTtsSmallDownloader {
    private static let logger = AppLogger(category: "PocketTtsSmallDownloader")

    /// Files a small-model pack needs on disk. The compiled bundle and the
    /// constants directory are checked through files INSIDE them, because
    /// `ModelHub.download` creates the directories as soon as their first file
    /// lands and an interrupted download must not look complete.
    static let requiredFiles: [String] = [
        ModelNames.PocketTTS.pocketStateFile + "/coremldata.bin",
        ModelNames.PocketTTS.pocketStateFile + "/model.mil",
        ModelNames.PocketTTS.mimiDecoderFile + "/coremldata.bin",
        ModelNames.PocketTTS.mimiDecoderFile + "/model.mil",
        "arch.json", "pocket_state.json",
        ModelNames.PocketTTS.constantsBinDir + "/tokenizer.model",
        ModelNames.PocketTTS.constantsBinDir + "/text_embed_table.bin",
    ]

    /// `requiredFiles` plus the model's default voice (a KV snapshot in `constants_bin/`).
    static func requiredFiles(for model: PocketTtsSmallModel) -> [String] {
        requiredFiles + [ModelNames.PocketTTS.constantsBinDir + "/\(model.defaultVoice).safetensors"]
    }

    public static func ensurePack(
        _ model: PocketTtsSmallModel, directory: URL? = nil, progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let base = try directory ?? TtsCacheDirectory.ensure()
        let repoDir = base.appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(Repo.pocketTts.folderName)
        let pack = repoDir.appendingPathComponent(model.repoSubdirectory)
        if requiredFiles(for: model).allSatisfy({
            FileManager.default.fileExists(atPath: pack.appendingPathComponent($0).path)
        }) {
            logger.info("PocketTTS \(model.rawValue) pack found in cache")
            return pack
        }
        logger.info("Downloading PocketTTS \(model.rawValue) pack (\(model.repoSubdirectory))...")
        try await ModelHub.download(
            .pocketTts, subdirectory: model.repoSubdirectory, to: repoDir,
            progressHandler: progressHandler, shouldSkip: { shouldSkipAsset(at: $0) })
        return pack
    }

    /// Keep `pocket_state.mlmodelc`, `mimi_decoder.mlmodelc`, the JSON
    /// descriptors and `constants_bin/`; skip the other fp32 I/O `.mlmodelc`
    /// bundles, every `.mlpackage` and the `constants/` intermediates.
    static func shouldSkipAsset(at path: String) -> Bool {
        let components = path.split(separator: "/").map(String.init)
        let basename = components.last ?? path
        if basename == ".DS_Store" || basename.hasSuffix(".wav") { return true }
        if components.contains(where: { $0.hasSuffix(".mlpackage") }) { return true }
        if let bundle = components.first(where: { $0.hasSuffix(".mlmodelc") }),
            bundle != ModelNames.PocketTTS.pocketStateFile, bundle != ModelNames.PocketTTS.mimiDecoderFile
        {
            return true
        }
        if components.contains("constants") { return true }
        if basename == "manifest.json" { return true }
        return false
    }
}
#endif
