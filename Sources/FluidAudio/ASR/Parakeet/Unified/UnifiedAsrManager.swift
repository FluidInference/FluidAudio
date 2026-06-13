import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Pure chunk-layout math for the unified offline (15 s window) batch path.
///
/// Long audio is split into frame-aligned windows that fit the fixed 15 s
/// encoder export, overlapping by 2 s so adjacent windows can be merged with
/// `ChunkProcessor.mergeChunks` (the same overlap-dedup machinery the TDT
/// pipeline uses).
struct UnifiedBatchLayout {
    let config: UnifiedConfig

    /// Fixed encoder window (15 s @ 16 kHz).
    var windowSamples: Int { 15 * config.sampleRate }
    /// Frame-aligned audio decoded per window.
    var chunkSamples: Int { windowSamples / config.frameSamples * config.frameSamples }
    /// Frame-aligned overlap between adjacent windows (2 s).
    var overlapSamples: Int {
        let requested = 2 * config.sampleRate
        return min(requested, chunkSamples / 2) / config.frameSamples * config.frameSamples
    }
    var strideSamples: Int { chunkSamples - overlapSamples }

    /// Start offsets (frame-aligned) of every window needed to cover `totalSamples`.
    func chunkStarts(totalSamples: Int) -> [Int] {
        guard totalSamples > 0 else { return [] }
        var starts = [0]
        var start = strideSamples
        while start < totalSamples {
            // A window is only needed if it adds samples beyond the previous one.
            if start + overlapSamples < totalSamples {
                starts.append(start)
            }
            start += strideSamples
        }
        return starts
    }
}

/// Offline batch ASR manager for Parakeet Unified 0.6B (FastConformer-RNNT).
///
/// Uses the full-attention 15 s offline encoder (better WER than the chunked
/// streaming export: 1.82% vs 2.15% on LibriSpeech test-clean) and transcribes
/// long audio with overlapping windows: each window is decoded independently
/// with a fresh RNNT state, then adjacent token streams are merged on the 2 s
/// overlap via `ChunkProcessor.mergeChunks` (time-tolerant token matching with
/// SentencePiece word-boundary splicing).
public actor UnifiedAsrManager {
    private let logger = AppLogger(category: "UnifiedOffline")

    private var preprocessor: MLModel?
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var jointDecision: MLModel?
    private var rnntDecoder: UnifiedRnntDecoder?
    private var tokenizer: Tokenizer?

    private let audioConverter = AudioConverter()
    public let config: UnifiedConfig
    public let encoderPrecision: UnifiedEncoderPrecision
    private let layout: UnifiedBatchLayout

    // Buffered audio for the StreamingAsrManager conformance (batch-on-finish).
    private var bufferedSamples: [Float] = []
    private var lastTranscript: String = ""
    private var partialCallback: (@Sendable (String) -> Void)?

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(
        configuration: MLModelConfiguration? = nil,
        config: UnifiedConfig = UnifiedConfig(),
        encoderPrecision: UnifiedEncoderPrecision = .int8
    ) {
        self.mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        self.config = config
        self.encoderPrecision = encoderPrecision
        self.layout = UnifiedBatchLayout(config: config)
    }

    // MARK: - Loading

    /// Load models from a directory containing the parakeet_unified_* bundles and vocab.json.
    public func loadModels(from directory: URL) async throws {
        logger.info("Loading Parakeet Unified offline CoreML models from \(directory.path)...")

        let names = ModelNames.ParakeetUnified.self
        // See StreamingUnifiedAsrManager: RangeDim preprocessor and per-token
        // decoder/joint stay on CPU; only the encoder uses ANE/GPU.
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        self.preprocessor = try await MLModel.load(
            contentsOf: directory.appendingPathComponent(names.preprocessorFile),
            configuration: cpuConfig
        )
        // int8 encoders must not route to the GPU: under `.all` CoreML sends
        // the quantized ops to MPSGraph, which fails its MLIR pass and
        // aborts ("MPSGraphExecutable.mm: Error: MLIR pass manager failed").
        // Coerce the known-bad int8 default to CPU+ANE; fp16 runs fine on the
        // GPU, so its `.all` choice is left untouched.
        let encoderConfig: MLModelConfiguration
        if encoderPrecision == .int8, mlConfiguration.computeUnits == .all {
            encoderConfig = MLModelConfiguration()
            encoderConfig.computeUnits = .cpuAndNeuralEngine
        } else {
            encoderConfig = mlConfiguration
        }
        self.encoder = try await MLModel.load(
            contentsOf: directory.appendingPathComponent(
                names.offlineEncoderFile(precision: encoderPrecision)),
            configuration: encoderConfig
        )
        self.decoder = try await MLModel.load(
            contentsOf: directory.appendingPathComponent(names.decoderFile),
            configuration: cpuConfig
        )
        self.jointDecision = try await MLModel.load(
            contentsOf: directory.appendingPathComponent(names.jointDecisionFile),
            configuration: cpuConfig
        )
        self.tokenizer = try Tokenizer(vocabPath: directory.appendingPathComponent(names.vocab))
        self.rnntDecoder = try UnifiedRnntDecoder(
            decoderModel: decoder!, jointDecisionModel: jointDecision!, config: config
        )

        logger.info("Parakeet Unified offline models loaded (15 s window, 2 s overlap).")
    }

    /// Download models from HuggingFace (if needed) and load them.
    /// Uses the "offline" variant set, which includes the full-attention
    /// 15 s encoder instead of the streaming one.
    public func loadModels(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        if let configuration {
            self.mlConfiguration = configuration
        }

        let repo = Repo.parakeetUnified
        let modelsBaseDir =
            directory
            ?? FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first!
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)

        let cacheDir = modelsBaseDir.appendingPathComponent(repo.folderName)
        let encoderPath = cacheDir.appendingPathComponent(
            ModelNames.ParakeetUnified.offlineEncoderFile(precision: encoderPrecision))

        if !FileManager.default.fileExists(atPath: encoderPath.path) {
            logger.info("Downloading Parakeet Unified offline models to \(modelsBaseDir.path)...")
            try await DownloadUtils.downloadRepo(
                repo, to: modelsBaseDir,
                variant: encoderPrecision == .fp16 ? "offline-fp16" : "offline",
                progressHandler: progressHandler)
        } else {
            logger.info("Using cached Parakeet Unified offline models at \(cacheDir.path)")
        }

        try await loadModels(from: cacheDir)
    }

    // MARK: - Batch API

    /// Transcribe 16 kHz mono samples of arbitrary length using overlapping
    /// 15 s windows.
    public func transcribe(_ samples: [Float]) async throws -> String {
        guard let tokenizer = tokenizer else { throw ASRError.notInitialized }

        var merged: [ChunkProcessor.TokenWindow] = []
        let merger = ChunkProcessor(audioSamples: [])
        let spliceSafeTokenIds = ChunkProcessor.spliceSafeTokenIds(vocabulary: tokenizer.vocabulary)

        for chunkStart in layout.chunkStarts(totalSamples: samples.count) {
            let chunkEnd = min(chunkStart + layout.chunkSamples, samples.count)
            let windowTokens = try await transcribeWindow(
                samples: samples, chunkStart: chunkStart, chunkEnd: chunkEnd
            )
            merged =
                merged.isEmpty
                ? windowTokens
                : merger.mergeChunks(merged, windowTokens, spliceSafeTokenIds: spliceSafeTokenIds)
        }

        merged.sort { $0.timestamp < $1.timestamp }
        return tokenizer.decode(ids: merged.map(\.token))
    }

    /// Transcribe an audio buffer (any format; resampled to 16 kHz mono).
    public func transcribe(_ buffer: AVAudioPCMBuffer) async throws -> String {
        let samples = try audioConverter.resampleBuffer(buffer)
        return try await transcribe(samples)
    }

    private func transcribeWindow(
        samples: [Float], chunkStart: Int, chunkEnd: Int
    ) async throws -> [ChunkProcessor.TokenWindow] {
        guard let preprocessor = preprocessor, let encoder = encoder, let rnntDecoder = rnntDecoder
        else {
            throw ASRError.notInitialized
        }

        let validCount = chunkEnd - chunkStart
        let window = try MLMultiArray(
            shape: [1, NSNumber(value: layout.windowSamples)], dataType: .float32
        )
        window.reset(to: 0)
        window.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            samples.withUnsafeBufferPointer { src in
                ptr.baseAddress!.update(from: src.baseAddress! + chunkStart, count: validCount)
            }
        }
        let audioLength = try MLMultiArray(shape: [1], dataType: .int32)
        audioLength[0] = NSNumber(value: validCount)

        let preprocOutput = try await preprocessor.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "audio_signal": MLFeatureValue(multiArray: window),
                "audio_length": MLFeatureValue(multiArray: audioLength),
            ])
        )
        guard let mel = preprocOutput.featureValue(for: "mel")?.multiArrayValue,
            let melLength = preprocOutput.featureValue(for: "mel_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Unified preprocessor failed to produce mel output")
        }

        let encoderOutput = try await encoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: mel),
                "mel_length": MLFeatureValue(multiArray: melLength),
            ])
        )
        guard let encoded = encoderOutput.featureValue(for: "encoder")?.multiArrayValue,
            let encodedLength = encoderOutput.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Unified encoder failed to produce output")
        }

        // Each window decodes independently from a fresh RNNT state; the
        // overlap merge reconciles the seams (same design as the TDT path).
        try rnntDecoder.reset()
        let encoderLength = min(encodedLength[0].intValue, encoded.shape[2].intValue)
        let emissions = try rnntDecoder.decode(
            encoded: encoded,
            frameRange: 0..<encoderLength,
            globalFrameOffset: chunkStart / config.frameSamples
        )
        return emissions.map { (token: $0.token, timestamp: $0.frame, confidence: $0.prob, duration: 0) }
    }

    // MARK: - Reset / Cleanup

    public func reset() async throws {
        bufferedSamples.removeAll()
        lastTranscript = ""
        try rnntDecoder?.reset()
    }

    public func cleanup() async {
        try? await reset()
        preprocessor = nil
        encoder = nil
        decoder = nil
        jointDecision = nil
        rnntDecoder = nil
        tokenizer = nil
        logger.info("UnifiedAsrManager resources cleaned up")
    }
}

// MARK: - StreamingAsrManager Conformance (batch-on-finish)

/// Conformance so the offline batch path is reachable through the same
/// engine-variant plumbing (CLI `--parakeet-variant parakeet-unified-offline-15s`).
/// Audio is buffered as it arrives and transcribed in one overlapping-window
/// batch at `finish()` — there are no incremental partial results.
extension UnifiedAsrManager: StreamingAsrManager {
    public var displayName: String {
        "Parakeet Unified 0.6B (offline 15s batch)"
    }

    public func loadModels() async throws {
        try await loadModels(to: nil, configuration: nil, progressHandler: nil)
    }

    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        let converted = try audioConverter.resampleBuffer(buffer)
        bufferedSamples.append(contentsOf: converted)
    }

    public func processBufferedAudio() async throws {
        // Batch engine: all decoding happens in finish().
    }

    public func finish() async throws -> String {
        let transcript = try await transcribe(bufferedSamples)
        bufferedSamples.removeAll()
        lastTranscript = transcript
        partialCallback?(transcript)
        return transcript
    }

    public func getPartialTranscript() -> String {
        lastTranscript
    }

    public func setPartialTranscriptCallback(_ callback: @escaping @Sendable (String) -> Void) {
        self.partialCallback = callback
    }
}
