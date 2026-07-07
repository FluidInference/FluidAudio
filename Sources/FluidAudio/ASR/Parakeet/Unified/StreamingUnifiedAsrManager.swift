import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Streaming ASR manager for Parakeet Unified 0.6B (FastConformer-RNNT).
///
/// Unlike the cache-aware engines (EOU, Nemotron), the unified model's encoder
/// is stateless: each step re-encodes a `[left | chunk | right]` audio window
/// whose chunked attention mask was baked in at conversion time. Only the
/// RNNT decoder LSTM state and the last emitted token persist across chunks,
/// so the streamed transcript matches the model's offline output closely
/// (word-for-word on validation audio).
///
/// Default context [70, 13, 13] encoder frames = 5.6 s left / 1.04 s chunk /
/// 1.04 s right → 2.08 s theoretical latency.
public actor StreamingUnifiedAsrManager {
    private let logger = AppLogger(category: "UnifiedStreaming")

    // Models
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var jointDecision: MLModel?

    // Log-mel features are computed natively in Swift (`AudioMelSpectrogram` +
    // NeMo per_feature normalization); the model ships no CoreML preprocessor.
    // WER-equivalent to the original traced preprocessor on test-clean, ~17-21%
    // faster, and avoids the forced-CPU RangeDim mel stage.
    private var swiftMel: UnifiedMelExtractor?

    // Components
    private let audioConverter = AudioConverter()
    private var tokenizer: Tokenizer?

    public let config: UnifiedConfig
    public let encoderPrecision: UnifiedEncoderPrecision

    // Rolling audio storage. `samples[0]` corresponds to global sample index
    // `samplesGlobalStart`; audio older than one window behind the consumed
    // position is trimmed.
    private var samples: [Float] = []
    private var samplesGlobalStart: Int = 0
    private var windower: UnifiedStreamingWindower

    // Greedy RNNT loop; its LSTM state persists across chunks.
    private var rnntDecoder: UnifiedRnntDecoder?

    // Accumulated token IDs and the incrementally built transcript.
    // The transcript is appended per chunk instead of re-decoding the full
    // token history (which would be O(n^2) over a long session — this
    // engine is intended to run for hours).
    private var accumulatedTokenIds: [Int] = []
    private var transcriptCache: String = ""
    // Per-token timings (start/end in seconds) since the last `consumeTokenTimings()`.
    // The greedy RNNT decoder already reports each emission's global encoder frame;
    // surfacing it lets downstream consumers do word→speaker attribution without
    // re-decoding. RNNT tokens are emitted AT a frame (no intrinsic duration), so
    // each token's `endTime` is back-filled to the next token's start; the frontier
    // token gets a provisional one-frame end until the next emission arrives.
    // Drained by `consumeTokenTimings()` so it stays bounded over hour-long streams.
    private var pendingTokenTimings: [TokenTiming] = []

    private var partialCallback: (@Sendable (String) -> Void)?
    private var processedChunks: Int = 0

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(
        configuration: MLModelConfiguration? = nil,
        config: UnifiedConfig = UnifiedConfig(),
        encoderPrecision: UnifiedEncoderPrecision = .int8
    ) {
        self.mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        self.config = config
        self.encoderPrecision = encoderPrecision
        self.windower = UnifiedStreamingWindower(config: config)
    }

    // MARK: - Loading

    /// Load models from a directory containing the parakeet_unified_* bundles and vocab.json.
    public func loadModels(from directory: URL) async throws {
        logger.info("Loading Parakeet Unified CoreML models from \(directory.path)...")

        let names = ModelNames.ParakeetUnified.self
        // Decoder/joint run tiny per-token steps that stay on CPU; only the
        // encoder benefits from ANE/GPU. Mel is computed in Swift (no CoreML
        // preprocessor bundle).
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
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
                names.streamingEncoderFile(precision: encoderPrecision, contextSuffix: config.contextSuffix)),
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
        self.swiftMel = UnifiedMelExtractor(windowSamples: config.windowSamples, nMels: config.melFeatures)

        logger.info("Parakeet Unified models loaded (latency \(config.latencyMs)ms).")
    }

    /// Download models from HuggingFace (if needed) and load them.
    public func loadModels(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        progressHandler: ProgressHandler? = nil
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
        // Each [L,C,R] tier is a distinct encoder bundle (the attention mask is
        // baked in at conversion), so the cache check and download both target
        // this config's specific encoder rather than the default 70_13_13.
        let encoderFile = ModelNames.ParakeetUnified.streamingEncoderFile(
            precision: encoderPrecision, contextSuffix: config.contextSuffix)
        let encoderPath = cacheDir.appendingPathComponent(encoderFile)

        if !FileManager.default.fileExists(atPath: encoderPath.path) {
            logger.info("Downloading Parakeet Unified models to \(modelsBaseDir.path)...")
            try await ModelHub.download(
                repo, to: modelsBaseDir,
                variant: encoderPrecision == .fp16 ? "fp16" : nil,
                additionalModelNames: [encoderFile],
                progressHandler: progressHandler)
        } else {
            logger.info("Using cached Parakeet Unified models at \(cacheDir.path)")
        }

        try await loadModels(from: cacheDir)
    }

    // MARK: - Streaming API

    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        let converted = try audioConverter.resampleBuffer(buffer)
        samples.append(contentsOf: converted)
    }

    /// Process as many complete chunks as the buffered audio allows.
    public func processBufferedAudio() async throws {
        try await processAvailableWindows(isFinal: false)
    }

    /// Flush remaining audio and return the final transcript.
    public func finish() async throws -> String {
        guard tokenizer != nil else { throw ASRError.notInitialized }
        try await processAvailableWindows(isFinal: true)
        return currentTranscript()
    }

    public func getPartialTranscript() -> String {
        currentTranscript()
    }

    /// Returns the per-token timings (start/end in seconds) accumulated since the
    /// previous call and clears them, so the buffer stays bounded over long
    /// streams. Call after `processBufferedAudio()` / `finish()` to drain the
    /// timings for the audio decoded so far. Each `TokenTiming` carries the global
    /// encoder frame the RNNT decoder records per emission, for downstream
    /// word→speaker attribution. The final token of a drained batch may carry a
    /// provisional one-frame `endTime` (the next emission's start is not yet known).
    public func consumeTokenTimings() -> [TokenTiming] {
        defer { pendingTokenTimings.removeAll(keepingCapacity: true) }
        return pendingTokenTimings
    }

    /// Word-level timings since the previous call, draining the same buffer as
    /// `consumeTokenTimings()` (call one or the other per cycle). Sub-word tokens
    /// are grouped on their `▁` / leading-space boundaries; each word spans its
    /// first sub-word's start to its last sub-word's end. Useful for streaming
    /// diarized ASR (word→speaker attribution).
    public func consumeWordTimings() -> [WordTiming] {
        buildWordTimings(from: consumeTokenTimings())
    }

    public func reset() async throws {
        samples.removeAll()
        samplesGlobalStart = 0
        windower.reset()
        accumulatedTokenIds.removeAll()
        transcriptCache = ""
        pendingTokenTimings.removeAll()
        processedChunks = 0
        try rnntDecoder?.reset()
    }

    public func cleanup() async {
        try? await reset()
        encoder = nil
        decoder = nil
        jointDecision = nil
        rnntDecoder = nil
        tokenizer = nil
        swiftMel = nil
        logger.info("StreamingUnifiedAsrManager resources cleaned up")
    }

    // MARK: - Pipeline

    private func processAvailableWindows(isFinal: Bool) async throws {
        guard swiftMel != nil, encoder != nil, decoder != nil, jointDecision != nil else {
            throw ASRError.notInitialized
        }

        while let plan = windower.nextWindow(
            totalSamples: samplesGlobalStart + samples.count, isFinal: isFinal
        ) {
            try await processWindow(plan)
            trimSamples()
        }
    }

    private func processWindow(_ plan: UnifiedStreamingWindower.WindowPlan) async throws {
        guard let swiftMel = swiftMel, let encoder = encoder else {
            throw ASRError.notInitialized
        }

        // 1. Assemble the zero-padded encoder window from the rolling buffer.
        let localStart = plan.bufferStart - samplesGlobalStart
        let localEnd = plan.bufferEnd - samplesGlobalStart
        guard localStart >= 0, localEnd <= samples.count else {
            throw ASRError.processingFailed("Streaming window out of range (trimmed too aggressively)")
        }
        let validCount = localEnd - localStart

        // 2. Window → mel (native Swift `AudioMelSpectrogram` + per_feature norm).
        var buffer = [Float](repeating: 0, count: config.windowSamples)
        samples.withUnsafeBufferPointer { src in
            buffer.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: src.baseAddress! + localStart, count: validCount)
            }
        }
        let (mel, melLength) = try swiftMel.features(window: buffer, validCount: validCount)

        // 3. Streaming encoder (chunked attention mask baked in)
        let encoderOutput = try await encoder.prediction(
            from: UnifiedEncoderFeatureProvider(mel: mel, melLength: melLength)
        )
        guard let encoded = encoderOutput.featureValue(for: "encoder")?.multiArrayValue,
            let encodedLength = encoderOutput.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Unified encoder failed to produce output")
        }

        // 4. Greedy RNNT decode over the new frames only.
        let encoderLength = min(encodedLength[0].intValue, encoded.shape[2].intValue)
        guard let range = windower.decodeRange(encoderLength: encoderLength, plan: plan),
            let rnntDecoder = rnntDecoder
        else {
            processedChunks += 1
            return
        }
        let emissions = try rnntDecoder.decode(
            encoded: encoded, frameRange: range, globalFrameOffset: plan.bufferStartFrame
        )
        accumulatedTokenIds.append(contentsOf: emissions.map(\.token))
        if let tokenizer = tokenizer {
            let secondsPerFrame = Double(config.frameSamples) / Double(config.sampleRate)
            for emission in emissions {
                guard let piece = tokenizer.piece(forId: emission.token) else { continue }
                let text = piece.replacingOccurrences(of: "\u{2581}", with: " ")
                transcriptCache += text
                let start = Double(emission.frame) * secondsPerFrame
                // RNNT tokens have no intrinsic duration — back-fill the previous
                // token's end to this token's start so durations reflect real gaps.
                if let last = pendingTokenTimings.indices.last, pendingTokenTimings[last].endTime > start {
                    let prev = pendingTokenTimings[last]
                    pendingTokenTimings[last] = TokenTiming(
                        token: prev.token, tokenId: prev.tokenId,
                        startTime: prev.startTime, endTime: max(prev.startTime, start),
                        confidence: prev.confidence
                    )
                }
                // Frontier token: provisional one-frame end until the next emission.
                pendingTokenTimings.append(
                    TokenTiming(
                        token: text,
                        tokenId: emission.token,
                        startTime: start,
                        endTime: start + secondsPerFrame,
                        confidence: emission.prob
                    )
                )
            }
        }
        processedChunks += 1

        if !emissions.isEmpty, let callback = partialCallback {
            callback(currentTranscript())
        }
    }

    private func currentTranscript() -> String {
        transcriptCache.trimmingCharacters(in: .whitespaces)
    }

    /// Drop audio that can no longer appear in any future window.
    private func trimSamples() {
        let keepFrom = windower.consumedSamples - config.windowSamples
        guard keepFrom > samplesGlobalStart else { return }
        let dropCount = keepFrom - samplesGlobalStart
        guard dropCount > 0, dropCount <= samples.count else { return }
        samples.removeFirst(dropCount)
        samplesGlobalStart = keepFrom
    }
}

// MARK: - StreamingAsrManager Conformance

extension StreamingUnifiedAsrManager: StreamingAsrManager {
    public var displayName: String {
        "Parakeet Unified 0.6B (\(config.latencyMs)ms)"
    }

    public func loadModels() async throws {
        try await loadModels(to: nil, configuration: nil, progressHandler: nil)
    }

    public func setPartialTranscriptCallback(_ callback: @escaping @Sendable (String) -> Void) {
        self.partialCallback = callback
    }
}
