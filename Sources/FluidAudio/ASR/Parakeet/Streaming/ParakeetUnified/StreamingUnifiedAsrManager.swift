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
    private var preprocessor: MLModel?
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var jointDecision: MLModel?

    // Components
    private let audioConverter = AudioConverter()
    private var tokenizer: Tokenizer?

    public let config: UnifiedStreamingConfig

    // Rolling audio storage. `samples[0]` corresponds to global sample index
    // `samplesGlobalStart`; audio older than one window behind the consumed
    // position is trimmed.
    private var samples: [Float] = []
    private var samplesGlobalStart: Int = 0
    private var windower: UnifiedStreamingWindower

    // RNNT decoder state (persists across chunks)
    private var hState: MLMultiArray?
    private var cState: MLMultiArray?
    private var lastToken: Int32

    // Accumulated token IDs
    private var accumulatedTokenIds: [Int] = []

    private var partialCallback: (@Sendable (String) -> Void)?
    private var processedChunks: Int = 0

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(
        configuration: MLModelConfiguration? = nil,
        config: UnifiedStreamingConfig = UnifiedStreamingConfig()
    ) {
        self.mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        self.config = config
        self.windower = UnifiedStreamingWindower(config: config)
        self.lastToken = Int32(config.blankIdx)
    }

    // MARK: - Loading

    /// Load models from a directory containing the parakeet_unified_* bundles and vocab.json.
    public func loadModels(from directory: URL) async throws {
        logger.info("Loading Parakeet Unified CoreML models from \(directory.path)...")

        let names = ModelNames.ParakeetUnified.self
        // Decoder/joint run tiny per-token steps and the variable-length
        // (RangeDim) preprocessor trips E5RT shape inference on the ANE —
        // all three stay on CPU. Only the encoder benefits from ANE/GPU.
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        self.preprocessor = try await MLModel.load(
            contentsOf: directory.appendingPathComponent(names.preprocessorFile),
            configuration: cpuConfig
        )
        self.encoder = try await MLModel.load(
            contentsOf: directory.appendingPathComponent(names.streamingEncoderFile),
            configuration: mlConfiguration
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

        try resetDecoderState()
        logger.info("Parakeet Unified models loaded (latency \(config.latencyMs)ms).")
    }

    /// Download models from HuggingFace (if needed) and load them.
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
        let encoderPath = cacheDir.appendingPathComponent(ModelNames.ParakeetUnified.streamingEncoderFile)

        if !FileManager.default.fileExists(atPath: encoderPath.path) {
            logger.info("Downloading Parakeet Unified models to \(modelsBaseDir.path)...")
            try await DownloadUtils.downloadRepo(repo, to: modelsBaseDir, progressHandler: progressHandler)
        } else {
            logger.info("Using cached Parakeet Unified models at \(cacheDir.path)")
        }

        try await loadModels(from: cacheDir)
    }

    // MARK: - State

    private func resetDecoderState() throws {
        hState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        cState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        hState?.reset(to: 0)
        cState?.reset(to: 0)
        lastToken = Int32(config.blankIdx)
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
        guard let tokenizer = tokenizer else { throw ASRError.notInitialized }
        try await processAvailableWindows(isFinal: true)
        return tokenizer.decode(ids: accumulatedTokenIds)
    }

    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        return tokenizer.decode(ids: accumulatedTokenIds)
    }

    public func reset() async throws {
        samples.removeAll()
        samplesGlobalStart = 0
        windower.reset()
        accumulatedTokenIds.removeAll()
        processedChunks = 0
        try resetDecoderState()
    }

    public func cleanup() async {
        try? await reset()
        preprocessor = nil
        encoder = nil
        decoder = nil
        jointDecision = nil
        tokenizer = nil
        logger.info("StreamingUnifiedAsrManager resources cleaned up")
    }

    // MARK: - Pipeline

    private func processAvailableWindows(isFinal: Bool) async throws {
        guard preprocessor != nil, encoder != nil, decoder != nil, jointDecision != nil else {
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
        guard let preprocessor = preprocessor, let encoder = encoder else {
            throw ASRError.notInitialized
        }

        // 1. Assemble the zero-padded encoder window from the rolling buffer.
        let window = try MLMultiArray(
            shape: [1, NSNumber(value: config.windowSamples)], dataType: .float32
        )
        window.reset(to: 0)
        let localStart = plan.bufferStart - samplesGlobalStart
        let localEnd = plan.bufferEnd - samplesGlobalStart
        guard localStart >= 0, localEnd <= samples.count else {
            throw ASRError.processingFailed("Streaming window out of range (trimmed too aggressively)")
        }
        let validCount = localEnd - localStart
        window.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            samples.withUnsafeBufferPointer { src in
                ptr.baseAddress!.update(from: src.baseAddress! + localStart, count: validCount)
            }
        }

        let audioLength = try MLMultiArray(shape: [1], dataType: .int32)
        audioLength[0] = NSNumber(value: validCount)

        // 2. Preprocessor: window → mel
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

        // 3. Streaming encoder (chunked attention mask baked in)
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

        // 4. Greedy RNNT decode over the new frames only.
        let encoderLength = min(encodedLength[0].intValue, encoded.shape[2].intValue)
        guard let range = windower.decodeRange(encoderLength: encoderLength, plan: plan) else {
            processedChunks += 1
            return
        }
        let newTokens = try await decodeFrames(encoded: encoded, range: range)
        processedChunks += 1

        if !newTokens.isEmpty, let callback = partialCallback, let tokenizer = tokenizer {
            callback(tokenizer.decode(ids: accumulatedTokenIds))
        }
    }

    private func decodeFrames(encoded: MLMultiArray, range: Range<Int>) async throws -> [Int] {
        guard let decoder = decoder, let jointDecision = jointDecision,
            var currentH = hState, var currentC = cState
        else {
            throw ASRError.notInitialized
        }
        var currentToken = lastToken
        var newTokens: [Int] = []

        // Decoder output for the current token, lazily refreshed after each emission.
        var decoderStep = try await runDecoder(
            decoder, token: currentToken, h: currentH, c: currentC
        )

        for t in range {
            let encStep = try extractEncoderStep(from: encoded, timeIndex: t)

            for _ in 0..<config.maxSymbolsPerFrame {
                let jointOutput = try await jointDecision.prediction(
                    from: MLDictionaryFeatureProvider(dictionary: [
                        "encoder_step": MLFeatureValue(multiArray: encStep),
                        "decoder_step": MLFeatureValue(multiArray: decoderStep.output),
                    ])
                )
                guard let tokenArray = jointOutput.featureValue(for: "token_id")?.multiArrayValue else {
                    throw ASRError.processingFailed("Unified joint decision missing token_id")
                }
                let tokenId = tokenArray[0].int32Value

                if tokenId == Int32(config.blankIdx) {
                    break
                }
                newTokens.append(Int(tokenId))
                accumulatedTokenIds.append(Int(tokenId))
                currentToken = tokenId
                currentH = decoderStep.h
                currentC = decoderStep.c
                decoderStep = try await runDecoder(
                    decoder, token: currentToken, h: currentH, c: currentC
                )
            }
        }

        // Persist decoder state atomically after the chunk completes.
        lastToken = currentToken
        hState = currentH
        cState = currentC
        return newTokens
    }

    private struct DecoderStep {
        let output: MLMultiArray
        let h: MLMultiArray
        let c: MLMultiArray
    }

    private func runDecoder(
        _ decoder: MLModel, token: Int32, h: MLMultiArray, c: MLMultiArray
    ) async throws -> DecoderStep {
        let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        targets[0] = NSNumber(value: token)
        let targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = 1

        let output = try await decoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "targets": MLFeatureValue(multiArray: targets),
                "target_length": MLFeatureValue(multiArray: targetLength),
                "h_in": MLFeatureValue(multiArray: h),
                "c_in": MLFeatureValue(multiArray: c),
            ])
        )
        guard let decoderOut = output.featureValue(for: "decoder")?.multiArrayValue,
            let hOut = output.featureValue(for: "h_out")?.multiArrayValue,
            let cOut = output.featureValue(for: "c_out")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Unified decoder failed")
        }
        // Decoder output is [1, 640, 1] (U=1 export); h/c become the state for
        // the NEXT decoder call only after this token is accepted.
        return DecoderStep(output: decoderOut, h: hOut, c: cOut)
    }

    private func extractEncoderStep(from encoded: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        let dim = encoded.shape[1].intValue
        let step = try MLMultiArray(shape: [1, NSNumber(value: dim), 1], dataType: .float32)

        let srcPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dstPtr = step.dataPointer.bindMemory(to: Float.self, capacity: step.count)
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue

        for c in 0..<dim {
            dstPtr[c] = srcPtr[c * stride1 + timeIndex * stride2]
        }
        return step
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
