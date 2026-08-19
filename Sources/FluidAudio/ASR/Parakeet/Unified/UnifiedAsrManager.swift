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

    private var encoder: MLModel?
    private var decoder: MLModel?
    private var jointDecision: MLModel?
    private var rnntDecoder: UnifiedRnntDecoder?
    private var tokenizer: Tokenizer?

    // Log-mel features are computed natively in Swift (`AudioMelSpectrogram` +
    // NeMo per_feature normalization); the model ships no CoreML preprocessor.
    private var swiftMel: UnifiedMelExtractor?

    private let audioConverter = AudioConverter()
    public let config: UnifiedConfig
    public let encoderPrecision: UnifiedEncoderPrecision
    private let layout: UnifiedBatchLayout

    // Buffered audio for the StreamingAsrManager conformance (batch-on-finish).
    private var bufferedSamples: [Float] = []
    private var lastTranscript: String = ""
    private var partialCallback: (@Sendable (String) -> Void)?

    // Vocabulary boosting (issue #851): configured via
    // configureVocabularyBoosting() before transcription.
    private var vocabularyBoosting: VocabularyBoostingSession?

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
        // See StreamingUnifiedAsrManager: per-token decoder/joint stay on CPU;
        // only the encoder uses ANE/GPU. Mel is computed in Swift.
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
        do {
            self.encoder = try await MLModel.load(
                contentsOf: directory.appendingPathComponent(
                    names.offlineEncoderFile(precision: encoderPrecision)),
                configuration: encoderConfig
            )
        } catch {
            if encoderPrecision == .int8 {
                // Same A-series caveat as the streaming manager (issue #828):
                // the int8 encoder can fail on every compute unit from an
                // intact download; name the fp16 escape hatch here.
                logger.error(
                    "int8 unified encoder failed to load. On some A-series chips (A16 verified) it cannot build an execution plan on any compute unit even from an intact download — retry with encoderPrecision: .fp16 (issue #828). Underlying error: \(error.localizedDescription)"
                )
            }
            throw error
        }
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
        self.swiftMel = UnifiedMelExtractor(windowSamples: layout.windowSamples, nMels: config.melFeatures)

        logger.info("Parakeet Unified offline models loaded (15 s window, 2 s overlap).")
    }

    /// Download models from HuggingFace (if needed) and load them.
    /// Uses the "offline" variant set, which includes the full-attention
    /// 15 s encoder instead of the streaming one.
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

        // Completeness-checked download + purge-and-retry on load failure: a
        // bare directory-existence gate mistook an interrupted encoder fetch
        // for a warm cache and bricked loading permanently (issue #819).
        try await ModelHub.loadWithRecovery(
            repo, directory: modelsBaseDir,
            requiredFiles: [
                ModelNames.ParakeetUnified.offlineEncoderFile(precision: encoderPrecision),
                ModelNames.ParakeetUnified.decoderFile,
                ModelNames.ParakeetUnified.jointDecisionFile,
                ModelNames.ParakeetUnified.vocab,
            ],
            variant: encoderPrecision == .fp16 ? "offline-fp16" : "offline",
            progressHandler: progressHandler
        ) {
            try await self.loadModels(from: cacheDir)
        }
    }

    // MARK: - Vocabulary Boosting

    /// Configure vocabulary boosting for batch transcription.
    ///
    /// When configured, the final merged transcript of each `transcribe` call
    /// is rescored against CTC acoustic evidence: a separate CTC model runs
    /// over the same audio and vocabulary terms replace misrecognized words
    /// where the acoustics support it. Same pipeline as
    /// `SlidingWindowAsrManager.configureVocabularyBoosting`.
    ///
    /// - Parameters:
    ///   - vocabulary: Custom vocabulary context with terms to detect
    ///     (tokenize via `CustomVocabularyContext.loadWithCtcTokens`)
    ///   - ctcModels: Pre-loaded CTC models for keyword spotting
    ///   - config: Optional rescorer configuration (default:
    ///     `VocabularyBoostingSession.itnDefaultConfig` — this engine's ITN
    ///     output needs the #702 spotter-rescue similarity floors)
    /// - Throws: Error if rescorer initialization fails
    public func configureVocabularyBoosting(
        vocabulary: CustomVocabularyContext,
        ctcModels: CtcModels,
        config: VocabularyRescorer.Config? = nil
    ) async throws {
        self.vocabularyBoosting = try await VocabularyBoostingSession(
            vocabulary: vocabulary, ctcModels: ctcModels,
            config: config ?? VocabularyBoostingSession.itnDefaultConfig
        )
        logger.info("Vocabulary boosting configured with \(vocabulary.terms.count) terms")
    }

    // MARK: - Batch API

    /// A transcript and the per-token timings behind it.
    public struct TranscriptionWithTimings: Sendable {
        public let text: String
        public let tokenTimings: [TokenTiming]

        public init(text: String, tokenTimings: [TokenTiming]) {
            self.text = text
            self.tokenTimings = tokenTimings
        }
    }

    /// Transcribe 16 kHz mono samples of arbitrary length using overlapping
    /// 15 s windows.
    public func transcribe(_ samples: [Float]) async throws -> String {
        guard let tokenizer = tokenizer else { throw ASRError.notInitialized }
        let merged = try await decodedTokens(samples, tokenizer: tokenizer)
        let text = tokenizer.decode(ids: merged.map(\.token))
        return await rescoreIfConfigured(text: text, merged: merged, samples: samples)
    }

    /// Transcribe as `transcribe(_:)` does, additionally reporting the encoder
    /// frame each token was emitted at, converted to seconds.
    ///
    /// The offline path already carries these frames — the greedy RNNT decoder
    /// records one per emission and the overlap merge preserves them — so this
    /// costs nothing beyond the conversion. It is the batch counterpart to
    /// `StreamingUnifiedAsrManager.consumeTokenTimings()`, and its output feeds
    /// `buildWordTimings(from:)` the same way, for callers that need to align
    /// the transcript back to the audio (seeking, playback highlighting,
    /// word→speaker attribution).
    ///
    /// As in the streaming manager, RNNT tokens are emitted *at* a frame and
    /// have no intrinsic duration, so every token gets a provisional one-frame
    /// end that is clamped back only when it would overrun its successor. The
    /// spans are therefore not contiguous: a real pause stays visible as a gap
    /// between one token's end and the next one's start. The last token keeps
    /// its provisional end, clamped to the end of the clip. These are emission
    /// times rather than forced-alignment boundaries: the decoder emits once it
    /// has heard enough context, so a token's start can sit slightly after the
    /// word's true onset.
    public func transcribeWithTimings(_ samples: [Float]) async throws -> TranscriptionWithTimings {
        guard let tokenizer = tokenizer else { throw ASRError.notInitialized }
        let merged = try await decodedTokens(samples, tokenizer: tokenizer)
        let timings = Self.tokenTimings(
            from: merged,
            secondsPerFrame: Double(config.frameSamples) / Double(config.sampleRate),
            vocabulary: tokenizer.vocabulary,
            clipDuration: Double(samples.count) / Double(config.sampleRate)
        )
        var text = tokenizer.decode(ids: merged.map(\.token))
        // Rescored text can replace words, so token timings no longer decode
        // to the text verbatim; they remain the raw emissions, which is what
        // timing consumers (seek, attribution) want.
        if let boosting = vocabularyBoosting,
            let rescored = await boosting.rescore(text: text, tokenTimings: timings, audioSamples: samples)
        {
            text = rescored.text
        }
        return TranscriptionWithTimings(text: text, tokenTimings: timings)
    }

    /// Apply vocabulary rescoring to a finished transcript when boosting is
    /// configured; otherwise return the transcript unchanged.
    private func rescoreIfConfigured(
        text: String, merged: [ChunkProcessor.TokenWindow], samples: [Float]
    ) async -> String {
        guard let boosting = vocabularyBoosting, let tokenizer = tokenizer else { return text }
        let timings = Self.tokenTimings(
            from: merged,
            secondsPerFrame: Double(config.frameSamples) / Double(config.sampleRate),
            vocabulary: tokenizer.vocabulary,
            clipDuration: Double(samples.count) / Double(config.sampleRate)
        )
        let rescored = await boosting.rescore(text: text, tokenTimings: timings, audioSamples: samples)
        return rescored?.text ?? text
    }

    /// Emission frames → seconds. Pure, so the back-fill rule can be tested
    /// without loading a 600M parameter model.
    ///
    /// `clipDuration` bounds the last token's provisional end, which the batch
    /// path can do because it knows the sample count up front and the streaming
    /// one cannot. Pass `nil` to leave it unbounded.
    static func tokenTimings(
        from emissions: [ChunkProcessor.TokenWindow],
        secondsPerFrame: Double,
        vocabulary: [Int: String],
        clipDuration: Double? = nil
    ) -> [TokenTiming] {
        var timings: [TokenTiming] = []
        timings.reserveCapacity(emissions.count)
        for emission in emissions {
            guard let piece = vocabulary[emission.token] else { continue }
            let start = Double(emission.timestamp) * secondsPerFrame
            // RNNT tokens have no intrinsic duration — back-fill the previous
            // token's end to this token's start so durations reflect real gaps.
            if let last = timings.indices.last, timings[last].endTime > start {
                let previous = timings[last]
                timings[last] = TokenTiming(
                    token: previous.token, tokenId: previous.tokenId,
                    startTime: previous.startTime, endTime: max(previous.startTime, start),
                    confidence: previous.confidence
                )
            }
            // Frontier token: provisional one-frame end, as in the streaming manager.
            timings.append(
                TokenTiming(
                    token: piece.replacingOccurrences(of: "\u{2581}", with: " "),
                    tokenId: emission.token,
                    startTime: start,
                    endTime: start + secondsPerFrame,
                    confidence: emission.confidence
                )
            )
        }
        // The frontier token's one-frame end is a guess; offline knows where the
        // audio actually stops, so don't hand callers a seek target past EOF.
        if let clipDuration, let last = timings.indices.last, timings[last].endTime > clipDuration {
            let previous = timings[last]
            timings[last] = TokenTiming(
                token: previous.token, tokenId: previous.tokenId,
                startTime: previous.startTime, endTime: max(previous.startTime, clipDuration),
                confidence: previous.confidence
            )
        }
        return timings
    }

    /// The merged, seam-collapsed token stream for the whole recording, in
    /// emission order. Shared by both batch entry points so the text they
    /// return cannot drift apart.
    private func decodedTokens(
        _ samples: [Float], tokenizer: Tokenizer
    ) async throws -> [ChunkProcessor.TokenWindow] {
        var merged: [ChunkProcessor.TokenWindow] = []
        // Reuse the TDT overlap merger to dedupe adjacent windows.
        let merger = ChunkProcessor(audioSamples: samples)
        let spliceSafeTokenIds = ChunkProcessor.spliceSafeTokenIds(vocabulary: tokenizer.vocabulary)
        let caseVariantIds = ChunkProcessor.caseVariantCanonicalIds(vocabulary: tokenizer.vocabulary)

        // Fixed-stride 15 s / 2 s grid. Silence-aligned starts were measured to
        // cost ~1 WER point on the 15 s offline encoder (Earnings-22 long-form)
        // with no artifact benefit, so the seam artifacts (#706) are handled
        // purely by the merge: case-folded matching + word-level collapse below.
        for chunkStart in layout.chunkStarts(totalSamples: samples.count) {
            let chunkEnd = min(chunkStart + layout.chunkSamples, samples.count)
            let windowTokens = try await transcribeWindow(
                samples: samples, chunkStart: chunkStart, chunkEnd: chunkEnd
            )
            merged =
                merged.isEmpty
                ? windowTokens
                : merger.mergeChunks(
                    merged,
                    windowTokens,
                    spliceSafeTokenIds: spliceSafeTokenIds,
                    caseVariantIds: caseVariantIds
                )
        }

        merged.sort { $0.timestamp < $1.timestamp }
        return merger.collapseSeamWordDuplicates(merged, vocabulary: tokenizer.vocabulary)
    }

    /// Transcribe an audio buffer (any format; resampled to 16 kHz mono).
    public func transcribe(_ buffer: AVAudioPCMBuffer) async throws -> String {
        let samples = try audioConverter.resampleBuffer(buffer)
        return try await transcribe(samples)
    }

    private func transcribeWindow(
        samples: [Float], chunkStart: Int, chunkEnd: Int
    ) async throws -> [ChunkProcessor.TokenWindow] {
        guard let swiftMel = swiftMel, let encoder = encoder, let rnntDecoder = rnntDecoder
        else {
            throw ASRError.notInitialized
        }

        let validCount = chunkEnd - chunkStart

        // Window → mel (native Swift `AudioMelSpectrogram` + per_feature norm).
        var buffer = [Float](repeating: 0, count: layout.windowSamples)
        samples.withUnsafeBufferPointer { src in
            buffer.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: src.baseAddress! + chunkStart, count: validCount)
            }
        }
        let (mel, melLength) = try swiftMel.features(window: buffer, validCount: validCount)

        let encoderOutput = try await encoder.prediction(
            from: UnifiedEncoderFeatureProvider(mel: mel, melLength: melLength)
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
        encoder = nil
        decoder = nil
        jointDecision = nil
        rnntDecoder = nil
        swiftMel = nil
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
