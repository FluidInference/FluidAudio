#if TTS
@preconcurrency import CoreML
import Foundation

/// A streaming session over a small Pocket TTS model: enqueue text, receive
/// 80 ms `AudioFrame`s. Text is chunked with the shared Pocket TTS chunker;
/// every chunk re-injects the voice KV snapshot, prefills the text once and
/// then generates one frame per engine call (`generate` on the ANE + the fp32
/// Mimi decoder on the CPU by default, or one fused `generate_decode` dispatch).
@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
public actor PocketTtsSmallSession {

    private static let logger = AppLogger(category: "PocketTtsSmallSession")

    public nonisolated let frames: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>

    public nonisolated func enqueue(_ text: String) { textContinuation.yield(text) }
    public nonisolated func finish() { textContinuation.finish() }

    public func cancel() async {
        generationTask?.cancel()
        textContinuation.finish()
        await generationTask?.value
    }

    private nonisolated let textContinuation: AsyncStream<String>.Continuation
    private let textStream: AsyncStream<String>
    private let frameContinuation: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>.Continuation
    private var generationTask: Task<Void, Never>?

    private let engine: PocketTtsSmallEngine
    private let constants: PocketTtsSmallConstants
    private let voiceSnapshot: PocketTtsSmallEngine.Fp16Snapshot
    private let voicePosition: Int
    private let temperature: Float
    private var rng: SeededRNG
    /// Text normalization applied to each enqueued text before chunking (Phonon:
    /// Gradium's English normalizer, numbers / currency / emails / URLs / symbols).
    /// nil keeps the kyutai behaviour (character replacements only).
    private let textNormalizer: (@Sendable (String) -> String)?

    init(
        engine: PocketTtsSmallEngine,
        constants: PocketTtsSmallConstants,
        voiceSnapshot: PocketTtsSmallEngine.Fp16Snapshot,
        temperature: Float?,
        seed: UInt64,
        textNormalizer: (@Sendable (String) -> String)? = nil
    ) {
        self.textNormalizer = textNormalizer
        self.engine = engine
        self.constants = constants
        self.voiceSnapshot = voiceSnapshot
        self.voicePosition = Int(voiceSnapshot.position)
        self.temperature = temperature ?? constants.arch.temperature
        self.rng = SeededRNG(seed: seed)
        let (textStream, textContinuation) = AsyncStream.makeStream(of: String.self)
        self.textStream = textStream
        self.textContinuation = textContinuation
        let (frames, frameContinuation) = AsyncThrowingStream.makeStream(
            of: PocketTtsSynthesizer.AudioFrame.self)
        self.frames = frames
        self.frameContinuation = frameContinuation
    }

    func start() {
        // Strong capture on purpose: one-shot callers keep only `frames`, so
        // the session object may be released before this task body runs. A
        // weak capture then exits without finishing the stream and the
        // consumer waits forever. The task holds the session until the text
        // stream is finished or `cancel()` is called.
        generationTask = Task {
            await self.generateLoop()
        }
        frameContinuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task { await self.cancel() }
        }
    }

    /// Upstream replaces characters absent from the training data before
    /// tokenizing; the small models add the dashes.
    static func replaceUntrainedCharacters(_ text: String) -> String {
        text.replacingOccurrences(of: "\u{2013}", with: ",")
            .replacingOccurrences(of: "\u{2014}", with: ",")
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
    }

    private func generateLoop() async {
        var utteranceIndex = 0
        do {
            for await raw in textStream {
                if Task.isCancelled { break }
                let normalized = textNormalizer.map { $0(raw) } ?? Self.replaceUntrainedCharacters(raw)
                let text = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                let cacheLen = constants.state.kvCacheLen
                let maxTokens = try PocketTtsSynthesizer.effectiveMaxTokensPerChunk(
                    requested: PocketTtsConstants.maxTokensPerChunk, voiceCachePosition: voicePosition,
                    cacheLen: cacheLen)
                let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
                    text, tokenizer: constants.tokenizer, maxTokens: maxTokens,
                    preferredMaxTokens: min(PocketTtsConstants.preferredTokensPerChunk, maxTokens),
                    voiceCachePosition: voicePosition, language: .english, cacheLen: cacheLen)
                Self.logger.info("Small session enqueued '\(text)', \(chunks.count) chunk(s)")
                // Fresh Mimi decoder per utterance (upstream semantics).
                try await engine.beginUtterance()
                for (chunkIndex, chunk) in chunks.enumerated() {
                    if Task.isCancelled { break }
                    try await generateChunk(
                        text: chunk.text, isMidSentence: chunk.isMidSentence,
                        chunkIndex: chunkIndex, chunkCount: chunks.count, utteranceIndex: utteranceIndex)
                }
                utteranceIndex += 1
            }
            frameContinuation.finish()
        } catch {
            frameContinuation.finish(throwing: error)
        }
    }

    private func generateChunk(
        text: String, isMidSentence: Bool, chunkIndex: Int, chunkCount: Int, utteranceIndex: Int
    ) async throws {
        let (normalized, framesAfterEos) = PocketTtsSynthesizer.normalizeText(
            text, isMidSentence: isMidSentence, language: .english)
        let tokenIds = constants.tokenizer.encode(normalized)
        let flat = constants.embed(tokenIds)

        try await engine.reset(with: voiceSnapshot)
        try await engine.prefill(flatConditioning: flat, tokenCount: tokenIds.count)
        let prefilled = Int(await engine.position)
        Self.logger.info("Small chunk \(chunkIndex): '\(normalized)' (\(tokenIds.count) tokens, KV at \(prefilled))")

        let cacheLen = constants.state.kvCacheLen
        let maxGenLen = PocketTtsSynthesizer.boundedGenerationFrameCount(
            text: normalized, cachePosition: prefilled, cacheLen: cacheLen)
        let arch = constants.arch
        let totalFramesAfterEos = framesAfterEos + arch.framesAfterEosExtra
        let fadeRamp = arch.fadeInRamp
        var sequence = constants.bosEmbedding
        var eosStep: Int?
        let scale = sqrtf(temperature)

        for step in 0..<maxGenLen {
            if Task.isCancelled { break }
            var localRng = rng
            var noise = [Float](repeating: 0, count: arch.ldim)
            for i in 0..<arch.ldim {
                noise[i] = Float.gaussianRandom(using: &localRng) * scale
            }
            rng = localRng

            let frame = try await engine.generateDecodeFrame(sequence: sequence, noise: noise)

            if eosStep == nil, step >= arch.minFramesBeforeEos, frame.eosLogit > arch.eosThreshold {
                eosStep = step
                Self.logger.info("Small chunk \(chunkIndex) EOS at step \(step)")
            }
            if let eos = eosStep, step >= eos + totalFramesAfterEos {
                break
            }
            var samples = frame.samples
            if step == 0, !fadeRamp.isEmpty {
                // every chunk starts a fresh Mimi state; fade its cold-start click
                for i in 0..<min(fadeRamp.count, samples.count) { samples[i] *= fadeRamp[i] }
            }
            frameContinuation.yield(
                PocketTtsSynthesizer.AudioFrame(
                    samples: samples, frameIndex: step, chunkIndex: chunkIndex,
                    chunkCount: chunkCount, utteranceIndex: utteranceIndex))
            sequence = frame.latent
        }

        if !Task.isCancelled {
            // Same contract as the 100M session: hitting the cache boundary
            // without EOS is an error, not a silently truncated utterance.
            try PocketTtsSynthesizer.validateGenerationCompleted(
                generatedFrameLimit: maxGenLen, cachePosition: prefilled, eosStep: eosStep,
                framesAfterEos: totalFramesAfterEos, cacheLen: cacheLen)
            if eosStep == nil {
                Self.logger.warning(
                    "Small chunk \(chunkIndex) reached the frame limit (\(maxGenLen)) without EOS")
            }
        }
    }
}
#endif
