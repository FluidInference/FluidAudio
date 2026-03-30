@preconcurrency import CoreML
import Foundation

/// A persistent TTS session that keeps the voice KV cache warm across utterances.
///
/// Creating a session performs the expensive voice prefill once (~125 tokens),
/// then each enqueued utterance only pays the text prefill cost. Mimi decoder
/// state persists across utterances for seamless audio continuity.
public actor PocketTtsSession {

    private static let logger = AppLogger(category: "PocketTtsSession")

    // MARK: - Public Interface

    /// Stream of generated audio frames (80ms / 1920 samples at 24kHz each).
    ///
    /// Frames are yielded as soon as they are generated. The stream completes
    /// after `finish()` is called and all enqueued text has been synthesized,
    /// or immediately if `cancel()` is called.
    public nonisolated let frames: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>

    /// Enqueue text for synthesis.
    ///
    /// Non-async and safe to call from any isolation context. Text is chunked
    /// internally if it exceeds the per-chunk token limit. Can be called
    /// multiple times to stream text as it arrives.
    public nonisolated func enqueue(_ text: String) {
        textContinuation.yield(text)
    }

    /// Signal that no more text will be enqueued.
    ///
    /// The session will finish generating all previously enqueued text,
    /// then complete the `frames` stream.
    public nonisolated func finish() {
        textContinuation.finish()
    }

    /// Cancel ongoing generation and finish the frames stream.
    ///
    /// Awaits until the generation task has fully stopped — after this returns,
    /// no more CoreML predictions are running and the Neural Engine is free.
    public func cancel() async {
        generationTask?.cancel()
        textContinuation.finish()
        await generationTask?.value
    }

    // MARK: - Internal State

    private nonisolated let textContinuation: AsyncStream<String>.Continuation
    private let textStream: AsyncStream<String>
    private let frameContinuation: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>.Continuation
    private var generationTask: Task<Void, Never>?

    // Models
    private let condModel: MLModel
    private let stepModel: MLModel
    private let flowModel: MLModel
    private let mimiModel: MLModel

    // Persistent state
    private let voiceKVSnapshot: PocketTtsSynthesizer.KVCacheState
    private let constants: PocketTtsConstantsBundle
    private let bosEmb: MLMultiArray
    private let temperature: Float
    private var mimiState: PocketTtsSynthesizer.MimiState
    private var rng: SeededRNG

    // MARK: - Initialization

    /// Create a session with pre-computed voice KV cache.
    ///
    /// This initializer is internal — use `PocketTtsManager.makeSession()` instead.
    init(
        voiceKVSnapshot: PocketTtsSynthesizer.KVCacheState,
        mimiState: PocketTtsSynthesizer.MimiState,
        constants: PocketTtsConstantsBundle,
        condModel: MLModel,
        stepModel: MLModel,
        flowModel: MLModel,
        mimiModel: MLModel,
        bosEmb: MLMultiArray,
        temperature: Float,
        seed: UInt64
    ) {
        self.voiceKVSnapshot = voiceKVSnapshot
        self.mimiState = mimiState
        self.constants = constants
        self.condModel = condModel
        self.stepModel = stepModel
        self.flowModel = flowModel
        self.mimiModel = mimiModel
        self.bosEmb = bosEmb
        self.temperature = temperature
        self.rng = SeededRNG(seed: seed)

        // Text queue channel
        let (textStream, textContinuation) = AsyncStream.makeStream(of: String.self)
        self.textStream = textStream
        self.textContinuation = textContinuation

        // Frame output stream
        let (frames, frameContinuation) = AsyncThrowingStream.makeStream(
            of: PocketTtsSynthesizer.AudioFrame.self
        )
        self.frames = frames
        self.frameContinuation = frameContinuation
    }

    /// Start the generation loop. Must be called once after init.
    func start() {
        generationTask = Task { [weak self] in
            guard let self else { return }
            await self.generateLoop()
        }
        frameContinuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task { await self.cancel() }
        }
    }

    // MARK: - Generation Loop

    private func generateLoop() async {
        var utteranceIndex = 0

        do {
            for await text in textStream {
                if Task.isCancelled { break }

                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }

                let chunks = PocketTtsSynthesizer.chunkText(
                    trimmed, tokenizer: constants.tokenizer
                )
                Self.logger.info(
                    "Session enqueued '\(trimmed)', \(chunks.count) chunk(s)")

                for (chunkIndex, chunkText) in chunks.enumerated() {
                    if Task.isCancelled { break }

                    try await generateChunk(
                        text: chunkText,
                        chunkIndex: chunkIndex,
                        chunkCount: chunks.count,
                        utteranceIndex: utteranceIndex
                    )
                }
                utteranceIndex += 1
            }
            frameContinuation.finish()
        } catch {
            frameContinuation.finish(throwing: error)
        }
    }

    private func generateChunk(
        text: String,
        chunkIndex: Int,
        chunkCount: Int,
        utteranceIndex: Int
    ) async throws {
        let (normalizedChunk, framesAfterEos) = PocketTtsSynthesizer.normalizeText(text)
        Self.logger.info("Session chunk \(chunkIndex): '\(normalizedChunk)'")

        // Tokenize and embed
        let tokenIds = constants.tokenizer.encode(normalizedChunk)
        let textEmbeddings = PocketTtsSynthesizer.embedTokens(tokenIds, constants: constants)

        // Clone voice KV snapshot and prefill text tokens only
        var kvState = try PocketTtsSynthesizer.cloneKVCacheState(voiceKVSnapshot)
        kvState = try await PocketTtsSynthesizer.prefillKVCacheText(
            state: kvState, textEmbeddings: textEmbeddings, model: condModel
        )

        // Generation loop
        let maxGenLen = PocketTtsSynthesizer.estimateMaxFrames(text: text)
        var eosStep: Int?
        var sequence = try PocketTtsSynthesizer.createNaNSequence()
        let totalFramesAfterEos = framesAfterEos + PocketTtsConstants.extraFramesAfterDetection

        for step in 0..<maxGenLen {
            if Task.isCancelled { break }

            // FlowLM step with local KV cache copy-in/copy-out
            var localKV = kvState
            let (transformerOut, eosLogit) = try await PocketTtsSynthesizer.runFlowLMStep(
                sequence: sequence,
                bosEmb: bosEmb,
                state: &localKV,
                model: stepModel
            )
            kvState = localKV

            // EOS detection
            if eosLogit > PocketTtsConstants.eosThreshold && eosStep == nil {
                eosStep = step
                Self.logger.info("Session chunk \(chunkIndex) EOS at step \(step)")
            }
            if let eos = eosStep, step >= eos + totalFramesAfterEos {
                break
            }

            // Flow decode with actor-isolated RNG
            var localRng = rng
            let latent = try await PocketTtsSynthesizer.flowDecode(
                transformerOut: transformerOut,
                numSteps: PocketTtsConstants.numLsdSteps,
                temperature: temperature,
                model: flowModel,
                rng: &localRng
            )
            rng = localRng

            // Mimi decode with actor-isolated state
            var localMimi = mimiState
            let frameSamples = try await PocketTtsSynthesizer.runMimiDecoder(
                latent: latent,
                state: &localMimi,
                model: mimiModel
            )
            mimiState = localMimi

            // Yield frame
            frameContinuation.yield(
                PocketTtsSynthesizer.AudioFrame(
                    samples: frameSamples,
                    frameIndex: step,
                    chunkIndex: chunkIndex,
                    chunkCount: chunkCount,
                    utteranceIndex: utteranceIndex
                )
            )

            // Autoregressive feedback
            sequence = try PocketTtsSynthesizer.createSequenceFromLatent(latent)
        }
    }
}
