@preconcurrency import CoreML
import Foundation

/// PocketTTS flow-matching language model synthesizer.
///
/// Generates audio autoregressively: each generation step produces
/// an 80ms audio frame (1920 samples at 24kHz).
///
/// Long text is split into sentence-based chunks (≤50 tokens each)
/// to stay within the KV cache limit (512 positions).
///
/// Pipeline: text → chunk → [tokenize → embed → prefill KV → generate → flow decode → mimi decode] → WAV
public struct PocketTtsSynthesizer {

    static let logger = AppLogger(category: "PocketTtsSynthesizer")

    private enum Context {
        @TaskLocal static var modelStore: PocketTtsModelStore?
    }

    static func withModelStore<T>(
        _ store: PocketTtsModelStore,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$modelStore.withValue(store) {
            try await operation()
        }
    }

    static func currentModelStore() throws -> PocketTtsModelStore {
        guard let store = Context.modelStore else {
            throw PocketTTSError.processingFailed(
                "PocketTtsSynthesizer requires a model store context.")
        }
        return store
    }

    // MARK: - Public API

    /// Synthesize audio from text.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice identifier (default: "alba").
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - seed: Random seed for reproducibility (nil for random).
    ///   - deEss: Whether to apply de-essing post-processing.
    /// - Returns: A synthesis result containing WAV audio data.
    public static func synthesize(
        text: String,
        voice: String = PocketTtsConstants.defaultVoice,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil,
        deEss: Bool = true,
        maxTokensPerChunk: Int = PocketTtsConstants.maxTokensPerChunk,
        language: PocketTtsLanguage = .english
    ) async throws -> SynthesisResult {
        let store = try currentModelStore()
        let voiceData = try await store.voiceData(for: voice)
        return try await synthesize(
            text: text,
            voiceData: voiceData,
            temperature: temperature,
            seed: seed,
            deEss: deEss,
            maxTokensPerChunk: maxTokensPerChunk,
            language: language
        )
    }

    /// Synthesize audio from text using provided voice data.
    ///
    /// Use this overload for cloned voices without saving to disk first.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voiceData: Voice conditioning data (e.g., from cloneVoice).
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - seed: Random seed for reproducibility (nil for random).
    ///   - deEss: Whether to apply de-essing post-processing.
    /// - Returns: A synthesis result containing WAV audio data.
    public static func synthesize(
        text: String,
        voiceData: PocketTtsVoiceData,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil,
        deEss: Bool = true,
        maxTokensPerChunk: Int = PocketTtsConstants.maxTokensPerChunk,
        language: PocketTtsLanguage = .english
    ) async throws -> SynthesisResult {
        logger.info("PocketTTS synthesizing with custom voice: '\(text)'")
        let genStart = Date()

        // Buffer the streaming output. Both APIs share one chunk loop now,
        // so any change to prefill/generation logic only needs to land once.
        let stream = try await synthesizeStreaming(
            text: text,
            voiceData: voiceData,
            temperature: temperature,
            seed: seed,
            maxTokensPerChunk: maxTokensPerChunk,
            language: language
        )

        var allSamples: [Float] = []
        var frameCount = 0
        for try await frame in stream {
            allSamples.append(contentsOf: frame.samples)
            frameCount += 1
        }

        let genElapsed = Date().timeIntervalSince(genStart)
        logger.info("Generated \(frameCount) frames in \(String(format: "%.2f", genElapsed))s")

        // De-essing (no peak normalization — preserve natural levels)
        if deEss {
            AudioPostProcessor.applyTtsPostProcessing(
                &allSamples,
                sampleRate: Float(PocketTtsConstants.audioSampleRate),
                deEssAmount: -3.0,
                smoothing: false
            )
        }

        // Encode WAV
        let audioData = try AudioWAV.data(
            from: allSamples,
            sampleRate: Double(PocketTtsConstants.audioSampleRate)
        )

        let duration = Double(allSamples.count) / Double(PocketTtsConstants.audioSampleRate)
        logger.info("Audio duration: \(String(format: "%.2f", duration))s")

        return SynthesisResult(
            audio: audioData,
            samples: allSamples,
            frameCount: frameCount,
            eosStep: nil
        )
    }

    // MARK: - Streaming API

    /// An audio frame produced during streaming synthesis.
    ///
    /// Each frame contains 80ms of audio (1920 samples at 24kHz).
    public struct AudioFrame: Sendable {
        /// Raw Float32 audio samples for this frame.
        public let samples: [Float]
        /// Zero-based frame index within the current text chunk.
        public let frameIndex: Int
        /// Zero-based index of the text chunk being synthesized.
        public let chunkIndex: Int
        /// Total number of text chunks for the current utterance.
        public let chunkCount: Int
        /// Zero-based index of the enqueued utterance that produced this frame.
        /// Only set in session mode; `nil` for one-shot and streaming synthesis.
        public let utteranceIndex: Int?
    }

    /// Synthesize audio as a stream of 80ms frames.
    ///
    /// Each frame contains 1920 Float32 samples at 24kHz. Frames are yielded
    /// as soon as they are generated, enabling real-time playback to start
    /// before the full utterance is complete.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice identifier (default: "alba").
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - seed: Random seed for reproducibility (nil for random).
    /// - Returns: An `AsyncThrowingStream` of audio frames. Throws if a model
    ///   inference error occurs during generation.
    ///
    /// Example:
    /// ```swift
    /// let stream = try await PocketTtsSynthesizer.synthesizeStreaming(text: "Hello, world!")
    /// for try await frame in stream {
    ///     playAudio(frame.samples)  // Play each 80ms frame immediately
    /// }
    /// ```
    public static func synthesizeStreaming(
        text: String,
        voice: String = PocketTtsConstants.defaultVoice,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil,
        maxTokensPerChunk: Int = PocketTtsConstants.maxTokensPerChunk,
        language: PocketTtsLanguage = .english
    ) async throws -> AsyncThrowingStream<AudioFrame, Error> {
        let store = try currentModelStore()
        let voiceData = try await store.voiceData(for: voice)
        return try await synthesizeStreaming(
            text: text,
            voiceData: voiceData,
            temperature: temperature,
            seed: seed,
            maxTokensPerChunk: maxTokensPerChunk,
            language: language
        )
    }

    /// Synthesize audio as a stream using custom voice data.
    ///
    /// Use this overload for cloned voices without saving to disk first.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voiceData: Voice conditioning data (e.g., from cloneVoice).
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - seed: Random seed for reproducibility (nil for random).
    /// - Returns: An `AsyncThrowingStream` of audio frames. Throws if a model
    ///   inference error occurs during generation.
    public static func synthesizeStreaming(
        text: String,
        voiceData: PocketTtsVoiceData,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil,
        maxTokensPerChunk: Int = PocketTtsConstants.maxTokensPerChunk,
        language: PocketTtsLanguage = .english
    ) async throws -> AsyncThrowingStream<AudioFrame, Error> {
        let store = try currentModelStore()

        logger.info("PocketTTS streaming synthesis with custom voice: '\(text)'")

        let constants = try await store.constants()
        let chunks = chunkTextWithMetadata(
            text, tokenizer: constants.tokenizer,
            maxTokens: maxTokensPerChunk, language: language)
        let condModel = try await store.condStep()
        let hasCondPrefill = await store.hasCondPrefill()
        let stepModel = try await store.flowlmStep()
        let flowModel = try await store.flowDecoder()
        let mimiModel = try await store.mimiDecoder()
        let condLayerKeys = try await store.condStepLayerKeys()
        let condPrefillLayerKeys = await store.condPrefillStepLayerKeys()
        let useCondPrefill = hasCondPrefill && condPrefillLayerKeys != nil
        let condPrefillModel = useCondPrefill ? try await store.condPrefill() : condModel
        let flowlmLayerKeys = try await store.flowLMStepLayerKeys()
        let mimiKeys = try await store.mimiDecoderKeys()
        let repoDir = try await store.repoDir()
        let mimiInitialState = try loadMimiInitialState(from: repoDir, mimiKeys: mimiKeys)
        let bosEmb = try createBosEmbedding(constants.bosEmbedding)
        let seedValue = seed ?? UInt64.random(in: 0...UInt64.max)
        let chunkCount = chunks.count

        let generator = StreamingGenerator(
            constants: constants,
            voiceData: voiceData,
            chunks: chunks,
            condModel: condModel,
            condPrefillModel: condPrefillModel,
            useCondPrefill: useCondPrefill,
            stepModel: stepModel,
            flowModel: flowModel,
            mimiModel: mimiModel,
            condLayerKeys: condLayerKeys,
            condPrefillLayerKeys: condPrefillLayerKeys,
            flowlmLayerKeys: flowlmLayerKeys,
            mimiKeys: mimiKeys,
            mimiInitialState: mimiInitialState,
            bosEmb: bosEmb,
            seedValue: seedValue,
            chunkCount: chunkCount,
            temperature: temperature,
            language: language
        )

        return makeStream(generator: generator)
    }

    // MARK: - Session API

    /// Create a persistent TTS session that keeps the voice KV cache warm.
    ///
    /// Performs the expensive voice prefill once (~125 tokens), then returns a
    /// session where each enqueued utterance only pays the text prefill cost.
    ///
    /// Must be called within a `withModelStore` context.
    static func makeSession(
        voiceData: PocketTtsVoiceData,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil,
        language: PocketTtsLanguage = .english
    ) async throws -> PocketTtsSession {
        let store = try currentModelStore()

        let constants = try await store.constants()
        let condModel = try await store.condStep()
        let hasCondPrefill = await store.hasCondPrefill()
        let stepModel = try await store.flowlmStep()
        let flowModel = try await store.flowDecoder()
        let mimiModel = try await store.mimiDecoder()
        let condLayerKeys = try await store.condStepLayerKeys()
        let condPrefillLayerKeys = await store.condPrefillStepLayerKeys()
        let useCondPrefill = hasCondPrefill && condPrefillLayerKeys != nil
        let condPrefillModel = useCondPrefill ? try await store.condPrefill() : condModel
        let flowlmLayerKeys = try await store.flowLMStepLayerKeys()
        let mimiKeys = try await store.mimiDecoderKeys()
        let repoDir = try await store.repoDir()
        let mimiState = try loadMimiInitialState(from: repoDir, mimiKeys: mimiKeys)
        let bosEmb = try createBosEmbedding(constants.bosEmbedding)
        let seedValue = seed ?? UInt64.random(in: 0...UInt64.max)

        // One-time voice prefill. Two paths matching `prefillKVCache`:
        //  - Shipped voices (cacheSnapshot != nil): drop pre-baked K/V into
        //    cache, skip cond_step entirely (`promptLength == 0`, so the
        //    loop in `prefillKVCacheVoice` would be a no-op anyway).
        //  - Cloned voices (flat audio prompt): feed `bos_before_voice`
        //    plus every voice token through cond_step.
        let voiceKVSnapshot: KVCacheState
        if let snapshot = voiceData.cacheSnapshot {
            voiceKVSnapshot = try kvCacheStateFromSnapshot(
                snapshot, layers: condLayerKeys.layerCount)
        } else {
            let emptyState = try emptyKVCacheState(layers: condLayerKeys.layerCount)
            voiceKVSnapshot = try await prefillKVCacheVoice(
                state: emptyState, voiceData: voiceData,
                bosBeforeVoice: constants.bosBeforeVoice,
                model: condModel, layerKeys: condLayerKeys,
                prefillModel: condPrefillModel, prefillLayerKeys: condPrefillLayerKeys,
                useFastPrefill: useCondPrefill
            )
        }

        logger.info(
            "Session voice prefill at position \(Int(voiceKVSnapshot.positions[0][0].floatValue))"
        )

        let session = PocketTtsSession(
            voiceKVSnapshot: voiceKVSnapshot,
            mimiState: mimiState,
            constants: constants,
            condModel: condModel,
            condPrefillModel: condPrefillModel,
            useCondPrefill: useCondPrefill,
            stepModel: stepModel,
            flowModel: flowModel,
            mimiModel: mimiModel,
            condLayerKeys: condLayerKeys,
            condPrefillLayerKeys: condPrefillLayerKeys,
            flowlmLayerKeys: flowlmLayerKeys,
            mimiKeys: mimiKeys,
            bosEmb: bosEmb,
            temperature: temperature,
            seed: seedValue,
            language: language
        )
        await session.start()
        return session
    }

    // MARK: - Streaming Internals

    /// Actor that owns all non-Sendable CoreML state for streaming generation.
    ///
    /// Using an actor ensures the non-Sendable `MLModel` and `MLMultiArray` types
    /// are properly isolated. The `Task` in `makeStream()` only captures this
    /// actor (which is `Sendable`) and the stream continuation.
    private actor StreamingGenerator {
        let constants: PocketTtsConstantsBundle
        let voiceData: PocketTtsVoiceData
        let chunks: [TextChunk]
        let condModel: MLModel
        let condPrefillModel: MLModel
        let useCondPrefill: Bool
        let stepModel: MLModel
        let flowModel: MLModel
        let mimiModel: MLModel
        let condLayerKeys: PocketTtsLayerKeys
        let condPrefillLayerKeys: PocketTtsLayerKeys?
        let flowlmLayerKeys: PocketTtsLayerKeys
        let mimiKeys: PocketTtsMimiKeys
        var mimiState: MimiState
        let bosEmb: MLMultiArray
        var rng: SeededRNG
        let chunkCount: Int
        let temperature: Float
        let language: PocketTtsLanguage

        init(
            constants: PocketTtsConstantsBundle,
            voiceData: PocketTtsVoiceData,
            chunks: [TextChunk],
            condModel: MLModel,
            condPrefillModel: MLModel,
            useCondPrefill: Bool,
            stepModel: MLModel,
            flowModel: MLModel,
            mimiModel: MLModel,
            condLayerKeys: PocketTtsLayerKeys,
            condPrefillLayerKeys: PocketTtsLayerKeys?,
            flowlmLayerKeys: PocketTtsLayerKeys,
            mimiKeys: PocketTtsMimiKeys,
            mimiInitialState: MimiState,
            bosEmb: MLMultiArray,
            seedValue: UInt64,
            chunkCount: Int,
            temperature: Float,
            language: PocketTtsLanguage
        ) {
            self.constants = constants
            self.voiceData = voiceData
            self.chunks = chunks
            self.condModel = condModel
            self.condPrefillModel = condPrefillModel
            self.useCondPrefill = useCondPrefill
            self.stepModel = stepModel
            self.flowModel = flowModel
            self.mimiModel = mimiModel
            self.condLayerKeys = condLayerKeys
            self.condPrefillLayerKeys = condPrefillLayerKeys
            self.flowlmLayerKeys = flowlmLayerKeys
            self.mimiKeys = mimiKeys
            self.mimiState = mimiInitialState
            self.bosEmb = bosEmb
            self.rng = SeededRNG(seed: seedValue)
            self.chunkCount = chunkCount
            self.temperature = temperature
            self.language = language
        }

        /// Flow decode using actor-isolated RNG state.
        ///
        /// Copies `rng` out before the async call and writes it back after,
        /// avoiding the `inout` restriction on actor-isolated properties.
        private func flowDecodeStep(
            transformerOut: MLMultiArray
        ) async throws -> [Float] {
            var localRng = rng
            let result = try await PocketTtsSynthesizer.flowDecode(
                transformerOut: transformerOut,
                numSteps: PocketTtsConstants.numLsdSteps,
                temperature: temperature,
                model: flowModel,
                rng: &localRng
            )
            rng = localRng
            return result
        }

        /// Mimi decode using actor-isolated streaming state.
        ///
        /// Copies `mimiState` out before the async call and writes it back after,
        /// avoiding the `inout` restriction on actor-isolated properties.
        private func mimiDecodeStep(latent: [Float]) async throws -> [Float] {
            var localState = mimiState
            let result = try await PocketTtsSynthesizer.runMimiDecoder(
                latent: latent,
                state: &localState,
                model: mimiModel,
                mimiKeys: mimiKeys
            )
            mimiState = localState
            return result
        }

        /// FlowLM step with local KV cache copy-in/copy-out.
        private func flowLMStep(
            sequence: MLMultiArray,
            kvState: inout KVCacheState
        ) async throws -> (transformerOut: MLMultiArray, eosLogit: Float) {
            var localState = kvState
            let result = try await PocketTtsSynthesizer.runFlowLMStep(
                sequence: sequence,
                bosEmb: bosEmb,
                state: &localState,
                model: stepModel,
                layerKeys: flowlmLayerKeys
            )
            kvState = localState
            return result
        }

        func generate(
            continuation: AsyncThrowingStream<AudioFrame, Error>.Continuation
        ) async {
            do {
                for (chunkIdx, chunk) in chunks.enumerated() {
                    let (normalizedChunk, framesAfterEos) =
                        PocketTtsSynthesizer.normalizeText(
                            chunk.text,
                            isMidSentence: chunk.isMidSentence,
                            language: language)
                    PocketTtsSynthesizer.logger.info(
                        "Stream chunk \(chunkIdx + 1)/\(chunkCount): '\(normalizedChunk)'"
                    )

                    let tokenIds = constants.tokenizer.encode(normalizedChunk)
                    let textEmbeddings = PocketTtsSynthesizer.embedTokens(
                        tokenIds, constants: constants)

                    var kvState = try await PocketTtsSynthesizer.prefillKVCache(
                        voiceData: voiceData,
                        textEmbeddings: textEmbeddings,
                        bosBeforeVoice: constants.bosBeforeVoice,
                        model: condModel,
                        layerKeys: condLayerKeys,
                        prefillModel: condPrefillModel,
                        prefillLayerKeys: condPrefillLayerKeys,
                        useFastPrefill: useCondPrefill
                    )

                    let maxGenLen = PocketTtsSynthesizer.estimateMaxFrames(text: chunk.text)
                    var eosStep: Int?
                    var sequence = try PocketTtsSynthesizer.createNaNSequence()
                    let totalFramesAfterEos =
                        framesAfterEos + PocketTtsConstants.extraFramesAfterDetection

                    for step in 0..<maxGenLen {
                        if Task.isCancelled { break }

                        let (transformerOut, eosLogit) = try await flowLMStep(
                            sequence: sequence,
                            kvState: &kvState
                        )

                        if eosLogit > PocketTtsConstants.eosThreshold && eosStep == nil {
                            eosStep = step
                            PocketTtsSynthesizer.logger.info(
                                "Stream chunk \(chunkIdx + 1) EOS at step \(step)")
                        }
                        if let eos = eosStep, step >= eos + totalFramesAfterEos {
                            break
                        }

                        let latent = try await flowDecodeStep(
                            transformerOut: transformerOut
                        )

                        let frameSamples = try await mimiDecodeStep(latent: latent)

                        continuation.yield(
                            AudioFrame(
                                samples: frameSamples,
                                frameIndex: step,
                                chunkIndex: chunkIdx,
                                chunkCount: chunkCount,
                                utteranceIndex: nil
                            ))

                        sequence = try PocketTtsSynthesizer.createSequenceFromLatent(latent)
                    }

                    if Task.isCancelled { break }
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        /// Cross-engine pipelined variant of `generate`.
        ///
        /// The per-frame chain is flowlm(GPU) → flow(ANE) → latent → [fed back to
        /// flowlm] + mimi(CPU). mimi's audio output feeds NOTHING back, so mimi[N]
        /// can run concurrently with flowlm[N+1]+flow[N+1]. This moves mimi onto its
        /// own actor + a detached consumer so the CPU decode overlaps the GPU/ANE
        /// critical path → per-frame wall ≈ max(mimi, flowlm+flow) instead of the sum.
        ///
        /// OPT-IN and UNVERIFIED on-device (gated by
        /// `PocketTtsSynthesizer.useCrossEnginePipeline`). Output is identical to
        /// `generate`; only scheduling differs. Verify timing + ordering on-device
        /// before making it the default.
        func generatePipelined(
            continuation: AsyncThrowingStream<AudioFrame, Error>.Continuation
        ) async {
            let mimi = MimiDecodeActor(
                model: mimiModel, keys: mimiKeys, initialState: mimiState)
            let totalChunks = chunkCount

            struct LatentWork: Sendable {
                let latent: [Float]
                let frameIndex: Int
                let chunkIndex: Int
            }
            let (latents, latentCont) = AsyncStream.makeStream(of: LatentWork.self)

            // CONSUMER (detached → off this actor): decode each latent on the mimi
            // actor (CPU) in order and yield audio. While it awaits mimi.decode,
            // the producer below runs flowlm/flow on GPU/ANE — the overlap.
            let consumer = Task.detached {
                do {
                    for await w in latents {
                        if Task.isCancelled { break }
                        let audio = try await mimi.decode(w.latent)
                        continuation.yield(
                            AudioFrame(
                                samples: audio, frameIndex: w.frameIndex,
                                chunkIndex: w.chunkIndex, chunkCount: totalChunks,
                                utteranceIndex: nil))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            // PRODUCER: flowlm(GPU)→flow(ANE) recurrence. Hands each latent off and
            // continues immediately — never waits for mimi.
            do {
                for (chunkIdx, chunk) in chunks.enumerated() {
                    if Task.isCancelled { break }
                    let (normalizedChunk, framesAfterEos) =
                        PocketTtsSynthesizer.normalizeText(
                            chunk.text, isMidSentence: chunk.isMidSentence, language: language)
                    let tokenIds = constants.tokenizer.encode(normalizedChunk)
                    let textEmbeddings = PocketTtsSynthesizer.embedTokens(
                        tokenIds, constants: constants)
                    var kvState = try await PocketTtsSynthesizer.prefillKVCache(
                        voiceData: voiceData, textEmbeddings: textEmbeddings,
                        bosBeforeVoice: constants.bosBeforeVoice, model: condModel,
                        layerKeys: condLayerKeys, prefillModel: condPrefillModel,
                        prefillLayerKeys: condPrefillLayerKeys, useFastPrefill: useCondPrefill)

                    let maxGenLen = PocketTtsSynthesizer.estimateMaxFrames(text: chunk.text)
                    var eosStep: Int?
                    var sequence = try PocketTtsSynthesizer.createNaNSequence()
                    let totalFramesAfterEos =
                        framesAfterEos + PocketTtsConstants.extraFramesAfterDetection

                    for step in 0..<maxGenLen {
                        if Task.isCancelled { break }
                        let (transformerOut, eosLogit) = try await flowLMStep(
                            sequence: sequence, kvState: &kvState)
                        if eosLogit > PocketTtsConstants.eosThreshold && eosStep == nil {
                            eosStep = step
                        }
                        if let eos = eosStep, step >= eos + totalFramesAfterEos { break }
                        let latent = try await flowDecodeStep(transformerOut: transformerOut)
                        latentCont.yield(
                            LatentWork(latent: latent, frameIndex: step, chunkIndex: chunkIdx))
                        sequence = try PocketTtsSynthesizer.createSequenceFromLatent(latent)
                    }
                    if Task.isCancelled { break }
                }
                latentCont.finish()
            } catch {
                latentCont.finish()
                consumer.cancel()
                continuation.finish(throwing: error)
                return
            }
            _ = await consumer.value
        }
    }

    /// Mimi streaming codec on its own actor so its CPU decode runs concurrently
    /// with the flowlm(GPU)→flow(ANE) critical path in `generatePipelined`.
    private actor MimiDecodeActor {
        private let model: MLModel
        private let keys: PocketTtsMimiKeys
        private var state: MimiState
        init(model: MLModel, keys: PocketTtsMimiKeys, initialState: MimiState) {
            self.model = model
            self.keys = keys
            self.state = initialState
        }
        func decode(_ latent: [Float]) async throws -> [Float] {
            var local = state
            let out = try await PocketTtsSynthesizer.runMimiDecoder(
                latent: latent, state: &local, model: model, mimiKeys: keys)
            state = local  // sequential codec state, single in-order consumer
            return out
        }
    }

    /// Opt-in cross-engine pipelining (mimi overlaps flowlm/flow). UNVERIFIED on
    /// device — leave `false` until timing + audio ordering are confirmed.
    static let useCrossEnginePipeline = false

    /// Create the AsyncThrowingStream and spawn the generation task.
    private static func makeStream(
        generator: StreamingGenerator
    ) -> AsyncThrowingStream<AudioFrame, Error> {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AudioFrame.self)

        let task = Task {
            if PocketTtsSynthesizer.useCrossEnginePipeline {
                await generator.generatePipelined(continuation: continuation)
            } else {
                await generator.generate(continuation: continuation)
            }
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return stream
    }

    // MARK: - Text Processing

    /// Metadata describing how a chunk was produced from the source text.
    ///
    /// Used by `normalizeText` to decide whether to capitalize the first letter
    /// and whether to append a sentence-ending period. Mid-sentence chunks
    /// (produced by clause- or word-boundary splits inside a longer sentence)
    /// should preserve their original casing and not gain artificial sentence
    /// punctuation, which would otherwise create unnatural pauses and prosody
    /// arcs (see issue #584).
    public struct TextChunk: Sendable, Equatable {
        /// The chunk's text, with surrounding whitespace trimmed.
        public let text: String
        /// True when this chunk is a continuation of a sentence — i.e., it
        /// came from a clause- or word-boundary split, not a sentence boundary.
        public let isMidSentence: Bool

        public init(text: String, isMidSentence: Bool) {
            self.text = text
            self.isMidSentence = isMidSentence
        }
    }

    /// Replace Unicode smart quotes with their ASCII equivalents.
    ///
    /// PocketTTS's SentencePiece vocabulary is trained on ASCII apostrophes/
    /// quotes; smart quotes (U+2018/U+2019/U+201C/U+201D) typically fall back
    /// to byte-level pieces, which inflates the per-sentence token count and
    /// triggers unnecessary clause splits. Modern keyboards auto-convert
    /// ASCII apostrophes to U+2019, so French (`d'aboutir`) and English
    /// contractions (`don't`) commonly hit this path. See issue #584.
    static func normalizeSmartQuotes(_ text: String) -> String {
        text.replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
    }

    /// Language-specific pre-normalization applied before the shared
    /// smart-quote pass.
    ///
    /// English is a no-op — the shared normalizer already handles every
    /// punctuation form that affects English tokenization.
    ///
    /// French additionally normalizes:
    /// - Guillemets (`«` U+00AB, `»` U+00BB) → ASCII `"`. The SentencePiece
    ///   vocab doesn't include guillemets, so they fall back to byte pieces.
    /// - Non-breaking space (U+00A0) → regular space. French typography uses
    ///   NBSP before `! ? : ;` and inside thousand separators; the tokenizer
    ///   has no NBSP piece.
    /// - Narrow non-breaking space (U+202F) → regular space (same rationale).
    static func normalizeForLanguage(
        _ text: String, language: PocketTtsLanguage
    ) -> String {
        switch language {
        case .english, .german, .german24L, .italian, .italian24L,
            .portuguese, .portuguese24L, .spanish, .spanish24L:
            return text
        case .french24L:
            return
                text
                .replacingOccurrences(of: "\u{00AB}", with: "\"")
                .replacingOccurrences(of: "\u{00BB}", with: "\"")
                .replacingOccurrences(of: "\u{00A0}", with: " ")
                .replacingOccurrences(of: "\u{202F}", with: " ")
        }
    }

    /// Normalize a text chunk for PocketTTS (matching Python `prepare_text_prompt`).
    ///
    /// For chunks that are continuations of a longer sentence (mid-sentence
    /// clause/word splits), pass `isMidSentence: true` to preserve the chunk's
    /// original casing and avoid appending an artificial sentence-ending
    /// period. This prevents the synthesizer from rendering mid-phrase
    /// fragments as standalone sentences (issue #584).
    ///
    /// The `language` parameter selects language-specific punctuation
    /// normalization (e.g., French guillemets and NBSP). English is the
    /// default and applies only the shared smart-quote pass.
    static func normalizeText(
        _ text: String,
        isMidSentence: Bool = false,
        language: PocketTtsLanguage = .english
    ) -> (text: String, framesAfterEos: Int) {
        var result = normalizeForLanguage(
            normalizeSmartQuotes(
                text.trimmingCharacters(in: .whitespacesAndNewlines)),
            language: language)
        // Collapse whitespace
        result = result.replacingOccurrences(
            of: "\\s+", with: " ", options: .regularExpression)

        if !isMidSentence {
            // Strip trailing clause punctuation (commas, semicolons, colons)
            // before adding sentence-ending punctuation
            while let last = result.last, ",;:".contains(last) {
                result = String(result.dropLast())
            }
            result = result.trimmingCharacters(in: .whitespaces)

            // Capitalize first letter
            if let first = result.first, first.isLetter {
                result = first.uppercased() + result.dropFirst()
            }

            // Add period if no terminal punctuation
            if let last = result.last, !".!?".contains(last) {
                result += "."
            }
        }

        // Pad short texts for better prosody — but only for full sentences.
        // Mid-sentence chunks (clause/word-boundary continuations) must skip
        // the leading-space padding and the extra trailing frames; otherwise
        // each short fragment introduces ~80ms+ of silence at the seam, which
        // re-creates the prosody break we're trying to remove (issue #584).
        let wordCount = result.split(separator: " ").count
        let framesAfterEos: Int
        if !isMidSentence, wordCount < PocketTtsConstants.shortTextWordThreshold {
            result = String(repeating: " ", count: 8) + result
            framesAfterEos = PocketTtsConstants.shortTextPadFrames
        } else {
            framesAfterEos = PocketTtsConstants.longTextExtraFrames
        }

        return (result, framesAfterEos)
    }

    /// Split text into chunks that fit within the KV cache token limit.
    ///
    /// Splits at sentence boundaries (`.!?`) and groups sentences into chunks
    /// where each chunk tokenizes to ≤ `maxTokens` tokens. Oversized single
    /// sentences are further split at clause and word boundaries.
    ///
    /// Smart quotes are normalized to ASCII before chunking so that French
    /// contractions like `d'aboutir` do not get inflated token counts (#584).
    static func chunkText(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int = PocketTtsConstants.maxTokensPerChunk,
        language: PocketTtsLanguage = .english
    ) -> [String] {
        chunkTextWithMetadata(
            text, tokenizer: tokenizer, maxTokens: maxTokens, language: language
        ).map { $0.text }
    }

    /// Like `chunkText`, but tags each chunk with `isMidSentence` so callers
    /// can preserve casing/punctuation for clause- or word-boundary splits.
    ///
    /// The `language` parameter selects language-specific abbreviation and
    /// punctuation tables. English is the default.
    static func chunkTextWithMetadata(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int = PocketTtsConstants.maxTokensPerChunk,
        language: PocketTtsLanguage = .english
    ) -> [TextChunk] {
        let normalized = normalizeForLanguage(
            normalizeSmartQuotes(
                text.trimmingCharacters(in: .whitespacesAndNewlines)),
            language: language)

        // If it fits in one chunk, return as-is. A single-chunk input is
        // never mid-sentence — it's whatever the caller passed in.
        let tokenCount = tokenizer.encode(normalized).count
        if tokenCount <= maxTokens {
            return [TextChunk(text: normalized, isMidSentence: false)]
        }

        // Split into sentences at .!? boundaries, using the language's
        // abbreviation table so e.g. French `M.` doesn't end a sentence.
        let sentences = splitSentences(normalized, language: language)

        // Further split any oversized sentences at clause/word boundaries.
        // Track which pieces came from mid-sentence splits so the synthesizer
        // doesn't capitalize them or append a period.
        var pieces: [TextChunk] = []
        for sentence in sentences {
            let sentenceTokens = tokenizer.encode(sentence).count
            if sentenceTokens <= maxTokens {
                pieces.append(TextChunk(text: sentence, isMidSentence: false))
            } else {
                let subPieces = splitOversizedSentence(
                    sentence, tokenizer: tokenizer, maxTokens: maxTokens)
                // The first sub-piece keeps the sentence's leading capital and
                // is treated as a sentence-start; subsequent sub-pieces are
                // mid-sentence continuations. The last sub-piece carries the
                // sentence's terminal punctuation (if any) and is also a
                // continuation — its prosody should flow from the prior piece.
                for (index, piece) in subPieces.enumerated() {
                    pieces.append(
                        TextChunk(
                            text: piece,
                            isMidSentence: index > 0
                        ))
                }
            }
        }

        // Group pieces into chunks that fit the token limit. Two pieces can
        // merge only if their mid-sentence flags are compatible; otherwise
        // we'd lose the boundary information needed for correct prosody.
        var chunks: [TextChunk] = []
        var current: TextChunk?

        for piece in pieces {
            guard let existing = current else {
                current = piece
                continue
            }

            // Don't merge a sentence-start piece onto a mid-sentence chunk —
            // we'd lose the sentence boundary cue.
            if existing.isMidSentence != piece.isMidSentence {
                chunks.append(existing)
                current = piece
                continue
            }

            let candidate = existing.text + " " + piece.text
            let candidateTokens = tokenizer.encode(candidate).count
            if candidateTokens <= maxTokens {
                current = TextChunk(
                    text: candidate, isMidSentence: existing.isMidSentence)
            } else {
                chunks.append(existing)
                current = piece
            }
        }

        if let last = current {
            chunks.append(last)
        }

        return chunks.isEmpty
            ? [TextChunk(text: normalized, isMidSentence: false)]
            : chunks
    }

    /// Split an oversized sentence to fit within the token limit.
    ///
    /// First tries splitting at clause boundaries (commas, semicolons, colons).
    /// Falls back to word-boundary splitting for clauses that still exceed the limit.
    static func splitOversizedSentence(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int
    ) -> [String] {
        // First try: split at clause boundaries
        let clauseParts = splitAtClauseBoundaries(text)

        // Group clause parts into chunks that fit
        var result: [String] = []
        var currentPart = ""

        for part in clauseParts {
            let candidate = currentPart.isEmpty ? part : currentPart + " " + part
            let candidateTokens = tokenizer.encode(candidate).count

            if candidateTokens <= maxTokens {
                currentPart = candidate
            } else {
                if !currentPart.isEmpty {
                    result.append(currentPart)
                }
                // If single clause part still exceeds limit, split at word boundaries
                if tokenizer.encode(part).count > maxTokens {
                    result.append(contentsOf: splitAtWordBoundaries(part, tokenizer: tokenizer, maxTokens: maxTokens))
                    currentPart = ""
                } else {
                    currentPart = part
                }
            }
        }

        if !currentPart.isEmpty {
            result.append(currentPart)
        }

        return result.isEmpty ? [text] : result
    }

    /// Split text at clause punctuation (commas, semicolons, colons).
    ///
    /// Does not split at commas within numbers (e.g., "3,500").
    static func splitAtClauseBoundaries(_ text: String) -> [String] {
        let clauseBreaks: Set<Character> = [",", ";", ":"]
        var parts: [String] = []
        var current = ""
        let chars = Array(text)

        for (i, char) in chars.enumerated() {
            current.append(char)

            guard clauseBreaks.contains(char) else { continue }

            // Don't split at commas between digits (e.g., "3,500")
            if char == "," {
                let prevIsDigit = i > 0 && chars[i - 1].isNumber
                let nextIsDigit = i + 1 < chars.count && chars[i + 1].isNumber
                if prevIsDigit && nextIsDigit {
                    continue
                }
            }

            let trimmed = current.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                parts.append(trimmed)
            }
            current = ""
        }

        let trimmed = current.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            parts.append(trimmed)
        }

        return parts
    }

    /// Split text at word boundaries to fit within the token limit.
    ///
    /// Avoids orphaning a single trailing word ("…stations-service de" +
    /// "TotalEnergies") by pre-budgeting one word back from the head chunk
    /// when the tail would otherwise be a single short word. See issue #584.
    static func splitAtWordBoundaries(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int
    ) -> [String] {
        let words = text.split(separator: " ").map(String.init)
        guard words.count > 1 else { return [text] }

        var chunks: [String] = []
        var currentWords: [String] = []

        for word in words {
            let candidate = (currentWords + [word]).joined(separator: " ")
            let tokens = tokenizer.encode(candidate).count

            if tokens > maxTokens && !currentWords.isEmpty {
                chunks.append(currentWords.joined(separator: " "))
                currentWords = [word]
            } else {
                currentWords.append(word)
            }
        }

        if !currentWords.isEmpty {
            chunks.append(currentWords.joined(separator: " "))
        }

        // If the tail is a single short word (likely orphaned by the greedy
        // split), shift one word back from the preceding chunk so the tail
        // has at least two words and prosody is less jarring. Only applies
        // when the preceding chunk has multiple words to give up.
        if chunks.count >= 2, let tail = chunks.last,
            tail.split(separator: " ").count == 1
        {
            let prevIndex = chunks.count - 2
            let prevWords = chunks[prevIndex].split(separator: " ").map(String.init)
            if prevWords.count >= 2 {
                let donated = prevWords.last!
                let newPrev = prevWords.dropLast().joined(separator: " ")
                let newTail = donated + " " + tail
                chunks[prevIndex] = newPrev
                chunks[chunks.count - 1] = newTail
            }
        }

        return chunks
    }

    /// Common English abbreviations that end with a period but don't end a
    /// sentence. Used as the default abbreviation set; other languages
    /// override via `abbreviations(for:)`.
    static let abbreviations: Set<String> = [
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc",
        "inc", "ltd", "co", "corp", "dept", "univ", "govt", "approx",
        "avg", "est", "gen", "gov", "hon", "sgt", "cpl", "pvt", "capt",
        "lt", "col", "maj", "cmdr", "adm", "rev", "sen", "rep",
    ]

    /// French abbreviations that end with a period but don't end a sentence.
    ///
    /// Includes the common civility titles (`M.`, `Mme`, `Mlle`, `Mtre`),
    /// honorifics (`Dr.`, `Pr.`), saints (`St.`, `Ste.`), reference markers
    /// (`p.`, `pp.`, `vol.`, `chap.`, `cf.`, `cf`, `ibid.`, `op.`, `cit.`,
    /// `etc.`), and address terms (`av.`, `bd.`, `bld.`, `rte.`).
    static let frenchAbbreviations: Set<String> = [
        "m", "mm", "mme", "mmes", "mlle", "mlles", "mtre", "mtres",
        "dr", "drs", "pr", "prs", "me", "mes",
        "st", "ste", "sts", "stes",
        "etc", "cf", "ibid", "op", "cit", "ndlr", "nb",
        "p", "pp", "vol", "chap", "tome", "fig",
        "av", "bd", "bld", "rte", "no", "nos",
    ]

    /// Return the abbreviation set for a given language. English is the
    /// default; French gets its own table. Other languages currently share
    /// the English table until their corpora warrant a custom list.
    static func abbreviations(for language: PocketTtsLanguage) -> Set<String> {
        switch language {
        case .french24L:
            return frenchAbbreviations
        case .english, .german, .german24L, .italian, .italian24L,
            .portuguese, .portuguese24L, .spanish, .spanish24L:
            return abbreviations
        }
    }

    /// Split text into sentences at `.!?` boundaries.
    ///
    /// Handles abbreviations (e.g., "Dr.", "Prof.") by not splitting after them.
    ///
    /// The `language` parameter selects the abbreviation table; English is the
    /// default. French (`.french24L`) uses `frenchAbbreviations` for civility
    /// titles, address terms, and reference markers.
    static func splitSentences(
        _ text: String,
        language: PocketTtsLanguage = .english
    ) -> [String] {
        let abbrevSet = abbreviations(for: language)
        var sentences: [String] = []
        var current = ""
        let chars = Array(text)

        for (i, char) in chars.enumerated() {
            current.append(char)

            guard ".!?".contains(char) else { continue }

            // For periods, check if this is an abbreviation
            if char == "." {
                let trimmed = current.trimmingCharacters(in: .whitespaces)
                // Get the last word before the period
                let withoutPeriod = String(trimmed.dropLast())
                let lastWord = withoutPeriod.split(separator: " ").last.map(String.init) ?? withoutPeriod

                // Skip if it's a known abbreviation
                if abbrevSet.contains(lastWord.lowercased()) {
                    continue
                }

                // Skip if it's a single uppercase letter (e.g., "J." in initials)
                if lastWord.count == 1, lastWord.first?.isUppercase == true {
                    continue
                }

                // Skip if followed by a digit (e.g., "3.5")
                if i + 1 < chars.count, chars[i + 1].isNumber {
                    continue
                }
            }

            let trimmed = current.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                sentences.append(trimmed)
            }
            current = ""
        }

        // Remaining text without terminal punctuation
        let trimmed = current.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            sentences.append(trimmed)
        }

        return sentences
    }

    // MARK: - Embedding

    /// Look up text token embeddings from the embedding table.
    ///
    /// Vocab size is derived from the actual loaded table because each
    /// language pack ships its own `text_embed_table` with potentially
    /// different row counts (`PocketTtsConstants.vocabSize` is only the
    /// English row count).
    static func embedTokens(
        _ tokenIds: [Int], constants: PocketTtsConstantsBundle
    ) -> [[Float]] {
        let dim = PocketTtsConstants.embeddingDim
        let vocabSize = constants.textEmbedTable.count / dim
        return tokenIds.map { id in
            guard id >= 0, id < vocabSize else {
                logger.warning("Token ID \(id) out of range [0, \(vocabSize)), clamping")
                let clampedId = min(max(id, 0), vocabSize - 1)
                let offset = clampedId * dim
                return Array(constants.textEmbedTable[offset..<(offset + dim)])
            }
            let offset = id * dim
            return Array(constants.textEmbedTable[offset..<(offset + dim)])
        }
    }

    // MARK: - Helpers

    /// Estimate maximum generation frames based on text length.
    ///
    /// At 80ms per frame, 12.5 frames ≈ 1 second of audio per word.
    /// The +2 adds margin for pauses and trailing silence.
    static func estimateMaxFrames(text: String) -> Int {
        let wordCount = text.split(separator: " ").count
        let genLenSec = Double(wordCount) + 2.0
        return Int(genLenSec * 12.5)
    }

    /// Create the BOS embedding as an MLMultiArray [32].
    static func createBosEmbedding(_ bos: [Float]) throws -> MLMultiArray {
        let dim = PocketTtsConstants.latentDim
        let array = try MLMultiArray(shape: [NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        bos.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            ptr.update(from: base, count: dim)
        }
        return array
    }

    /// Create a NaN-filled sequence `[1, 1, 32]` to signal beginning-of-sequence.
    ///
    /// The first generation step has no previous audio latent. NaN values tell
    /// the model to use the BOS embedding instead, triggering the start of speech.
    static func createNaNSequence() throws -> MLMultiArray {
        let dim = PocketTtsConstants.latentDim
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        for i in 0..<dim {
            ptr[i] = .nan
        }
        return array
    }

    /// Create a sequence `[1, 1, 32]` from a latent vector.
    ///
    /// Autoregressive feedback: each generated audio latent becomes the input
    /// for the next flowlm_step, so the model conditions on its own output.
    static func createSequenceFromLatent(_ latent: [Float]) throws -> MLMultiArray {
        let dim = PocketTtsConstants.latentDim
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        latent.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            ptr.update(from: base, count: dim)
        }
        return array
    }

}
