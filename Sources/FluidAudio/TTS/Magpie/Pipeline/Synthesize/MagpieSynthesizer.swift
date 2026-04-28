import Accelerate
@preconcurrency import CoreML
import Foundation

/// Orchestrates one Magpie synthesis call end-to-end.
///
/// Pipeline (mirroring `generate_coreml.generate`):
///   1. Tokenize text → padded ids (256) + mask.
///   2. `text_encoder.predict` → encoderOutput (1, 256, 768).
///   3. (CFG) make zero-context encoder pair.
///   4. Prefill: 110 step-by-step `decoder_step` calls with speaker embedding rows.
///   5. AR loop (≤ 500 steps):
///        embed current 8 codes → `decoder_step` → LT sample → new codes.
///   6. NanoCodec decode → fp32 PCM 22 kHz.
///   7. Peak-normalize to 0.9 when `options.peakNormalize`.
public actor MagpieSynthesizer {

    private let logger = AppLogger(category: "MagpieSynthesizer")

    private let store: MagpieModelStore
    private let tokenizer: MagpieTokenizer

    public init(store: MagpieModelStore, tokenizer: MagpieTokenizer) {
        self.store = store
        self.tokenizer = tokenizer
    }

    /// One-shot CoreML graph warmup. Runs a minimal `text_encoder` →
    /// prefill → AR loop (~4–8 steps) → `nanocodec_decoder` pass on a
    /// throwaway "." input so each model's first-call specialization (Metal
    /// dispatch, ANE compile, output-backing layout) is paid here instead of
    /// at the user's first `synthesize` call.
    ///
    /// Discards all generated audio. CFG is forced off so the unconditional
    /// branch isn't warmed unless the user actually opts into it.
    public func warmup() async throws {
        // minFrames > maxSteps forbids EOS for the entire warmup, guaranteeing
        // we hit `maxSteps` decoder_step calls (instead of stopping at step 4
        // when topK=1 picks EOS). 16 steps trades full graph specialization
        // for shorter init time — covers the first-call dispatch overhead on
        // text_encoder + decoder_step + nanocodec without doubling load time.
        let warmupOpts = MagpieSynthesisOptions(
            temperature: 1.0,
            topK: 1,
            maxSteps: 16,
            minFrames: 32,
            cfgScale: 1.0,
            seed: 0,
            peakNormalize: false,
            allowIpaOverride: false)
        let tokenized = try await tokenizer.tokenize(
            ".", language: .english, options: warmupOpts)
        let frames = try await synthesizeFrames(
            tokenized: tokenized, speaker: .john, options: warmupOpts)

        // Force a nanocodec dispatch even if the AR loop stopped on EOS at
        // step 4. nanocodec pads to maxFrames internally so any T works.
        let constants = try await store.constants()
        let nanoModel = try await store.nanocodecDecoder()
        let nano = MagpieNanocodec(
            model: nanoModel, numCodebooks: constants.config.numCodebooks)
        let rows: [[Int32]]
        if frames.numFrames > 0 {
            rows = frames.codebookRows
        } else {
            rows = Swift.Array(
                repeating: Swift.Array<Int32>(repeating: 0, count: 1),
                count: constants.config.numCodebooks)
        }
        _ = try nano.decode(frames: rows)
        logger.info("Warmup complete (text_encoder + prefill + decoder_step + nanocodec)")
    }

    /// Synthesize from plain text (honors `|...|` IPA override per `options`).
    ///
    /// Long inputs are split into sentence-level chunks via `MagpieChunker` so
    /// each piece fits inside the NanoCodec 256-frame static-shape cap (~11.9 s
    /// of audio). Chunks are synthesized **pipelined**: the actor runs the AR
    /// loop for chunk N+1 on `decoder_step` (GPU/Metal) while a detached task
    /// runs `nanocodec_decoder` (CPU) for chunk N — the two stages don't share
    /// compute, so the wall time becomes `Σ AR + last nanocodec` instead of
    /// `Σ (AR + nanocodec)`.
    public func synthesize(
        text: String, speaker: MagpieSpeaker, language: MagpieLanguage,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let chunks = MagpieChunker.chunk(text: text, language: language)
        logger.info(
            "Chunker produced \(chunks.count) chunk(s) "
                + "from \(text.count)-char input (lang=\(language.rawValue))")
        if chunks.count <= 1 {
            let tokenized = try await tokenizer.tokenize(
                text, language: language, options: options)
            return try await synthesize(
                tokenized: tokenized, speaker: speaker, options: options)
        }

        // Disable per-chunk peak normalization; apply once globally so chunk
        // boundaries don't get rescaled inconsistently.
        var perChunkOptions = options
        perChunkOptions.peakNormalize = false

        let sampleRate = MagpieConstants.audioSampleRate
        let nanocodecModel = try await store.nanocodecDecoder()
        let numCodebooks = try await store.constants().config.numCodebooks

        // Pre-allocate ordered slots so detached nanocodec tasks can deposit
        // their PCM into the right chunk index regardless of completion order.
        var chunkSamples: [[Float]] = Array(repeating: [], count: chunks.count)
        var totalCodes = 0
        var lastFinishedOnEos = false
        var sumTextEnc: Double = 0
        var sumPrefill: Double = 0
        var sumArLoop: Double = 0
        var sumDecoder: Double = 0
        var sumSampler: Double = 0
        var sumNano: Double = 0

        // The nanocodec future for the *previous* chunk. While the actor runs
        // synthesizeFrames for the current chunk on Metal, this task converts
        // codes → PCM on CPU in parallel.
        var pendingNano: Task<NanocodecJobResult, Error>? = nil

        for (i, chunk) in chunks.enumerated() {
            logger.info(
                "Synthesizing chunk \(i + 1)/\(chunks.count) "
                    + "(\(chunk.text.count) chars, est \(chunk.estimatedCodes) codes)")
            let tokenized = try await tokenizer.tokenize(
                chunk.text, language: language, options: perChunkOptions)
            let frames = try await synthesizeFrames(
                tokenized: tokenized, speaker: speaker, options: perChunkOptions)
            totalCodes += frames.numFrames
            lastFinishedOnEos = frames.finishedOnEos
            sumTextEnc += frames.textEncoderSeconds
            sumPrefill += frames.prefillSeconds
            sumArLoop += frames.arLoopSeconds
            sumDecoder += frames.decoderStepSeconds
            sumSampler += frames.samplerSeconds

            // Spawn nanocodec for *this* chunk on a background task so the
            // next iteration's AR loop can start immediately on the actor.
            let chunkIdx = i
            let rows = frames.codebookRows
            let model = nanocodecModel
            let codebooks = numCodebooks
            // Use .utility priority: decoder_step on Metal needs CPU
            // bandwidth for its Metal driver thread, and an aggressive nano
            // task throttles it. .utility lets the actor's AR loop keep
            // priority while still running nano in parallel.
            let newTask = Self.startNanoChunkTask(
                model: model, numCodebooks: codebooks,
                rows: rows, chunkIndex: chunkIdx)

            // Drain the previous chunk's nanocodec while we set up the next.
            if let prev = pendingNano {
                let result = try await prev.value
                chunkSamples[result.chunkIndex] = result.samples
                sumNano += result.seconds
            }
            pendingNano = newTask
        }

        // Drain the last in-flight nanocodec.
        if let last = pendingNano {
            let result = try await last.value
            chunkSamples[result.chunkIndex] = result.samples
            sumNano += result.seconds
        }

        // Concatenate ordered chunks with punctuation-aware silence between.
        var totalLen = 0
        for (i, s) in chunkSamples.enumerated() {
            totalLen += s.count
            if i < chunks.count - 1 {
                totalLen += (chunks[i].pauseAfterMs * sampleRate) / 1_000
            }
        }
        var combined = Swift.Array<Float>()
        combined.reserveCapacity(totalLen)
        for (i, s) in chunkSamples.enumerated() {
            combined.append(contentsOf: s)
            if i < chunks.count - 1 {
                let silenceCount = (chunks[i].pauseAfterMs * sampleRate) / 1_000
                if silenceCount > 0 {
                    combined.append(
                        contentsOf: Swift.Array<Float>(repeating: 0, count: silenceCount))
                }
            }
        }

        if options.peakNormalize {
            var peak: Float = 0
            for s in combined where abs(s) > peak { peak = abs(s) }
            if peak > 0 {
                let scale = MagpieConstants.peakTarget / peak
                for i in 0..<combined.count { combined[i] *= scale }
            }
        }

        let timings = MagpieSynthesisTimings(
            textEncoderSeconds: sumTextEnc,
            prefillSeconds: sumPrefill,
            arLoopSeconds: sumArLoop,
            decoderStepSeconds: sumDecoder,
            samplerSeconds: sumSampler,
            nanocodecSeconds: sumNano)

        return MagpieSynthesisResult(
            samples: combined,
            sampleRate: sampleRate,
            codeCount: totalCodes,
            finishedOnEos: lastFinishedOnEos,
            timings: timings)
    }

    /// Result returned by a detached nanocodec task. Sendable so it can cross
    /// the actor boundary cleanly.
    private struct NanocodecJobResult: Sendable {
        let chunkIndex: Int
        let samples: [Float]
        let seconds: Double
    }

    /// Streaming variant of `synthesize(text:...)`: yields `MagpieAudioChunk`s
    /// as each chunk's codec decode finishes, instead of returning a single
    /// concatenated buffer at the end.
    ///
    /// The chunker uses a smaller cap on the *first* chunk
    /// (`MagpieChunker.streamingFirstChunkCap`, ~50 codec frames ≈ 2.3 s) so
    /// time-to-first-audio drops from "all-text AR loop" to roughly
    /// `prefill + (50-step AR loop) + nanocodec ≈ 1.5–4 s` on M-series. Later
    /// chunks pack normally (≤ `MagpieChunker.maxCodesPerChunk` = 220 codes).
    ///
    /// Producer/consumer split internal:
    ///   - Producer task runs the AR loop on the actor and pushes per-chunk
    ///     `ChunkFrames` into an internal `AsyncThrowingStream`.
    ///   - Consumer (this method's body) drains those, runs `nanocodec_decoder`
    ///     on a detached task per chunk, applies a 5 ms edge-fade, and yields
    ///     the resulting `MagpieAudioChunk` to the public stream.
    ///
    /// AR (slow, GPU/Metal) overlaps nanocodec (fast, CPU) via actor
    /// reentrancy: while the consumer is awaiting a detached nanocodec task,
    /// the actor is free for the producer to run the next AR loop.
    ///
    /// Cancellation: cancelling the consumer task (or breaking out of the
    /// `for try await` loop) terminates the producer task as well.
    ///
    /// Note: `peakNormalize` is force-disabled for the stream (we can't peak-
    /// normalize incrementally without changing chunk gain mid-stream).
    public nonisolated func synthesizeStream(
        text: String, speaker: MagpieSpeaker, language: MagpieLanguage,
        options: MagpieSynthesisOptions
    ) -> AsyncThrowingStream<MagpieAudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.produceStream(
                        text: text, speaker: speaker, language: language,
                        options: options, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func produceStream(
        text: String, speaker: MagpieSpeaker, language: MagpieLanguage,
        options: MagpieSynthesisOptions,
        continuation: AsyncThrowingStream<MagpieAudioChunk, Error>.Continuation
    ) async throws {
        let chunks = MagpieChunker.chunkForStreaming(text: text, language: language)
        logger.info(
            "[stream] chunker produced \(chunks.count) chunk(s) "
                + "from \(text.count)-char input (lang=\(language.rawValue))")
        guard !chunks.isEmpty else { return }

        var perChunkOptions = options
        perChunkOptions.peakNormalize = false

        let sampleRate = MagpieConstants.audioSampleRate
        let nanoModel = try await store.nanocodecDecoder()
        let numCodebooks = try await store.constants().config.numCodebooks
        let totalChunks = chunks.count

        // Producer/consumer pipe.
        let (framesStream, framesContinuation) = AsyncThrowingStream<
            ProducedFrames, Error
        >.makeStream()

        // Producer: AR loop on actor. Pushes each chunk's frames as soon as
        // they're ready.
        let producerTokenizer = self.tokenizer
        let producer = Task {
            do {
                for (i, chunk) in chunks.enumerated() {
                    try Task.checkCancellation()
                    let tokenized = try await producerTokenizer.tokenize(
                        chunk.text, language: language, options: perChunkOptions)
                    let frames = try await self.synthesizeFrames(
                        tokenized: tokenized, speaker: speaker,
                        options: perChunkOptions)
                    framesContinuation.yield(
                        ProducedFrames(index: i, frames: frames, chunk: chunk))
                }
                framesContinuation.finish()
            } catch {
                framesContinuation.finish(throwing: error)
            }
        }
        defer { producer.cancel() }

        // Consumer: drain frames, run nano per chunk, yield audio in order.
        var consumed = 0
        for try await produced in framesStream {
            try Task.checkCancellation()
            let isFinal = (produced.index == totalChunks - 1)
            let rows = produced.frames.codebookRows
            let pauseMs = produced.chunk.pauseAfterMs
            let nanoTask = Self.startNanoFramesTask(
                model: nanoModel, numCodebooks: numCodebooks, rows: rows)
            var samples = try await nanoTask.value
            Self.applyEdgeFade(&samples, sampleRate: sampleRate)
            if !isFinal && pauseMs > 0 {
                let pad = (pauseMs * sampleRate) / 1_000
                if pad > 0 {
                    samples.append(
                        contentsOf: Swift.Array<Float>(repeating: 0, count: pad))
                }
            }
            let audioChunk = MagpieAudioChunk(
                samples: samples,
                sampleRate: sampleRate,
                sequenceIndex: produced.index,
                isFinal: isFinal,
                text: produced.chunk.text,
                codeCount: produced.frames.numFrames,
                finishedOnEos: produced.frames.finishedOnEos)
            continuation.yield(audioChunk)
            consumed += 1
        }

        if consumed != totalChunks {
            logger.warning(
                "[stream] producer ended early: \(consumed)/\(totalChunks) chunks yielded")
        }
    }

    private struct ProducedFrames: Sendable {
        let index: Int
        let frames: ChunkFrames
        let chunk: MagpieTextChunk
    }

    /// Nonisolated factory for the chunked-path detached task. Creating
    /// `Task.detached` from a nonisolated static context (rather than from
    /// inside an actor method) keeps the closure itself nonisolated, which
    /// avoids the Swift 6 region-based isolation error
    /// "'self'-isolated value of type ... passed as a strongly transferred
    /// parameter; later accesses could race".
    nonisolated private static func startNanoChunkTask(
        model: sending MLModel, numCodebooks: Int, rows: sending [[Int32]], chunkIndex: Int
    ) -> Task<NanocodecJobResult, Error> {
        Task.detached(priority: .utility) {
            try Self.decodeNanoChunk(
                model: model, numCodebooks: numCodebooks,
                rows: rows, chunkIndex: chunkIndex)
        }
    }

    /// Nonisolated factory for the streaming-path detached task. See
    /// `startNanoChunkTask` for the rationale.
    nonisolated private static func startNanoFramesTask(
        model: sending MLModel, numCodebooks: Int, rows: sending [[Int32]]
    ) -> Task<[Float], Error> {
        Task.detached(priority: .utility) {
            try Self.decodeNanoFrames(
                model: model, numCodebooks: numCodebooks, rows: rows)
        }
    }

    /// Nonisolated nanocodec wrapper used by detached tasks in the chunked
    /// non-streaming path. Returning a `Sendable` `NanocodecJobResult` lets
    /// the result cross the actor boundary cleanly without tripping Swift 6
    /// region-based isolation.
    nonisolated private static func decodeNanoChunk(
        model: MLModel, numCodebooks: Int, rows: [[Int32]], chunkIndex: Int
    ) throws -> NanocodecJobResult {
        let nano = MagpieNanocodec(model: model, numCodebooks: numCodebooks)
        let start = Date()
        let samples = try nano.decode(frames: rows)
        return NanocodecJobResult(
            chunkIndex: chunkIndex,
            samples: samples,
            seconds: Date().timeIntervalSince(start))
    }

    /// Nonisolated nanocodec wrapper used by detached tasks in the streaming
    /// path. Returns the raw `[Float]` PCM buffer (Sendable).
    nonisolated private static func decodeNanoFrames(
        model: MLModel, numCodebooks: Int, rows: [[Int32]]
    ) throws -> [Float] {
        let nano = MagpieNanocodec(model: model, numCodebooks: numCodebooks)
        return try nano.decode(frames: rows)
    }

    /// 5 ms linear fade-in/out at chunk boundaries to mask zero-crossing pops
    /// when chunks are concatenated by the consumer.
    private static func applyEdgeFade(_ samples: inout [Float], sampleRate: Int) {
        let fadeMs = 5
        let fadeLen = (fadeMs * sampleRate) / 1_000
        let n = min(fadeLen, samples.count / 2)
        guard n > 0 else { return }
        for i in 0..<n {
            let t = Float(i) / Float(n)
            samples[i] *= t
            samples[samples.count - 1 - i] *= t
        }
    }

    /// Codebook rows + per-stage timings for a single chunk; the nanocodec
    /// stage is intentionally not run here so the caller can pipeline it.
    private struct ChunkFrames: Sendable {
        let codebookRows: [[Int32]]
        let numFrames: Int
        let finishedOnEos: Bool
        let textEncoderSeconds: Double
        let prefillSeconds: Double
        let arLoopSeconds: Double
        let decoderStepSeconds: Double
        let samplerSeconds: Double
    }

    /// Synthesize from pre-tokenized phoneme ids.
    public func synthesize(
        phonemes: MagpiePhonemeTokens, speaker: MagpieSpeaker,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let tokenized = try await tokenizer.pad(phonemes: phonemes)
        return try await synthesize(tokenized: tokenized, speaker: speaker, options: options)
    }

    // MARK: - Core

    private func synthesize(
        tokenized: MagpieTokenizedText, speaker: MagpieSpeaker,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let frames = try await synthesizeFrames(
            tokenized: tokenized, speaker: speaker, options: options)
        let nanocodecModel = try await store.nanocodecDecoder()
        let numCodebooks = try await store.constants().config.numCodebooks
        let nano = MagpieNanocodec(model: nanocodecModel, numCodebooks: numCodebooks)
        let nanocodecStart = Date()
        var samples = try nano.decode(frames: frames.codebookRows)
        let nanocodecSeconds = Date().timeIntervalSince(nanocodecStart)
        logger.info(
            "nanocodec done in \(String(format: "%.0f", nanocodecSeconds * 1000))ms")

        if options.peakNormalize {
            var peak: Float = 0
            for s in samples where abs(s) > peak { peak = abs(s) }
            if peak > 0 {
                let scale = MagpieConstants.peakTarget / peak
                for i in 0..<samples.count { samples[i] *= scale }
            }
        }

        let timings = MagpieSynthesisTimings(
            textEncoderSeconds: frames.textEncoderSeconds,
            prefillSeconds: frames.prefillSeconds,
            arLoopSeconds: frames.arLoopSeconds,
            decoderStepSeconds: frames.decoderStepSeconds,
            samplerSeconds: frames.samplerSeconds,
            nanocodecSeconds: nanocodecSeconds)

        return MagpieSynthesisResult(
            samples: samples,
            sampleRate: MagpieConstants.audioSampleRate,
            codeCount: frames.numFrames,
            finishedOnEos: frames.finishedOnEos,
            timings: timings)
    }

    /// Run text_encoder + prefill + AR loop only; return per-codebook rows
    /// without invoking nanocodec. Lets the chunked path overlap nanocodec
    /// (CPU) with the next chunk's AR loop (GPU/Metal).
    private func synthesizeFrames(
        tokenized: MagpieTokenizedText, speaker: MagpieSpeaker,
        options: MagpieSynthesisOptions
    ) async throws -> ChunkFrames {
        let constants = try await store.constants()
        let ltWeights = try await store.localTransformer()
        let textEncoder = try await store.textEncoder()
        let decoderStep = try await store.decoderStep()

        let dModel = constants.config.dModel
        let maxTextLen = MagpieConstants.maxTextLength
        let numCodebooks = constants.config.numCodebooks
        let audioBosId = constants.config.audioBosId
        let audioEosId = constants.config.audioEosId
        let speakerContextLength = constants.config.speakerContextLength

        let speakerIndex = speaker.rawValue
        guard speakerIndex >= 0 && speakerIndex < constants.speakerEmbeddings.count else {
            throw MagpieError.invalidSpeakerIndex(speakerIndex)
        }

        // 1. text_encoder
        let textEncoderStart = Date()
        let encResult = try runTextEncoder(
            tokenized: tokenized, maxTextLen: maxTextLen, model: textEncoder)
        let encoderOutput = encResult.encoderOutput
        let encoderMask = encResult.encoderMask
        let textEncoderSeconds = Date().timeIntervalSince(textEncoderStart)
        logger.info(
            "text_encoder done in \(String(format: "%.0f", textEncoderSeconds * 1000))ms")

        let useCfg = options.cfgScale != 1.0
        let uncond: (encoderOutput: MLMultiArray, encoderMask: MLMultiArray)?
        if useCfg {
            uncond = try MagpiePrefill.makeUnconditional(
                encoderOutputShape: encoderOutput.shape, maxTextLen: maxTextLen)
        } else {
            uncond = nil
        }

        // 2. KV caches (conditional + optional unconditional).
        let condCache = try MagpieKvCache(
            numLayers: constants.config.numDecoderLayers,
            maxCacheLength: constants.config.maxCacheLength,
            numHeads: constants.config.numHeads,
            headDim: constants.config.headDim)
        let uncondCache: MagpieKvCache? =
            useCfg
            ? try MagpieKvCache(
                numLayers: constants.config.numDecoderLayers,
                maxCacheLength: constants.config.maxCacheLength,
                numHeads: constants.config.numHeads,
                headDim: constants.config.headDim)
            : nil

        // 3. Prefill (fast batched path when decoder_prefill is available).
        let prefill = MagpiePrefill(decoderStep: decoderStep)
        let hasPrefill = await store.hasDecoderPrefill()
        let prefillStart = Date()
        if hasPrefill {
            let decoderPrefill = try await store.decoderPrefill()
            try prefill.prefillFast(
                decoderPrefill: decoderPrefill,
                speakerEmbedding: constants.speakerEmbeddings[speakerIndex],
                speakerContextLength: speakerContextLength,
                dModel: dModel,
                encoderOutput: encoderOutput,
                encoderMask: encoderMask,
                cache: condCache)
            if let uncondTensors = uncond, let uncondCache = uncondCache {
                let zeroSpeaker = Swift.Array<Float>(
                    repeating: 0, count: speakerContextLength * dModel)
                try prefill.prefillFast(
                    decoderPrefill: decoderPrefill,
                    speakerEmbedding: zeroSpeaker,
                    speakerContextLength: speakerContextLength,
                    dModel: dModel,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache)
            }
        } else {
            try prefill.prefill(
                speakerEmbedding: constants.speakerEmbeddings[speakerIndex],
                speakerContextLength: speakerContextLength,
                dModel: dModel,
                encoderOutput: encoderOutput,
                encoderMask: encoderMask,
                cache: condCache)
            if let uncondTensors = uncond, let uncondCache = uncondCache {
                let zeroSpeaker = Swift.Array<Float>(
                    repeating: 0, count: speakerContextLength * dModel)
                try prefill.prefill(
                    speakerEmbedding: zeroSpeaker,
                    speakerContextLength: speakerContextLength,
                    dModel: dModel,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache)
            }
        }
        let prefillElapsed = Date().timeIntervalSince(prefillStart)
        logger.info(
            "Prefill done in \(String(format: "%.2f", prefillElapsed))s "
                + "(\(hasPrefill ? "fast batched" : "slow loop"))")

        // 4. AR loop.
        let sampler = MagpieLocalSampler(
            localTransformer: MagpieLocalTransformer(weights: ltWeights),
            audioEmbeddings: constants.audioEmbeddings)

        var currentCodes = Swift.Array<Int32>(repeating: audioBosId, count: numCodebooks)
        var allFrames: [[Int32]] = []
        var finishedOnEos = false

        let rng = MagpieSamplerRng(seed: options.seed)

        // Allocate audio_embed buffer once; refill in-place each step (vDSP).
        let audioEmbed = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dModel)], dataType: .float32)

        // Pre-allocate the decoder_hidden output backing once. CoreML writes
        // straight into this array each step; we then read it fp16 → fp32 via
        // vImage. Shape: [1, 1, dModel] fp16 (per decoder_step.mlmodelc).
        let condHiddenBacking = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dModel)], dataType: .float16)
        condHiddenBacking.zeroFillFloat16()
        let uncondHiddenBacking: MLMultiArray? =
            useCfg
            ? {
                let arr = try? MLMultiArray(
                    shape: [1, 1, NSNumber(value: dModel)], dataType: .float16)
                arr?.zeroFillFloat16()
                return arr
            }() : nil

        let arLoopStart = Date()
        var decoderStepNanos: UInt64 = 0
        var samplerNanos: UInt64 = 0

        for step in 0..<options.maxSteps {
            try fillAudioEmbed(
                audioEmbed, codes: currentCodes,
                tables: constants.audioEmbeddings, dModel: dModel)

            let dsStart = DispatchTime.now()
            let condHidden = try runDecoderStep(
                audioEmbed: audioEmbed,
                encoderOutput: encoderOutput, encoderMask: encoderMask,
                cache: condCache, hiddenBacking: condHiddenBacking,
                dModel: dModel, model: decoderStep)

            let uncondHidden: [Float]?
            if useCfg, let uncondTensors = uncond, let uncondCache = uncondCache,
                let uncondBacking = uncondHiddenBacking
            {
                uncondHidden = try runDecoderStep(
                    audioEmbed: audioEmbed,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache, hiddenBacking: uncondBacking,
                    dModel: dModel, model: decoderStep)
            } else {
                uncondHidden = nil
            }
            decoderStepNanos &+= DispatchTime.now().uptimeNanoseconds &- dsStart.uptimeNanoseconds

            let forbidEos = step < options.minFrames
            let smpStart = DispatchTime.now()
            let next = sampler.sample(
                decoderHidden: condHidden,
                uncondDecoderHidden: uncondHidden,
                forbidEos: forbidEos,
                options: options,
                rng: rng)
            samplerNanos &+= DispatchTime.now().uptimeNanoseconds &- smpStart.uptimeNanoseconds

            let isEos = next.contains(audioEosId)
            if isEos && step >= options.minFrames {
                finishedOnEos = true
                logger.info("EOS at step \(step)")
                break
            }
            allFrames.append(next)
            currentCodes = next
        }

        let numFrames = allFrames.count
        guard numFrames > 0 else {
            throw MagpieError.inferenceFailed(
                stage: "synthesize", underlying: "no audio frames generated")
        }
        let arLoopElapsed = Date().timeIntervalSince(arLoopStart)
        if arLoopElapsed > 0 {
            let dsMs = Double(decoderStepNanos) / 1_000_000.0
            let smpMs = Double(samplerNanos) / 1_000_000.0
            logger.info(
                "AR loop: \(numFrames) frames in "
                    + "\(String(format: "%.2f", arLoopElapsed))s "
                    + "(\(String(format: "%.1f", Double(numFrames) / arLoopElapsed)) fps) "
                    + "decoder=\(String(format: "%.0f", dsMs))ms "
                    + "(\(String(format: "%.1f", dsMs / Double(numFrames)))ms/step) "
                    + "sampler=\(String(format: "%.0f", smpMs))ms "
                    + "(\(String(format: "%.1f", smpMs / Double(numFrames)))ms/step)")
        }

        // 5. Reshape (numFrames × numCodebooks) into per-codebook rows; the
        //    actual nanocodec decode is run by the caller (so the chunked
        //    path can overlap it with the next chunk's AR loop).
        var codebookRows = Swift.Array(
            repeating: Swift.Array<Int32>(repeating: 0, count: numFrames),
            count: numCodebooks)
        for t in 0..<numFrames {
            let row = allFrames[t]
            for cb in 0..<numCodebooks {
                codebookRows[cb][t] = row[cb]
            }
        }

        return ChunkFrames(
            codebookRows: codebookRows,
            numFrames: numFrames,
            finishedOnEos: finishedOnEos,
            textEncoderSeconds: textEncoderSeconds,
            prefillSeconds: prefillElapsed,
            arLoopSeconds: arLoopElapsed,
            decoderStepSeconds: Double(decoderStepNanos) / 1_000_000_000.0,
            samplerSeconds: Double(samplerNanos) / 1_000_000_000.0)
    }

    // MARK: - Model runners

    nonisolated private func runTextEncoder(
        tokenized: MagpieTokenizedText, maxTextLen: Int, model: MLModel
    ) throws -> (encoderOutput: MLMultiArray, encoderMask: MLMultiArray) {
        let tokenArr = try MLMultiArray(
            shape: [1, NSNumber(value: maxTextLen)], dataType: .int32)
        tokenArr.withUnsafeMutableBytes { ptr, _ in
            let base = ptr.bindMemory(to: Int32.self).baseAddress!
            for i in 0..<maxTextLen { base[i] = tokenized.paddedIds[i] }
        }
        let maskArr = try MLMultiArray(
            shape: [1, NSNumber(value: maxTextLen)], dataType: .float32)
        maskArr.withUnsafeMutableBytes { ptr, _ in
            let base = ptr.bindMemory(to: Float.self).baseAddress!
            for i in 0..<maxTextLen { base[i] = tokenized.mask[i] }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "text_tokens": MLFeatureValue(multiArray: tokenArr),
            "text_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let out = try model.prediction(from: provider)
        guard let encoderOutput = out.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw MagpieError.inferenceFailed(
                stage: "text_encoder", underlying: "missing encoder_output key")
        }
        return (encoderOutput, maskArr)
    }

    nonisolated private func runDecoderStep(
        audioEmbed: MLMultiArray,
        encoderOutput: MLMultiArray,
        encoderMask: MLMultiArray,
        cache: MagpieKvCache,
        hiddenBacking: MLMultiArray,
        dModel: Int,
        model: MLModel
    ) throws -> [Float] {
        var inputs: [String: MLMultiArray] = [
            "audio_embed": audioEmbed,
            "encoder_output": encoderOutput,
            "encoder_mask": encoderMask,
        ]
        cache.addInputs(to: &inputs)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })

        // Bind every output to a pre-allocated MLMultiArray so CoreML writes
        // in place instead of allocating ~18.9 MB of fresh fp16 buffers per
        // step. The cache provides 24 K/V + 12 position back-buffers, the
        // synthesizer provides the 1 hidden buffer. After the call,
        // `swapBackings` promotes back→front for the next step's inputs.
        var backings: [String: Any] = [:]
        cache.addOutputBackings(to: &backings)
        backings[MagpieKvCache.decoderHiddenKey] = hiddenBacking
        let predOpts = MLPredictionOptions()
        predOpts.outputBackings = backings

        _ = try model.prediction(from: provider, options: predOpts)
        cache.swapBackings()

        // Hidden state lives in `hiddenBacking` after the call. Convert fp16
        // → fp32 via vImage into a fresh [Float] result buffer (the sampler
        // wants `[Float]`).
        let dim = dModel  // hiddenBacking shape = [1, 1, dModel]
        var result = Swift.Array<Float>(repeating: 0, count: dim)
        hiddenBacking.withUnsafeBytes { raw in
            guard let src = raw.baseAddress else { return }
            var srcBuffer = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: src),
                height: 1, width: vImagePixelCount(dim), rowBytes: dim * 2)
            result.withUnsafeMutableBufferPointer { dst in
                var dstBuffer = vImage_Buffer(
                    data: dst.baseAddress, height: 1,
                    width: vImagePixelCount(dim), rowBytes: dim * 4)
                _ = vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
            }
        }
        return result
    }

    /// In-place mean-of-codebook-embeddings into a pre-allocated MLMultiArray.
    /// Replaces the per-step alloc + manual loop with vDSP primitives:
    ///   - `vDSP_vclr` zeros the buffer.
    ///   - `vDSP_vadd` accumulates each codebook's embedding row.
    ///   - `vDSP_vsmul` applies the `1 / numCodebooks` scale.
    private func fillAudioEmbed(
        _ arr: MLMultiArray, codes: [Int32], tables: [[Float]], dModel: Int
    ) throws {
        arr.withUnsafeMutableBytes { ptr, _ in
            guard let base = ptr.bindMemory(to: Float.self).baseAddress else { return }
            vDSP_vclr(base, 1, vDSP_Length(dModel))
            let numCodebooks = codes.count
            for cb in 0..<numCodebooks {
                let row = Int(codes[cb])
                tables[cb].withUnsafeBufferPointer { tablePtr in
                    guard let tableBase = tablePtr.baseAddress else { return }
                    let rowPtr = tableBase.advanced(by: row * dModel)
                    vDSP_vadd(base, 1, rowPtr, 1, base, 1, vDSP_Length(dModel))
                }
            }
            var inv = 1.0 / Float(numCodebooks)
            vDSP_vsmul(base, 1, &inv, base, 1, vDSP_Length(dModel))
        }
    }

}
