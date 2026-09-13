@preconcurrency import CoreML
import Foundation

/// Drives the MOSS-TTS-Nano CoreML graphs frame by frame.
///
/// Per text chunk:
///   1. `Prefill(input_ids [1,512,17], input_len)` → `hidden`, `kv_k`, `kv_v`
///   2. loop: `Frame(hidden, randoms, sampling params, seen, greedy)` →
///      `should_continue`, 16 codes → `CodecStep(codes, frame_index, caches…)` →
///      3840 stereo samples → `Step(row, kv, cur_len)` → next `hidden`, kv.
///
/// Sampling happens inside the Frame graph (top-k / top-p / temperature /
/// repetition penalty via inverse-CDF); the host only supplies uniform randoms.
final class MossTtsNanoSynthesizer {

    private let logger = AppLogger(category: "MossTtsNanoSynthesizer")

    private let prefill: MLModel
    private let step: MLModel
    private let frame: MLModel
    private let codecStep: MLModel
    let config: MossTtsNanoConfig
    let tokenizer: MossTtsNanoTokenizer
    let builder: MossTtsNanoPromptBuilder
    private let codecCacheShapes: [(name: String, shape: [NSNumber])]

    init(
        prefill: MLModel, step: MLModel, frame: MLModel, codecStep: MLModel,
        config: MossTtsNanoConfig, tokenizer: MossTtsNanoTokenizer
    ) {
        self.prefill = prefill
        self.step = step
        self.frame = frame
        self.codecStep = codecStep
        self.config = config
        self.tokenizer = tokenizer
        self.builder = MossTtsNanoPromptBuilder(config: config)
        self.codecCacheShapes = codecStep.modelDescription.inputDescriptionsByName
            .filter { $0.key != "codes" && $0.key != "frame_index" }
            .compactMap { name, desc in
                desc.multiArrayConstraint.map { (name: name, shape: $0.shape) }
            }
            .sorted { $0.name < $1.name }
    }

    // MARK: - Planning

    /// Split `text` into chunks that fit the prefill graph next to `voice`.
    func planChunks(text: String, voice: MossTtsNanoVoice, options: MossTtsNanoSamplingOptions) throws -> [String] {
        let overhead = builder.voiceCloneOverheadRows(voice: voice)
        let capacity = config.coreml.prefillRows - overhead
        guard capacity > 0 else {
            throw MossTtsNanoError.promptTooLong(rows: overhead, capacity: config.coreml.prefillRows)
        }
        var budget = options.maxTextTokens > 0 ? options.maxTextTokens : capacity
        budget = min(budget, capacity)
        let chunks = try MossTtsNanoTextChunker.chunk(text, maxTokens: budget) { [tokenizer] in
            tokenizer.encode($0).count
        }
        for chunk in chunks {
            let rows = overhead + tokenizer.encode(chunk).count
            if rows > config.coreml.prefillRows {
                throw MossTtsNanoError.promptTooLong(rows: rows, capacity: config.coreml.prefillRows)
            }
        }
        return chunks
    }

    // MARK: - Generation

    /// Generate one chunk, calling `emit` per decoded 80 ms frame. Returns the frame count.
    func generateChunk(
        text: String,
        voice: MossTtsNanoVoice,
        options: MossTtsNanoSamplingOptions,
        rng: inout MossTtsNanoRandom,
        chunkIndex: Int,
        chunkCount: Int,
        emit: (MossTtsNanoAudioFrame) throws -> Void
    ) throws -> Int {
        let nVq = config.model.nVq
        let rowWidth = config.model.rowWidth
        let prefillRows = config.coreml.prefillRows
        let textIds = tokenizer.encode(text)
        guard !textIds.isEmpty else { throw MossTtsNanoError.emptyText }
        let rows = builder.voiceCloneRows(textIds: textIds, voice: voice)
        guard rows.count <= prefillRows else {
            throw MossTtsNanoError.promptTooLong(rows: rows.count, capacity: prefillRows)
        }

        // --- Prefill --- //
        var padded = [Int32](repeating: Int32(config.tokens.audioPad), count: prefillRows * rowWidth)
        for r in 0..<prefillRows where r >= rows.count {
            padded[r * rowWidth] = Int32(config.tokens.pad)
        }
        for (r, row) in rows.enumerated() {
            for (c, v) in row.enumerated() { padded[r * rowWidth + c] = v }
        }
        let prefillOut = try MossTtsNanoTensor.predict(
            prefill,
            [
                "input_ids": try MossTtsNanoTensor.int32(padded, shape: [1, prefillRows, rowWidth], stage: "prefill"),
                "input_len": try MossTtsNanoTensor.int32([Int32(rows.count)], shape: [1], stage: "prefill"),
            ], stage: "prefill")
        var hidden = try MossTtsNanoTensor.output(prefillOut, "hidden", stage: "prefill")
        var kvK = try MossTtsNanoTensor.output(prefillOut, "kv_k", stage: "prefill")
        var kvV = try MossTtsNanoTensor.output(prefillOut, "kv_v", stage: "prefill")

        // --- Per-frame state --- //
        var caches: [String: MLMultiArray] = [:]
        for (name, shape) in codecCacheShapes {
            caches[name] = try MossTtsNanoTensor.zeros(shape: shape)
        }
        let seen = try MossTtsNanoTensor.zeros(shape: [1, nVq, config.model.audioCodebookSize].map(NSNumber.init))
        let seenPtr = seen.dataPointer.bindMemory(to: Float.self, capacity: nVq * config.model.audioCodebookSize)
        let textTemperature = try MossTtsNanoTensor.float32([options.textTemperature], shape: [1], stage: "frame")
        let audioTemperature = try MossTtsNanoTensor.float32([options.audioTemperature], shape: [1], stage: "frame")
        let audioTopP = try MossTtsNanoTensor.float32([options.audioTopP], shape: [1], stage: "frame")
        let repetitionPenalty = try MossTtsNanoTensor.float32([options.repetitionPenalty], shape: [1], stage: "frame")
        let greedy = try MossTtsNanoTensor.float32([options.greedy ? 1 : 0], shape: [1], stage: "frame")

        let maxFrames = min(options.maxNewFrames, config.coreml.maxLen - rows.count - 1)
        var curLen = rows.count
        var produced = 0
        let samplesPerChannel = config.model.samplesPerFramePerChannel

        for t in 0..<max(0, maxFrames) {
            // --- Frame: stop decision + 16 codes --- //
            var audioU = [Float](repeating: 0, count: nVq)
            for i in 0..<nVq { audioU[i] = rng.uniform() }
            let frameOut = try MossTtsNanoTensor.predict(
                frame,
                [
                    "global_hidden": hidden,
                    "text_u": try MossTtsNanoTensor.float32([rng.uniform()], shape: [1], stage: "frame"),
                    "audio_u": try MossTtsNanoTensor.float32(audioU, shape: [1, nVq], stage: "frame"),
                    "text_temperature": textTemperature,
                    "audio_temperature": audioTemperature,
                    "audio_top_p": audioTopP,
                    "repetition_penalty": repetitionPenalty,
                    "seen": seen,
                    "greedy": greedy,
                ], stage: "frame")
            let shouldContinue =
                try MossTtsNanoTensor.ints(
                    MossTtsNanoTensor.output(frameOut, "should_continue", stage: "frame")
                ).first ?? 0
            if shouldContinue == 0 { break }
            let codes = try MossTtsNanoTensor.ints(MossTtsNanoTensor.output(frameOut, "frame", stage: "frame"))
            guard codes.count == nVq else {
                throw MossTtsNanoError.invalidTensorShape(stage: "frame", expected: "\(nVq)", got: "\(codes.count)")
            }
            for (c, code) in codes.enumerated() where code >= 0 && Int(code) < config.model.audioCodebookSize {
                seenPtr[c * config.model.audioCodebookSize + Int(code)] = 1
            }

            // --- Codec step: 16 codes → 80 ms stereo --- //
            var codecInputs: [String: MLMultiArray] = [
                "codes": try MossTtsNanoTensor.int32(codes, shape: [nVq, 1, 1], stage: "codec_step"),
                "frame_index": try MossTtsNanoTensor.int32([Int32(t)], shape: [1], stage: "codec_step"),
            ]
            for (name, cache) in caches { codecInputs[name] = cache }
            let codecOut = try MossTtsNanoTensor.predict(codecStep, codecInputs, stage: "codec_step")
            let audio = MossTtsNanoTensor.floats(try MossTtsNanoTensor.output(codecOut, "audio", stage: "codec_step"))
            guard audio.count == 2 * samplesPerChannel else {
                throw MossTtsNanoError.invalidTensorShape(
                    stage: "codec_step", expected: "\(2 * samplesPerChannel)", got: "\(audio.count)")
            }
            for (name, _) in codecCacheShapes {
                caches[name] = try MossTtsNanoTensor.output(codecOut, "\(name)_out", stage: "codec_step")
            }
            try emit(
                MossTtsNanoAudioFrame(
                    left: Array(audio[0..<samplesPerChannel]),
                    right: Array(audio[samplesPerChannel..<(2 * samplesPerChannel)]),
                    frameIndex: t, chunkIndex: chunkIndex, chunkCount: chunkCount, isPause: false))
            produced += 1

            // --- Step: advance the global LM by one row --- //
            let row = builder.generationRow(codes: codes)
            let stepOut = try MossTtsNanoTensor.predict(
                step,
                [
                    "input_ids": try MossTtsNanoTensor.int32(row, shape: [1, 1, rowWidth], stage: "step"),
                    "kv_k": kvK,
                    "kv_v": kvV,
                    "cur_len": try MossTtsNanoTensor.int32([Int32(curLen)], shape: [1], stage: "step"),
                ], stage: "step")
            hidden = try MossTtsNanoTensor.output(stepOut, "hidden", stage: "step")
            kvK = try MossTtsNanoTensor.output(stepOut, "kv_k_out", stage: "step")
            kvV = try MossTtsNanoTensor.output(stepOut, "kv_v_out", stage: "step")
            curLen += 1
        }
        if produced == maxFrames && maxFrames > 0 {
            logger.warning(
                "chunk \(chunkIndex + 1)/\(chunkCount) hit the \(maxFrames)-frame budget without a stop token")
        }
        return produced
    }

    /// Encode a 48 kHz stereo clip into voice codes with the fp32 codec encoder.
    static func encodeVoice(
        encoder: MLModel, left: [Float], right: [Float], name: String, nVq: Int, samplesPerFrame: Int
    ) throws -> MossTtsNanoVoice {
        let n = min(left.count, right.count)
        let padded = ((n + samplesPerFrame - 1) / samplesPerFrame) * samplesPerFrame
        var audio = [Float](repeating: 0, count: 2 * padded)
        for i in 0..<n {
            audio[i] = left[i]
            audio[padded + i] = right[i]
        }
        let out = try MossTtsNanoTensor.predict(
            encoder,
            ["audio": try MossTtsNanoTensor.float32(audio, shape: [1, 2, padded], stage: "codec_encoder")],
            stage: "codec_encoder")
        let flat = try MossTtsNanoTensor.ints(MossTtsNanoTensor.output(out, "codes", stage: "codec_encoder"))
        let frames = padded / samplesPerFrame
        guard flat.count == nVq * frames else {
            throw MossTtsNanoError.invalidTensorShape(
                stage: "codec_encoder", expected: "\(nVq * frames)", got: "\(flat.count)")
        }
        var codes = [[Int32]](repeating: [Int32](repeating: 0, count: nVq), count: frames)
        for q in 0..<nVq {
            for t in 0..<frames { codes[t][q] = flat[q * frames + t] }
        }
        return MossTtsNanoVoice(name: name, codes: codes)
    }
}
