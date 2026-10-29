#if TTS && arch(arm64)
import Accelerate
@preconcurrency import CoreML
import Foundation

/// Single-consumer queue of speech frames between the LM loop and the codec: `next()` returns every frame queued
/// since the last call (the first frame alone, for time-to-first-audio), so the codec catches up with F=25 / F=4
/// calls instead of falling behind one F=1 call per frame.
actor LFM3AudioFrameQueue {
    private var pending: [[Int32]] = []
    private var finished = false
    private var waiter: CheckedContinuation<[[Int32]]?, Never>?

    func push(_ codes: [Int32]) {
        if let waiter {
            self.waiter = nil
            waiter.resume(returning: [codes])
        } else {
            pending.append(codes)
        }
    }

    func finish() {
        finished = true
        if let waiter {
            self.waiter = nil
            waiter.resume(returning: pending.isEmpty ? nil : pending)
            pending = []
        }
    }

    /// Queued frames, waiting for at least one; `nil` once finished and drained.
    func next() async -> [[Int32]]? {
        if !pending.isEmpty {
            defer { pending = [] }
            return pending
        }
        if finished { return nil }
        return await withCheckedContinuation { waiter = $0 }
    }
}

/// Streaming LiquidCodec decoder: `CodecDecoder-fp16` (functions f1 / f4 / f25 with explicit conv / K / V state)
/// + the host fp32 synthesis head and hop-320 overlap-add with HF streaming trims. Speech frames (12.5 Hz) in,
/// 24 kHz samples out. Runs in its own actor so audio is decoded (on the ANE) while the LM generates.
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
actor LFM3AudioCodecWorker {
    private final class Function {
        let model: MLModel
        let inputs: [String: MLMultiArray]
        let outputs: [[String: MLMultiArray]]
        let providers: [LFM3AudioInputs]
        let options: [MLPredictionOptions]

        init(model: MLModel) throws {
            self.model = model
            inputs = try LFM3AudioArrays.zeroInputs(model.modelDescription)
            var outputs: [[String: MLMultiArray]] = []
            var options: [MLPredictionOptions] = []
            for _ in 0..<2 {
                let backing = try LFM3AudioArrays.outputBackings(model.modelDescription)
                let option = MLPredictionOptions()
                option.outputBackings = backing
                outputs.append(backing)
                options.append(option)
            }
            self.outputs = outputs
            self.options = options
            providers = [LFM3AudioInputs(inputs), LFM3AudioInputs(inputs)]
        }
    }

    private static let hop = 320
    private static let trim = 480
    private static let tail = 960

    private let functions: [Int: Function]
    private let head: LFM3AudioCodecHead
    private let fsq: [Float]  // [2016, 4] FSQ latents per code
    private var conv: MLMultiArray
    private var k: MLMultiArray
    private var v: MLMultiArray
    private var flip = 0
    private var decodedRows = 0
    private var audioTail: [Float]?
    private var envelopeTail: [Float]?
    private var started = false

    init(models: [Int: MLModel], head: LFM3AudioCodecHead) throws {
        let functions = try models.mapValues { try Function(model: $0) }
        guard let any = functions[1] else { throw LFM3AudioError.modelNotFound("CodecDecoder function f1") }
        self.functions = functions
        self.head = head
        conv = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "conv").shape.map(\.intValue))
        k = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "k").shape.map(\.intValue))
        v = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "v").shape.map(\.intValue))
        // FSQ levels [8, 7, 6, 6]: digit d of code c = (c / basis[d]) % levels[d]; latent = (digit - L/2) / (L/2)
        let levels = [8, 7, 6, 6]
        let basis = [1, 8, 56, 336]
        var table = [Float](repeating: 0, count: 2016 * 4)
        for c in 0..<2016 {
            for d in 0..<4 {
                let half = levels[d] / 2
                table[c * 4 + d] = Float((c / basis[d]) % levels[d] - half) / Float(half)
            }
        }
        fsq = table
    }

    /// Decode `frames` (12 codes each, no EOS frame) into new samples; `finalize` flushes the overlap tail.
    func push(_ frames: [[Int32]], finalize: Bool) throws -> [Float] {
        var out: [Float] = []
        var t = 0
        while t < frames.count {
            let left = frames.count - t
            let size = (left >= 25 && functions[25] != nil) ? 25 : (left >= 4 && functions[4] != nil) ? 4 : 1
            out += try decode(Array(frames[t..<(t + size)]), finalize: finalize && t + size >= frames.count)
            t += size
        }
        if finalize && frames.isEmpty { out += overlapAdd(nil, finalize: true) }
        return out
    }

    private func decode(_ frames: [[Int32]], finalize: Bool) throws -> [Float] {
        let count = frames.count
        guard let fn = functions[count] else { throw LFM3AudioError.modelNotFound("CodecDecoder function f\(count)") }
        let latents = try LFM3AudioArrays.require(fn.inputs, "lat48").lfm3Float16
        for (i, codes) in frames.enumerated() {
            for q in 0..<LFM3AudioSpeechConfig.codebooks {
                let code = Int(codes[q])
                guard code >= 0, code < 2016 else { throw LFM3AudioError.processingFailed("codec code \(code)") }
                for d in 0..<4 { latents[i * 48 + q * 4 + d] = Float16(fsq[code * 4 + d]) }
            }
        }
        let mask = LFM3AudioEncoderPlanner.fifoMask(previous: decodedRows, q: 6 * count, window: 30, fifo: 29)
        let maskPointer = try LFM3AudioArrays.require(fn.inputs, "mask").lfm3Float16
        for i in 0..<mask.count { maskPointer[i] = Float16(mask[i]) }
        let provider = fn.providers[flip]
        provider.arrays["conv"] = conv
        provider.arrays["k"] = k
        provider.arrays["v"] = v
        _ = try fn.model.prediction(from: provider, options: fn.options[flip])
        let out = fn.outputs[flip]
        conv = try LFM3AudioArrays.require(out, "conv_out")
        k = try LFM3AudioArrays.require(out, "k_out")
        v = try LFM3AudioArrays.require(out, "v_out")
        flip ^= 1
        decodedRows += 6 * count
        let hidden = try LFM3AudioArrays.require(out, "hidden")
        return overlapAdd(synthesisFrames(hidden.lfm3Float16, rows: 6 * count), finalize: finalize)
    }

    /// hidden `[rows, 512]` fp16 -> windowed irfft frames `[rows, 1280]` fp32.
    private func synthesisFrames(_ h: UnsafePointer<Float16>, rows: Int) -> [Float] {
        let latent = LFM3AudioCodecHead.latent
        let projection = LFM3AudioCodecHead.projection
        let n = LFM3AudioCodecHead.nFFT
        let bins = projection / 2
        var x = [Float](repeating: 0, count: rows * latent)
        for i in 0..<(rows * latent) { x[i] = Float(h[i]) }
        var y = [Float](repeating: 0, count: rows * projection)
        for r in 0..<rows { y.replaceSubrange((r * projection)..<((r + 1) * projection), with: head.linearBias) }
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(projection), Int32(latent), 1, x,
            Int32(latent), head.linearWeight, Int32(latent), 1, &y, Int32(projection))
        // y = [log magnitude 641 | angle 641] -> z = [Re 641 | Im 641]
        var z = [Float](repeating: 0, count: rows * projection)
        var count = Int32(bins)
        var magnitude = [Float](repeating: 0, count: bins)
        var sine = [Float](repeating: 0, count: bins)
        var cosine = [Float](repeating: 0, count: bins)
        for r in 0..<rows {
            y.withUnsafeBufferPointer { yb in
                guard let base = yb.baseAddress.map({ $0 + r * projection }) else { return }
                vvexpf(&magnitude, base, &count)
                vvsincosf(&sine, &cosine, base + bins, &count)
            }
            for j in 0..<bins {
                z[r * projection + j] = magnitude[j] * cosine[j]
                z[r * projection + bins + j] = magnitude[j] * sine[j]
            }
        }
        var frames = [Float](repeating: 0, count: rows * n)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), Int32(n), Int32(projection), 1, z,
            Int32(projection), head.basis, Int32(n), 0, &frames, Int32(n))
        return frames
    }

    /// Streaming overlap-add == HF `decode(streaming_cache)`.
    private func overlapAdd(_ frames: [Float]?, finalize: Bool) -> [Float] {
        let n = LFM3AudioCodecHead.nFFT
        let hop = Self.hop
        let tail = Self.tail
        var audio: [Float]
        var envelope: [Float]
        if let frames, !frames.isEmpty {
            let count = frames.count / n
            let length = (count - 1) * hop + n
            audio = [Float](repeating: 0, count: length)
            envelope = [Float](repeating: 0, count: length)
            for t in 0..<count {
                for j in 0..<n {
                    audio[t * hop + j] += frames[t * n + j]
                    envelope[t * hop + j] += head.window[j] * head.window[j]
                }
            }
        } else {
            guard finalize && started else { return [] }
            audio = [Float](repeating: 0, count: tail)
            envelope = [Float](repeating: 0, count: tail)
        }
        if let audioTail, let envelopeTail {
            for i in 0..<tail {
                audio[i] += audioTail[i]
                envelope[i] += envelopeTail[i]
            }
        }
        if !started {
            audio.removeFirst(Self.trim)
            envelope.removeFirst(Self.trim)
            started = true
        }
        if finalize {
            audio.removeLast(Self.trim)
            envelope.removeLast(Self.trim)
            audioTail = nil
            envelopeTail = nil
        } else {
            audioTail = Array(audio.suffix(tail))
            envelopeTail = Array(envelope.suffix(tail))
            audio.removeLast(tail)
            envelope.removeLast(tail)
        }
        var out = [Float](repeating: 0, count: audio.count)
        vDSP_vdiv(envelope, 1, audio, 1, &out, 1, vDSP_Length(audio.count))
        return out
    }
}
#endif
