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

/// Streaming overlap-add of LiquidCodec synthesis frames == HF `decode(streaming_cache)`: hop 320 over windowed
/// 1280-sample frames, divided by the overlapped window² envelope; the first call drops 480 samples, every call
/// holds back a 960-sample tail for the next one, and `finalize` drops the last 480. Buffers are allocated once.
final class LFM3AudioOverlapAdd {
    static let hop = 320
    static let trim = 480
    static let tail = 960
    let maxRows: Int
    private let n: Int
    private let windowSquared: UnsafeMutablePointer<Float>
    private let audio: UnsafeMutablePointer<Float>
    private let envelope: UnsafeMutablePointer<Float>
    private let audioTail: UnsafeMutablePointer<Float>
    private let envelopeTail: UnsafeMutablePointer<Float>
    private var started = false
    private var hasTail = false

    init(window: [Float], maxRows: Int) {
        n = window.count
        self.maxRows = maxRows
        let length = max((maxRows - 1) * Self.hop + window.count, Self.tail)
        func buffer(_ count: Int) -> UnsafeMutablePointer<Float> {
            let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
            p.initialize(repeating: 0, count: count)
            return p
        }
        let squared = buffer(window.count)
        for (j, w) in window.enumerated() { squared[j] = w * w }
        windowSquared = squared
        audio = buffer(length)
        envelope = buffer(length)
        audioTail = buffer(Self.tail)
        envelopeTail = buffer(Self.tail)
    }

    deinit {
        for p in [windowSquared, audio, envelope, audioTail, envelopeTail] { p.deallocate() }
    }

    func reset() {
        started = false
        hasTail = false
    }

    /// Adds `rows` frames (`[rows, window.count]` fp32) and returns the samples that are complete. With `rows` 0 and
    /// `finalize`, flushes the held-back tail; after the stream is finalized it returns no samples.
    func add(_ frames: UnsafePointer<Float>, rows: Int, finalize: Bool) throws -> [Float] {
        guard rows <= maxRows else {
            throw LFM3AudioError.processingFailed("overlap-add of \(rows) rows (max \(maxRows))")
        }
        let hop = Self.hop
        let tail = Self.tail
        let length: Int
        if rows > 0 {
            length = (rows - 1) * hop + n
            audio.update(repeating: 0, count: length)
            envelope.update(repeating: 0, count: length)
            for t in 0..<rows {  // frame by frame, as the scalar overlap-add
                vDSP_vadd(audio + t * hop, 1, frames + t * n, 1, audio + t * hop, 1, vDSP_Length(n))
                vDSP_vadd(envelope + t * hop, 1, windowSquared, 1, envelope + t * hop, 1, vDSP_Length(n))
            }
        } else {
            guard finalize && started && hasTail else { return [] }
            length = tail
            audio.update(repeating: 0, count: tail)
            envelope.update(repeating: 0, count: tail)
        }
        if hasTail {
            for i in 0..<tail {
                audio[i] += audioTail[i]
                envelope[i] += envelopeTail[i]
            }
        }
        var lo = 0
        var hi = length
        if !started {
            lo = Self.trim
            started = true
        }
        if finalize {
            hi -= Self.trim
            hasTail = false
        } else {
            audioTail.update(from: audio + hi - tail, count: tail)
            envelopeTail.update(from: envelope + hi - tail, count: tail)
            hasTail = true
            hi -= tail
        }
        guard hi > lo else { return [] }
        var out = [Float](repeating: 0, count: hi - lo)
        vDSP_vdiv(envelope + lo, 1, audio + lo, 1, &out, 1, vDSP_Length(hi - lo))
        return out
    }
}

/// Streaming LiquidCodec decoder: `CodecDecoder-fp16` (functions f1 / f4 / f25 with explicit conv / K / V state)
/// + the host fp32 synthesis head and the streaming overlap-add. Speech frames (12.5 Hz) in, 24 kHz samples out.
/// Runs in its own actor so audio is decoded (on the ANE) while the LM generates; one worker serves every utterance
/// of a manager (`reset()` between them), so its buffers are allocated once.
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
actor LFM3AudioCodecWorker {
    private final class Function {
        let model: MLModel
        let inputs: [String: MLMultiArray]
        let outputs: [[String: MLMultiArray]]
        let providers: [LFM3AudioInputs]
        let options: [MLPredictionOptions]
        /// The fifo mask stops changing once 29 rows are decoded; set when that steady mask is in `inputs`.
        var steadyMask = false

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

    /// Synthesis-head buffers, sized for the largest loaded function.
    private final class Scratch {
        let latents: UnsafeMutablePointer<Float>
        let projected: UnsafeMutablePointer<Float>
        let spectrum: UnsafeMutablePointer<Float>
        let magnitude: UnsafeMutablePointer<Float>
        let sine: UnsafeMutablePointer<Float>
        let cosine: UnsafeMutablePointer<Float>
        let frames: UnsafeMutablePointer<Float>

        init(rows: Int) {
            func buffer(_ count: Int) -> UnsafeMutablePointer<Float> {
                let p = UnsafeMutablePointer<Float>.allocate(capacity: count)
                p.initialize(repeating: 0, count: count)
                return p
            }
            let bins = LFM3AudioCodecHead.projection / 2
            latents = buffer(rows * LFM3AudioCodecHead.latent)
            projected = buffer(rows * LFM3AudioCodecHead.projection)
            spectrum = buffer(rows * LFM3AudioCodecHead.projection)
            magnitude = buffer(bins)
            sine = buffer(bins)
            cosine = buffer(bins)
            frames = buffer(rows * LFM3AudioCodecHead.nFFT)
        }

        deinit {
            for p in [latents, projected, spectrum, magnitude, sine, cosine, frames] { p.deallocate() }
        }
    }

    private static let maskFifo = 29

    private let functions: [Int: Function]
    private let head: LFM3AudioCodecHead
    private let fsq: [Float]  // [2016, 4] FSQ latents per code
    private let scratch: Scratch
    private let overlapAdd: LFM3AudioOverlapAdd
    private let zeroConv: MLMultiArray
    private let zeroK: MLMultiArray
    private let zeroV: MLMultiArray
    private var conv: MLMultiArray
    private var k: MLMultiArray
    private var v: MLMultiArray
    private var flip = 0
    private var decodedRows = 0

    init(models: [Int: MLModel], head: LFM3AudioCodecHead) throws {
        let functions = try models.mapValues { try Function(model: $0) }
        guard let any = functions[1], let largest = functions.keys.max() else {
            throw LFM3AudioError.modelNotFound("CodecDecoder function f1")
        }
        self.functions = functions
        self.head = head
        zeroConv = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "conv").shape.map(\.intValue))
        zeroK = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "k").shape.map(\.intValue))
        zeroV = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "v").shape.map(\.intValue))
        conv = zeroConv
        k = zeroK
        v = zeroV
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
        scratch = Scratch(rows: 6 * largest)
        overlapAdd = LFM3AudioOverlapAdd(window: head.window, maxRows: 6 * largest)
    }

    /// Start a new utterance: zero codec state, rows decoded and the overlap-add.
    func reset() {
        for array in [zeroConv, zeroK, zeroV] { array.lfm3Float16.update(repeating: 0, count: array.count) }
        conv = zeroConv
        k = zeroK
        v = zeroV
        flip = 0
        decodedRows = 0
        for fn in functions.values { fn.steadyMask = false }
        overlapAdd.reset()
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
        if finalize && frames.isEmpty { out += try overlapAdd.add(scratch.frames, rows: 0, finalize: true) }
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
        // fifoMask(previous:) only depends on min(previous, 29): write it until it reaches its steady value.
        let steady = decodedRows >= Self.maskFifo
        if !(steady && fn.steadyMask) {
            let mask = LFM3AudioEncoderPlanner.fifoMask(
                previous: decodedRows, q: 6 * count, window: 30, fifo: Self.maskFifo)
            let maskPointer = try LFM3AudioArrays.require(fn.inputs, "mask").lfm3Float16
            for i in 0..<mask.count { maskPointer[i] = Float16(mask[i]) }
            fn.steadyMask = steady
        }
        let provider = fn.providers[flip]
        provider.set(conv, for: "conv")
        provider.set(k, for: "k")
        provider.set(v, for: "v")
        _ = try fn.model.prediction(from: provider, options: fn.options[flip])
        let out = fn.outputs[flip]
        conv = try LFM3AudioArrays.require(out, "conv_out")
        k = try LFM3AudioArrays.require(out, "k_out")
        v = try LFM3AudioArrays.require(out, "v_out")
        flip ^= 1
        decodedRows += 6 * count
        let hidden = try LFM3AudioArrays.require(out, "hidden")
        synthesisFrames(hidden.lfm3Float16, rows: 6 * count)
        return try overlapAdd.add(scratch.frames, rows: 6 * count, finalize: finalize)
    }

    /// hidden `[rows, 512]` fp16 -> windowed irfft frames `[rows, 1280]` fp32 in `scratch.frames`.
    private func synthesisFrames(_ h: UnsafePointer<Float16>, rows: Int) {
        let latent = LFM3AudioCodecHead.latent
        let projection = LFM3AudioCodecHead.projection
        let n = LFM3AudioCodecHead.nFFT
        let bins = projection / 2
        let b = scratch
        h.withMemoryRebound(to: UInt16.self, capacity: rows * latent) {
            Float16Conversion.toFloat32(src: $0, dst: b.latents, count: rows * latent)
        }
        head.linearBias.withUnsafeBufferPointer { bias in
            guard let base = bias.baseAddress else { return }
            for r in 0..<rows { (b.projected + r * projection).update(from: base, count: projection) }
        }
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(projection), Int32(latent), 1, b.latents,
            Int32(latent), head.linearWeight, Int32(latent), 1, b.projected, Int32(projection))
        // projected = [log magnitude 641 | angle 641] -> spectrum = [Re 641 | Im 641]
        var count = Int32(bins)
        for r in 0..<rows {
            let y = b.projected + r * projection
            let z = b.spectrum + r * projection
            vvexpf(b.magnitude, y, &count)
            vvsincosf(b.sine, b.cosine, y + bins, &count)
            vDSP_vmul(b.magnitude, 1, b.cosine, 1, z, 1, vDSP_Length(bins))
            vDSP_vmul(b.magnitude, 1, b.sine, 1, z + bins, 1, vDSP_Length(bins))
        }
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), Int32(n), Int32(projection), 1, b.spectrum,
            Int32(projection), head.basis, Int32(n), 0, b.frames, Int32(n))
    }
}
#endif
