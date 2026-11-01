#if arch(arm64)
import Accelerate
import Foundation

/// `[128, T]` log-mel matrix (mel-major, row-major) with column slicing for the encoder planner.
struct LFM3AudioMelBuffer: Sendable {
    static let mels = 128
    var data: [Float]
    var frames: Int

    init() {
        data = []
        frames = 0
    }

    init(data: [Float], frames: Int) {
        self.data = data
        self.frames = frames
    }

    mutating func append(_ other: LFM3AudioMelBuffer) {
        guard other.frames > 0 else { return }
        let total = frames + other.frames
        var merged = [Float](repeating: 0, count: Self.mels * total)
        for m in 0..<Self.mels {
            for t in 0..<frames { merged[m * total + t] = data[m * frames + t] }
            for t in 0..<other.frames { merged[m * total + frames + t] = other.data[m * other.frames + t] }
        }
        data = merged
        frames = total
    }

    func columns(_ lo: Int, _ hi: Int) -> LFM3AudioMelBuffer {
        let n = hi - lo
        var out = [Float](repeating: 0, count: Self.mels * n)
        for m in 0..<Self.mels { for t in 0..<n { out[m * n + t] = data[m * frames + lo + t] } }
        return LFM3AudioMelBuffer(data: out, frames: n)
    }

    mutating func dropFirst(_ n: Int) { self = columns(n, frames) }
}

/// fp32 log-mel front end of LFM3-Audio (== the HF feature extractor): periodic Hann(400), n_fft 400, hop 160,
/// center=False, |rfft|^2 (201 bins) -> 128 Slaney mel filters (0-8 kHz) -> log10 clamped at -12 -> (x + 4) / 4.
/// The 400-point DFT is one sgemm with the window folded into stacked cos/sin bases (400 is not a vDSP size).
struct LFM3AudioLogMel: Sendable {
    static let nFFT = 400
    static let hop = 160
    static let bins = 201

    private let basis: [Float]  // [402, 400]
    private let filters: [Float]  // [128, 201]

    init() {
        let n = Self.nFFT
        let nb = Self.bins
        var b = [Float](repeating: 0, count: 2 * nb * n)
        for k in 0..<nb {
            for t in 0..<n {
                let w = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(t) / Double(n))
                let angle = 2.0 * Double.pi * Double((k * t) % n) / Double(n)
                b[k * n + t] = Float(cos(angle) * w)
                b[(nb + k) * n + t] = Float(-sin(angle) * w)
            }
        }
        basis = b
        filters = Self.melFilters()
    }

    private static func hzToMel(_ f: Double) -> Double {
        f >= 1000 ? 15.0 + log(max(f, 1e-30) / 1000.0) * (27.0 / log(6.4)) : 3.0 * f / 200.0
    }

    private static func melToHz(_ m: Double) -> Double {
        m >= 15 ? 1000.0 * exp((log(6.4) / 27.0) * (m - 15.0)) : 200.0 * m / 3.0
    }

    /// Slaney-normalized filter bank `[128, 201]`, computed in fp64 then rounded to fp32.
    private static func melFilters(mels: Int = LFM3AudioMelBuffer.mels, sampleRate: Double = 16_000) -> [Float] {
        let lo = hzToMel(0)
        let hi = hzToMel(sampleRate / 2)
        let points = (0..<(mels + 2)).map { lo + (hi - lo) * Double($0) / Double(mels + 1) }.map(melToHz)
        let fft = (0..<bins).map { Double($0) * (sampleRate / 2) / Double(bins - 1) }
        var out = [Float](repeating: 0, count: mels * bins)
        for m in 0..<mels {
            let norm = 2.0 / (points[m + 2] - points[m])
            for k in 0..<bins {
                let down = -(points[m] - fft[k]) / (points[m + 1] - points[m])
                let up = (points[m + 2] - fft[k]) / (points[m + 2] - points[m + 1])
                out[m * bins + k] = Float(max(0, min(down, up)) * norm)
            }
        }
        return out
    }

    static func paddedLength(_ n: Int) -> Int {
        n < nFFT ? nFFT : n + ((-(n - nFFT)) % hop + hop) % hop
    }

    static func frameCount(samples n: Int) -> Int { 1 + (paddedLength(n) - nFFT) / hop }

    /// Encoder tokens (12.5 Hz) for `frames` mel frames (100 Hz): the stride-8 encoder plus the HF +7 flush.
    static func tokenCount(frames: Int) -> Int { (frames + 7) / 8 }

    /// Log-mel of an already padded signal: frame t covers samples `[160 t, 160 t + 400)`.
    func frames(_ x: UnsafePointer<Float>, count: Int) -> LFM3AudioMelBuffer {
        let n = Self.nFFT
        let nb = Self.bins
        let nm = LFM3AudioMelBuffer.mels
        guard count >= n else { return LFM3AudioMelBuffer() }
        let frames = 1 + (count - n) / Self.hop
        var framed = [Float](repeating: 0, count: n * frames)
        framed.withUnsafeMutableBufferPointer { f in
            for t in 0..<frames {
                let src = x + t * Self.hop
                for i in 0..<n { f[i * frames + t] = src[i] }
            }
        }
        var spectrum = [Float](repeating: 0, count: 2 * nb * frames)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(2 * nb), Int32(frames), Int32(n), 1, basis, Int32(n),
            framed, Int32(frames), 0, &spectrum, Int32(frames))
        var power = [Float](repeating: 0, count: nb * frames)
        spectrum.withUnsafeBufferPointer { s in
            power.withUnsafeMutableBufferPointer { p in
                guard let re = s.baseAddress, let out = p.baseAddress else { return }
                let im = re + nb * frames
                vDSP_vmma(re, 1, re, 1, im, 1, im, 1, out, 1, vDSP_Length(nb * frames))  // re^2 + im^2
            }
        }
        var mel = [Float](repeating: 0, count: nm * frames)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(nm), Int32(frames), Int32(nb), 1, filters, Int32(nb),
            power, Int32(frames), 0, &mel, Int32(frames))
        var total = Int32(nm * frames)
        mel.withUnsafeMutableBufferPointer { m in
            guard let p = m.baseAddress else { return }
            vvlog10f(p, p, &total)
            var lo: Float = -12
            var hi: Float = .greatestFiniteMagnitude
            vDSP_vclip(p, 1, &lo, &hi, p, 1, vDSP_Length(nm * frames))
            var scale: Float = 0.25  // (x + 4) / 4
            var offset: Float = 1
            vDSP_vsmsa(p, 1, &scale, &offset, p, 1, vDSP_Length(nm * frames))
        }
        return LFM3AudioMelBuffer(data: mel, frames: frames)
    }

    /// Offline clip (right-padded like the feature extractor) -> `[128, T]`.
    func callAsFunction(_ samples: ArraySlice<Float>) -> LFM3AudioMelBuffer {
        var padded = [Float](repeating: 0, count: Self.paddedLength(samples.count))
        padded.replaceSubrange(0..<samples.count, with: samples)
        return padded.withUnsafeBufferPointer { p in
            guard let base = p.baseAddress else { return LFM3AudioMelBuffer() }
            return frames(base, count: p.count)
        }
    }
}

/// Streaming log-mel (FE.stream): push sample pieces, get the newly completed frames.
struct LFM3AudioMelStream {
    private let mel: LFM3AudioLogMel
    private var pending: [Float] = []
    private var totalSamples = 0
    private var emittedFrames = 0
    private(set) var finished = false

    init(_ mel: LFM3AudioLogMel) { self.mel = mel }

    mutating func push(_ chunk: [Float], isFinal: Bool = false) -> LFM3AudioMelBuffer {
        guard !finished else { return LFM3AudioMelBuffer() }
        var samples = pending + chunk
        totalSamples += chunk.count
        let count: Int
        if isFinal {
            count = LFM3AudioLogMel.frameCount(samples: totalSamples) - emittedFrames
            if count > 0 {
                let size = (count - 1) * LFM3AudioLogMel.hop + LFM3AudioLogMel.nFFT
                if samples.count < size { samples += [Float](repeating: 0, count: size - samples.count) }
            }
        } else {
            count =
                samples.count < LFM3AudioLogMel.nFFT
                ? 0 : 1 + (samples.count - LFM3AudioLogMel.nFFT) / LFM3AudioLogMel.hop
        }
        var out = LFM3AudioMelBuffer()
        if count > 0 {
            let length = (count - 1) * LFM3AudioLogMel.hop + LFM3AudioLogMel.nFFT
            out = samples.withUnsafeBufferPointer { p in
                guard let base = p.baseAddress else { return LFM3AudioMelBuffer() }
                return mel.frames(base, count: length)
            }
        }
        pending = isFinal ? [] : Array(samples[min(samples.count, max(count, 0) * LFM3AudioLogMel.hop)...])
        emittedFrames += max(count, 0)
        finished = isFinal
        return out
    }
}
#endif
