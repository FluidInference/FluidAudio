#if TTS
import Accelerate
import Foundation

/// Native equivalent of the toolkit's deterministic CoreMLSineGenV2 and forward STFT.
/// Inspired by mattmireles/kokoro-coreml's vDSP source/STFT separation (Apache-2.0).
/// Uses checkpoint-exported DFT coefficients and retains our deterministic noise convention.
struct KokoroAneV3NativeSource {
    struct Parameters: Decodable {
        let linearWeights: [Float]
        let linearBias: Float
        let realBasis: [Float]
        let imagBasis: [Float]
        enum CodingKeys: String, CodingKey {
            case linearWeights = "linear_weights"
            case linearBias = "linear_bias"
            case realBasis = "real_basis"
            case imagBasis = "imag_basis"
        }
    }
    private let parameters: Parameters
    init(parametersURL: URL) throws {
        let p = try JSONDecoder().decode(Parameters.self, from: Data(contentsOf: parametersURL))
        guard p.linearWeights.count == 9, p.realBasis.count == 220, p.imagBasis.count == 220,
            (p.linearWeights + p.realBasis + p.imagBasis + [p.linearBias]).allSatisfy(\.isFinite)
        else {
            throw KokoroAneV3Error.invalidInput("Invalid native source parameters")
        }
        parameters = p
    }

    func waveform(f0: [Float]) throws -> [Float] {
        guard !f0.isEmpty, f0.count <= 4000, f0.allSatisfy(\.isFinite) else {
            throw KokoroAneV3Error.invalidInput("Expected 1...4000 finite F0 frames")
        }
        let scale = 300
        let inverseScale = 1 / Float(scale)
        let count = f0.count * scale
        var merged = [Float](repeating: parameters.linearBias, count: count)
        var phase = [Float](repeating: 0, count: count)
        var sines = [Float](repeating: 0, count: count)
        // Match the deterministic export's FP32 pooling/cumsum/interpolation.
        // No sample-rate phase integration: accumulate only at F0 frame rate.
        for harmonic in 1...9 {
            var framePhase = [Float](repeating: 0, count: f0.count)
            var accumulated: Double = 0
            for t in f0.indices {
                let increment = f0[t] * Float(harmonic) / 24000
                var pooled: Float = 0
                for _ in 0..<scale { pooled += increment }
                accumulated += Double(pooled / Float(scale))
                framePhase[t] = Float(accumulated) * (2 * Float.pi) * Float(scale)
            }
            for i in 0..<count {
                // Match PyTorch's align_corners=false index arithmetic: it
                // multiplies by the rounded FP32 reciprocal, rather than dividing.
                let position = max(0, min(Float(f0.count - 1), inverseScale * (Float(i) + 0.5) - 0.5))
                let lo = Int(position)
                let hi = min(lo + 1, f0.count - 1)
                let fraction = position - Float(lo)
                phase[i] = framePhase[lo] * (1 - fraction) + framePhase[hi] * fraction
            }
            var n = Int32(count)
            vvsinf(&sines, phase, &n)
            let weight = parameters.linearWeights[harmonic - 1]
            for t in f0.indices {
                let voiced: Float = f0[t] > 10 ? 1 : 0
                let deterministicNoise: Float = (voiced * 0.003 + (1 - voiced) * (0.1 / 3)) * 0.01
                for j in 0..<scale {
                    let i = t * scale + j
                    merged[i] += (sines[i] * 0.1 * voiced + deterministicNoise) * weight
                }
            }
        }
        var n = Int32(count)
        vvtanhf(&merged, merged, &n)
        return merged
    }

    /// Channel-major [1,22,T*60+1] magnitude and phase. Positive pi at negative-real DC/Nyquist.
    func spectrum(waveform: [Float]) throws -> [Float] {
        guard !waveform.isEmpty, waveform.allSatisfy(\.isFinite) else {
            throw KokoroAneV3Error.invalidInput("Expected finite nonempty waveform")
        }
        let frames = waveform.count / 5 + 1
        let padded =
            [Float](repeating: waveform[0], count: 10) + waveform
            + [Float](repeating: waveform[waveform.count - 1], count: 10)
        var result = [Float](repeating: 0, count: frames * 22)
        var real = [Float](repeating: 0, count: frames)
        var imaginary = real
        var squared = real
        var epsilon: Float = 1e-14
        var n = Int32(frames)
        // Use the same im2col + SGEMM formulation as the reference Conv1d.
        // Different dot-product reduction orders can flip low-energy STFT
        // phase by 2*pi, which is significant to the learned noise network.
        var windows = [Float](repeating: 0, count: 20 * frames)
        for tap in 0..<20 {
            for frame in 0..<frames { windows[tap * frames + frame] = padded[frame * 5 + tap] }
        }
        var components = [Float](repeating: 0, count: 22 * frames)
        let basis = parameters.realBasis + parameters.imagBasis
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            22, Int32(frames), 20, 1, basis, 20, windows, Int32(frames),
            0, &components, Int32(frames))
        result.withUnsafeMutableBufferPointer { output in
            guard let outputBase = output.baseAddress else { return }
            for k in 0..<11 {
                real = Array(components[(k * frames)..<((k + 1) * frames)])
                if k == 0 || k == 10 {
                    vDSP_vclr(&imaginary, 1, vDSP_Length(frames))
                } else {
                    imaginary = Array(components[((k + 11) * frames)..<((k + 12) * frames)])
                }
                vDSP_vsq(real, 1, &squared, 1, vDSP_Length(frames))
                vDSP_vma(imaginary, 1, imaginary, 1, squared, 1, &squared, 1, vDSP_Length(frames))
                let magnitude = outputBase + k * frames
                let phase = outputBase + (k + 11) * frames
                vDSP_vsadd(squared, 1, &epsilon, magnitude, 1, vDSP_Length(frames))
                vvsqrtf(magnitude, magnitude, &n)
                vvatan2f(phase, imaginary, real, &n)
            }
        }
        return result
    }

    func build(f0: [Float]) throws -> [Float] {
        try spectrum(waveform: waveform(f0: f0))
    }
}

enum KokoroAneV3Error: Error {
    case invalidInput(String)
    case missingOutput(String)
    case missingModel(String)
}
#endif
