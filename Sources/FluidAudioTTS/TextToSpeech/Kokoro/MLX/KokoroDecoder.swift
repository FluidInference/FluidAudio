import Accelerate
import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Accelerate ISTFT

/// Performs inverse STFT using Accelerate's vDSP or direct DFT for small sizes
final class AccelerateISTFT {
    let nFft: Int
    let hopSize: Int
    let window: [Float]
    private let fftSetup: vDSP_DFT_Setup?
    private let useDirectDFT: Bool
    // Precomputed twiddle factors for direct DFT
    private let twiddleReal: [[Float]]
    private let twiddleImag: [[Float]]

    init(nFft: Int, hopSize: Int) {
        self.nFft = nFft
        self.hopSize = hopSize

        // Create periodic Hann window (matching Python: window_fn(N+1)[:-1])
        // This is different from symmetric window: uses N instead of N-1 in denominator
        // w[n] = 0.5 * (1 - cos(2πn / N)) for n = 0..N-1
        var hannWindow = [Float](repeating: 0, count: nFft)
        for n in 0..<nFft {
            let angle = 2.0 * Float.pi * Float(n) / Float(nFft)  // Note: nFft, not nFft-1
            hannWindow[n] = 0.5 * (1.0 - cos(angle))
        }
        self.window = hannWindow

        // Try to create vDSP setup, fall back to direct DFT if not supported
        if let setup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(nFft),
            vDSP_DFT_Direction.INVERSE
        ) {
            self.fftSetup = setup
            self.useDirectDFT = false
            self.twiddleReal = []
            self.twiddleImag = []
        } else {
            // Use direct DFT for unsupported sizes
            self.fftSetup = nil
            self.useDirectDFT = true

            // Precompute twiddle factors: e^(i*2*pi*k*n/N) for inverse DFT
            var tReal: [[Float]] = []
            var tImag: [[Float]] = []
            let twoPiOverN = 2.0 * Float.pi / Float(nFft)
            for n in 0..<nFft {
                var rowReal: [Float] = []
                var rowImag: [Float] = []
                for k in 0..<nFft {
                    let angle = twoPiOverN * Float(k) * Float(n)
                    rowReal.append(cos(angle))
                    rowImag.append(sin(angle))
                }
                tReal.append(rowReal)
                tImag.append(rowImag)
            }
            self.twiddleReal = tReal
            self.twiddleImag = tImag
        }
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// Perform direct IDFT: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(i*2*pi*k*n/N)
    private func directIDFT(realInput: [Float], imagInput: [Float]) -> [Float] {
        var realOutput = [Float](repeating: 0, count: nFft)
        let scale = 1.0 / Float(nFft)

        for n in 0..<nFft {
            var sumReal: Float = 0
            for k in 0..<nFft {
                // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                // We only need real part for real output
                sumReal += realInput[k] * twiddleReal[n][k] - imagInput[k] * twiddleImag[n][k]
            }
            realOutput[n] = sumReal * scale
        }

        return realOutput
    }

    /// Phase unwrapping along time axis to ensure temporal coherence
    /// This is critical for ISTFT - without it, phase discontinuities cause noise
    private func unwrapPhase(_ phase: [[Float]]) -> [[Float]] {
        let numBins = phase.count
        guard numBins > 0 else { return phase }
        let numFrames = phase[0].count

        var unwrapped: [[Float]] = Array(repeating: [Float](repeating: 0, count: numFrames), count: numBins)

        for bin in 0..<numBins {
            unwrapped[bin][0] = phase[bin][0]
            for t in 1..<numFrames {
                var diff = phase[bin][t] - phase[bin][t - 1]

                // Wrap diff to [-π, π]
                while diff > Float.pi {
                    diff -= 2 * Float.pi
                }
                while diff < -Float.pi {
                    diff += 2 * Float.pi
                }

                unwrapped[bin][t] = unwrapped[bin][t - 1] + diff
            }
        }

        return unwrapped
    }

    /// Perform ISTFT on magnitude and phase spectrograms
    /// - Parameters:
    ///   - magnitude: [batch, nFft/2+1, numFrames]
    ///   - phase: [batch, nFft/2+1, numFrames]
    /// - Returns: Audio waveform [batch, samples]
    func inverse(_ magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        // Evaluate to ensure data is ready
        eval(magnitude, phase)

        let batchSize = magnitude.dim(0)
        let numBins = magnitude.dim(1)
        let numFrames = magnitude.dim(2)
        let outputLength = (numFrames - 1) * hopSize + nFft

        var batchOutput: [[Float]] = []

        for b in 0..<batchSize {
            var output = [Float](repeating: 0, count: outputLength)
            var windowSum = [Float](repeating: 0, count: outputLength)

            // Get magnitude and phase for this batch
            let magBatch = magnitude[b]
            let phaseBatch = phase[b]
            eval(magBatch, phaseBatch)

            // Extract all phase data for unwrapping (shape: [numBins, numFrames])
            var phaseData: [[Float]] = []
            for bin in 0..<numBins {
                let binPhase = phaseBatch[bin].asArray(Float.self)
                phaseData.append(binPhase)
            }

            // Apply phase unwrapping along time axis (matching Python mlx_unwrap)
            phaseData = unwrapPhase(phaseData)

            for frameIdx in 0..<numFrames {
                // Extract frame
                let magFrame = magBatch[0..., frameIdx]
                eval(magFrame)

                let magData = magFrame.asArray(Float.self)

                // Convert magnitude/phase to complex (real/imag)
                // Phase from sin(x) is in [-1, 1], use directly as phase angle
                // (Python unwrap has no effect for values in [-1, 1] since |diff| < π)
                var realInput = [Float](repeating: 0, count: nFft)
                var imagInput = [Float](repeating: 0, count: nFft)

                // Fill positive frequencies (DC to Nyquist)
                // Phase is sin(phaseRaw) which is used as the angle for cos/sin
                // This matches Python: real = mag * cos(sin(phaseRaw)), imag = mag * sin(sin(phaseRaw))
                for i in 0..<numBins {
                    let phaseValue = phaseData[i][frameIdx]  // This is sin(phaseRaw)
                    realInput[i] = magData[i] * cos(phaseValue)
                    imagInput[i] = magData[i] * sin(phaseValue)
                }

                // Mirror for negative frequencies (conjugate symmetry for real signal)
                // For IDFT of real signal: X[N-k] = X[k]*, so real[N-k] = real[k], imag[N-k] = -imag[k]
                for i in 1..<numBins {
                    let mirrorIdx = nFft - i
                    if mirrorIdx < nFft && mirrorIdx != i {
                        realInput[mirrorIdx] = realInput[i]
                        imagInput[mirrorIdx] = -imagInput[i]
                    }
                }

                // Perform inverse FFT
                var realOutput: [Float]

                if useDirectDFT {
                    realOutput = directIDFT(realInput: realInput, imagInput: imagInput)
                } else {
                    realOutput = [Float](repeating: 0, count: nFft)
                    var imagOutput = [Float](repeating: 0, count: nFft)

                    vDSP_DFT_Execute(
                        fftSetup!,
                        realInput, imagInput,
                        &realOutput, &imagOutput
                    )

                    // Scale by 1/N (vDSP doesn't normalize)
                    var scale = Float(1.0) / Float(nFft)
                    vDSP_vsmul(realOutput, 1, &scale, &realOutput, 1, vDSP_Length(nFft))
                }

                // Apply window and overlap-add
                // Python: updates_reconstructed = (frames_time * w).flatten()
                //         updates_window = mx.tile(w, (num_frames,)).flatten()
                let start = frameIdx * hopSize
                for i in 0..<nFft {
                    let outIdx = start + i
                    if outIdx < outputLength {
                        output[outIdx] += realOutput[i] * window[i]
                        windowSum[outIdx] += window[i]  // Just w, not w*w
                    }
                }
            }

            // Normalize by window sum (standard ISTFT normalization)
            let minWindowSum: Float = 1e-8
            for i in 0..<outputLength {
                if windowSum[i] > minWindowSum {
                    output[i] /= windowSum[i]
                }
            }

            // Remove edge artifacts (center=True in Python ISTFT)
            // Removes nFft/2 samples from each end
            let trimStart = nFft / 2
            let trimEnd = outputLength - nFft / 2
            if trimEnd > trimStart {
                let trimmed = Array(output[trimStart..<trimEnd])
                batchOutput.append(trimmed)
            } else {
                batchOutput.append(output)
            }
        }

        // Convert back to MLXArray
        if batchSize == 1 {
            return MLXArray(batchOutput[0]).expandedDimensions(axis: 0)
        } else {
            let stacked = batchOutput.map { MLXArray($0) }
            return MLX.stacked(stacked, axis: 0)
        }
    }
}

// MARK: - Leaky ReLU Helper

/// Apply leaky ReLU activation
private func leakyRelu(_ x: MLXArray, negativeSlope: Float = 0.01) -> MLXArray {
    MLX.maximum(x, x * negativeSlope)
}

/// Flip array along an axis
private func flip(_ x: MLXArray, axis: Int) -> MLXArray {
    let axisInt = axis < 0 ? x.ndim + axis : axis
    let size = x.dim(axisInt)
    let indices = MLXArray((0..<size).reversed().map { Int32($0) })
    return x.take(indices, axis: axisInt)
}

// MARK: - Generator (ISTFTNet Vocoder)

/// Generates audio from mel-spectrograms using ISTFT with harmonic-plus-noise synthesis
final class KokoroGenerator: Module {
    let styleDim: Int
    let resblockKernelSizes: [Int]
    let upsampleRates: [Int]
    let upsampleInitialChannel: Int
    let resblockDilationSizes: [[Int]]
    let upsampleKernelSizes: [Int]
    let genIstftNFft: Int
    let genIstftHopSize: Int
    let numKernels: Int
    let numUpsamples: Int
    let f0UpsampleScale: Int

    var ups: [ConvTransposed1dWeighted]
    var resblocks: [ResBlock1]  // Flat array
    // swiftlint:disable identifier_name
    var noise_convs: [Conv1d]
    var noise_res: [ResBlock1]
    let m_source: SourceModuleHnNSF
    let conv_post: ConvWeighted
    // swiftlint:enable identifier_name

    let reflection: ReflectionPad1d
    let istft: AccelerateISTFT
    let stft: MLXSTFT

    init(
        styleDim: Int,
        resblockKernelSizes: [Int],
        upsampleRates: [Int],
        upsampleInitialChannel: Int,
        resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int],
        genIstftNFft: Int,
        genIstftHopSize: Int
    ) {
        self.styleDim = styleDim
        self.resblockKernelSizes = resblockKernelSizes
        self.upsampleRates = upsampleRates
        self.upsampleInitialChannel = upsampleInitialChannel
        self.resblockDilationSizes = resblockDilationSizes
        self.upsampleKernelSizes = upsampleKernelSizes
        self.genIstftNFft = genIstftNFft
        self.genIstftHopSize = genIstftHopSize
        self.numKernels = resblockKernelSizes.count
        self.numUpsamples = upsampleRates.count

        // Calculate F0 upsample scale: product of all upsample rates * hop size
        self.f0UpsampleScale = upsampleRates.reduce(1, *) * genIstftHopSize

        self.ups = []
        self.resblocks = []
        self.noise_convs = []
        self.noise_res = []

        // Source module for harmonic-plus-noise synthesis
        // harmonicNum=8 matches Python
        self.m_source = SourceModuleHnNSF(
            samplingRate: 24000,
            upsampleScale: f0UpsampleScale,
            harmonicNum: 8
        )

        // STFT for harmonic source transform
        self.stft = MLXSTFT(filterLength: genIstftNFft, hopLength: genIstftHopSize, winLength: genIstftNFft)

        // Input channels for noise_convs: nFft + 2 (magnitude + phase concatenated)
        let noiseInputChannels = genIstftNFft + 2

        for (i, (u, k)) in zip(upsampleRates, upsampleKernelSizes).enumerated() {
            let inCh = upsampleInitialChannel / Int(pow(2.0, Double(i)))
            let outCh = upsampleInitialChannel / Int(pow(2.0, Double(i + 1)))

            ups.append(
                ConvTransposed1dWeighted(
                    inputChannels: inCh,
                    outputChannels: outCh,
                    kernelSize: k,
                    stride: u,
                    padding: (k - u) / 2
                )
            )

            let ch = outCh

            // Resblocks for this upsample level
            for (j, d) in resblockDilationSizes.enumerated() {
                resblocks.append(
                    ResBlock1(channels: ch, kernelSize: resblockKernelSizes[j], dilations: d, styleDim: styleDim)
                )
            }

            // Noise convolution - input channels = nFft + 2 (harmonic spec + phase)
            if i + 1 < numUpsamples {
                let strideF0 = Int(upsampleRates[(i + 1)...].reduce(1, *))
                noise_convs.append(
                    Conv1d(
                        inputChannels: noiseInputChannels,
                        outputChannels: ch,
                        kernelSize: strideF0 * 2,
                        stride: strideF0,
                        padding: (strideF0 + 1) / 2
                    )
                )
                noise_res.append(
                    ResBlock1(channels: ch, kernelSize: 7, dilations: [1, 3, 5], styleDim: styleDim)
                )
            } else {
                noise_convs.append(
                    Conv1d(inputChannels: noiseInputChannels, outputChannels: ch, kernelSize: 1)
                )
                noise_res.append(
                    ResBlock1(channels: ch, kernelSize: 11, dilations: [1, 3, 5], styleDim: styleDim)
                )
            }
        }

        // ISTFT output layers
        let finalCh = upsampleInitialChannel / Int(pow(2.0, Double(numUpsamples)))
        self.reflection = ReflectionPad1d(padding: (1, 0))
        self.conv_post = ConvWeighted(inChannels: finalCh, outChannels: genIstftNFft + 2, kernelSize: 7, padding: 3)
        self.istft = AccelerateISTFT(nFft: genIstftNFft, hopSize: genIstftHopSize)
    }

    func callAsFunction(_ x: MLXArray, s: MLXArray, f0: MLXArray) -> MLXArray {
        var out = x

        // Enable harmonic source injection for full synthesis
        let skipHarmonicSource = false
        var har: MLXArray? = nil

        if !skipHarmonicSource {
            // Upsample F0 to audio sample rate
            let f0Upsampled = upsampleF0(f0)

            // Generate harmonic source from F0
            let (harSource, _, _) = m_source(f0Upsampled)

            // harSource: [batch, length, 1] -> squeeze to [batch, length]
            let harSourceSqueezed = harSource.squeezed(axis: -1)

            // Get STFT of harmonic source
            let (harSpec, harPhase) = stft.transform(harSourceSqueezed)

            // Concatenate magnitude and phase
            har = MLX.concatenated([harSpec, harPhase], axis: 1).swappedAxes(1, 2)
        }

        for i in 0..<numUpsamples {
            // Apply leaky relu and upsample
            out = leakyRelu(out, negativeSlope: 0.1)

            // Upsample main signal
            out = ups[i](out.swappedAxes(1, 2)).swappedAxes(1, 2)

            // Add reflection pad for last upsample
            if i == numUpsamples - 1 {
                out = reflection(out.swappedAxes(1, 2)).swappedAxes(1, 2)
            }

            // Process and add harmonic source if enabled
            if let harArray = har {
                var xSource = noise_convs[i](harArray)
                xSource = xSource.swappedAxes(1, 2)  // [batch, channels, frames]
                xSource = noise_res[i](xSource, s: s)

                // Match dimensions
                let outLen = out.dim(2)
                let srcLen = xSource.dim(2)
                if srcLen != outLen {
                    if srcLen > outLen {
                        xSource = xSource[0..., 0..., 0..<outLen]
                    } else {
                        let pad = MLXArray.zeros([xSource.dim(0), xSource.dim(1), outLen - srcLen])
                        xSource = MLX.concatenated([xSource, pad], axis: 2)
                    }
                }
                out = out + xSource
            }

            // Apply resblocks
            let startIdx = i * numKernels
            var xs: MLXArray? = nil
            for j in 0..<numKernels {
                let blockOut = resblocks[startIdx + j](out, s: s)
                xs = xs.map { $0 + blockOut } ?? blockOut
            }
            out = xs! / Float(numKernels)
        }

        out = leakyRelu(out, negativeSlope: 0.01)

        // conv_post: output magnitude + phase
        out = out.swappedAxes(1, 2)
        let combined = conv_post(out)
        let combinedT = combined.swappedAxes(1, 2)

        // Split into magnitude and phase
        let numBins = genIstftNFft / 2 + 1
        let specRaw = combinedT[0..., 0..<numBins, 0...]
        let phaseRaw = combinedT[0..., numBins..., 0...]

        let spec = exp(specRaw)
        let phase = sin(phaseRaw)

        // ISTFT to audio
        let audio = istft.inverse(spec, phase: phase)

        return audio
    }

    /// Upsample F0 to audio sample rate using nearest neighbor
    private func upsampleF0(_ f0: MLXArray) -> MLXArray {
        // f0: [batch, length]
        eval(f0)
        let batchSize = f0.dim(0)
        let length = f0.dim(1)
        let newLength = length * f0UpsampleScale

        var result = [Float](repeating: 0, count: batchSize * newLength)
        let f0Data = f0.asArray(Float.self)

        for b in 0..<batchSize {
            for i in 0..<newLength {
                let srcIdx = i / f0UpsampleScale
                result[b * newLength + i] = f0Data[b * length + min(srcIdx, length - 1)]
            }
        }

        // Return as [batch, newLength, 1]
        return MLXArray(result).reshaped([batchSize, newLength, 1])
    }
}

// MARK: - ResBlock1 (StyleAdaIN version)

/// Weight-normalized 1D convolution for ResBlock
/// Property names match the model's safetensors keys exactly
final class ConvWeighted1d: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let padding: Int
    let dilation: Int

    // swiftlint:disable identifier_name
    var weight_v: MLXArray
    var weight_g: MLXArray
    // swiftlint:enable identifier_name
    var bias: MLXArray?

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.padding = padding
        self.dilation = dilation

        self.weight_v = MLXRandom.normal([outChannels, kernelSize, inChannels])
        self.weight_g = MLXArray.ones([outChannels, 1, 1])

        if bias {
            self.bias = MLXArray.zeros([outChannels])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let vNormSquared = (weight_v * weight_v).sum(axes: [1, 2], keepDims: true)
        let vNorm = sqrt(vNormSquared + 1e-12)
        let weight = weight_g * weight_v / vNorm

        var result = conv1d(x, weight, stride: 1, padding: padding, dilation: dilation)

        if let b = bias {
            result = result + b
        }

        return result
    }
}

/// AdaIN1d for ResBlock (simplified version)
final class ResBlockAdaIN1d: Module {
    let fc: Linear

    init(styleDim: Int, numFeatures: Int) {
        self.fc = Linear(styleDim, numFeatures * 2)
    }

    func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
        let h = fc(s)

        let split = MLX.split(h, parts: 2, axis: -1)
        let gamma = split[0].expandedDimensions(axis: -1)
        let beta = split[1].expandedDimensions(axis: -1)

        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + 1e-5)

        // Use (1 + gamma) as per mlx_audio implementation
        return (1 + gamma) * normalized + beta
    }
}

/// Wrapper for a single alpha parameter
final class AlphaParam: Module {
    var weight: MLXArray

    init(channels: Int) {
        self.weight = MLXArray.ones([1, channels, 1])
    }
}

/// ResBlock1 with StyleAdaIN (matches Kokoro model weights)
/// Property names match the model's safetensors keys exactly
final class ResBlock1: Module {
    let channels: Int
    let kernelSize: Int
    let dilations: [Int]
    let styleDim: Int

    var convs1: [ConvWeighted1d]
    var convs2: [ConvWeighted1d]
    var adain1: [ResBlockAdaIN1d]
    var adain2: [ResBlockAdaIN1d]
    var alpha1: [AlphaParam]
    var alpha2: [AlphaParam]

    init(channels: Int, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5], styleDim: Int = 128) {
        self.channels = channels
        self.kernelSize = kernelSize
        self.dilations = dilations
        self.styleDim = styleDim

        self.convs1 = []
        self.convs2 = []
        self.adain1 = []
        self.adain2 = []
        self.alpha1 = []
        self.alpha2 = []

        for d in dilations {
            let padding = (kernelSize * d - d) / 2
            convs1.append(
                ConvWeighted1d(
                    inChannels: channels,
                    outChannels: channels,
                    kernelSize: kernelSize,
                    padding: padding,
                    dilation: d
                )
            )
            convs2.append(
                ConvWeighted1d(
                    inChannels: channels,
                    outChannels: channels,
                    kernelSize: kernelSize,
                    padding: (kernelSize - 1) / 2
                )
            )
            adain1.append(ResBlockAdaIN1d(styleDim: styleDim, numFeatures: channels))
            adain2.append(ResBlockAdaIN1d(styleDim: styleDim, numFeatures: channels))
            alpha1.append(AlphaParam(channels: channels))
            alpha2.append(AlphaParam(channels: channels))
        }
    }

    func callAsFunction(_ x: MLXArray, s: MLXArray? = nil) -> MLXArray {
        var out = x
        for i in 0..<convs1.count {
            var xt = out
            if let style = s {
                xt = adain1[i](xt, s: style)
            }
            // Snake activation: x + (1/alpha) * sin(alpha * x)^2
            let a1 = alpha1[i].weight
            xt = xt + (1 / a1) * MLX.pow(sin(a1 * xt), 2)

            xt = convs1[i](xt.swappedAxes(1, 2)).swappedAxes(1, 2)

            if let style = s {
                xt = adain2[i](xt, s: style)
            }
            // Snake activation
            let a2 = alpha2[i].weight
            xt = xt + (1 / a2) * MLX.pow(sin(a2 * xt), 2)

            xt = convs2[i](xt.swappedAxes(1, 2)).swappedAxes(1, 2)

            // Simple residual (as per Python implementation)
            out = xt + out
        }
        return out
    }
}

// MARK: - ReflectionPad1d

final class ReflectionPad1d: Module {
    let paddingLeft: Int
    let paddingRight: Int

    init(padding: (Int, Int)) {
        self.paddingLeft = padding.0
        self.paddingRight = padding.1
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, length, channels] (after swapAxes)
        let length = x.dim(1)

        var result = x

        // Left padding (reflection)
        if paddingLeft > 0 {
            let leftPad = x[0..., 1...(paddingLeft), 0...]
            let leftReversed = flip(leftPad, axis: 1)
            result = MLX.concatenated([leftReversed, result], axis: 1)
        }

        // Right padding (reflection)
        if paddingRight > 0 {
            let rightStart = length - paddingRight - 1
            let rightPad = x[0..., rightStart..<(length - 1), 0...]
            let rightReversed = flip(rightPad, axis: 1)
            result = MLX.concatenated([result, rightReversed], axis: 1)
        }

        return result
    }
}

// MARK: - MLXSTFT

/// STFT class for harmonic source transform (matching Python mlx_audio)
final class MLXSTFT {
    let filterLength: Int
    let hopLength: Int
    let winLength: Int
    let window: [Float]

    init(filterLength: Int, hopLength: Int, winLength: Int) {
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.winLength = winLength

        // Create Hann window
        var hannWindow = [Float](repeating: 0, count: winLength)
        for n in 0..<winLength {
            let angle = 2.0 * Float.pi * Float(n) / Float(winLength)
            hannWindow[n] = 0.5 * (1.0 - cos(angle))
        }
        self.window = hannWindow
    }

    /// Compute STFT magnitude and phase
    func transform(_ inputData: MLXArray) -> (MLXArray, MLXArray) {
        // Input: [batch, length] or [length]
        eval(inputData)

        let input2d = inputData.ndim == 1 ? inputData.expandedDimensions(axis: 0) : inputData
        let batchSize = input2d.dim(0)
        let length = input2d.dim(1)

        // Apply center padding (reflect mode, like Python)
        let padSize = filterLength / 2
        let paddedLength = length + 2 * padSize

        let numBins = filterLength / 2 + 1
        let numFrames = max(1, (paddedLength - filterLength) / hopLength + 1)

        var magnitudes: [Float] = []
        var phases: [Float] = []

        for b in 0..<batchSize {
            let rawSignal = input2d[b].asArray(Float.self)

            // Apply reflect padding: signal[padSize-1:0:-1] + signal + signal[-2:-padSize-2:-1]
            var signal = [Float](repeating: 0, count: paddedLength)
            // Prefix: reflect the first padSize samples
            for i in 0..<padSize {
                let srcIdx = padSize - i
                signal[i] = srcIdx < rawSignal.count ? rawSignal[srcIdx] : 0
            }
            // Main signal
            for i in 0..<rawSignal.count {
                signal[padSize + i] = rawSignal[i]
            }
            // Suffix: reflect the last padSize samples
            for i in 0..<padSize {
                let srcIdx = rawSignal.count - 2 - i
                if srcIdx >= 0 {
                    signal[padSize + rawSignal.count + i] = rawSignal[srcIdx]
                }
            }
            var batchMag: [[Float]] = Array(repeating: [Float](repeating: 0, count: numFrames), count: numBins)
            var batchPhase: [[Float]] = Array(repeating: [Float](repeating: 0, count: numFrames), count: numBins)

            for frameIdx in 0..<numFrames {
                let start = frameIdx * hopLength
                var frame = [Float](repeating: 0, count: filterLength)

                // Apply window and extract frame
                for i in 0..<filterLength {
                    let idx = start + i
                    if idx < signal.count {
                        frame[i] = signal[idx] * window[i % winLength]
                    }
                }

                // Compute DFT for each bin
                for k in 0..<numBins {
                    var real: Float = 0
                    var imag: Float = 0
                    let freqMult = 2.0 * Float.pi * Float(k) / Float(filterLength)

                    for n in 0..<filterLength {
                        let angle = freqMult * Float(n)
                        real += frame[n] * cos(angle)
                        imag -= frame[n] * sin(angle)
                    }

                    let magnitude = sqrt(real * real + imag * imag)
                    let phase = atan2(imag, real)

                    batchMag[k][frameIdx] = magnitude
                    batchPhase[k][frameIdx] = phase
                }
            }

            // Flatten batch results into 1D array
            for bin in 0..<numBins {
                magnitudes.append(contentsOf: batchMag[bin])
            }
            for bin in 0..<numBins {
                phases.append(contentsOf: batchPhase[bin])
            }
        }

        // Reshape to [batch, numBins, numFrames]
        let magArray = MLXArray(magnitudes).reshaped([batchSize, numBins, numFrames])
        let phaseArray = MLXArray(phases).reshaped([batchSize, numBins, numFrames])

        return (magArray, phaseArray)
    }
}

// MARK: - Upsample1d

/// 1D upsampling using linear interpolation
final class Upsample1d: Module {
    let scaleFactor: Int

    init(scaleFactor: Int) {
        self.scaleFactor = scaleFactor
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, length] or [batch, length, channels]
        // Assuming [batch, length, channels] format
        let batchSize = x.dim(0)
        let length = x.dim(1)
        let channels = x.dim(2)
        let newLength = length * scaleFactor

        eval(x)

        // Simple nearest-neighbor upsampling
        var result = [Float](repeating: 0, count: batchSize * newLength * channels)

        for b in 0..<batchSize {
            for c in 0..<channels {
                let channelData = x[b, 0..., c].asArray(Float.self)
                for i in 0..<newLength {
                    let srcIdx = i / scaleFactor
                    result[b * newLength * channels + i * channels + c] = channelData[min(srcIdx, length - 1)]
                }
            }
        }

        return MLXArray(result).reshaped([batchSize, newLength, channels])
    }
}

// MARK: - SourceModuleHnNSF

/// Source module for harmonic-plus-noise synthesis
final class SourceModuleHnNSF: Module {
    // swiftlint:disable identifier_name
    let l_sin_gen: SineGen
    let l_linear: Linear
    // swiftlint:enable identifier_name
    let sineAmp: Float

    init(samplingRate: Int = 24000, upsampleScale: Int, harmonicNum: Int = 8) {
        self.sineAmp = 0.1
        self.l_sin_gen = SineGen(
            samplingRate: samplingRate,
            upsampleScale: upsampleScale,
            harmonicNum: harmonicNum,
            sineAmp: sineAmp,
            noiseStd: 0.003,
            voicedThreshold: 10
        )
        self.l_linear = Linear(harmonicNum + 1, 1)
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // f0: [batch, length, 1]
        let (sineWavs, uv, _) = l_sin_gen(f0)
        let sineMerge = tanh(l_linear(sineWavs))

        // Noise source in same shape as uv
        let noise = MLXRandom.normal(uv.shape) * (sineAmp / 3.0)

        return (sineMerge, noise, uv)
    }
}

// MARK: - SineGen

/// Sine generator for F0 conditioning - generates sine waves from F0
final class SineGen: Module {
    let samplingRate: Int
    let harmonicNum: Int
    let sineAmp: Float
    let noiseStd: Float
    let voicedThreshold: Float
    let dim: Int
    let upsampleScale: Int

    init(
        samplingRate: Int = 24000,
        upsampleScale: Int = 1,
        harmonicNum: Int = 0,
        sineAmp: Float = 0.1,
        noiseStd: Float = 0.003,
        voicedThreshold: Float = 0
    ) {
        self.samplingRate = samplingRate
        self.upsampleScale = upsampleScale
        self.harmonicNum = harmonicNum
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.voicedThreshold = voicedThreshold
        self.dim = harmonicNum + 1
    }

    /// Convert F0 to voiced/unvoiced mask
    private func f02uv(_ f0: MLXArray) -> MLXArray {
        (f0 .> voicedThreshold).asType(.float32)
    }

    /// Linear interpolation helper (like Python's interpolate with mode="linear")
    private func interpolate1d(_ x: MLXArray, scaleFactor: Float) -> MLXArray {
        // x: [batch, channels, length]
        let batchSize = x.dim(0)
        let channels = x.dim(1)
        let oldLength = x.dim(2)
        let newLength = Int(Float(oldLength) * scaleFactor)

        if newLength == oldLength {
            return x
        }

        // Linear interpolation
        var result = [[Float]](repeating: [Float](repeating: 0, count: newLength), count: batchSize * channels)

        eval(x)
        let xFlat = x.reshaped([batchSize * channels, oldLength])

        for bc in 0..<(batchSize * channels) {
            let row = xFlat[bc].asArray(Float.self)
            for i in 0..<newLength {
                let srcPos = Float(i) * Float(oldLength - 1) / Float(max(1, newLength - 1))
                let srcIdx = Int(srcPos)
                let frac = srcPos - Float(srcIdx)

                if srcIdx + 1 < oldLength {
                    result[bc][i] = row[srcIdx] * (1 - frac) + row[srcIdx + 1] * frac
                } else {
                    result[bc][i] = row[min(srcIdx, oldLength - 1)]
                }
            }
        }

        let flatResult = result.flatMap { $0 }
        return MLXArray(flatResult).reshaped([batchSize, channels, newLength])
    }

    /// Convert F0 to sine waveforms
    private func f02sine(_ f0Values: MLXArray) -> MLXArray {
        // f0Values: [batch, length, dim] where dim = harmonicNum + 1
        let batchSize = f0Values.dim(0)

        // Convert to rad values (normalized frequency)
        // radValues = (f0 / samplingRate) % 1.0
        let normalized = f0Values / Float(samplingRate)
        var radValues = normalized - MLX.floor(normalized)

        // Add initial phase noise (no noise for fundamental)
        var randIni = MLXRandom.normal([batchSize, dim])
        // Zero out first column (fundamental)
        let zeros = MLXArray.zeros([batchSize, 1])
        let rest = randIni[0..., 1...]
        randIni = MLX.concatenated([zeros, rest], axis: 1)

        // Add initial phase to first time step
        let firstStep = radValues[0..., 0, 0...] + randIni
        let restSteps = radValues[0..., 1..., 0...]
        radValues = MLX.concatenated(
            [firstStep.expandedDimensions(axis: 1), restSteps],
            axis: 1
        )

        // Apply interpolation like Python (downsample, cumsum, upsample)
        if upsampleScale > 1 {
            // Transpose to [batch, dim, length] for interpolation
            let radT = radValues.transposed(0, 2, 1)

            // Downsample by 1/upsample_scale
            let radDown = interpolate1d(radT, scaleFactor: 1.0 / Float(upsampleScale))

            // Transpose back to [batch, length, dim] for cumsum
            let radDownT = radDown.transposed(0, 2, 1)

            // Compute cumulative phase on downsampled
            var phase = MLX.cumsum(radDownT, axis: 1) * (2 * Float.pi)

            // Transpose for upsampling
            let phaseT = phase.transposed(0, 2, 1)

            // Upsample back and scale
            let phaseUp = interpolate1d(phaseT * Float(upsampleScale), scaleFactor: Float(upsampleScale))

            // Transpose back
            phase = phaseUp.transposed(0, 2, 1)

            return sin(phase)
        } else {
            // No interpolation needed
            let phase = MLX.cumsum(radValues, axis: 1) * (2 * Float.pi)
            return sin(phase)
        }
    }

    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // f0: [batch, length, 1]
        let batchSize = f0.dim(0)
        let length = f0.dim(1)

        // Create harmonics: f0 * [1, 2, 3, ..., harmonicNum+1]
        let harmonicMult = MLXArray((1...dim).map { Float($0) }).expandedDimensions(axes: [0, 1])
        let fn = f0 * harmonicMult  // [batch, length, dim]

        // Generate sine waveforms
        var sineWavs = f02sine(fn) * sineAmp

        // Get UV mask
        let uv = f02uv(f0)

        // Generate noise
        let noiseAmp = uv * noiseStd + (1 - uv) * (sineAmp / 3.0)
        let noise = noiseAmp * MLXRandom.normal(sineWavs.shape)

        // Apply UV mask: voiced uses sine, unvoiced uses noise
        sineWavs = sineWavs * uv + noise

        return (sineWavs, uv, noise)
    }
}

// MARK: - Decoder

/// ISTFTNet-based decoder for audio generation
/// Property names match the model's safetensors keys exactly
final class KokoroDecoder: Module {
    let dimIn: Int
    let styleDim: Int
    let dimOut: Int

    let encode: AdainResBlk1d
    var decode: [AdainResBlk1d]
    // swiftlint:disable identifier_name
    let F0_conv: ConvWeighted
    let N_conv: ConvWeighted
    var asr_res: [ConvWeighted]
    // swiftlint:enable identifier_name
    let generator: KokoroGenerator

    init(
        dimIn: Int,
        styleDim: Int,
        dimOut: Int,
        resblockKernelSizes: [Int],
        upsampleRates: [Int],
        upsampleInitialChannel: Int,
        resblockDilationSizes: [[Int]],
        upsampleKernelSizes: [Int],
        genIstftNFft: Int,
        genIstftHopSize: Int
    ) {
        self.dimIn = dimIn
        self.styleDim = styleDim
        self.dimOut = dimOut

        self.encode = AdainResBlk1d(inDim: dimIn + 2, outDim: 1024, styleDim: styleDim)

        self.decode = [
            AdainResBlk1d(inDim: 1024 + 2 + 64, outDim: 1024, styleDim: styleDim),
            AdainResBlk1d(inDim: 1024 + 2 + 64, outDim: 1024, styleDim: styleDim),
            AdainResBlk1d(inDim: 1024 + 2 + 64, outDim: 1024, styleDim: styleDim),
            AdainResBlk1d(inDim: 1024 + 2 + 64, outDim: 512, styleDim: styleDim, upsample: true),
        ]

        self.F0_conv = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1)
        self.N_conv = ConvWeighted(inChannels: 1, outChannels: 1, kernelSize: 3, stride: 2, padding: 1)
        self.asr_res = [ConvWeighted(inChannels: 512, outChannels: 64, kernelSize: 1)]

        self.generator = KokoroGenerator(
            styleDim: styleDim,
            resblockKernelSizes: resblockKernelSizes,
            upsampleRates: upsampleRates,
            upsampleInitialChannel: upsampleInitialChannel,
            resblockDilationSizes: resblockDilationSizes,
            upsampleKernelSizes: upsampleKernelSizes,
            genIstftNFft: genIstftNFft,
            genIstftHopSize: genIstftHopSize
        )
    }

    func callAsFunction(
        _ asr: MLXArray,
        f0Curve: MLXArray,
        n: MLXArray,
        s: MLXArray
    ) -> MLXArray {
        // Process F0 and N
        var f0 = f0Curve.expandedDimensions(axis: 1)
        f0 = F0_conv(f0.swappedAxes(1, 2)).swappedAxes(1, 2)

        var noise = n.expandedDimensions(axis: 1)
        noise = N_conv(noise.swappedAxes(1, 2)).swappedAxes(1, 2)

        // Concatenate and encode
        var x = MLX.concatenated([asr, f0, noise], axis: 1)
        x = encode(x, s: s)

        // ASR residual
        let asrResOut = asr_res[0](asr.swappedAxes(1, 2)).swappedAxes(1, 2)

        // Decode blocks
        var addRes = true
        for block in decode {
            if addRes {
                x = MLX.concatenated([x, asrResOut, f0, noise], axis: 1)
            }
            x = block(x, s: s)
            if block.doUpsample {
                addRes = false
            }
        }

        // Generate audio
        let audio = generator(x, s: s, f0: f0Curve)

        return audio
    }
}
