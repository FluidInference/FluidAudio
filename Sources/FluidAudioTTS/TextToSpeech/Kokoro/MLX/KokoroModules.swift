import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - LinearNorm

/// Linear layer with optional gain initialization
/// Property names match the model's safetensors keys exactly
final class LinearNorm: Module, UnaryLayer {
    // swiftlint:disable identifier_name
    let linear_layer: Linear
    // swiftlint:enable identifier_name

    init(inDim: Int, outDim: Int, bias: Bool = true) {
        self.linear_layer = Linear(inDim, outDim, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear_layer(x)
    }
}

// MARK: - Bidirectional LSTM

/// Bidirectional LSTM implementation for MLX
/// Property names match the model's safetensors keys exactly
final class BiLSTM: Module {
    let inputSize: Int
    let hiddenSize: Int
    let hasBias: Bool

    // Forward direction weights (names match safetensors keys)
    // swiftlint:disable identifier_name
    var Wx_forward: MLXArray
    var Wh_forward: MLXArray
    var bias_ih_forward: MLXArray?
    var bias_hh_forward: MLXArray?

    // Backward direction weights (names match safetensors keys)
    var Wx_backward: MLXArray
    var Wh_backward: MLXArray
    var bias_ih_backward: MLXArray?
    var bias_hh_backward: MLXArray?
    // swiftlint:enable identifier_name

    init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.hasBias = bias

        let scale = 1.0 / Float(hiddenSize).squareRoot()

        // Forward direction
        self.Wx_forward = MLXRandom.uniform(
            low: -scale, high: scale,
            [4 * hiddenSize, inputSize]
        )
        self.Wh_forward = MLXRandom.uniform(
            low: -scale, high: scale,
            [4 * hiddenSize, hiddenSize]
        )
        if bias {
            self.bias_ih_forward = MLXRandom.uniform(
                low: -scale, high: scale,
                [4 * hiddenSize]
            )
            self.bias_hh_forward = MLXRandom.uniform(
                low: -scale, high: scale,
                [4 * hiddenSize]
            )
        }

        // Backward direction
        self.Wx_backward = MLXRandom.uniform(
            low: -scale, high: scale,
            [4 * hiddenSize, inputSize]
        )
        self.Wh_backward = MLXRandom.uniform(
            low: -scale, high: scale,
            [4 * hiddenSize, hiddenSize]
        )
        if bias {
            self.bias_ih_backward = MLXRandom.uniform(
                low: -scale, high: scale,
                [4 * hiddenSize]
            )
            self.bias_hh_backward = MLXRandom.uniform(
                low: -scale, high: scale,
                [4 * hiddenSize]
            )
        }
    }

    private func forwardDirection(
        _ x: MLXArray,
        hidden: MLXArray? = nil,
        cell: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let batchSize = x.dim(0)
        let seqLen = x.dim(1)

        // Pre-compute input projections
        var xProj: MLXArray
        if let biasIh = bias_ih_forward, let biasHh = bias_hh_forward {
            xProj = MLX.addmm(biasIh + biasHh, x, Wx_forward.T)
        } else {
            xProj = MLX.matmul(x, Wx_forward.T)
        }

        var h = hidden ?? MLX.zeros([batchSize, hiddenSize])
        var c = cell ?? MLX.zeros([batchSize, hiddenSize])

        var allHidden: [MLXArray] = []

        for idx in 0..<seqLen {
            var ifgo = xProj[0..., idx, 0...]
            ifgo = ifgo + MLX.matmul(h, Wh_forward.T)

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            c = f * c + i * g
            h = o * MLX.tanh(c)

            allHidden.append(h)
        }

        return (MLX.stacked(allHidden, axis: 1), c)
    }

    private func backwardDirection(
        _ x: MLXArray,
        hidden: MLXArray? = nil,
        cell: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let batchSize = x.dim(0)
        let seqLen = x.dim(1)

        // Pre-compute input projections
        var xProj: MLXArray
        if let biasIh = bias_ih_backward, let biasHh = bias_hh_backward {
            xProj = MLX.addmm(biasIh + biasHh, x, Wx_backward.T)
        } else {
            xProj = MLX.matmul(x, Wx_backward.T)
        }

        var h = hidden ?? MLX.zeros([batchSize, hiddenSize])
        var c = cell ?? MLX.zeros([batchSize, hiddenSize])

        var allHidden: [MLXArray] = []

        // Process in reverse
        for idx in stride(from: seqLen - 1, through: 0, by: -1) {
            var ifgo = xProj[0..., idx, 0...]
            ifgo = ifgo + MLX.matmul(h, Wh_backward.T)

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            c = f * c + i * g
            h = o * MLX.tanh(c)

            allHidden.insert(h, at: 0)
        }

        return (MLX.stacked(allHidden, axis: 1), c)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
        var input = x
        if x.ndim == 2 {
            input = input.expandedDimensions(axis: 0)
        }

        let (forwardHidden, forwardCell) = forwardDirection(input)
        let (backwardHidden, backwardCell) = backwardDirection(input)

        // Concatenate outputs
        let output = MLX.concatenated([forwardHidden, backwardHidden], axis: -1)

        return (
            output,
            (
                (forwardHidden[0..., -1, 0...], forwardCell),
                (backwardHidden[0..., 0, 0...], backwardCell)
            )
        )
    }
}

// MARK: - AdaLayerNorm

/// Adaptive Layer Normalization
final class AdaLayerNorm: Module {
    let channels: Int
    let eps: Float
    let fc: Linear

    init(styleDim: Int, channels: Int, eps: Float = 1e-5) {
        self.channels = channels
        self.eps = eps
        self.fc = Linear(styleDim, channels * 2)
    }

    func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
        var h = fc(s)
        h = h.reshaped([h.dim(0), h.dim(1), 1])

        let split = MLX.split(h, parts: 2, axis: 1)
        var gamma = split[0]
        var beta = split[1]

        gamma = gamma.transposed(2, 0, 1)
        beta = beta.transposed(2, 0, 1)

        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)

        return (1 + gamma) * normalized + beta
    }
}

// MARK: - ConvWeighted

/// Weight-normalized 1D convolution
/// Property names match the model's safetensors keys exactly
final class ConvWeighted: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let stride: Int
    let padding: Int
    let groups: Int

    // swiftlint:disable identifier_name
    var weight_v: MLXArray
    var weight_g: MLXArray
    // swiftlint:enable identifier_name
    var bias: MLXArray?

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.groups = groups

        // Weight normalization: weight = g * v / ||v||
        self.weight_v = MLXRandom.normal([outChannels, kernelSize, inChannels / groups])
        self.weight_g = MLXArray.ones([outChannels, 1, 1])

        if bias {
            self.bias = MLXArray.zeros([outChannels])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute normalized weight
        let vNormSquared = (weight_v * weight_v).sum(axes: [1, 2], keepDims: true)
        let vNorm = sqrt(vNormSquared + 1e-12)
        let weight = weight_g * weight_v / vNorm

        // Apply convolution
        var result = conv1d(x, weight, stride: stride, padding: padding, groups: groups)

        if let b = bias {
            result = result + b
        }

        return result
    }
}

// MARK: - DepthwiseConvTransposed1d

/// Depthwise transposed convolution for upsampling
/// Weight format: (channels, kernel_size, 1) where each channel has its own kernel
final class DepthwiseConvTransposed1d: Module {
    let channels: Int
    let kernelSize: Int
    let stride: Int
    let padding: Int
    let outputPadding: Int

    // swiftlint:disable identifier_name
    var weight_v: MLXArray
    var weight_g: MLXArray
    // swiftlint:enable identifier_name
    var bias: MLXArray?

    init(
        channels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        bias: Bool = true
    ) {
        self.channels = channels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding

        // Weight shape: (channels, kernel, 1) for depthwise
        self.weight_v = MLXRandom.normal([channels, kernelSize, 1])
        self.weight_g = MLXArray.ones([channels, 1, 1])

        if bias {
            self.bias = MLXArray.zeros([channels])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute normalized weight
        let vNormSquared = (weight_v * weight_v).sum(axes: [1, 2], keepDims: true)
        let vNorm = sqrt(vNormSquared + 1e-12)
        let weight = weight_g * weight_v / vNorm

        // Depthwise transposed convolution using groups
        // Apply transposed conv then pad if needed
        var result = convTransposed1d(x, weight, stride: stride, padding: padding, groups: channels)

        // Apply output padding if needed (to match target size)
        if outputPadding > 0 {
            let paddingArray = MLXArray.zeros([result.dim(0), outputPadding, result.dim(2)])
            result = MLX.concatenated([result, paddingArray], axis: 1)
        }

        if let b = bias {
            result = result + b
        }

        return result
    }
}

// MARK: - AdainResBlk1d

/// Adaptive Instance Normalization Residual Block (1D)
/// Property names match the model's safetensors keys exactly
final class AdainResBlk1d: Module {
    let inDim: Int
    let outDim: Int
    let styleDim: Int
    let doUpsample: Bool
    let dropoutP: Float

    let conv1: ConvWeighted
    let conv2: ConvWeighted
    let norm1: AdaIN1d
    let norm2: AdaIN1d
    let actv: LeakyReLU
    let dropout: Dropout

    var conv1x1: ConvWeighted?
    var pool: DepthwiseConvTransposed1d?  // Used for upsampling path

    init(
        inDim: Int,
        outDim: Int,
        styleDim: Int,
        upsample: Bool = false,
        dropoutP: Float = 0.0
    ) {
        self.inDim = inDim
        self.outDim = outDim
        self.styleDim = styleDim
        self.doUpsample = upsample
        self.dropoutP = dropoutP

        self.conv1 = ConvWeighted(inChannels: inDim, outChannels: outDim, kernelSize: 3, padding: 1)
        self.conv2 = ConvWeighted(inChannels: outDim, outChannels: outDim, kernelSize: 3, padding: 1)
        self.norm1 = AdaIN1d(styleDim: styleDim, numFeatures: inDim)
        self.norm2 = AdaIN1d(styleDim: styleDim, numFeatures: outDim)
        self.actv = LeakyReLU(negativeSlope: 0.2)
        self.dropout = Dropout(p: dropoutP)

        if inDim != outDim {
            self.conv1x1 = ConvWeighted(inChannels: inDim, outChannels: outDim, kernelSize: 1, padding: 0)
        }

        if upsample {
            // Pool is depthwise transposed conv: in=1, out=inDim, k=3, stride=2, groups=inDim
            // No output padding - we add padding separately after pool
            self.pool = DepthwiseConvTransposed1d(
                channels: inDim, kernelSize: 3, stride: 2, padding: 1, outputPadding: 0
            )
        }
    }

    func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
        // Residual path (matches Python _residual)
        var out = norm1(x, s: s)
        out = actv(out)

        // Pool (transposed conv) upsampling on residual path, with left-padding
        if let poolConv = pool {
            out = out.swappedAxes(1, 2)
            out = poolConv(out)
            // Pad left side by 1 to match Python: mx.pad(x, ((0, 0), (1, 0), (0, 0)))
            out = MLX.padded(out, widths: [[0, 0], [1, 0], [0, 0]])
            out = out.swappedAxes(1, 2)
        }

        // Conv1 with dropout BEFORE conv (matches Python)
        out = out.swappedAxes(1, 2)
        out = dropout(out)
        out = conv1(out)
        out = out.swappedAxes(1, 2)

        // Second conv block
        out = norm2(out, s: s)
        out = actv(out)
        out = out.swappedAxes(1, 2)
        out = conv2(out)
        out = out.swappedAxes(1, 2)

        // Shortcut path (matches Python _shortcut)
        var skip = x

        // Skip uses nearest neighbor upsampling (not pool)
        // Python: x.swapaxes(2,1) -> upsample -> x.swapaxes(2,1)
        // Our upsample1d operates on last dimension, so we can skip the swaps
        if doUpsample {
            skip = upsample1d(skip)
        }

        // Then apply 1x1 conv if dimensions differ
        if let convS = conv1x1 {
            skip = skip.swappedAxes(1, 2)
            skip = convS(skip)
            skip = skip.swappedAxes(1, 2)
        }

        // Normalize by sqrt(2) as per mlx_audio implementation
        return (out + skip) / sqrt(MLXArray(2.0))
    }

    private func upsample1d(_ x: MLXArray) -> MLXArray {
        // Simple nearest neighbor upsampling (2x)
        let expanded = x.expandedDimensions(axis: -1)
        let repeated = MLX.repeat(expanded, count: 2, axis: -1)
        let reshaped = repeated.reshaped([x.dim(0), x.dim(1), x.dim(2) * 2])
        return reshaped
    }
}

// MARK: - ConvTransposed1dWeighted

/// Weight-normalized 1D transposed convolution
/// Property names match the model's safetensors keys exactly
final class ConvTransposed1dWeighted: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let stride: Int
    let padding: Int

    // swiftlint:disable identifier_name
    var weight_v: MLXArray
    var weight_g: MLXArray
    // swiftlint:enable identifier_name
    var bias: MLXArray?

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = true
    ) {
        self.inChannels = inputChannels
        self.outChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding

        // Weight normalization: weight = g * v / ||v||
        // Shape for transposed conv: [outputChannels, kernelSize, inputChannels] to match MLX format
        self.weight_v = MLXRandom.normal([outputChannels, kernelSize, inputChannels])
        self.weight_g = MLXArray.ones([outputChannels, 1, 1])

        if bias {
            self.bias = MLXArray.zeros([outputChannels])
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute normalized weight (weight_v is in (in, kernel, out) format from model)
        let vNormSquared = (weight_v * weight_v).sum(axes: [1, 2], keepDims: true)
        let vNorm = sqrt(vNormSquared + 1e-12)
        var weight = weight_g * weight_v / vNorm

        // Check if we need to transpose weight for convTransposed1d
        // convTransposed1d expects (out, kernel, in) but weight is (in, kernel, out)
        // Transpose if input channels don't match weight's last dimension
        if x.dim(-1) != weight.dim(-1) {
            // Transpose from (in, kernel, out) to (out, kernel, in)
            weight = weight.transposed(2, 1, 0)
        }

        // Apply transposed convolution
        var result = convTransposed1d(x, weight, stride: stride, padding: padding)

        if let b = bias {
            result = result + b
        }

        return result
    }
}

// MARK: - AdaIN1d

/// Adaptive Instance Normalization 1D
final class AdaIN1d: Module {
    let styleDim: Int
    let numFeatures: Int
    let eps: Float

    let fc: Linear

    init(styleDim: Int, numFeatures: Int, eps: Float = 1e-5) {
        self.styleDim = styleDim
        self.numFeatures = numFeatures
        self.eps = eps
        self.fc = Linear(styleDim, numFeatures * 2)
    }

    func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
        let h = fc(s)

        // Split into gamma and beta
        let split = MLX.split(h, parts: 2, axis: -1)
        let gamma = split[0].expandedDimensions(axis: -1)
        let beta = split[1].expandedDimensions(axis: -1)

        // Instance normalization
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)

        // Use (1 + gamma) as per mlx_audio implementation
        return (1 + gamma) * normalized + beta
    }
}
