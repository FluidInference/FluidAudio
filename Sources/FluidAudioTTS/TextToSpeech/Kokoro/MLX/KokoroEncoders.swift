import Foundation
import MLX
import MLXNN

// MARK: - TextEncoder

/// Encodes phoneme sequences into hidden representations
final class KokoroTextEncoder: Module {
    let channels: Int
    let kernelSize: Int
    let depth: Int

    let embedding: Embedding
    var cnn: [[Module]]
    let lstm: BiLSTM

    init(channels: Int, kernelSize: Int, depth: Int, nSymbols: Int) {
        self.channels = channels
        self.kernelSize = kernelSize
        self.depth = depth

        self.embedding = Embedding(embeddingCount: nSymbols, dimensions: channels)

        let padding = (kernelSize - 1) / 2
        var cnnLayers: [[Module]] = []
        for _ in 0..<depth {
            let block: [Module] = [
                ConvWeighted(
                    inChannels: channels,
                    outChannels: channels,
                    kernelSize: kernelSize,
                    padding: padding
                ),
                LayerNorm(dimensions: channels),
                LeakyReLU(negativeSlope: 0.2),
                Dropout(p: 0.2),
            ]
            cnnLayers.append(block)
        }
        self.cnn = cnnLayers
        self.lstm = BiLSTM(inputSize: channels, hiddenSize: channels / 2)
    }

    func callAsFunction(
        _ x: MLXArray,
        inputLengths: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        var out = embedding(x)
        out = out.transposed(0, 2, 1)

        let expandedMask = mask.expandedDimensions(axis: 1)
        out = MLX.where(expandedMask, MLXArray(0.0), out)

        for block in cnn {
            for layer in block {
                if layer is ConvWeighted || layer is LayerNorm {
                    out = out.swappedAxes(1, 2)
                    if let conv = layer as? ConvWeighted {
                        out = conv(out)
                    } else if let norm = layer as? LayerNorm {
                        out = norm(out)
                    }
                    out = out.swappedAxes(1, 2)
                } else if let actv = layer as? LeakyReLU {
                    out = actv(out)
                } else if let drop = layer as? Dropout {
                    out = drop(out)
                }
                out = MLX.where(expandedMask, MLXArray(0.0), out)
            }
        }

        out = out.swappedAxes(1, 2)
        let (lstmOut, _) = lstm(out)
        out = lstmOut.swappedAxes(1, 2)

        // Pad to mask length if needed
        let maxLen = mask.dim(-1)
        if out.dim(-1) < maxLen {
            let padSize = maxLen - out.dim(-1)
            let padding = MLX.zeros([out.dim(0), out.dim(1), padSize])
            out = MLX.concatenated([out, padding], axis: -1)
        }

        out = MLX.where(expandedMask, MLXArray(0.0), out)
        return out
    }
}

// MARK: - DurationEncoder

/// Encodes duration information with style conditioning
final class DurationEncoder: Module {
    let styleDim: Int
    let dModel: Int
    let nLayers: Int
    let dropoutRate: Float

    var lstms: [Module]

    init(styleDim: Int, dModel: Int, nLayers: Int, dropout: Float = 0.1) {
        self.styleDim = styleDim
        self.dModel = dModel
        self.nLayers = nLayers
        self.dropoutRate = dropout

        var layers: [Module] = []
        for _ in 0..<nLayers {
            layers.append(BiLSTM(inputSize: dModel + styleDim, hiddenSize: dModel / 2))
            layers.append(AdaLayerNorm(styleDim: styleDim, channels: dModel))
        }
        self.lstms = layers
    }

    func callAsFunction(
        _ x: MLXArray,
        style: MLXArray,
        textLengths: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        var out = x.transposed(2, 0, 1)

        // Broadcast style to sequence
        let s = MLX.broadcast(
            style,
            to: [out.dim(0), out.dim(1), style.dim(-1)]
        )

        out = MLX.concatenated([out, s], axis: -1)

        let maskExpanded = mask.expandedDimensions(axis: -1).transposed(1, 0, 2)
        out = MLX.where(maskExpanded, MLXArray(0.0), out)
        out = out.transposed(1, 2, 0)

        for (idx, layer) in lstms.enumerated() {
            if let norm = layer as? AdaLayerNorm {
                out = norm(out.transposed(0, 2, 1), s: style).transposed(0, 2, 1)
                out = MLX.concatenated([out, s.transposed(1, 2, 0)], axis: 1)
                out = MLX.where(mask.expandedDimensions(axis: -1).transposed(0, 2, 1), MLXArray(0.0), out)
            } else if let lstm = layer as? BiLSTM {
                out = out.transposed(0, 2, 1)[0]
                let (lstmOut, _) = lstm(out)
                out = lstmOut.transposed(0, 2, 1)

                // Pad if needed
                let maxLen = mask.dim(-1)
                if out.dim(-1) < maxLen {
                    let padSize = maxLen - out.dim(-1)
                    let padding = MLX.zeros([out.dim(0), out.dim(1), padSize])
                    out = MLX.concatenated([out, padding], axis: -1)
                }
            }
        }

        return out.transposed(0, 2, 1)
    }
}

// MARK: - ProsodyPredictor

/// Predicts duration, F0, and noise from text and style
/// Property names match the model's safetensors keys exactly
final class ProsodyPredictor: Module {
    let styleDim: Int
    let dHid: Int
    let nLayers: Int
    let maxDur: Int
    let dropoutRate: Float

    // swiftlint:disable identifier_name
    let text_encoder: DurationEncoder
    let lstm: BiLSTM
    let duration_proj: LinearNorm
    let shared: BiLSTM

    var F0: [AdainResBlk1d]
    var N: [AdainResBlk1d]

    let F0_proj: Conv1d
    let N_proj: Conv1d
    // swiftlint:enable identifier_name

    init(styleDim: Int, dHid: Int, nLayers: Int, maxDur: Int = 50, dropout: Float = 0.1) {
        self.styleDim = styleDim
        self.dHid = dHid
        self.nLayers = nLayers
        self.maxDur = maxDur
        self.dropoutRate = dropout

        self.text_encoder = DurationEncoder(
            styleDim: styleDim,
            dModel: dHid,
            nLayers: nLayers,
            dropout: dropout
        )

        self.lstm = BiLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)
        self.duration_proj = LinearNorm(inDim: dHid, outDim: maxDur)
        self.shared = BiLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2)

        // F0 prediction blocks
        self.F0 = [
            AdainResBlk1d(inDim: dHid, outDim: dHid, styleDim: styleDim, dropoutP: dropout),
            AdainResBlk1d(inDim: dHid, outDim: dHid / 2, styleDim: styleDim, upsample: true, dropoutP: dropout),
            AdainResBlk1d(inDim: dHid / 2, outDim: dHid / 2, styleDim: styleDim, dropoutP: dropout),
        ]

        // N prediction blocks
        self.N = [
            AdainResBlk1d(inDim: dHid, outDim: dHid, styleDim: styleDim, dropoutP: dropout),
            AdainResBlk1d(inDim: dHid, outDim: dHid / 2, styleDim: styleDim, upsample: true, dropoutP: dropout),
            AdainResBlk1d(inDim: dHid / 2, outDim: dHid / 2, styleDim: styleDim, dropoutP: dropout),
        ]

        self.F0_proj = Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0)
        self.N_proj = Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0)
    }

    func f0nTrain(_ x: MLXArray, s: MLXArray) -> (MLXArray, MLXArray) {
        let out = x.transposed(0, 2, 1)
        let (sharedOut, _) = shared(out)

        // F0 prediction
        var f0 = sharedOut.transposed(0, 2, 1)
        for block in F0 {
            f0 = block(f0, s: s)
        }
        f0 = f0.swappedAxes(1, 2)
        f0 = F0_proj(f0)
        f0 = f0.swappedAxes(1, 2)

        // N prediction
        var n = sharedOut.transposed(0, 2, 1)
        for block in N {
            n = block(n, s: s)
        }
        n = n.swappedAxes(1, 2)
        n = N_proj(n)
        n = n.swappedAxes(1, 2)

        return (f0.squeezed(axis: 1), n.squeezed(axis: 1))
    }
}
