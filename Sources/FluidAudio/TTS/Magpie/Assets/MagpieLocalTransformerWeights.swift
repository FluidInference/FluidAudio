import Foundation

/// Weights for the Swift-side 1-layer Local Transformer that samples the 8
/// codebook tokens per frame.
///
/// Shapes match the NumPy reference in `mobius/models/tts/magpie/coreml/generate_coreml.py`
/// (fn `local_transformer_forward`). All arrays are kept row-major fp32 so the
/// Accelerate + BNNS forward pass can consume them directly.
public struct MagpieLocalTransformerWeights: Sendable {
    // Input projection: (localDim, dModel) weight + (localDim,) bias.
    public let inProjWeight: [Float]
    public let inProjBias: [Float]
    /// Positional embedding slots: (maxPositions, localDim).
    public let posEmbedding: [Float]
    /// RMSNorm / LayerNorm weights: (localDim,) each.
    public let norm1Weight: [Float]
    public let norm2Weight: [Float]
    /// Self-attention QKV weight: (3*localDim, localDim).
    public let saQkvWeight: [Float]
    /// Self-attention output weight: (localDim, localDim).
    public let saOWeight: [Float]
    /// FFN conv kernel=1: (ffnDim, localDim) then (localDim, ffnDim).
    public let ffnConv1Weight: [Float]
    public let ffnConv2Weight: [Float]
    /// Per-codebook output heads: 8× (numCodesPerCodebook, localDim) + (numCodesPerCodebook,).
    public let outProjWeights: [[Float]]
    public let outProjBiases: [[Float]]

    // Cached dimensions for convenience.
    public let localDim: Int
    public let dModel: Int
    public let ffnDim: Int
    public let maxPositions: Int
    public let numCodebooks: Int
    public let numCodesPerCodebook: Int
}

public enum MagpieLocalTransformerLoader {

    private static let logger = AppLogger(category: "MagpieLocalTransformerLoader")

    /// Loads all `local_transformer/*.npy` files from `constantsDir`.
    public static func load(
        from constantsDir: URL,
        config: MagpieModelConfig
    ) throws -> MagpieLocalTransformerWeights {
        let ltDir = constantsDir.appendingPathComponent(MagpieConstants.Files.localTransformerDir)
        guard FileManager.default.fileExists(atPath: ltDir.path) else {
            throw MagpieError.modelFileNotFound(MagpieConstants.Files.localTransformerDir)
        }

        let localDim = MagpieConstants.localTransformerDim
        let ffnDim = MagpieConstants.localTransformerFfnDim
        let maxPositions = MagpieConstants.localTransformerMaxPositions
        let dModel = config.dModel
        let numCodebooks = config.numCodebooks
        let numCodesPerCodebook = config.numCodesPerCodebook

        func loadNpy(_ name: String, expecting shape: [Int]) throws -> [Float] {
            let url = ltDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MagpieError.modelFileNotFound("\(MagpieConstants.Files.localTransformerDir)/\(name)")
            }
            let array = try NpyReader.read(from: url)
            try array.assertShape(shape, label: name)
            return array.data
        }

        let inProjWeight = try loadNpy(
            MagpieConstants.Files.LocalTransformer.inProjWeight,
            expecting: [localDim, dModel])
        let inProjBias = try loadNpy(
            MagpieConstants.Files.LocalTransformer.inProjBias,
            expecting: [localDim])
        let posEmbedding = try loadNpy(
            MagpieConstants.Files.LocalTransformer.posEmb,
            expecting: [maxPositions, localDim])
        let norm1Weight = try loadNpy(
            MagpieConstants.Files.LocalTransformer.norm1Weight,
            expecting: [localDim])
        let norm2Weight = try loadNpy(
            MagpieConstants.Files.LocalTransformer.norm2Weight,
            expecting: [localDim])
        let saQkvWeight = try loadNpy(
            MagpieConstants.Files.LocalTransformer.saQkvWeight,
            expecting: [3 * localDim, localDim])
        let saOWeight = try loadNpy(
            MagpieConstants.Files.LocalTransformer.saOWeight,
            expecting: [localDim, localDim])
        // Conv1d kernel=1 is effectively (out, in) matmul; the exporter keeps
        // the trailing kernel dim so we accept either [out, in] or [out, in, 1].
        let ffnConv1Weight = try loadFlexible(
            name: MagpieConstants.Files.LocalTransformer.ffnConv1Weight,
            directory: ltDir,
            primary: [ffnDim, localDim],
            alternate: [ffnDim, localDim, 1])
        let ffnConv2Weight = try loadFlexible(
            name: MagpieConstants.Files.LocalTransformer.ffnConv2Weight,
            directory: ltDir,
            primary: [localDim, ffnDim],
            alternate: [localDim, ffnDim, 1])

        var outProjWeights: [[Float]] = []
        var outProjBiases: [[Float]] = []
        outProjWeights.reserveCapacity(numCodebooks)
        outProjBiases.reserveCapacity(numCodebooks)
        for cb in 0..<numCodebooks {
            let w = try loadNpy(
                MagpieConstants.Files.LocalTransformer.outProjWeight(codebook: cb),
                expecting: [numCodesPerCodebook, localDim])
            let b = try loadNpy(
                MagpieConstants.Files.LocalTransformer.outProjBias(codebook: cb),
                expecting: [numCodesPerCodebook])
            outProjWeights.append(w)
            outProjBiases.append(b)
        }

        logger.info(
            "Loaded local transformer weights: localDim=\(localDim), ffnDim=\(ffnDim), maxPositions=\(maxPositions), codebooks=\(numCodebooks)"
        )

        return MagpieLocalTransformerWeights(
            inProjWeight: inProjWeight,
            inProjBias: inProjBias,
            posEmbedding: posEmbedding,
            norm1Weight: norm1Weight,
            norm2Weight: norm2Weight,
            saQkvWeight: saQkvWeight,
            saOWeight: saOWeight,
            ffnConv1Weight: ffnConv1Weight,
            ffnConv2Weight: ffnConv2Weight,
            outProjWeights: outProjWeights,
            outProjBiases: outProjBiases,
            localDim: localDim,
            dModel: dModel,
            ffnDim: ffnDim,
            maxPositions: maxPositions,
            numCodebooks: numCodebooks,
            numCodesPerCodebook: numCodesPerCodebook
        )
    }

    /// Loads a `.npy` file accepting either `primary` or `alternate` shape. Returns
    /// the raw float buffer; callers treat both shapes as equivalent (conv1d
    /// kernel=1 vs plain matmul).
    private static func loadFlexible(
        name: String, directory: URL, primary: [Int], alternate: [Int]
    ) throws -> [Float] {
        let url = directory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieError.modelFileNotFound(
                "\(MagpieConstants.Files.localTransformerDir)/\(name)")
        }
        let array = try NpyReader.read(from: url)
        if array.shape == primary || array.shape == alternate {
            return array.data
        }
        throw MagpieError.invalidNpyFile(
            path: name,
            reason: "expected shape \(primary) or \(alternate), got \(array.shape)")
    }
}
