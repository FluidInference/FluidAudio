import Foundation
import MLX
import MLXNN

// MARK: - Albert Configuration

struct AlbertModelArgs {
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let embeddingSize: Int
    let innerGroupNum: Int
    let numHiddenGroups: Int
    let hiddenDropoutProb: Float
    let attentionProbsDropoutProb: Float
    let typeVocabSize: Int
    let layerNormEps: Float
    let vocabSize: Int

    init(
        vocabSize: Int = 30522,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        hiddenSize: Int = 768,
        intermediateSize: Int = 3072,
        maxPositionEmbeddings: Int = 512,
        embeddingSize: Int = 128,
        innerGroupNum: Int = 1,
        numHiddenGroups: Int = 1,
        hiddenDropoutProb: Float = 0.1,
        attentionProbsDropoutProb: Float = 0.1,
        typeVocabSize: Int = 2,
        layerNormEps: Float = 1e-12
    ) {
        self.vocabSize = vocabSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.embeddingSize = embeddingSize
        self.innerGroupNum = innerGroupNum
        self.numHiddenGroups = numHiddenGroups
        self.hiddenDropoutProb = hiddenDropoutProb
        self.attentionProbsDropoutProb = attentionProbsDropoutProb
        self.typeVocabSize = typeVocabSize
        self.layerNormEps = layerNormEps
    }
}

// MARK: - AlbertEmbeddings

/// Property names match the model's safetensors keys exactly
final class AlbertEmbeddings: Module {
    // swiftlint:disable identifier_name
    let word_embeddings: Embedding
    let position_embeddings: Embedding
    let token_type_embeddings: Embedding
    let LayerNorm: LayerNorm
    // swiftlint:enable identifier_name
    let dropout: Dropout

    init(config: AlbertModelArgs) {
        self.word_embeddings = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.embeddingSize
        )
        self.position_embeddings = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.embeddingSize
        )
        self.token_type_embeddings = Embedding(
            embeddingCount: config.typeVocabSize,
            dimensions: config.embeddingSize
        )
        self.LayerNorm = MLXNN.LayerNorm(dimensions: config.embeddingSize, eps: config.layerNormEps)
        self.dropout = Dropout(p: config.hiddenDropoutProb)
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        tokenTypeIds: MLXArray? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let seqLength = inputIds.dim(1)

        let posIds = positionIds ?? MLXArray(Int32(0)..<Int32(seqLength)).expandedDimensions(axis: 0)
        let typeIds = tokenTypeIds ?? MLXArray.zeros(like: inputIds)

        let wordEmb = word_embeddings(inputIds)
        let posEmb = position_embeddings(posIds)
        let typeEmb = token_type_embeddings(typeIds)

        var embeddings = wordEmb + posEmb + typeEmb
        embeddings = LayerNorm(embeddings)
        embeddings = dropout(embeddings)

        return embeddings
    }
}

// MARK: - AlbertSelfAttention

/// Property names match the model's safetensors keys exactly
final class AlbertSelfAttention: Module {
    let numAttentionHeads: Int
    let attentionHeadSize: Int
    let allHeadSize: Int

    let query: Linear
    let key: Linear
    let value: Linear
    let dense: Linear
    // swiftlint:disable identifier_name
    let LayerNorm: LayerNorm
    // swiftlint:enable identifier_name
    let dropout: Dropout

    init(config: AlbertModelArgs) {
        self.numAttentionHeads = config.numAttentionHeads
        self.attentionHeadSize = config.hiddenSize / config.numAttentionHeads
        self.allHeadSize = numAttentionHeads * attentionHeadSize

        self.query = Linear(config.hiddenSize, allHeadSize)
        self.key = Linear(config.hiddenSize, allHeadSize)
        self.value = Linear(config.hiddenSize, allHeadSize)
        self.dense = Linear(config.hiddenSize, config.hiddenSize)
        self.LayerNorm = MLXNN.LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.dropout = Dropout(p: config.attentionProbsDropoutProb)
    }

    private func transposeForScores(_ x: MLXArray) -> MLXArray {
        let newShape = [x.dim(0), x.dim(1), numAttentionHeads, attentionHeadSize]
        let reshaped = x.reshaped(newShape)
        return reshaped.transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let queryLayer = transposeForScores(query(hiddenStates))
        let keyLayer = transposeForScores(key(hiddenStates))
        let valueLayer = transposeForScores(value(hiddenStates))

        var attentionScores = MLX.matmul(queryLayer, keyLayer.transposed(0, 1, 3, 2))
        attentionScores = attentionScores / Float(attentionHeadSize).squareRoot()

        if let mask = attentionMask {
            attentionScores = attentionScores + mask
        }

        var attentionProbs = MLX.softmax(attentionScores, axis: -1)
        attentionProbs = dropout(attentionProbs)

        var contextLayer = MLX.matmul(attentionProbs, valueLayer)
        contextLayer = contextLayer.transposed(0, 2, 1, 3)

        let newShape = [contextLayer.dim(0), contextLayer.dim(1), allHeadSize]
        contextLayer = contextLayer.reshaped(newShape)

        contextLayer = dense(contextLayer)
        contextLayer = LayerNorm(contextLayer + hiddenStates)

        return contextLayer
    }
}

// MARK: - AlbertLayer

/// Property names match the model's safetensors keys exactly
final class AlbertLayer: Module {
    let attention: AlbertSelfAttention
    // swiftlint:disable identifier_name
    let full_layer_layer_norm: LayerNorm
    let ffn: Linear
    let ffn_output: Linear
    // swiftlint:enable identifier_name
    let activation: GELU

    init(config: AlbertModelArgs) {
        self.attention = AlbertSelfAttention(config: config)
        self.full_layer_layer_norm = MLXNN.LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.ffn = Linear(config.hiddenSize, config.intermediateSize)
        self.ffn_output = Linear(config.intermediateSize, config.hiddenSize)
        self.activation = GELU()
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
        let ffnOut = ffChunk(attentionOutput)
        return full_layer_layer_norm(ffnOut + attentionOutput)
    }

    private func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
        var out = ffn(attentionOutput)
        out = activation(out)
        out = ffn_output(out)
        return out
    }
}

// MARK: - AlbertLayerGroup

/// Property names match the model's safetensors keys exactly
final class AlbertLayerGroup: Module {
    // swiftlint:disable identifier_name
    var albert_layers: [AlbertLayer]
    // swiftlint:enable identifier_name

    init(config: AlbertModelArgs) {
        self.albert_layers = (0..<config.innerGroupNum).map { _ in
            AlbertLayer(config: config)
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var out = hiddenStates
        for layer in albert_layers {
            out = layer(out, attentionMask: attentionMask)
        }
        return out
    }
}

// MARK: - AlbertEncoder

/// Property names match the model's safetensors keys exactly
final class AlbertEncoder: Module {
    let config: AlbertModelArgs
    // swiftlint:disable identifier_name
    let embedding_hidden_mapping_in: Linear
    var albert_layer_groups: [AlbertLayerGroup]
    // swiftlint:enable identifier_name

    init(config: AlbertModelArgs) {
        self.config = config
        self.embedding_hidden_mapping_in = Linear(config.embeddingSize, config.hiddenSize)
        self.albert_layer_groups = (0..<config.numHiddenGroups).map { _ in
            AlbertLayerGroup(config: config)
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var out = embedding_hidden_mapping_in(hiddenStates)

        for i in 0..<config.numHiddenLayers {
            let groupIdx = i / (config.numHiddenLayers / config.numHiddenGroups)
            out = albert_layer_groups[groupIdx](out, attentionMask: attentionMask)
        }

        return out
    }
}

// MARK: - CustomAlbert

/// BERT-like encoder for Kokoro TTS
final class CustomAlbert: Module {
    let config: AlbertModelArgs
    let embeddings: AlbertEmbeddings
    let encoder: AlbertEncoder
    let pooler: Linear

    init(config: AlbertModelArgs) {
        self.config = config
        self.embeddings = AlbertEmbeddings(config: config)
        self.encoder = AlbertEncoder(config: config)
        self.pooler = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let embeddingOutput = embeddings(inputIds, tokenTypeIds: tokenTypeIds)

        var mask: MLXArray? = nil
        if let attnMask = attentionMask {
            // [batch, 1, 1, seq_len]
            mask = attnMask.expandedDimensions(axes: [1, 2])
            mask = (1.0 - mask!) * -10000.0
        }

        let encoderOutput = encoder(embeddingOutput, attentionMask: mask)
        let sequenceOutput = encoderOutput

        // Pooler: take first token and apply tanh
        let firstToken = sequenceOutput[0..., 0, 0...]
        let pooledOutput = MLX.tanh(pooler(firstToken))

        return (sequenceOutput, pooledOutput)
    }
}
