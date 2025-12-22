import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Model Configuration

/// Configuration for Kokoro TTS model
public struct KokoroModelConfig: Codable {
    public let dimIn: Int
    public let dropout: Float
    public let hiddenDim: Int
    public let maxConvDim: Int
    public let maxDur: Int
    public let multispeaker: Bool
    public let nLayer: Int
    public let nMels: Int
    public let nToken: Int
    public let styleDim: Int
    public let textEncoderKernelSize: Int
    public let sampleRate: Int
    public let vocab: [String: Int]

    // ISTFTNet config
    public let istftnet: ISTFTNetConfig

    // PLBERT config
    public let plbert: PLBertConfig

    public struct ISTFTNetConfig: Codable {
        public let resblockKernelSizes: [Int]
        public let upsampleRates: [Int]
        public let upsampleInitialChannel: Int
        public let resblockDilationSizes: [[Int]]
        public let upsampleKernelSizes: [Int]
        public let genIstftNFft: Int
        public let genIstftHopSize: Int

        enum CodingKeys: String, CodingKey {
            case resblockKernelSizes = "resblock_kernel_sizes"
            case upsampleRates = "upsample_rates"
            case upsampleInitialChannel = "upsample_initial_channel"
            case resblockDilationSizes = "resblock_dilation_sizes"
            case upsampleKernelSizes = "upsample_kernel_sizes"
            case genIstftNFft = "gen_istft_n_fft"
            case genIstftHopSize = "gen_istft_hop_size"
        }
    }

    public struct PLBertConfig: Codable {
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int

        enum CodingKeys: String, CodingKey {
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case maxPositionEmbeddings = "max_position_embeddings"
        }
    }

    enum CodingKeys: String, CodingKey {
        case dimIn = "dim_in"
        case dropout
        case hiddenDim = "hidden_dim"
        case maxConvDim = "max_conv_dim"
        case maxDur = "max_dur"
        case multispeaker
        case nLayer = "n_layer"
        case nMels = "n_mels"
        case nToken = "n_token"
        case styleDim = "style_dim"
        case textEncoderKernelSize = "text_encoder_kernel_size"
        case sampleRate = "sample_rate"
        case vocab
        case istftnet
        case plbert
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        dimIn = try container.decode(Int.self, forKey: .dimIn)
        dropout = try container.decode(Float.self, forKey: .dropout)
        hiddenDim = try container.decode(Int.self, forKey: .hiddenDim)
        maxConvDim = try container.decode(Int.self, forKey: .maxConvDim)
        maxDur = try container.decode(Int.self, forKey: .maxDur)
        multispeaker = try container.decode(Bool.self, forKey: .multispeaker)
        nLayer = try container.decode(Int.self, forKey: .nLayer)
        nMels = try container.decode(Int.self, forKey: .nMels)
        nToken = try container.decode(Int.self, forKey: .nToken)
        styleDim = try container.decode(Int.self, forKey: .styleDim)
        textEncoderKernelSize = try container.decode(Int.self, forKey: .textEncoderKernelSize)
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        vocab = try container.decode([String: Int].self, forKey: .vocab)
        istftnet = try container.decode(ISTFTNetConfig.self, forKey: .istftnet)
        plbert = try container.decode(PLBertConfig.self, forKey: .plbert)
    }
}

// MARK: - Kokoro Model

/// Main Kokoro TTS model for MLX
/// Property names match the model's safetensors keys exactly
public final class KokoroModel: Module {
    public let config: KokoroModelConfig
    public let vocab: [String: Int]

    let bert: CustomAlbert
    // swiftlint:disable identifier_name
    let bert_encoder: Linear
    let text_encoder: KokoroTextEncoder
    // swiftlint:enable identifier_name
    let predictor: ProsodyPredictor
    let decoder: KokoroDecoder

    let contextLength: Int

    public init(config: KokoroModelConfig) {
        self.config = config
        self.vocab = config.vocab

        let albertConfig = AlbertModelArgs(
            vocabSize: config.nToken,
            numHiddenLayers: config.plbert.numHiddenLayers,
            numAttentionHeads: config.plbert.numAttentionHeads,
            hiddenSize: config.plbert.hiddenSize,
            intermediateSize: config.plbert.intermediateSize,
            maxPositionEmbeddings: config.plbert.maxPositionEmbeddings
        )

        self.bert = CustomAlbert(config: albertConfig)
        self.bert_encoder = Linear(albertConfig.hiddenSize, config.hiddenDim)
        self.contextLength = albertConfig.maxPositionEmbeddings

        self.predictor = ProsodyPredictor(
            styleDim: config.styleDim,
            dHid: config.hiddenDim,
            nLayers: config.nLayer,
            maxDur: config.maxDur,
            dropout: config.dropout
        )

        self.text_encoder = KokoroTextEncoder(
            channels: config.hiddenDim,
            kernelSize: config.textEncoderKernelSize,
            depth: config.nLayer,
            nSymbols: config.nToken
        )

        self.decoder = KokoroDecoder(
            dimIn: config.hiddenDim,
            styleDim: config.styleDim,
            dimOut: config.nMels,
            resblockKernelSizes: config.istftnet.resblockKernelSizes,
            upsampleRates: config.istftnet.upsampleRates,
            upsampleInitialChannel: config.istftnet.upsampleInitialChannel,
            resblockDilationSizes: config.istftnet.resblockDilationSizes,
            upsampleKernelSizes: config.istftnet.upsampleKernelSizes,
            genIstftNFft: config.istftnet.genIstftNFft,
            genIstftHopSize: config.istftnet.genIstftHopSize
        )
    }

    /// Generate audio from phoneme string and voice embedding
    /// - Parameters:
    ///   - phonemes: Phoneme string (bopomofo for Chinese)
    ///   - refS: Voice pack with shape (max_phonemes, 1, 256) or (batch, 256)
    ///   - speed: Speech speed multiplier
    public func callAsFunction(
        phonemes: String,
        refS: MLXArray,
        speed: Float = 1.0
    ) -> KokoroOutput {
        // Convert phonemes to input IDs
        var inputIds: [Int] = [0]  // Start token
        for phoneme in phonemes {
            if let id = vocab[String(phoneme)] {
                inputIds.append(id)
            }
        }
        inputIds.append(0)  // End token

        assert(inputIds.count <= contextLength, "Input too long: \(inputIds.count) > \(contextLength)")

        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        let inputLengths = MLXArray([Int32(inputIds.count)])

        // Create text mask
        let maxLen = inputIds.count
        var textMask = MLXArray(Int32(0)..<Int32(maxLen)).expandedDimensions(axis: 0)
        textMask = textMask + 1 .> inputLengths.expandedDimensions(axis: 1)

        // BERT encoding
        let (bertDur, _) = bert(inputIdsTensor, attentionMask: 1 - textMask.asType(.int32))
        let dEn = bert_encoder(bertDur).transposed(0, 2, 1)

        // Voice pack indexing: select style based on phoneme length
        // Voice pack shape: (max_phonemes, 1, 256) -> index by len(phonemes) - 1
        // Selected ref shape: (1, 256)
        let selectedRef: MLXArray
        if refS.ndim == 3 {
            // Voice pack format: (max_phonemes, batch, style_dim)
            let idx = max(0, phonemes.count - 1)
            selectedRef = refS[idx]  // Shape: (1, 256)
        } else {
            // Already 2D format: (batch, style_dim)
            selectedRef = refS
        }

        // Style extraction: second half of ref for prosody predictor
        let s = selectedRef[0..., 128...]

        // Duration prediction
        let d = predictor.text_encoder(dEn, style: s, textLengths: inputLengths, mask: textMask)
        let (lstmOut, _) = predictor.lstm(d)
        var duration = predictor.duration_proj(lstmOut)
        duration = sigmoid(duration).sum(axis: -1) / speed

        let predDur = clip(round(duration), min: 1).asType(.int32)[0]

        // Build alignment
        var indices: [Int32] = []
        for (i, n) in predDur.asArray(Int32.self).enumerated() {
            for _ in 0..<Int(n) {
                indices.append(Int32(i))
            }
        }

        let indicesArray = MLXArray(indices)
        var predAlnTrg = MLXArray.zeros([inputIds.count, indices.count])
        // Set alignment values (simplified - proper implementation needs scatter)
        for (j, i) in indices.enumerated() {
            predAlnTrg[Int(i), j] = MLXArray(1.0)
        }
        let predAlnTrgBatch = predAlnTrg.expandedDimensions(axis: 0)

        // Compute aligned encoding
        let en = matmul(d.transposed(0, 2, 1), predAlnTrgBatch)

        // F0 and N prediction
        let (f0Pred, nPred) = predictor.f0nTrain(en, s: s)

        // Text encoding
        let tEn = text_encoder(inputIdsTensor, inputLengths: inputLengths, mask: textMask)
        let asr = matmul(tEn, predAlnTrgBatch)

        // Decode to audio: first half of ref for decoder
        let audio = decoder(asr, f0Curve: f0Pred, n: nPred, s: selectedRef[0..., ..<128])

        eval(audio, predDur)

        let finalAudio = audio[0]

        return KokoroOutput(audio: finalAudio, predDur: predDur)
    }

    /// Load model from safetensors file
    public static func load(from url: URL, config: KokoroModelConfig) throws -> KokoroModel {
        let model = KokoroModel(config: config)

        // Load weights from safetensors
        let weights = try loadSafetensors(from: url)
        let sanitized = model.sanitize(weights)

        // Convert flat weights to nested structure and apply
        let nested = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: nested, verify: [.noUnusedKeys])

        return model
    }

    /// Load config from JSON file
    public static func loadConfig(from url: URL) throws -> KokoroModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(KokoroModelConfig.self, from: data)
    }

    // MARK: - Weight Sanitization

    private func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue  // Skip position_ids
            }

            if key.hasPrefix("bert") || key.hasPrefix("bert_encoder") {
                sanitized[key] = value
            }

            if key.hasPrefix("text_encoder") {
                if key.hasSuffix(".gamma") {
                    let newKey = key.replacingOccurrences(of: ".gamma", with: ".weight")
                    sanitized[newKey] = value
                } else if key.hasSuffix(".beta") {
                    let newKey = key.replacingOccurrences(of: ".beta", with: ".bias")
                    sanitized[newKey] = value
                } else if key.contains("lstm") && key.contains("weight_v") {
                    // Only transpose LSTM weight_v
                    sanitized[key] = sanitizeLSTMWeight(key, value)
                } else if key.contains("weight_v") {
                    // Conv weight_v - no transpose needed
                    sanitized[key] = value
                } else {
                    sanitized.merge(sanitizeLSTMKeys(key, value)) { _, new in new }
                }
            }

            if key.hasPrefix("predictor") {
                if key.contains("F0_proj.weight") || key.contains("N_proj.weight") {
                    // Weights are already in MLX format (out, kernel, in), no transpose needed
                    sanitized[key] = value
                } else if key.contains("lstm") && key.contains("weight_v") {
                    // Only transpose LSTM weight_v, not conv weight_v
                    sanitized[key] = sanitizeLSTMWeight(key, value)
                } else if key.contains("weight_v") {
                    // Conv weight_v - no transpose needed, already in MLX format (out, kernel, in)
                    sanitized[key] = value
                } else {
                    sanitized.merge(sanitizeLSTMKeys(key, value)) { _, new in new }
                }
            }

            if key.hasPrefix("decoder") {
                // Handle alpha parameters - wrap in .weight for AlphaParam module
                if key.contains(".alpha1.") || key.contains(".alpha2.") {
                    // decoder.generator.resblocks.0.0.alpha1.0 -> decoder.generator.resblocks.0.0.alpha1.0.weight
                    let newKey = key + ".weight"
                    sanitized[newKey] = value
                } else {
                    sanitized[key] = sanitizeDecoderWeight(key, value)
                }
            }
        }

        return sanitized
    }

    private func sanitizeLSTMWeight(_ key: String, _ value: MLXArray) -> MLXArray {
        if value.ndim == 3 && value.dim(1) > value.dim(2) {
            return value.transposed(0, 2, 1)
        }
        return value
    }

    private func sanitizeLSTMKeys(_ key: String, _ value: MLXArray) -> [String: MLXArray] {
        let weightMap: [String: String] = [
            "weight_ih_l0_reverse": "Wx_backward",
            "weight_hh_l0_reverse": "Wh_backward",
            "bias_ih_l0_reverse": "bias_ih_backward",
            "bias_hh_l0_reverse": "bias_hh_backward",
            "weight_ih_l0": "Wx_forward",
            "weight_hh_l0": "Wh_forward",
            "bias_ih_l0": "bias_ih_forward",
            "bias_hh_l0": "bias_hh_forward",
        ]

        for (suffix, newSuffix) in weightMap {
            if key.hasSuffix(suffix) {
                let baseKey = String(key.dropLast(suffix.count + 1))
                return ["\(baseKey).\(newSuffix)": value]
            }
        }

        return [key: value]
    }

    private func sanitizeDecoderWeight(_ key: String, _ value: MLXArray) -> MLXArray {
        // noise_convs weights are already in MLX Conv1d format: (out_channels, kernel_size, in_channels)
        // No transpose needed
        // Conv weight_v - no transpose needed, already in MLX format (out, kernel, in)
        // Ups weights are in (in, kernel, out) format - will be transposed at runtime
        return value
    }
}

// MARK: - Output

public struct KokoroOutput {
    public let audio: MLXArray
    public let predDur: MLXArray
}

// MARK: - Safetensors Loading

private func loadSafetensors(from url: URL) throws -> [String: MLXArray] {
    // Use MLX's built-in safetensors loading
    let weights = try MLX.loadArrays(url: url)
    return weights
}
