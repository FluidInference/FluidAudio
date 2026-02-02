import Foundation

// MARK: - Qwen3-ASR Model Configuration

/// Configuration constants for the Qwen3-ASR-0.6B CoreML model.
///
/// Architecture: audio_encoder -> embedding + merge -> 28 decoder layers -> lm_head
/// The audio encoder processes mel spectrograms in fixed-size windows.
/// The LLM decoder generates text autoregressively with KV-cache.
public struct Qwen3AsrConfig: Sendable {
    // MARK: Audio

    public let sampleRate: Int = 16000
    public let numMelBins: Int = 128

    /// Audio encoder window size in mel frames (n_window * 2).
    /// The encoder processes chunks of this size.
    public let melWindowSize: Int = 100

    /// Conv2D downsampling factor: 3 layers of stride-2 -> 8x reduction.
    public let convDownsampleFactor: Int = 8

    /// Output frames per mel window: ceil(100/8) = 13.
    public var outputFramesPerWindow: Int { (melWindowSize + convDownsampleFactor - 1) / convDownsampleFactor }

    // MARK: Encoder

    public let encoderDModel: Int = 896
    public let encoderOutputDim: Int = 1024
    public let encoderNumLayers: Int = 18
    public let encoderNumHeads: Int = 14

    // MARK: Decoder (LLM)

    public let hiddenSize: Int = 1024
    public let intermediateSize: Int = 3072
    public let numDecoderLayers: Int = 28
    public let numAttentionHeads: Int = 16
    public let numKVHeads: Int = 8
    public let headDim: Int = 128
    public let vocabSize: Int = 151_936
    public let ropeTheta: Double = 1_000_000
    public let mropeSection: [Int] = [24, 20, 20]

    // MARK: Special Tokens

    public let audioStartTokenId: Int = 151_669
    public let audioEndTokenId: Int = 151_670
    public let audioTokenId: Int = 151_676
    public let eosTokenIds: Set<Int> = [151_645, 151_643]

    // MARK: Chat Template Tokens

    public let imStartTokenId: Int = 151_644
    public let imEndTokenId: Int = 151_645
    public let systemTokenId: Int = 8948
    public let userTokenId: Int = 872
    public let assistantTokenId: Int = 77_091
    public let newlineTokenId: Int = 198

    // MARK: Generation

    public let maxSequenceLength: Int = 4096
    public let maxAudioSeconds: Double = 30.0

    public static let `default` = Qwen3AsrConfig()
}
