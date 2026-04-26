import Foundation

/// Constants for the NVIDIA Magpie TTS Multilingual 357M backend.
///
/// Source: https://huggingface.co/nvidia/magpie_tts_multilingual_357m
/// Architecture: encoder-decoder transformer + NanoCodec vocoder, produces 22 kHz audio.
public enum MagpieConstants {

    // MARK: - Audio

    /// NanoCodec output sample rate (Hz).
    public static let audioSampleRate: Int = 22_050
    /// Samples per codec frame (NanoCodec is 21.5 fps at 22050 Hz ⇒ ~1024 samples/frame).
    public static let codecSamplesPerFrame: Int = 1_024
    /// Peak-normalize audio to this level before returning samples.
    public static let peakTarget: Float = 0.9

    // MARK: - Model dimensions

    /// Transformer hidden dim (decoder input + output, encoder output).
    public static let dModel: Int = 768
    /// Decoder transformer layers.
    public static let numDecoderLayers: Int = 12
    /// Number of heads in decoder attention.
    public static let numHeads: Int = 12
    /// Head dimension (dModel / numHeads).
    public static let headDim: Int = 64
    /// Max KV cache length used when the decoder_step model was converted.
    public static let maxCacheLength: Int = 512
    /// Max text tokens after padding (matches traceable text_encoder input shape).
    public static let maxTextLength: Int = 256

    // MARK: - NanoCodec

    /// Number of codebooks the decoder emits per frame.
    public static let numCodebooks: Int = 8
    /// Number of codes per codebook (NanoCodec FSQ size).
    public static let numCodesPerCodebook: Int = 2_024
    /// Max frames NanoCodec accepts in a single forward pass.
    public static let maxNanocodecFrames: Int = 256

    // MARK: - Special audio token ids

    /// BOS for audio codebooks (never sampled).
    public static let audioBosId: Int32 = 2_016
    /// End-of-sequence: if sampled in any codebook, generation stops.
    public static let audioEosId: Int32 = 2_017
    /// Forbidden auxiliary tokens (CTX_BOS, CTX_EOS, MASK, reserved).
    public static let forbiddenAudioIds: [Int32] = [2_016, 2_018, 2_019, 2_020, 2_021, 2_022, 2_023]

    // MARK: - Speaker context

    /// Context length per speaker embedding (T_ctx).
    public static let speakerContextLength: Int = 110
    /// Number of built-in speakers (John, Sofia, Aria, Jason, Leo).
    public static let numSpeakers: Int = 5

    // MARK: - Local Transformer (Swift-side sampling head)

    /// Hidden dim of the 1-layer local transformer.
    public static let localTransformerDim: Int = 256
    /// FFN hidden dim inside the local transformer.
    public static let localTransformerFfnDim: Int = 1_024
    /// Max positional embedding slots (num_codebooks + 2 for BOS alignment).
    public static let localTransformerMaxPositions: Int = 10

    // MARK: - Generation defaults

    /// Max decoder steps per utterance (hard cap, ~11.9 s of audio).
    public static let maxSteps: Int = 500
    /// Number of steps EOS is masked out at the start (avoids empty audio).
    public static let minFrames: Int = 4
    /// Default sampling temperature.
    public static let defaultTemperature: Float = 0.6
    /// Default top-k truncation.
    public static let defaultTopK: Int = 80
    /// Default CFG scale. `1.0` disables the unconditional path entirely.
    ///
    /// The Python reference ships `cfg_scale = 2.5` (in `constants.json`) which doubles
    /// `decoder_step` calls per frame (cond + uncond). Default is now `1.0` so the
    /// Swift port runs at half the wall time out-of-the-box; opt back in via
    /// `MagpieSynthesisOptions.cfgScale = 2.5` (or `--cfg 2.5` on the CLI) when guidance
    /// quality matters more than throughput.
    public static let defaultCfgScale: Float = 1.0

    // MARK: - Repository

    /// HuggingFace repository id that ships the compiled CoreML artifacts + constants.
    public static let huggingFaceRepo: String = "FluidInference/magpie-tts-multilingual-357m-coreml"

    // MARK: - File names

    public enum Files {
        // Models
        public static let textEncoder = "text_encoder.mlmodelc"
        public static let decoderPrefill = "decoder_prefill.mlmodelc"  // optional
        public static let decoderStep = "decoder_step.mlmodelc"
        public static let nanocodecDecoder = "nanocodec_decoder.mlmodelc"

        // Constants
        public static let constantsDir = "constants"
        public static let constantsJson = "constants.json"
        public static let tokenizerMetadataJson = "tokenizer_metadata.json"

        public static func speakerEmbedding(index: Int) -> String { "speaker_\(index).npy" }
        public static func audioEmbedding(codebook: Int) -> String { "audio_embedding_\(codebook).npy" }

        // Local transformer weights (under constants/local_transformer/)
        public static let localTransformerDir = "local_transformer"
        public enum LocalTransformer {
            public static let inProjWeight = "in_proj_weight.npy"
            public static let inProjBias = "in_proj_bias.npy"
            public static let posEmb = "pos_emb.npy"
            public static let norm1Weight = "norm1_weight.npy"
            public static let norm2Weight = "norm2_weight.npy"
            public static let saQkvWeight = "sa_qkv_weight.npy"
            public static let saOWeight = "sa_o_weight.npy"
            public static let ffnConv1Weight = "ffn_conv1_weight.npy"
            public static let ffnConv2Weight = "ffn_conv2_weight.npy"
            public static func outProjWeight(codebook: Int) -> String { "out_proj_\(codebook)_weight.npy" }
            public static func outProjBias(codebook: Int) -> String { "out_proj_\(codebook)_bias.npy" }
        }

        // Tokenizer data (under tokenizer/)
        public static let tokenizerDir = "tokenizer"
    }
}
