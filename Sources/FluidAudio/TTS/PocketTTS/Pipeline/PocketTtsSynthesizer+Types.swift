import Foundation

extension PocketTtsSynthesizer {

    /// Result of a PocketTTS synthesis operation.
    public struct SynthesisResult: Sendable {
        /// WAV audio data (24kHz, 16-bit mono).
        public let audio: Data
        /// Raw Float32 audio samples.
        public let samples: [Float]
        /// Number of 80ms frames generated.
        public let frameCount: Int
        /// Generation step at which EOS was detected (nil if max length reached).
        public let eosStep: Int?
    }

    /// CoreML output key names for the conditioning and generation step models
    /// are discovered at model-load time via `PocketTtsLayerKeys.discover(...)`
    /// because CoreML auto-generates names that differ between 6L and 24L
    /// language packs. See `PocketTtsLayerKeys.swift`.

    /// CoreML output key names for the Mimi decoder model.
    enum MimiKeys {
        static let audioOutput = "var_821"
    }

    /// Mimi decoder streaming state key mappings (input name → output name).
    ///
    /// 26 state tensors that carry the decoder's streaming context across frames:
    /// - Upsampling: `upsample_partial` — partial output buffer for upsampling layers
    /// - Attention: `attn{0,1}_cache/offset/end_offset` — causal attention KV caches
    /// - Convolutions: `conv*_prev/first` — causal conv padding buffers
    /// - Residual blocks: `res{0,1,2}_conv{0,1}_prev/first` — residual conv state
    /// - Transposed convs: `convtr{0,1,2}_partial` — transposed conv overlap buffers
    ///
    /// 3 zero-length tensors (`res{0,1,2}_conv1_prev`) are pass-throughs where
    /// input and output names are identical.
    static let mimiStateMapping: [(input: String, output: String)] = [
        ("upsample_partial", "var_82"),
        ("attn0_cache", "var_262"),
        ("attn0_offset", "var_840"),
        ("attn0_end_offset", "new_end_offset_1"),
        ("attn1_cache", "var_479"),
        ("attn1_offset", "var_843"),
        ("attn1_end_offset", "new_end_offset"),
        ("conv0_prev", "var_607"),
        ("conv0_first", "conv0_first"),
        ("convtr0_partial", "var_634"),
        ("res0_conv0_prev", "var_660"),
        ("res0_conv0_first", "res0_conv0_first"),
        ("res0_conv1_prev", "res0_conv1_prev"),
        ("res0_conv1_first", "res0_conv1_first"),
        ("convtr1_partial", "var_700"),
        ("res1_conv0_prev", "var_726"),
        ("res1_conv0_first", "res1_conv0_first"),
        ("res1_conv1_prev", "res1_conv1_prev"),
        ("res1_conv1_first", "res1_conv1_first"),
        ("convtr2_partial", "var_766"),
        ("res2_conv0_prev", "var_792"),
        ("res2_conv0_first", "res2_conv0_first"),
        ("res2_conv1_prev", "res2_conv1_prev"),
        ("res2_conv1_first", "res2_conv1_first"),
        ("conv_final_prev", "var_824"),
        ("conv_final_first", "conv_final_first"),
    ]
}
