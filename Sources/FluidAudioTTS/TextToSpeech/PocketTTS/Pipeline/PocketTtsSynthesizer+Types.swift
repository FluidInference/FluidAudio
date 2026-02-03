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

    /// CoreML output key names for the conditioning step model.
    enum CondStepKeys {
        static let cacheKeys: [String] = [
            "new_cache_1_internal_tensor_assign_2",
            "new_cache_3_internal_tensor_assign_2",
            "new_cache_5_internal_tensor_assign_2",
            "new_cache_7_internal_tensor_assign_2",
            "new_cache_9_internal_tensor_assign_2",
            "new_cache_internal_tensor_assign_2",
        ]
        static let positionKeys: [String] = [
            "var_445", "var_864", "var_1283", "var_1702", "var_2121", "var_2365",
        ]
    }

    /// CoreML output key names for the generation step model.
    enum FlowLMStepKeys {
        /// CoreML assigned this output the name "input" during model tracing —
        /// it is the transformer hidden state output, not an input tensor.
        static let transformerOut = "input"
        static let eosLogit = "var_2582"
        static let cacheKeys: [String] = [
            "new_cache_1_internal_tensor_assign_2",
            "new_cache_3_internal_tensor_assign_2",
            "new_cache_5_internal_tensor_assign_2",
            "new_cache_7_internal_tensor_assign_2",
            "new_cache_9_internal_tensor_assign_2",
            "new_cache_internal_tensor_assign_2",
        ]
        static let positionKeys: [String] = [
            "var_458", "var_877", "var_1296", "var_1715", "var_2134", "var_2553",
        ]
    }

    /// CoreML output key names for the Mimi decoder model.
    enum MimiKeys {
        static let audioOutput = "var_821"
    }

    /// Mimi decoder streaming state key mappings (input name → output name).
    ///
    /// 26 state tensors including 3 zero-length tensors (res{0,1,2}_conv1_prev)
    /// whose input and output names are identical pass-throughs.
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
