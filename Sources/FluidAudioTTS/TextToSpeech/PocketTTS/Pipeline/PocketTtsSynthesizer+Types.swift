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
        static let audioOutput = "var_1445"
    }

    /// Mimi decoder streaming state key mappings (input name → output name).
    ///
    /// 23 state tensors (26 original minus 3 zero-length tensors that were removed
    /// from the model to fix CoreML Espresso crash with zero-element blobs).
    static let mimiStateMapping: [(input: String, output: String)] = [
        ("upsample_partial", "y_end_1"),
        ("attn0_cache", "new_cache_1_internal_tensor_assign_2"),
        ("attn0_offset", "var_402"),
        ("attn0_end_offset", "new_end_offset_1"),
        ("attn1_cache", "new_cache_internal_tensor_assign_2"),
        ("attn1_offset", "var_825"),
        ("attn1_end_offset", "new_end_offset"),
        ("conv0_prev", "var_998"),
        ("conv0_first", "var_1006"),
        ("convtr0_partial", "var_1048"),
        ("res0_conv0_prev", "var_1105"),
        ("res0_conv0_first", "var_1113"),
        ("res0_conv1_first", "var_1134"),
        ("convtr1_partial", "var_1178"),
        ("res1_conv0_prev", "var_1235"),
        ("res1_conv0_first", "var_1243"),
        ("res1_conv1_first", "var_1264"),
        ("convtr2_partial", "var_1308"),
        ("res2_conv0_prev", "var_1365"),
        ("res2_conv0_first", "var_1373"),
        ("res2_conv1_first", "var_1394"),
        ("conv_final_prev", "var_1450"),
        ("conv_final_first", "var_1458"),
    ]
}
