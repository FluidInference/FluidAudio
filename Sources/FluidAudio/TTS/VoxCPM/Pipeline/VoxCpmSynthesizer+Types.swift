import Foundation

/// CoreML I/O key names for VoxCPM models.
extension VoxCpmSynthesizer {

    /// Input/output key names for base_lm_step (24 layers, 48 KV cache tensors).
    enum BaseLmKeys {
        static let embed = "embed"
        static let position = "position"
        static let lmHidden = "lm_hidden"
        static let lmHiddenFsq = "lm_hidden_fsq"
        static let stopLogit = "stop_logit"

        static func k(_ layer: Int) -> String { "k\(layer)" }
        static func v(_ layer: Int) -> String { "v\(layer)" }
        static func outK(_ layer: Int) -> String { "out_k\(layer)" }
        static func outV(_ layer: Int) -> String { "out_v\(layer)" }
    }

    /// Input/output key names for residual_lm_step (8 layers, 16 KV cache tensors).
    enum ResidualLmKeys {
        static let embed = "embed"
        static let position = "position"
        static let resHidden = "res_hidden"

        static func k(_ layer: Int) -> String { "k\(layer)" }
        static func v(_ layer: Int) -> String { "v\(layer)" }
        static func outK(_ layer: Int) -> String { "out_k\(layer)" }
        static func outV(_ layer: Int) -> String { "out_v\(layer)" }
    }

    /// Input/output key names for locdit_step.
    enum LocDiTKeys {
        static let x = "x"
        static let mu = "mu"
        static let t = "t"
        static let cond = "cond"
        static let dt = "dt"
        static let velocity = "velocity"
    }

    /// Input/output key names for feat_encoder.
    enum FeatEncoderKeys {
        static let feat = "feat"
        static let embedding = "embedding"
    }

    /// Input/output key names for audio_vae_encoder.
    enum VaeEncoderKeys {
        static let audio = "audio"
        static let latent = "latent"
    }

    /// Input/output key names for audio_vae_decoder.
    enum VaeDecoderKeys {
        static let latent = "latent"
        static let audio = "audio"
    }

    /// Result of a VoxCPM synthesis operation.
    public struct SynthesisResult: Sendable {
        /// WAV audio data at 44.1 kHz mono.
        public let audio: Data
        /// Raw Float32 audio samples.
        public let samples: [Float]
        /// Number of generated latent patches.
        public let patchCount: Int
        /// Duration in seconds.
        public let duration: Double
    }
}
