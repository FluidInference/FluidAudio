import Foundation

/// Fixed pipeline parameters for the Inflect v2 CoreML backend. Values mirror
/// the upstream `config.json` and the mobius conversion (`inflect-v2/coreml`).
public enum InflectConstants {

    /// Output sample rate (24 kHz mono, matches `data.sampling_rate`).
    public static let sampleRate = 24_000

    /// HiFiGAN total upsampling factor (8·8·2·2) — samples per mel frame.
    public static let hopLength = 256

    /// Fixed token axis of the encoder bundle (`tokens`/`x_mask` shape `[1, 512]`).
    public static let encoderTokens = 512

    /// Latent channel count fed to the synthesizer (`inter_channels`).
    /// Micro = 192, Nano = 128.
    public static func interChannels(for variant: InflectVariant) -> Int {
        switch variant {
        case .micro: return 192
        case .nano: return 128
        }
    }

    /// Synthesizer frame buckets (each `synthesizer_f<N>.mlmodelc`). The
    /// smallest bucket ≥ the predicted frame length is used; audio is trimmed
    /// to the exact length afterwards.
    public static let frameBuckets = [256, 384, 512, 640, 768, 896, 1024, 2048]

    /// Largest bucket — a chunk whose predicted duration exceeds this throws.
    public static var maxFrames: Int { frameBuckets.last! }

    /// Max phoneme characters per synthesis chunk. Interspersed with blanks
    /// this stays under the 512-token encoder axis (`2·255 + 1 = 511`); the
    /// text path splits longer input at punctuation/whitespace.
    public static let maxPhonemeChunkChars = 240

    /// Prior sampling temperature (`noise_scale`, VITS `z_p` stddev multiplier).
    public static let defaultNoiseScale: Float = 0.667

    /// Speech-rate multiplier. Durations scale by `1 / speed`.
    public static let defaultSpeed: Float = 1.0

    /// Edge fade applied to each chunk (ms), matching upstream `edge_fade`.
    public static let edgeFadeMs: Double = 5.0
}
