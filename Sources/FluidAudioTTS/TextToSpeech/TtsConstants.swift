import FluidAudio
import Foundation

/// Constants for the TTS (Text-to-Speech) system.
///
/// - Note: **Beta:** The TTS system is currently in beta and only supports American English.
///   While voice identifiers for other languages are included in the model, only American English
///   voices are currently tested and supported. Additional language support is planned for future releases.
public enum TtsConstants {

    /// Voice identifier we regression-test and ship by default.
    /// This is the recommended American English voice for production use.
    public static let recommendedVoice = "af_heart"

    /// Canonical voice identifiers bundled with the Kokoro CoreML release.
    ///
    /// - Important: Only American English voices (af_*, am_*) are currently supported and tested.
    ///   Only `recommendedVoice` is covered by automated QA; all other voices are experimental.
    ///   Non-English voices are present in the model but are not yet quality-assured for production use.
    public static let availableVoices: [String] = [
        // American English (supported, beta)
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
        "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
        // British English (experimental, not tested)
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        // Spanish (LATAM) (experimental, not tested)
        "ef_dora", "em_alex", "em_santa",
        // French (experimental, not tested)
        "ff_siwis",
        // Hindi (experimental, not tested)
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        // Italian (experimental, not tested)
        "if_sara", "im_nicola",
        // Japanese (experimental, not tested)
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        // Brazilian Portuguese (experimental, not tested)
        "pf_dora", "pm_alex", "pm_santa",
        // Mandarin Chinese (experimental, not tested)
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia",
        "zm_yunyang",
    ]

    /// Characters to drop from synthesis input while preserving neighboring text.
    public static let delimiterCharacters: Set<Character> = ["(", ")", "[", "]", "{", "}"]

    /// Collapses multiple whitespace runs into a single space for clean synthesis input.
    public static let whitespacePattern = try! NSRegularExpression(pattern: "\\s+", options: [])

    /// Sample rate expected by Kokoro's CoreML models and downstream consumers.
    public static let audioSampleRate: Int = 24_000

    /// Core Kokoro tuning parameters. For 5s model configuration specifically
    public static let kokoroFrameSamples: Int = 600
    public static let shortVariantGuardThresholdSeconds: Double = 4.5
    public static let shortVariantGuardFrameCount: Int = 4
    public static let shortSentenceMergeTokenThreshold: Int = 242

    /// Model fetch configuration.
    public static let defaultRepository: String = "FluidInference/kokoro-82m-coreml"
    public static let defaultModelsSubdirectory: String = "Models"
}
