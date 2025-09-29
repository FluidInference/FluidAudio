import Foundation

public enum TtsConstants {

    /// Canonical voice identifiers bundled with the Kokoro CoreML release.
    public static let availableVoices: [String] = [
        // American English
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
        "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
        // British English
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        // Spanish (LATAM)
        "ef_dora", "em_alex", "em_santa",
        // French
        "ff_siwis",
        // Hindi
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        // Italian
        "if_sara", "im_nicola",
        // Japanese
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        // Brazilian Portuguese
        "pf_dora", "pm_alex", "pm_santa",
        // Mandarin Chinese
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
