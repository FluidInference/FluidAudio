import Foundation

/// Compile-time constants for the laishere/kokoro 7-stage CoreML chain.
///
/// Source of truth: mobius/models/tts/kokoro/laishere-coreml/convert-coreml.py
/// (specifically `compute_shape_bounds(max_frames=2000)` and the per-stage
/// I/O contracts).
public enum KokoroAneConstants {

    /// Default voice id for the English (`ANE/`) variant.
    public static let defaultVoice = "af_heart"

    /// Default voice id for the Mandarin (`ANE-zh/`) variant.
    public static let defaultVoiceMandarin = "zf_001"

    /// Default voice id for the Japanese (`ANE-ja/`) variant.
    public static let defaultVoiceJapanese = "jf_alpha"

    /// Voice packs published for the English (`ANE/`) variant. Only
    /// `af_heart.bin` ships pre-converted; every other name is the Kokoro-82M
    /// v1.0 pack hosted as `voices/<name>.json` at the repository root, which
    /// `KokoroAneResourceDownloader.ensureVoicePack` converts to the flat
    /// `[510, 256]` fp32 layout on first use (#896). The 7-stage chain takes
    /// the style vectors as runtime inputs, so any v1.0 pack works with it;
    /// non-English-prefixed packs (`zf_*`, `jf_*`, …) still speak English
    /// phonemes here, just with that voice's timbre.
    /// Listing as of 2026-09-09 (huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/voices).
    public static let englishVoices: [String] = [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam",
        "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
        "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex",
        "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
        "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni",
        "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]

    /// Voice packs in the Mandarin (`ANE-zh/voices/`) bundle, as of 2026-09-09.
    public static let mandarinVoices: [String] = [
        "af_maple", "af_sol", "bf_vale", "zf_001", "zf_002", "zf_003",
        "zf_004", "zf_005", "zf_006", "zf_007", "zf_008", "zf_017",
        "zf_018", "zf_019", "zf_021", "zf_022", "zf_023", "zf_024",
        "zf_026", "zf_027", "zf_028", "zf_032", "zf_036", "zf_038",
        "zf_039", "zf_040", "zf_042", "zf_043", "zf_044", "zf_046",
        "zf_047", "zf_048", "zf_049", "zf_051", "zf_059", "zf_060",
        "zf_067", "zf_070", "zf_071", "zf_072", "zf_073", "zf_074",
        "zf_075", "zf_076", "zf_077", "zf_078", "zf_079", "zf_083",
        "zf_084", "zf_085", "zf_086", "zf_087", "zf_088", "zf_090",
        "zf_092", "zf_093", "zf_094", "zf_099", "zm_009", "zm_010",
        "zm_011", "zm_012", "zm_013", "zm_014", "zm_015", "zm_016",
        "zm_020", "zm_025", "zm_029", "zm_030", "zm_031", "zm_033",
        "zm_034", "zm_035", "zm_037", "zm_041", "zm_045", "zm_050",
        "zm_052", "zm_053", "zm_054", "zm_055", "zm_056", "zm_057",
        "zm_058", "zm_061", "zm_062", "zm_063", "zm_064", "zm_065",
        "zm_066", "zm_068", "zm_069", "zm_080", "zm_081", "zm_082",
        "zm_089", "zm_091", "zm_095", "zm_096", "zm_097", "zm_098",
        "zm_100",
    ]

    /// Voice packs in the Japanese (`ANE-ja/voices/`) bundle, as of 2026-09-09.
    public static let japaneseVoices: [String] = [
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    ]

    /// Output sample rate of the iSTFT in `KokoroTail_v2.mlpackage`.
    public static let sampleRate = 24_000

    /// BOS / EOS token id used by both `convert-coreml.py` and the iOS demo.
    public static let bosTokenId: Int32 = 0
    public static let eosTokenId: Int32 = 0

    /// ALBERT context window — input_ids cannot exceed this, so the IPA
    /// phoneme sequence (excluding BOS/EOS) must be ≤ 510.
    public static let maxInputTokens = 512
    public static let maxPhonemeLength = 510

    /// Voice pack rows × columns. The pack is stored flat as `[510, 256]` fp32:
    ///   * row index = `min(max(phonemeCount - 1, 0), 509)` — bucketed by the
    ///     raw phoneme-string length (BOS/EOS excluded), matching
    ///     `convert.py:get_ref_data`.
    ///   * cols `[0..<128]`   = `style_timbre` (→ Noise + Vocoder)
    ///   * cols `[128..<256]` = `style_s`      (→ PostAlbert + Prosody)
    public static let voicePackRows = 510
    public static let voicePackCols = 256

    /// `--max-frames` baked into the converted models. Sentences whose `T_a`
    /// exceeds this must be skipped or chunked.
    public static let maxAcousticFrames = 2_000

    /// Default playback speed factor for PostAlbert.
    public static let defaultSpeed: Float = 1.0

    // MARK: - Mandarin G2P assets

    /// Local subdirectory (relative to the cached `ANE-zh/` repo dir) for
    /// the Mandarin G2P binary dictionaries.
    public static let g2pSubdir = "g2p"

    /// Single-Hanzi pinyin dict, fetched by
    /// `KokoroAneResourceDownloader.ensureMandarinG2P` and cached at
    /// `<repoDir>/g2p/pinyin_single.bin`.
    public static let g2pPinyinSingleFile = "pinyin_single.bin"

    /// Hanzi-phrase pinyin dict.
    public static let g2pPinyinPhrasesFile = "pinyin_phrases.bin"

    /// HuggingFace repo that hosts the Mandarin G2P binary fixtures.
    /// Co-located with the CoreML weights so the Mandarin variant has a
    /// single HF dependency.
    public static let g2pRemoteRepo = "FluidInference/kokoro-82m-coreml"

    /// Subdirectory inside `g2pRemoteRepo` containing the `.bin` payloads.
    public static let g2pRemoteSubdir = "ANE-zh/assets"

    /// Remote artefact names (uncompressed — ~10 MB total, dwarfed by the
    /// 7 mlmodelc bundles already in this repo).
    public static let g2pPinyinSingleRemoteFile = "pinyin_single.bin"
    public static let g2pPinyinPhrasesRemoteFile = "pinyin_phrases.bin"

    // MARK: - Jieba HMM tables

    /// Local filenames for the three jieba HMM tables (start /
    /// transition / emission), cached alongside the pinyin dicts under
    /// `<repoDir>/g2p/`. Format documented on
    /// `MandarinJiebaHmmTables`.
    public static let jiebaHmmStartFile = "jieba_hmm_start.bin"
    public static let jiebaHmmTransFile = "jieba_hmm_trans.bin"
    public static let jiebaHmmEmitFile = "jieba_hmm_emit.bin"

    /// Remote artefact names — uploaded to the same `ANE-zh/assets/`
    /// folder as the pinyin dicts. Combined size is ≈ 3 MB (emit table
    /// dominates; ~7 800 codepoints × 16 bytes plus headers).
    public static let jiebaHmmStartRemoteFile = "jieba_hmm_start.bin"
    public static let jiebaHmmTransRemoteFile = "jieba_hmm_trans.bin"
    public static let jiebaHmmEmitRemoteFile = "jieba_hmm_emit.bin"

    // MARK: - Mandarin g2pW polyphone disambiguator

    /// Local subdirectory (relative to the cached `ANE-zh/` repo dir) for
    /// the g2pW BERT classifier + its tokenizer / catalog assets.
    public static let g2pwSubdir = "g2pw"

    /// Compiled CoreML bundle name. Matches the upstream HF folder.
    public static let g2pwModelBundle = "g2pw.mlmodelc"

    /// `bert-base-chinese` vocab co-located with the model.
    public static let g2pwVocabFile = "vocab.txt"

    /// Per-character allowed-phoneme map shipped alongside the model.
    public static let g2pwPolyphonicCharsFile = "POLYPHONIC_CHARS.txt"

    /// Subdirectory inside `g2pRemoteRepo` containing the g2pW assets.
    public static let g2pwRemoteSubdir = "ANE-zh/g2pw"

    /// Remote artefact filenames (mirrors the local names — no rename).
    public static let g2pwVocabRemoteFile = "vocab.txt"
    public static let g2pwPolyphonicCharsRemoteFile = "POLYPHONIC_CHARS.txt"
}

/// Language variant of the laishere/kokoro 7-stage CoreML chain.
///
/// The 7-stage chain is language-agnostic by construction (input ids, voice
/// slices, and per-stage I/O contracts are identical across variants). Only
/// the embedding vocab, HF subdirectory, voice-file layout, and the default
/// voice id differ.
///
/// | Variant      | HF subdir  | Default voice | Voice layout                  | Text frontend                |
/// |--------------|------------|---------------|-------------------------------|------------------------------|
/// | `.english`   | `ANE/`     | `af_heart`    | flat (`<voice>.bin`)          | `KokoroAneEnglishPhonemizer` |
/// | `.mandarin`  | `ANE-zh/`  | `zf_001`      | nested (`voices/<voice>.bin`) | `MandarinG2P`                |
/// | `.japanese`  | `ANE-ja/`  | `jf_alpha`    | nested (`voices/<voice>.bin`) | none — phoneme bypass only   |
///
/// The Japanese variant ships **no in-process text → phoneme frontend**.
/// `synthesize(text:)` / `phonemes(for:)` throw for `.japanese`; callers feed
/// pre-computed IPA through ``KokoroAneManager/synthesizeFromPhonemes(_:voice:speed:)``
/// (the bypass path the 7-stage chain already supports). See issue #698.
public enum KokoroAneVariant: String, CaseIterable, Sendable {
    case english
    case mandarin
    case japanese

    /// Default voice id shipped with the variant's HF bundle.
    public var defaultVoice: String {
        switch self {
        case .english: return KokoroAneConstants.defaultVoice
        case .mandarin: return KokoroAneConstants.defaultVoiceMandarin
        case .japanese: return KokoroAneConstants.defaultVoiceJapanese
        }
    }

    /// True if voice packs live under a `voices/` subdirectory inside the repo
    /// bundle (Mandarin / Japanese); false if they sit at the bundle root
    /// (English).
    public var useVoicesSubdir: Bool {
        switch self {
        case .english: return false
        case .mandarin, .japanese: return true
        }
    }

    /// Voice ids known to be available for this variant (see
    /// `KokoroAneConstants.englishVoices` etc.). Used for error messages and
    /// the CLI listing; a name outside the list is still attempted.
    public var knownVoices: [String] {
        switch self {
        case .english: return KokoroAneConstants.englishVoices
        case .mandarin: return KokoroAneConstants.mandarinVoices
        case .japanese: return KokoroAneConstants.japaneseVoices
        }
    }

    /// HuggingFace repo case for this variant.
    public var repo: Repo {
        switch self {
        case .english: return .kokoroAne
        case .mandarin: return .kokoroAneZh
        case .japanese: return .kokoroAneJa
        }
    }
}
