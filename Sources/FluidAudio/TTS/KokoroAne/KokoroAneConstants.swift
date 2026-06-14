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

    /// Output sample rate of the iSTFT in `KokoroTail.mlpackage`.
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

    /// HuggingFace repo case for this variant.
    public var repo: Repo {
        switch self {
        case .english: return .kokoroAne
        case .mandarin: return .kokoroAneZh
        case .japanese: return .kokoroAneJa
        }
    }
}
