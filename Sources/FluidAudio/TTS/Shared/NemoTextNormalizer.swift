#if canImport(CNemoTextProcessing)
import CNemoTextProcessing
#endif
import Foundation

/// Byte-exact NeMo text normalization via the bundled compiled-FST engine
/// (`text-processing-rs`, `fst-engine` feature).
///
/// Converts written forms to their spoken reading *before* G2P — the standard
/// TTS frontend order — e.g. `"$5"` → `"five dollars"`, `"2024年"` →
/// `"二零二四年"`. Output matches NVIDIA NeMo's
/// `Normalizer(lang=…, deterministic=True)` exactly.
///
/// The engine is deterministic and rule-authored (no model, no inference); see
/// `docs/NEMO_PARITY.md` in text-processing-rs.
public enum NemoTextNormalizer {

    /// BCP-47-ish language codes the FST engine supports.
    public enum Language: String {
        case english = "en"
        case mandarin = "zh"
        case japanese = "ja"
        case french = "fr"
        case spanish = "es"
        case german = "de"
        case hindi = "hi"
    }

    /// Whether the engine is linked into this build. `false` when the package
    /// was resolved with the `NemoTextProcessing` trait disabled (#880, #888);
    /// `normalize` then returns its input unchanged.
    public static var isAvailable: Bool {
        #if canImport(CNemoTextProcessing)
        return true
        #else
        return false
        #endif
    }

    /// Normalize `text` for `language`. Returns `text` unchanged if the engine
    /// declines the input (its own out-of-domain passthrough), the underlying
    /// library was built without the `fst-engine` feature, or the engine is
    /// not linked (`isAvailable == false`) — so this is always safe to call as
    /// a frontend pre-pass.
    public static func normalize(_ text: String, language: Language) -> String {
        #if canImport(CNemoTextProcessing)
        guard let ptr = nemo_tn_fst(text, language.rawValue) else { return text }
        defer { nemo_free_string(ptr) }
        return String(cString: ptr)
        #else
        return text
        #endif
    }
}
