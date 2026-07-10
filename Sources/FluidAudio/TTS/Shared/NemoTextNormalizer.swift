import CNemoTextProcessing
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

    /// Normalize `text` for `language`. Returns `text` unchanged if the engine
    /// declines the input (its own out-of-domain passthrough) or the underlying
    /// library was built without the `fst-engine` feature — so this is always
    /// safe to call as a frontend pre-pass.
    public static func normalize(_ text: String, language: Language) -> String {
        guard let ptr = nemo_tn_fst(text, language.rawValue) else { return text }
        defer { nemo_free_string(ptr) }
        return String(cString: ptr)
    }
}
