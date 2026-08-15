#if os(macOS)
import Foundation

/// Multilingual ASR benchmark datasets supported by
/// `NemotronMultilingualFleursBenchmark`.
///
/// FLEURS is the default; LibriSpeech and Earnings22 provide English
/// cross-validation. All loaders produce the same
/// `FLEURSBenchmark.FLEURSSample` shape so the scoring pipeline is shared.
public enum MultilingualBenchmarkDataset: String, CaseIterable {
    case fleurs
    /// LibriSpeech test-clean / test-other / dev-* (English-only). Uses the
    /// per-flac transcripts from the `<chapter>.trans.txt` files that ship
    /// with the dataset. The exact subset is picked via `--librispeech-subset`.
    case librispeech
    /// Earnings22 — real multi-speaker financial earnings calls. Loads the
    /// chunked KWS subset (argmaxinc/contextual-earnings22) populated by
    /// `fluidaudio download --dataset earnings22-kws`. Each sample is one
    /// `<id>_chunk<N>.wav` + matching `<id>_chunk<N>.text.txt` reference.
    /// English-only. Real-world long-form benchmark (~120 calls, 772
    /// chunks; ~14.7s avg per chunk).
    case earnings22

    /// On-disk cache subdirectory under the user-supplied cache root.
    /// Lets multiple datasets coexist for the same set of languages.
    public var cacheSubdir: String {
        switch self {
        case .fleurs: return "FLEURS-full"
        case .librispeech: return "Datasets/LibriSpeech"
        case .earnings22: return "earnings22-kws"
        }
    }

    /// HuggingFace dataset repo identifier.
    public var hfRepo: String {
        switch self {
        case .fleurs: return "FluidInference/fleurs-full"
        case .librispeech: return "openslr/librispeech_asr"
        case .earnings22: return "argmaxinc/contextual-earnings22"
        }
    }

    /// URL the user should visit to accept dataset terms-of-service when
    /// HF returns 401/403 on a gated dataset.
    public var acceptTermsURL: String {
        switch self {
        case .fleurs: return "https://huggingface.co/datasets/FluidInference/fleurs-full"
        case .librispeech: return "https://www.openslr.org/12"
        case .earnings22: return "https://huggingface.co/datasets/argmaxinc/contextual-earnings22"
        }
    }

    /// Audio file extension as the loader reads it from disk.
    public var audioExtension: String {
        switch self {
        case .fleurs: return "wav"
        case .librispeech: return "flac"
        case .earnings22: return "wav"
        }
    }

    /// Map a FLEURS-style user-facing language code (e.g. `en_us`) to the
    /// per-dataset HuggingFace config name. Returns nil for languages outside
    /// the dataset's available configs.
    public func hfConfigName(forFleursCode code: String) -> String? {
        switch self {
        case .fleurs:
            return code  // FLEURS uses the same code as its config
        case .librispeech:
            // LibriSpeech is English-only.
            return code == "en_us" ? "en" : nil
        case .earnings22:
            // Earnings22 is English-only.
            return code == "en_us" ? "en" : nil
        }
    }

    /// Languages supported by this dataset (in FLEURS user-facing codes).
    /// FLEURS supports the full multilingual set (delegated to FLEURSBenchmark).
    public var supportedLanguages: [String] {
        switch self {
        case .fleurs:
            return []  // Validated by FLEURSBenchmark.supportedLanguages
        case .librispeech:
            return ["en_us"]
        case .earnings22:
            return ["en_us"]
        }
    }
}
#endif
