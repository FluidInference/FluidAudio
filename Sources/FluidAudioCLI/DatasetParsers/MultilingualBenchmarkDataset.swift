#if os(macOS)
import Foundation

/// Multilingual ASR benchmark datasets supported by
/// `NemotronMultilingualFleursBenchmark`.
///
/// FLEURS is the default; MCV (Mozilla Common Voice v17.0) and MLS
/// (Multilingual LibriSpeech) provide cross-validation against datasets without
/// FLEURS-specific eval quirks. All three loaders produce the same
/// `FLEURSBenchmark.FLEURSSample` shape so the scoring pipeline is shared.
public enum MultilingualBenchmarkDataset: String, CaseIterable {
    case fleurs
    case mcv
    case mls
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
        case .mcv: return "mcv-17"
        case .mls: return "mls-test-flac"
        case .librispeech: return "Datasets/LibriSpeech"
        case .earnings22: return "earnings22-kws"
        }
    }

    /// HuggingFace dataset repo identifier used by the datasets-server API.
    public var hfRepo: String {
        switch self {
        case .fleurs: return "FluidInference/fleurs-full"
        case .mcv: return "mozilla-foundation/common_voice_17_0"
        case .mls: return "facebook/multilingual_librispeech"
        case .librispeech: return "openslr/librispeech_asr"
        case .earnings22: return "argmaxinc/contextual-earnings22"
        }
    }

    /// URL the user should visit to accept dataset terms-of-service when
    /// HF returns 401/403 on a gated dataset.
    public var acceptTermsURL: String {
        switch self {
        case .fleurs: return "https://huggingface.co/datasets/FluidInference/fleurs-full"
        case .mcv: return "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
        case .mls: return "https://huggingface.co/datasets/facebook/multilingual_librispeech"
        case .librispeech: return "https://www.openslr.org/12"
        case .earnings22: return "https://huggingface.co/datasets/argmaxinc/contextual-earnings22"
        }
    }

    /// Audio file extension as the loader reads it from disk.
    /// FLEURS = wav, MCV-17 = mp3, MLS = flac (original-quality audio
    /// from `data/mls_<lang>/test/audio/*.tar.gz`; pre-populated via
    /// Scripts/prep_mls_flac.py — the HF datasets-server opus path used
    /// to live here but lossy transcoding cost 1-3pp WER vs NVIDIA).
    public var audioExtension: String {
        switch self {
        case .fleurs: return "wav"
        case .mcv: return "mp3"
        case .mls: return "flac"
        case .librispeech: return "flac"
        case .earnings22: return "wav"
        }
    }

    /// Map a FLEURS-style user-facing language code (e.g. `en_us`,
    /// `pt_br`) to the per-dataset HuggingFace config name. Returns nil for
    /// languages outside the dataset's available configs.
    ///
    /// Note: MLS has no English config (English is published separately as
    /// LibriSpeech, not under `multilingual_librispeech`). MCV-17 covers all
    /// 5 of en/es/fr/it/pt but is gated and lacks a per-row datasets-server
    /// viewer, so this enum's MCV path surfaces a clear error at download
    /// time rather than attempting per-row pagination.
    public func hfConfigName(forFleursCode code: String) -> String? {
        switch self {
        case .fleurs:
            return code  // FLEURS uses the same code as its config
        case .mcv:
            switch code {
            case "en_us": return "en"
            case "es_419": return "es"
            case "fr_fr": return "fr"
            case "it_it": return "it"
            case "pt_br": return "pt"
            default: return nil
            }
        case .mls:
            switch code {
            case "es_419": return "spanish"
            case "fr_fr": return "french"
            case "it_it": return "italian"
            case "pt_br": return "portuguese"
            // de_de / nl_nl / pl_pl are supported by MLS too, but we
            // currently only validate the en/es/fr/it/pt benchmark set.
            // `en_us` intentionally returns nil — MLS has no English config.
            default: return nil
            }
        case .librispeech:
            // LibriSpeech is English-only.
            return code == "en_us" ? "en" : nil
        case .earnings22:
            // Earnings22 is English-only.
            return code == "en_us" ? "en" : nil
        }
    }

    /// Languages supported by this dataset (in FLEURS user-facing codes).
    /// MCV: full 5 (en/es/fr/it/pt). MLS: 4 (es/fr/it/pt — no English).
    /// FLEURS supports the full multilingual set (delegated to FLEURSBenchmark).
    public var supportedLanguages: [String] {
        switch self {
        case .fleurs:
            return []  // Validated by FLEURSBenchmark.supportedLanguages
        case .mcv:
            return ["en_us", "es_419", "fr_fr", "it_it", "pt_br"]
        case .mls:
            return ["es_419", "fr_fr", "it_it", "pt_br"]
        case .librispeech:
            return ["en_us"]
        case .earnings22:
            return ["en_us"]
        }
    }
}
#endif
