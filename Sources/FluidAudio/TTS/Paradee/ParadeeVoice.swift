#if TTS
import Foundation

/// Paradee release. v1.2 keeps v1.1's frame-rate decoder head (about 3×
/// faster than v1.0's upsampling vocoder upstream) and adds three voices;
/// v1.3 is the same graph with four more voices in Hindi, French and Spanish.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public enum ParadeeVersion: String, Sendable, CaseIterable {
    /// Paradee-8M v1.0: `af_heart` only, upsampling iSTFTNet vocoder plus phase-lock filter.
    case v10 = "1.0"
    /// Paradee-8M v1.2: four English voices, frame-rate source-filter head.
    case v12 = "1.2"
    /// Paradee-8M v1.3 (default): v1.2's English voices plus Hindi, French and Spanish ones.
    case v13 = "1.3"

    /// Voices the release was trained on, in the order of the model's voice table.
    public var voices: [ParadeeVoice] {
        switch self {
        case .v10: return [.afHeart]
        case .v12: return [.afHeart, .bfIsabella, .amPuck, .bmFable]
        case .v13: return [.afHeart, .bfIsabella, .amPuck, .bmFable, .hfAlpha, .hmOmega, .ffSiwis, .emAlex]
        }
    }

    /// Whether the graphs take a `voice` input (v1.0 has a single fixed voice).
    var hasVoiceInput: Bool { self != .v10 }

    /// Whether text goes through Paradee's eSpeak-free G2P (``ParadeeG2P``):
    /// Hindi, French and Spanish, and the English words the Misaki lexicon
    /// lacks, as upstream 1.3.1 does. Earlier releases read English only, with
    /// the BART G2P for missing words.
    var usesParadeeG2P: Bool { self == .v13 }
}

/// Language of a Paradee voice; the raw value is the upstream G2P's language
/// code (`paradee_g2p.json`).
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public enum ParadeeLanguage: String, Sendable, CaseIterable {
    case americanEnglish = "en-us"
    case britishEnglish = "en-gb"
    case hindi = "hi"
    case french = "fr"
    case spanish = "es"

    public var isEnglish: Bool { self == .americanEnglish || self == .britishEnglish }
}

/// Paradee voice. As in Kokoro, the first letter is the language: `a`/`b`
/// American/British English (misaki phonemes; British lexicon for `b`), `h`
/// Hindi, `f` French, `e` Spanish (eSpeak-style phonemes from ``ParadeeG2P``).
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public enum ParadeeVoice: String, Sendable, CaseIterable {
    /// American English, female. The only voice of v1.0.
    case afHeart = "af_heart"
    /// British English, female.
    case bfIsabella = "bf_isabella"
    /// American English, male.
    case amPuck = "am_puck"
    /// British English, male.
    case bmFable = "bm_fable"
    /// Hindi, female (v1.3).
    case hfAlpha = "hf_alpha"
    /// Hindi, male (v1.3).
    case hmOmega = "hm_omega"
    /// French, female (v1.3).
    case ffSiwis = "ff_siwis"
    /// Spanish, male (v1.3).
    case emAlex = "em_alex"

    public var language: ParadeeLanguage {
        switch self {
        case .afHeart, .amPuck: return .americanEnglish
        case .bfIsabella, .bmFable: return .britishEnglish
        case .hfAlpha, .hmOmega: return .hindi
        case .ffSiwis: return .french
        case .emAlex: return .spanish
        }
    }

    public var isBritish: Bool { language == .britishEnglish }

    /// Row of the voice tables (`config.json` `voices` order). v1.3 keeps
    /// v1.2's rows 0–3 and adds 4–7.
    var modelIndex: Int32 {
        switch self {
        case .afHeart: return 0
        case .bfIsabella: return 1
        case .amPuck: return 2
        case .bmFable: return 3
        case .hfAlpha: return 4
        case .hmOmega: return 5
        case .ffSiwis: return 6
        case .emAlex: return 7
        }
    }
}
#endif
