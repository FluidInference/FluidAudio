#if TTS
import Foundation

/// Paradee release. v1.2 keeps v1.1's frame-rate decoder head (about 3×
/// faster than v1.0's upsampling vocoder upstream) and adds three voices.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public enum ParadeeVersion: String, Sendable, CaseIterable {
    /// Paradee-8M v1.0: `af_heart` only, upsampling iSTFTNet vocoder plus phase-lock filter.
    case v10 = "1.0"
    /// Paradee-8M v1.2 (default): four voices, frame-rate source-filter head.
    case v12 = "1.2"

    /// Voices the release was trained on, in the order of the model's voice table.
    public var voices: [ParadeeVoice] {
        switch self {
        case .v10: return [.afHeart]
        case .v12: return [.afHeart, .bfIsabella, .amPuck, .bmFable]
        }
    }

    /// Whether the graphs take a `voice` input (v1.0 has a single fixed voice).
    var hasVoiceInput: Bool { self != .v10 }
}

/// Paradee voice. The British voices (`bf_`, `bm_`) were trained on misaki's
/// British pronunciations, so text for them goes through the British lexicon.
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

    public var isBritish: Bool { rawValue.hasPrefix("b") }

    /// Row of the v1.2 voice tables (`config.json` `voices` order).
    var modelIndex: Int32 {
        switch self {
        case .afHeart: return 0
        case .bfIsabella: return 1
        case .amPuck: return 2
        case .bmFable: return 3
        }
    }
}
#endif
