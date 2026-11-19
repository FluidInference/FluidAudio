import Foundation

/// The 25 European languages supported by canary-1b-v2 for transcription and
/// speech translation (en ↔ X), keyed by ISO 639-1 code.
public enum CanaryLanguage: String, CaseIterable, Sendable {
    case bulgarian = "bg"
    case croatian = "hr"
    case czech = "cs"
    case danish = "da"
    case dutch = "nl"
    case english = "en"
    case estonian = "et"
    case finnish = "fi"
    case french = "fr"
    case german = "de"
    case greek = "el"
    case hungarian = "hu"
    case italian = "it"
    case latvian = "lv"
    case lithuanian = "lt"
    case maltese = "mt"
    case polish = "pl"
    case portuguese = "pt"
    case romanian = "ro"
    case russian = "ru"
    case slovak = "sk"
    case slovenian = "sl"
    case spanish = "es"
    case swedish = "sv"
    case ukrainian = "uk"

    /// Decoder id of this language's `<|xx|>` special token (from vocab.json).
    public var tokenId: Int32 {
        switch self {
        case .bulgarian: return 46
        case .croatian: return 58
        case .czech: return 59
        case .danish: return 60
        case .dutch: return 62
        case .english: return 64
        case .estonian: return 66
        case .finnish: return 70
        case .french: return 71
        case .german: return 78
        case .greek: return 79
        case .hungarian: return 89
        case .italian: return 99
        case .latvian: return 117
        case .lithuanian: return 120
        case .maltese: return 127
        case .polish: return 150
        case .portuguese: return 151
        case .romanian: return 154
        case .russian: return 157
        case .slovak: return 167
        case .slovenian: return 168
        case .spanish: return 171
        case .swedish: return 175
        case .ukrainian: return 192
        }
    }
}
