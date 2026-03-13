import Foundation

/// Languages supported by the CharsiuG2P ByT5 multilingual model,
/// mapped to Kokoro voice prefixes.
public enum MultilingualG2PLanguage: String, CaseIterable, Sendable {
    case americanEnglish = "eng-us"
    case britishEnglish = "eng-uk"
    case spanish = "spa"
    case french = "fra"
    case hindi = "hin"
    case italian = "ita"
    case japanese = "jpn"
    case brazilianPortuguese = "por-bz"
    case mandarinChinese = "cmn"

    /// The CharsiuG2P language code used in the model input prefix.
    public var charsiuCode: String { rawValue }

    /// The formatted prefix prepended to input words (e.g. `"<eng-us>: "`).
    public var prefix: String { "<\(charsiuCode)>: " }

    /// Infer the language from a Kokoro voice identifier.
    ///
    /// Kokoro voices use a two-character prefix indicating language and gender
    /// (e.g. `"af_heart"` → American English female). Returns `nil` for
    /// unrecognized prefixes.
    public static func fromKokoroVoice(_ voiceId: String) -> MultilingualG2PLanguage? {
        guard voiceId.count >= 2 else { return nil }
        let prefix = String(voiceId.prefix(2))
        switch prefix {
        case "af", "am":
            return .americanEnglish
        case "bf", "bm":
            return .britishEnglish
        case "ef", "em":
            return .spanish
        case "ff", "fm":
            return .french
        case "hf", "hm":
            return .hindi
        case "if", "im":
            return .italian
        case "jf", "jm":
            return .japanese
        case "pf", "pm":
            return .brazilianPortuguese
        case "zf", "zm":
            return .mandarinChinese
        default:
            return nil
        }
    }
}
