import Foundation

/// Result of SSML preprocessing - cleaned text with phonetic overrides
public struct SSMLProcessingResult: Sendable {
    public let text: String
    public let phoneticOverrides: [TtsPhoneticOverride]

    public init(text: String, phoneticOverrides: [TtsPhoneticOverride]) {
        self.text = text
        self.phoneticOverrides = phoneticOverrides
    }
}

/// Represents a parsed SSML tag with its position in the source text
struct SSMLParsedTag: Sendable {
    enum TagType: Sendable {
        case phoneme(alphabet: String, ph: String, content: String)
        case sub(alias: String, content: String)
        case sayAs(interpretAs: String, format: String?, content: String)
    }

    let type: TagType
    let range: Range<String.Index>
}
