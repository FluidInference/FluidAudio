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

// MARK: - Shared Utilities

/// Apostrophe characters that should be treated as part of a word
/// Used by both SSML processing and text preprocessing for consistent word boundary detection
let phoneticApostropheCharacters: Set<Character> = ["'", "'", "ʼ", "‛", "‵", "′"]

/// Check if a character is an emoji
/// Used by both SSML processing and text preprocessing for consistent word boundary detection
func isEmoji(_ character: Character) -> Bool {
    character.unicodeScalars.contains { scalar in
        scalar.properties.isEmojiPresentation || scalar.properties.isEmoji
    }
}

/// Check if a character is considered part of a word for TTS processing
/// Used for word boundary detection in SSML and text preprocessing
func isWordCharacter(
    _ char: Character,
    apostrophes: Set<Character> = phoneticApostropheCharacters
) -> Bool {
    char.isLetter || char.isNumber || apostrophes.contains(char) || isEmoji(char)
}

/// Shared NumberFormatter for spelling out numbers (expensive to create)
/// Used by SSML processing, text preprocessing, and chunking
let spellOutFormatter: NumberFormatter = {
    let formatter = NumberFormatter()
    formatter.numberStyle = .spellOut
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.maximumFractionDigits = 0
    formatter.roundingMode = .down
    return formatter
}()

/// Digit words for spelling individual digits (0-9)
/// Used by SSML processing and text preprocessing
let digitWords = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

/// Convert a digit character to its word representation
func digitToWord(_ char: Character) -> String? {
    guard let digit = Int(String(char)), digit >= 0, digit <= 9 else { return nil }
    return digitWords[digit]
}
