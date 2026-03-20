import Foundation
@preconcurrency import Tokenizers

/// Type alias to disambiguate from the local ASR `Tokenizer` class.
private typealias HFTokenizerProtocol = Tokenizers.Tokenizer

/// HuggingFace BPE tokenizer wrapper for VoxCPM 1.5.
///
/// Wraps `swift-transformers` `AutoTokenizer` loaded from a local directory
/// containing `tokenizer.json` and `tokenizer_config.json` (MiniCPM tokenizer).
///
/// Includes Chinese character splitting to match VoxCPM's
/// `mask_multichar_chinese_tokens` behavior.
public struct VoxCpmTokenizer: @unchecked Sendable {

    private let tokenizer: any HFTokenizerProtocol

    /// Load tokenizer from a local directory containing tokenizer.json.
    public static func load(from directory: URL) async throws -> VoxCpmTokenizer {
        guard
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("tokenizer.json").path
            )
        else {
            throw VoxCpmError.tokenizerFailed(
                "tokenizer.json not found in \(directory.lastPathComponent)")
        }
        let hfTokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return VoxCpmTokenizer(tokenizer: hfTokenizer)
    }

    /// Encode text to token IDs, applying Chinese character splitting.
    ///
    /// VoxCPM requires `mask_multichar_chinese_tokens` which splits multi-character
    /// Chinese tokens into individual characters. Without this, Chinese text
    /// produces garbage output.
    public func encode(_ text: String) -> [Int] {
        // Apply Chinese character splitting before tokenization
        let processed = maskMulticharChineseTokens(text)
        return tokenizer.encode(text: processed, addSpecialTokens: false)
    }

    // MARK: - Chinese Character Splitting

    /// Check if a character is in the CJK Unified Ideographs range.
    private func isChinese(_ char: Character) -> Bool {
        guard let scalar = char.unicodeScalars.first else { return false }
        let value = scalar.value
        // CJK Unified Ideographs: U+4E00 to U+9FFF
        // CJK Extension A: U+3400 to U+4DBF
        // CJK Extension B: U+20000 to U+2A6DF
        // CJK Compatibility Ideographs: U+F900 to U+FAFF
        return (0x4E00...0x9FFF).contains(value)
            || (0x3400...0x4DBF).contains(value)
            || (0x20000...0x2A6DF).contains(value)
            || (0xF900...0xFAFF).contains(value)
    }

    /// Split text so each Chinese character is separated by spaces.
    ///
    /// This matches VoxCPM's `mask_multichar_chinese_tokens` behavior:
    /// the tokenizer then sees each Chinese character as a separate token.
    private func maskMulticharChineseTokens(_ text: String) -> String {
        var result = ""
        for char in text {
            if isChinese(char) {
                result += " \(char) "
            } else {
                result.append(char)
            }
        }
        // Collapse multiple spaces
        return result.replacingOccurrences(
            of: "\\s+", with: " ", options: .regularExpression
        ).trimmingCharacters(in: .whitespaces)
    }
}
