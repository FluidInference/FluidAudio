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
public struct VoxCpmTokenizer: Sendable {

    private let tokenizer: any HFTokenizerProtocol

    /// Pre-computed set of multi-character pure-Chinese tokens from the vocabulary.
    /// These tokens need to be split into individual characters for VoxCPM.
    private let multicharChineseTokens: Set<String>

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

        // Pre-compute multi-character Chinese tokens from vocabulary.
        // This matches Python's mask_multichar_chinese_tokens which finds all vocab
        // entries with length >= 2 where every character is CJK Unified Ideographs.
        var multichar = Set<String>()
        // Iterate vocab by trying to decode each possible token ID
        // and checking if the result is a multi-char Chinese string.
        // We use a range covering typical vocab sizes.
        let vocabSize = VoxCpmConstants.vocabSize
        for id in 0..<vocabSize {
            guard let token = hfTokenizer.convertIdToToken(id) else { continue }
            // Strip subword prefix (▁) for checking
            let clean = token.hasPrefix("▁") ? String(token.dropFirst()) : token
            if clean.count >= 2 && clean.allSatisfy({ isCJK($0) }) {
                multichar.insert(clean)
            }
        }

        return VoxCpmTokenizer(
            tokenizer: hfTokenizer,
            multicharChineseTokens: multichar
        )
    }

    /// Encode text to token IDs, applying Chinese character splitting.
    ///
    /// Matches Python's `mask_multichar_chinese_tokens` behavior:
    /// 1. Tokenize text normally with BPE
    /// 2. For each token, strip `▁` prefix and check if it's a multi-character
    ///    pure-Chinese token
    /// 3. If yes, split into individual characters
    /// 4. Convert processed tokens to IDs
    public func encode(_ text: String) -> [Int] {
        let tokens = tokenizer.tokenize(text: text)

        var processed: [String] = []
        for token in tokens {
            let clean = token.hasPrefix("▁") ? String(token.dropFirst()) : token
            if multicharChineseTokens.contains(clean) {
                // Split multi-character Chinese token into individual characters
                for char in clean {
                    processed.append(String(char))
                }
            } else {
                processed.append(token)
            }
        }

        // Convert tokens to IDs
        let ids = tokenizer.convertTokensToIds(processed)
        return ids.map { $0 ?? 0 }
    }

    // MARK: - CJK Detection

    /// Check if a character is in the CJK Unified Ideographs range.
    private static func isCJK(_ char: Character) -> Bool {
        guard let scalar = char.unicodeScalars.first else { return false }
        let value = scalar.value
        // Match Python: '\u4e00' <= c <= '\u9fff'
        return (0x4E00...0x9FFF).contains(value)
    }
}
