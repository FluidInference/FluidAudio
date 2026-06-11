import Foundation

/// Decoded transcript with detected language.
public struct NemotronMultilingualDecoded: Sendable {
    /// Transcript text with `<xx-XX>` language tag tokens stripped.
    public let text: String
    /// First language-tag piece encountered (e.g. `"en-US"`), or `nil` if none.
    public let detectedLanguage: String?

    public init(text: String, detectedLanguage: String?) {
        self.text = text
        self.detectedLanguage = detectedLanguage
    }
}

/// Tokenizer wrapper for the Nemotron multilingual model.
///
/// The model emits a leading `<xx-XX>` language-tag token (one of
/// `langTagTokenIds`). This wrapper:
///   1. Surfaces the first such tag as `detectedLanguage` (without angle brackets).
///   2. Strips all language-tag tokens from the textual transcript so they don't
///      appear in the user-visible output.
///
/// Underlying vocab format is identical to the English variant: a flat
/// `{"id": "piece"}` JSON dictionary, decoded by the shared `Tokenizer`.
public final class NemotronMultilingualTokenizer: Sendable {
    private let base: Tokenizer
    private let langTagTokenIds: Set<Int>

    public init(vocabPath: URL, langTagTokenIds: Set<Int>) throws {
        self.base = try Tokenizer(vocabPath: vocabPath)
        self.langTagTokenIds = langTagTokenIds
    }

    /// Decode a sequence of token IDs into text + detected language.
    public func decode(ids: [Int]) -> NemotronMultilingualDecoded {
        var detected: String? = nil
        var filtered: [Int] = []
        filtered.reserveCapacity(ids.count)

        for id in ids {
            if langTagTokenIds.contains(id) {
                if detected == nil, let piece = base.piece(forId: id) {
                    detected = NemotronMultilingualTokenizer.stripAngleBrackets(piece)
                }
                continue
            }
            filtered.append(id)
        }

        // Collapse runs of spaces to one. The model can emit a standalone
        // word-boundary token (`▁`, id 2) — e.g. after sentence-final
        // punctuation — on top of the next word's own leading `▁`, which the
        // SentencePiece detokenizer would otherwise render as a double space.
        let raw = base.decode(ids: filtered)
        let text = raw.replacingOccurrences(
            of: " {2,}", with: " ", options: .regularExpression)
        return NemotronMultilingualDecoded(text: text, detectedLanguage: detected)
    }

    /// Strip leading `<` and trailing `>` from a piece like `"<en-US>"`.
    /// Returns the original string unchanged if the brackets are missing.
    static func stripAngleBrackets(_ piece: String) -> String {
        guard piece.hasPrefix("<"), piece.hasSuffix(">"), piece.count >= 2 else {
            return piece
        }
        return String(piece.dropFirst().dropLast())
    }

    /// Return the raw SentencePiece piece for a token id, INCLUDING the
    /// `▁` word-boundary marker, or `nil` if the id is not in the vocab.
    /// Unlike `decode(ids:)` / `tokenizerPiece`, this does NOT strip the marker
    /// or filter lang-tag tokens — callers grouping tokens into words need the
    /// original marker-bearing piece to detect word starts.
    public func rawToken(for id: Int) -> String? {
        return base.rawToken(for: id)
    }

    /// Look up the token id for a language-tag piece (e.g. `"en-US"` →
    /// the id whose piece equals `"<en-US>"`). Returns `nil` if no
    /// matching tag is present in `langTagTokenIds`.
    public func langTagTokenId(forLanguage language: String) -> Int? {
        let target = "<\(language)>"
        for id in langTagTokenIds {
            if base.piece(forId: id) == target {
                return id
            }
        }
        return nil
    }
}
