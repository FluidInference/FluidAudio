#if TTS
import Foundation

/// Minimal BERT WordPiece tokenizer for the g2pW polyphone disambiguator.
///
/// g2pW operates exclusively on Hanzi targets, so the tokenizer only
/// needs the char-level path from `bert-base-chinese`: each Hanzi maps
/// to a single token (no subword merges), unmapped characters fall
/// through to `[UNK]`, and the sequence is wrapped in `[CLS]` / `[SEP]`
/// padded to the model's input length. This keeps the implementation
/// well under 200 LOC versus a full WordPiece port.
///
/// Vocabulary file format: one token per line, line number = token id.
/// This matches `vocab.txt` from
/// https://huggingface.co/bert-base-chinese — the file the upstream
/// g2pW Python project uses verbatim.
public struct MandarinBertTokenizer: Sendable {

    /// Token → id lookup. Loaded from `vocab.txt` line-by-line.
    public let vocab: [String: Int32]

    /// Special-token ids resolved from `vocab` at load time.
    public let clsId: Int32
    public let sepId: Int32
    public let padId: Int32
    public let unkId: Int32

    /// Maximum total sequence length the downstream model accepts.
    /// `bert-base-chinese` defaults to 512.
    public static let defaultMaxLength = 512

    public enum LoadError: Swift.Error, LocalizedError {
        case missingSpecialToken(String)
        case emptyVocab(URL)

        public var errorDescription: String? {
            switch self {
            case .missingSpecialToken(let name):
                return "BERT vocab is missing required special token '\(name)'"
            case .emptyVocab(let url):
                return "BERT vocab at \(url.path) is empty"
            }
        }
    }

    /// Load `vocab.txt` from disk. Tokens are read in line order; the
    /// id of a token equals its zero-indexed line number, matching the
    /// `bert-base-chinese` convention.
    public static func load(vocabURL: URL) throws -> MandarinBertTokenizer {
        let raw = try String(contentsOf: vocabURL, encoding: .utf8)
        let lines = raw.split(omittingEmptySubsequences: false, whereSeparator: { $0 == "\n" })
        guard !lines.isEmpty else { throw LoadError.emptyVocab(vocabURL) }
        var vocab: [String: Int32] = [:]
        vocab.reserveCapacity(lines.count)
        for (idx, line) in lines.enumerated() {
            // Strip trailing CR (Windows line-endings) and the optional
            // trailing newline left by the splitter; everything else is
            // significant (BERT tokens may include leading `##` etc.).
            var token = String(line)
            if token.hasSuffix("\r") { token.removeLast() }
            if token.isEmpty && idx == lines.count - 1 {
                // Final blank line from a trailing newline — drop.
                continue
            }
            vocab[token] = Int32(idx)
        }
        return try MandarinBertTokenizer(vocab: vocab)
    }

    public init(vocab: [String: Int32]) throws {
        self.vocab = vocab
        guard let cls = vocab["[CLS]"] else {
            throw LoadError.missingSpecialToken("[CLS]")
        }
        guard let sep = vocab["[SEP]"] else {
            throw LoadError.missingSpecialToken("[SEP]")
        }
        guard let pad = vocab["[PAD]"] else {
            throw LoadError.missingSpecialToken("[PAD]")
        }
        guard let unk = vocab["[UNK]"] else {
            throw LoadError.missingSpecialToken("[UNK]")
        }
        self.clsId = cls
        self.sepId = sep
        self.padId = pad
        self.unkId = unk
    }

    /// Tokenization output: padded `inputIds` + `attentionMask` plus
    /// the (post-CLS) position of each input character so g2pW can
    /// pick its target index without re-deriving offsets.
    public struct Encoded: Equatable, Sendable {
        public let inputIds: [Int32]
        public let attentionMask: [Int32]
        public let tokenTypeIds: [Int32]
        /// `tokenPositionForChar[i]` = id-array index where `chars[i]`
        /// landed (`i + 1` after `[CLS]` until a dropped whitespace
        /// character shifts it; a whitespace character gets the index of
        /// the token after it), so callers don't bake the offset in.
        public let tokenPositionForChar: [Int]
    }

    /// Char-level encode of a Hanzi sentence. Truncates from the right
    /// if the sentence + `[CLS]` + `[SEP]` would exceed `maxLength`.
    /// Pads with `[PAD]` (`attentionMask = 0`) up to `maxLength`.
    ///
    /// Whitespace is dropped, as BERT's basic tokenizer and upstream g2pW
    /// do (the vocab has no space token, so it would otherwise be
    /// `[UNK]`); punctuation stays as its own token.
    public func encode(
        chars: [Character],
        maxLength: Int = defaultMaxLength
    ) -> Encoded {
        precondition(maxLength >= 2, "maxLength must hold at least [CLS] + [SEP]")
        let usable = max(0, maxLength - 2)

        var inputIds: [Int32] = []
        inputIds.reserveCapacity(maxLength)
        var positions: [Int] = []
        positions.reserveCapacity(min(chars.count, usable))

        inputIds.append(clsId)
        for ch in chars {
            if ch.isWhitespace {
                positions.append(inputIds.count)
                continue
            }
            guard inputIds.count - 1 < usable else { break }
            positions.append(inputIds.count)
            inputIds.append(vocab[String(ch)] ?? unkId)
        }
        inputIds.append(sepId)

        var attentionMask = Array(repeating: Int32(1), count: inputIds.count)
        if inputIds.count < maxLength {
            let padCount = maxLength - inputIds.count
            inputIds.append(contentsOf: Array(repeating: padId, count: padCount))
            attentionMask.append(contentsOf: Array(repeating: Int32(0), count: padCount))
        }
        let tokenTypeIds = Array(repeating: Int32(0), count: maxLength)
        return Encoded(
            inputIds: inputIds,
            attentionMask: attentionMask,
            tokenTypeIds: tokenTypeIds,
            tokenPositionForChar: positions
        )
    }
}
#endif
