import Foundation

/// Result of tokenizing text for Magpie: padded token ids + mask + pre-pad length.
public struct MagpieTokenizedText: Sendable {
    public let paddedIds: [Int32]
    public let mask: [Float]
    public let realLength: Int

    public init(paddedIds: [Int32], mask: [Float], realLength: Int) {
        self.paddedIds = paddedIds
        self.mask = mask
        self.realLength = realLength
    }
}

/// Common interface for per-language Magpie tokenizers.
public protocol MagpieLanguageTokenizer: Sendable {
    var language: MagpieLanguage { get }
    /// Convert raw text to a list of token ids (pre-padding). Must append the model's
    /// EOS id if the caller expects one — Magpie appends EOS downstream.
    func encode(_ text: String) throws -> [Int32]

    /// Encode a `|...|` IPA override region where `tokens` are space-separated IPA
    /// phoneme strings that must be looked up directly against the language's
    /// `token2id` map (with the language offset applied).
    func encodeIpaTokens(_ tokens: [String]) throws -> [Int32]
}

/// Top-level dispatcher that loads the appropriate language tokenizer on demand
/// and pads/truncates the result to `MagpieConstants.maxTextLength`.
public actor MagpieTokenizer {

    private let logger = AppLogger(category: "MagpieTokenizer")
    private let tokenizerDir: URL
    private let eosId: Int32

    private var cache: [MagpieLanguage: MagpieLanguageTokenizer] = [:]

    /// - Parameters:
    ///   - tokenizerDir: directory containing the per-language JSON lookup files.
    ///   - eosId: language-agnostic EOS token id (from `tokenizer_metadata.json`
    ///     or the constants bundle).
    public init(tokenizerDir: URL, eosId: Int32) {
        self.tokenizerDir = tokenizerDir
        self.eosId = eosId
    }

    /// Resolve (and cache) the tokenizer for `language`.
    public func tokenizer(for language: MagpieLanguage) throws -> MagpieLanguageTokenizer {
        if let cached = cache[language] { return cached }
        let tok = try makeTokenizer(for: language)
        cache[language] = tok
        return tok
    }

    /// Full encode: text → (padded ids + mask). Honors `|...|` IPA override when
    /// `options.allowIpaOverride == true`.
    public func tokenize(
        _ text: String, language: MagpieLanguage, options: MagpieSynthesisOptions
    ) throws -> MagpieTokenizedText {
        let tok = try tokenizer(for: language)

        var ids: [Int32] = []
        if options.allowIpaOverride {
            for segment in MagpieIpaOverride.segment(text) {
                switch segment {
                case .text(let str):
                    ids.append(contentsOf: try tok.encode(str))
                case .ipa(let tokens):
                    ids.append(contentsOf: try tok.encodeIpaTokens(tokens))
                }
            }
        } else {
            ids.append(contentsOf: try tok.encode(text))
        }

        // Append EOS unless the encoder already did so.
        if ids.last != eosId {
            ids.append(eosId)
        }

        let maxLen = MagpieConstants.maxTextLength
        if ids.count > maxLen {
            throw MagpieError.textTooLong(tokenCount: ids.count, maxLength: maxLen)
        }

        var padded = Swift.Array<Int32>(repeating: 0, count: maxLen)
        var mask = Swift.Array<Float>(repeating: 0, count: maxLen)
        for (i, v) in ids.enumerated() {
            padded[i] = v
            mask[i] = 1.0
        }
        return MagpieTokenizedText(paddedIds: padded, mask: mask, realLength: ids.count)
    }

    /// Pad/truncate pre-tokenized phoneme ids without running any G2P.
    public func pad(phonemes: MagpiePhonemeTokens) throws -> MagpieTokenizedText {
        var ids = phonemes.tokenIds
        if ids.last != eosId { ids.append(eosId) }
        let maxLen = MagpieConstants.maxTextLength
        if ids.count > maxLen {
            throw MagpieError.textTooLong(tokenCount: ids.count, maxLength: maxLen)
        }
        var padded = Swift.Array<Int32>(repeating: 0, count: maxLen)
        var mask = Swift.Array<Float>(repeating: 0, count: maxLen)
        for (i, v) in ids.enumerated() {
            padded[i] = v
            mask[i] = 1.0
        }
        return MagpieTokenizedText(paddedIds: padded, mask: mask, realLength: ids.count)
    }

    // MARK: - Factory

    private func makeTokenizer(for language: MagpieLanguage) throws -> MagpieLanguageTokenizer {
        switch language {
        case .english, .spanish, .german, .italian, .vietnamese:
            return try MagpiePhonemeTokenizer(
                language: language,
                tokenizerDir: tokenizerDir)
        case .french, .hindi:
            return try MagpieCharTokenizer(
                language: language,
                tokenizerDir: tokenizerDir)
        case .mandarin:
            return try MagpieMandarinTokenizer(tokenizerDir: tokenizerDir)
        }
    }
}
