import Foundation

/// A single custom vocabulary entry.
public struct CustomVocabularyTerm: Codable, Sendable {
    public let text: String
    public let weight: Float?
    public let aliases: [String]?
    /// Optional pre-tokenized Parakeet TDT vocabulary IDs for this phrase.
    /// When present, decode-time biasing can operate directly on RNNT/TDT token IDs.
    public let tokenIds: [Int]?
    /// Optional pre-tokenized CTC vocabulary IDs for this phrase.
    /// This is used by the auxiliary CTC keyword spotter path and is distinct from
    /// the Parakeet TDT token IDs above.
    public let ctcTokenIds: [Int]?

    /// Optional per-term minimum string similarity for CTC rescoring.
    ///
    /// Overrides the vocabulary-level `CustomVocabularyContext.minSimilarity`
    /// for this term only. Lower values widen recall (more aggressive
    /// correction); higher values demand a closer match (fewer false
    /// replacements). When `nil`, the vocabulary-level / size-aware default is
    /// used. Vocabulary-wide safety guards (short-word, stopword-span, and
    /// length-ratio floors) still apply on top of this value, so a per-term
    /// override can only loosen matching down to those guards.
    public let minSimilarity: Float?

    /// Pre-computed lowercased text for efficient comparison (not serialized).
    public let textLowercased: String

    // Only encode/decode the original properties, not the cached ones
    private enum CodingKeys: String, CodingKey {
        case text, weight, aliases, tokenIds, ctcTokenIds, minSimilarity
    }

    /// Create a custom vocabulary term.
    /// - Parameters:
    ///   - text: The word or phrase to boost
    ///   - weight: Optional weight for the term
    ///   - aliases: Optional alternative spellings
    ///   - tokenIds: Optional pre-tokenized Parakeet TDT token IDs
    ///   - ctcTokenIds: Optional pre-tokenized CTC token IDs
    ///   - minSimilarity: Optional per-term minimum string similarity override
    ///     (falls back to the vocabulary-level threshold when `nil`)
    public init(
        text: String,
        weight: Float? = nil,
        aliases: [String]? = nil,
        tokenIds: [Int]? = nil,
        ctcTokenIds: [Int]? = nil,
        minSimilarity: Float? = nil
    ) {
        self.text = text
        self.weight = weight
        self.aliases = aliases
        self.tokenIds = tokenIds
        self.ctcTokenIds = ctcTokenIds
        self.minSimilarity = minSimilarity.map { Self.clampSimilarity($0) }
        self.textLowercased = text.lowercased()
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try container.decode(String.self, forKey: .text)
        weight = try container.decodeIfPresent(Float.self, forKey: .weight)
        aliases = try container.decodeIfPresent([String].self, forKey: .aliases)
        tokenIds = try container.decodeIfPresent([Int].self, forKey: .tokenIds)
        ctcTokenIds = try container.decodeIfPresent([Int].self, forKey: .ctcTokenIds)
        minSimilarity = try container.decodeIfPresent(Float.self, forKey: .minSimilarity)
            .map { Self.clampSimilarity($0) }
        textLowercased = text.lowercased()
    }

    /// Clamp a similarity threshold into the valid [0, 1] range so malformed
    /// config values cannot disable or invert the rescoring gate.
    private static func clampSimilarity(_ value: Float) -> Float {
        min(1.0, max(0.0, value))
    }
}

/// Raw JSON model for on‑disk config (internal DTO for JSON decoding).
struct CustomVocabularyConfig: Codable, Sendable {
    let alpha: Float?
    let terms: [CustomVocabularyTerm]

    // CTC keyword boosting confidence thresholds
    let minCtcScore: Float?
    let minSimilarity: Float?
    let minCombinedConfidence: Float?

    /// Minimum character length for vocabulary terms (per NeMo CTC-WS paper)
    /// Terms shorter than this are skipped to reduce false positives (e.g., "or" → "VR")
    let minTermLength: Int?
}

/// Runtime context used by the decoder biasing system.
public struct CustomVocabularyContext: Sendable {
    public let terms: [CustomVocabularyTerm]
    public let alpha: Float

    // CTC keyword boosting confidence thresholds
    public let minCtcScore: Float
    public let minSimilarity: Float
    public let minCombinedConfidence: Float

    /// Minimum character length for vocabulary terms (per NeMo CTC-WS paper)
    /// Terms shorter than this are skipped to reduce false positives (e.g., "or" → "VR")
    public let minTermLength: Int

    public init(
        terms: [CustomVocabularyTerm],
        alpha: Float = ContextBiasingConstants.defaultAlpha,
        minCtcScore: Float = ContextBiasingConstants.defaultMinVocabCtcScore,
        minSimilarity: Float = ContextBiasingConstants.defaultMinSimilarity,
        minCombinedConfidence: Float = ContextBiasingConstants.defaultMinCombinedConfidence,
        minTermLength: Int = 3
    ) {
        self.terms = terms
        self.alpha = alpha
        self.minCtcScore = minCtcScore
        self.minSimilarity = minSimilarity
        self.minCombinedConfidence = minCombinedConfidence
        self.minTermLength = minTermLength
    }

    /// Load a custom vocabulary JSON file produced by the analysis tooling.
    public static func load(from url: URL) throws -> CustomVocabularyContext {
        let logger = AppLogger(category: "CustomVocabulary")
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(CustomVocabularyConfig.self, from: data)

        let alpha = config.alpha ?? ContextBiasingConstants.defaultAlpha
        let minCtcScore = config.minCtcScore ?? ContextBiasingConstants.defaultMinVocabCtcScore
        let minSimilarity = config.minSimilarity ?? ContextBiasingConstants.defaultMinSimilarity
        let minCombinedConfidence = config.minCombinedConfidence ?? ContextBiasingConstants.defaultMinCombinedConfidence
        let minTermLength = config.minTermLength ?? 3

        // Validate and normalize vocabulary terms
        var validatedTerms: [CustomVocabularyTerm] = []
        for term in config.terms {
            let (sanitized, warnings) = sanitizeVocabularyTerm(term.text)

            if !warnings.isEmpty {
                logger.warning("Term '\(term.text)': \(warnings.joined(separator: ", "))")
            }

            // Skip empty terms after sanitization
            guard !sanitized.isEmpty else {
                logger.warning("Term '\(term.text)' is empty after sanitization, skipping")
                continue
            }

            // Sanitize aliases too (remove control characters, skip empty)
            let sanitizedAliases = term.aliases?.compactMap { alias -> String? in
                let (sanitizedAlias, _) = sanitizeVocabularyTerm(alias)
                return sanitizedAlias.isEmpty ? nil : sanitizedAlias
            }

            // Use sanitized text and aliases
            let sanitizedTerm = CustomVocabularyTerm(
                text: sanitized,
                weight: term.weight,
                aliases: sanitizedAliases?.isEmpty == true ? nil : sanitizedAliases,
                tokenIds: term.tokenIds,
                ctcTokenIds: term.ctcTokenIds,
                minSimilarity: term.minSimilarity
            )
            validatedTerms.append(sanitizedTerm)
        }

        return CustomVocabularyContext(
            terms: validatedTerms,
            alpha: alpha,
            minCtcScore: minCtcScore,
            minSimilarity: minSimilarity,
            minCombinedConfidence: minCombinedConfidence,
            minTermLength: minTermLength
        )
    }

    /// Load a custom vocabulary from simple text format.
    /// Format: one word per line, optionally "word: alias1, alias2, ..."
    public static func loadFromSimpleFormat(from url: URL) throws -> CustomVocabularyContext {
        let contents = try String(contentsOf: url, encoding: .utf8)
        var terms: [CustomVocabularyTerm] = []

        for line in contents.split(whereSeparator: { $0.isNewline }) {
            let trimmed = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }

            // Parse "word: alias1, alias2, ..." format
            if let colonIndex = trimmed.firstIndex(of: ":") {
                let word = String(trimmed[..<colonIndex]).trimmingCharacters(in: .whitespaces)
                let aliasesPart = String(trimmed[trimmed.index(after: colonIndex)...])
                let rawAliases = aliasesPart.split(separator: ",").map {
                    String($0).trimmingCharacters(in: .whitespaces)
                }.filter { !$0.isEmpty }

                // Sanitize term and aliases
                let (sanitizedWord, _) = sanitizeVocabularyTerm(word)
                guard !sanitizedWord.isEmpty else { continue }

                let sanitizedAliases = rawAliases.compactMap { alias -> String? in
                    let (sanitized, _) = sanitizeVocabularyTerm(alias)
                    return sanitized.isEmpty ? nil : sanitized
                }

                terms.append(
                    CustomVocabularyTerm(
                        text: sanitizedWord,
                        weight: 10.0,  // Aggressive default weight for text list
                        aliases: sanitizedAliases.isEmpty ? nil : sanitizedAliases
                    ))
            } else {
                // Sanitize term
                let (sanitizedWord, _) = sanitizeVocabularyTerm(trimmed)
                guard !sanitizedWord.isEmpty else { continue }
                terms.append(CustomVocabularyTerm(text: sanitizedWord, weight: 10.0))
            }
        }

        return CustomVocabularyContext(terms: terms)
    }

    /// Sanitize a vocabulary term and return warnings about potential issues.
    private static func sanitizeVocabularyTerm(_ text: String) -> (sanitized: String, warnings: [String]) {
        var warnings: [String] = []
        var result = text

        // 1. Check for control characters
        if result.rangeOfCharacter(from: .controlCharacters) != nil {
            warnings.append("contains control characters")
            result = result.filter { !$0.isNewline && !$0.isWhitespace || $0 == " " }
        }

        // 2. Check for diacritics (informational, not blocking)
        if result.folding(options: .diacriticInsensitive, locale: nil) != result {
            warnings.append("contains diacritics - consider adding ASCII alias")
        }

        // 3. Check for numbers (informational)
        if result.rangeOfCharacter(from: .decimalDigits) != nil {
            warnings.append("contains numbers")
        }

        // 4. Check for unusual characters (not letters, spaces, hyphens, apostrophes)
        let allowedChars = CharacterSet.letters
            .union(.whitespaces)
            .union(CharacterSet(charactersIn: "-'"))

        if result.rangeOfCharacter(from: allowedChars.inverted) != nil {
            warnings.append("contains unusual characters")
        }

        return (result, warnings)
    }

    // MARK: - CTC Tokenization

    /// Load vocabulary from file and tokenize with CTC tokenizer.
    ///
    /// This is a convenience method that loads a vocabulary file and tokenizes
    /// each term with the CTC tokenizer for use with vocabulary boosting. The
    /// file may be either the structured JSON config (which supports per-term
    /// `minSimilarity` and vocabulary-level thresholds, see ``load(from:)``) or
    /// the simple one-term-per-line text format (see ``loadFromSimpleFormat(from:)``).
    /// The format is detected from the file contents.
    ///
    /// - Parameters:
    ///   - path: Path to the vocabulary file (JSON config or simple text list)
    ///   - ctcVariant: CTC model variant to use for tokenization (default: .ctc110m)
    /// - Returns: Tuple of tokenized vocabulary context and loaded CTC models
    /// - Throws: Error if vocabulary file cannot be read or CTC models fail to load
    public static func loadWithCtcTokens(
        from path: String,
        ctcVariant: CtcModelVariant = .ctc110m
    ) async throws -> (vocab: CustomVocabularyContext, models: CtcModels) {
        // Load CTC models
        let ctcModels = try await CtcModels.downloadAndLoad(variant: ctcVariant)

        // Load vocabulary from file (JSON config or simple text list).
        let vocabURL = URL(fileURLWithPath: path)
        let loadedVocab = try loadVocabularyFile(at: vocabURL)

        // Load CTC tokenizer
        let ctcTokenizer = try await CtcTokenizer.load(
            from: CtcModels.defaultCacheDirectory(for: ctcVariant)
        )

        // Tokenize each term with CTC tokenizer, preserving per-term settings.
        let tokenizedTerms = loadedVocab.terms.compactMap { term -> CustomVocabularyTerm? in
            let tokenIds = ctcTokenizer.encode(term.text)
            guard !tokenIds.isEmpty else { return nil }
            return CustomVocabularyTerm(
                text: term.text,
                weight: term.weight,
                aliases: term.aliases,
                tokenIds: nil,
                ctcTokenIds: tokenIds,
                minSimilarity: term.minSimilarity
            )
        }

        // Preserve vocabulary-level thresholds parsed from the JSON config so
        // a structured file is honored end-to-end (the simple-text path keeps
        // the defaults it always used).
        let tokenizedVocab = CustomVocabularyContext(
            terms: tokenizedTerms,
            alpha: loadedVocab.alpha,
            minCtcScore: loadedVocab.minCtcScore,
            minSimilarity: loadedVocab.minSimilarity,
            minCombinedConfidence: loadedVocab.minCombinedConfidence,
            minTermLength: loadedVocab.minTermLength
        )
        return (tokenizedVocab, ctcModels)
    }

    /// Load a vocabulary file, auto-detecting the structured JSON config vs the
    /// simple one-term-per-line text format. JSON config files begin with `{`
    /// (after optional leading whitespace); anything else is treated as simple
    /// text.
    static func loadVocabularyFile(at url: URL) throws -> CustomVocabularyContext {
        let data = try Data(contentsOf: url)
        let whitespace: Set<UInt8> = [0x20, 0x09, 0x0a, 0x0d]  // space, tab, LF, CR
        let firstMeaningfulByte = data.first { !whitespace.contains($0) }
        if firstMeaningfulByte == UInt8(ascii: "{") {
            return try load(from: url)
        }
        return try loadFromSimpleFormat(from: url)
    }
}
