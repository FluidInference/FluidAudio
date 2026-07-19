import Foundation

extension VocabularyRescorer {

    // MARK: - String Similarity

    /// Compute string similarity using Levenshtein distance
    static func stringSimilarity(_ a: String, _ b: String) -> Float {
        let aLower = a.lowercased()
        let bLower = b.lowercased()

        let distance = StringUtils.levenshteinDistance(aLower, bLower)
        let maxLen = max(aLower.count, bLower.count)

        guard maxLen > 0 else { return 1.0 }
        return 1.0 - Float(distance) / Float(maxLen)
    }

    /// Compute string similarity with length penalty for compound matches.
    /// Penalizes when compound length differs significantly from vocab term length.
    static func lengthPenalizedSimilarity(_ compound: String, _ vocabTerm: String) -> Float {
        let baseSimilarity = stringSimilarity(compound, vocabTerm)

        // Length ratio: how well do the lengths match?
        let compoundLen = Float(compound.count)
        let vocabLen = Float(vocabTerm.count)
        let lengthRatio = min(compoundLen, vocabLen) / max(compoundLen, vocabLen)

        // Apply square root to soften the penalty
        return baseSimilarity * sqrt(lengthRatio)
    }

    // MARK: - Normalized Forms

    /// Represents a normalized form of a vocabulary term (canonical or alias)
    struct NormalizedForm: Hashable {
        let normalized: String
        let wordCount: Int
        let matchedAlias: String?
    }

    /// Build normalized canonical and alias forms while retaining the exact alias source.
    static func normalizedForms(canonicalTerm: String, aliases: [String]) -> [NormalizedForm] {
        let rawForms: [(text: String, matchedAlias: String?)] =
            [(canonicalTerm, nil)] + aliases.map { ($0, $0) }
        var seen = Set<String>()
        var forms: [NormalizedForm] = []

        for rawForm in rawForms {
            let normalized = normalizeForSimilarity(rawForm.text)
            guard !normalized.isEmpty, seen.insert(normalized).inserted else { continue }

            forms.append(
                NormalizedForm(
                    normalized: normalized,
                    wordCount: normalized.split(separator: " ").count,
                    matchedAlias: rawForm.matchedAlias
                ))
        }

        return forms
    }

    /// Build all normalized forms (canonical + aliases) for a vocabulary term
    func buildNormalizedForms(for term: CustomVocabularyTerm) -> [NormalizedForm] {
        var aliases: [String] = []
        let termLower = term.textLowercased

        // Look up canonical term in vocabulary to get ALL aliases
        for vocabTerm in vocabulary.terms where vocabTerm.textLowercased == termLower {
            if let vocabularyAliases = vocabTerm.aliases {
                aliases.append(contentsOf: vocabularyAliases)
            }
        }
        // Also add aliases from the term itself
        if let termAliases = term.aliases {
            aliases.append(contentsOf: termAliases)
        }

        return Self.normalizedForms(canonicalTerm: term.text, aliases: aliases)
    }

    // MARK: - Similarity Thresholds

    /// Determine required similarity threshold based on span length and word length
    /// Note: Using permissive thresholds to avoid rejecting valid matches
    func requiredSimilarity(minSimilarity: Float, spanLength: Int) -> Float {
        // Multi-word spans: slightly higher threshold to avoid false positives
        if spanLength >= 2 {
            return max(minSimilarity, 0.55)
        }

        // Single words: use the configured minimum similarity
        // Note: The 0.85 threshold for short words was too aggressive (caused regression)
        return minSimilarity
    }

    // MARK: - Text Utilities

    /// Preserve capitalization from original word in replacement
    func preserveCapitalization(original: String, replacement: String) -> String {
        guard let firstChar = original.first else { return replacement }

        if firstChar.isUppercase && replacement.first?.isLowercase == true {
            return replacement.prefix(1).uppercased() + replacement.dropFirst()
        }
        return replacement
    }

    /// Normalize text for similarity checks: lowercase, collapse whitespace,
    /// and strip punctuation while preserving letters, numbers, apostrophes, and hyphens.
    static func normalizeForSimilarity(_ text: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "'-"))
        var result = ""
        var lastWasSpace = true

        for scalar in text.lowercased().unicodeScalars {
            if allowed.contains(scalar) {
                result.append(Character(scalar))
                lastWasSpace = false
            } else if scalar == " " || scalar == "\t" || scalar == "\n" {
                if !lastWasSpace && !result.isEmpty {
                    result.append(" ")
                    lastWasSpace = true
                }
            }
            // Skip other characters (punctuation)
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Align the complete internal word sequence to one byte-exact occurrence in the untouched transcript.
    ///
    /// Each returned range is half-open in UTF-8 bytes. It excludes recognized quotation wrappers
    /// and terminal punctuation while retaining technical boundary symbols such as `+`, `#`, `@`,
    /// `$`, and a leading `.`. When a lexical boundary is ambiguous, the complete sequence is
    /// absent, or more than one valid alignment exists, the affected range is `nil` rather than a
    /// guess.
    static func alignBaseWordsToUTF8Ranges(baseText: String, baseWords: [String]) -> [Range<Int>?] {
        guard !baseWords.isEmpty else { return [] }
        guard baseWords.allSatisfy({ !$0.isEmpty }) else {
            return Array(repeating: nil, count: baseWords.count)
        }

        var alignments: [[Range<String.Index>]] = []

        func search(
            wordIndex: Int,
            cursor: String.Index,
            ranges: [Range<String.Index>]
        ) {
            guard alignments.count < 2 else { return }
            guard wordIndex < baseWords.count else {
                guard isDelimiterOnly(baseText[cursor..<baseText.endIndex]) else { return }
                alignments.append(ranges)
                return
            }

            let word = baseWords[wordIndex]
            var searchStart = cursor
            while searchStart < baseText.endIndex,
                let match = baseText.range(
                    of: word,
                    options: .literal,
                    range: searchStart..<baseText.endIndex
                )
            {
                if isDelimiterOnly(baseText[cursor..<match.lowerBound]) {
                    search(
                        wordIndex: wordIndex + 1,
                        cursor: match.upperBound,
                        ranges: ranges + [match]
                    )
                }
                guard alignments.count < 2, match.lowerBound < baseText.endIndex else { return }
                searchStart = baseText.index(after: match.lowerBound)
            }
        }

        search(wordIndex: 0, cursor: baseText.startIndex, ranges: [])
        guard alignments.count == 1, let alignment = alignments.first else {
            return Array(repeating: nil, count: baseWords.count)
        }

        return alignment.map { lexicalUTF8Range(for: $0, in: baseText) }
    }

    /// Build one candidate span from aligned source words and validate it against the evaluated phrase.
    static func candidateBaseTextUTF8Range(
        wordRange: Range<Int>,
        alignedWordRanges: [Range<Int>?],
        baseText: String,
        basePhrase: String
    ) -> Range<Int>? {
        guard !wordRange.isEmpty,
            wordRange.lowerBound >= 0,
            wordRange.upperBound <= alignedWordRanges.count,
            let first = alignedWordRanges[wordRange.lowerBound],
            let last = alignedWordRanges[wordRange.upperBound - 1]
        else {
            return nil
        }

        for index in wordRange where alignedWordRanges[index] == nil {
            return nil
        }

        let candidateRange = first.lowerBound..<last.upperBound
        guard let candidateText = substring(in: baseText, utf8Range: candidateRange),
            normalizeForSimilarity(String(candidateText)) == normalizeForSimilarity(basePhrase)
        else {
            return nil
        }
        return candidateRange
    }

    /// Return a substring only when both UTF-8 offsets are valid `String` boundaries.
    static func substring(in text: String, utf8Range: Range<Int>) -> Substring? {
        guard utf8Range.lowerBound >= 0,
            utf8Range.upperBound >= utf8Range.lowerBound,
            utf8Range.upperBound <= text.utf8.count,
            let lowerUTF8 = text.utf8.index(
                text.utf8.startIndex,
                offsetBy: utf8Range.lowerBound,
                limitedBy: text.utf8.endIndex
            ),
            let upperUTF8 = text.utf8.index(
                text.utf8.startIndex,
                offsetBy: utf8Range.upperBound,
                limitedBy: text.utf8.endIndex
            ),
            let lower = String.Index(lowerUTF8, within: text),
            let upper = String.Index(upperUTF8, within: text)
        else {
            return nil
        }
        return text[lower..<upper]
    }

    private static func lexicalUTF8Range(
        for exactRange: Range<String.Index>,
        in text: String
    ) -> Range<Int>? {
        var lower = exactRange.lowerBound
        var upper = exactRange.upperBound

        while lower < upper, leadingQuotationWrappers.contains(text[lower]) {
            lower = text.index(after: lower)
        }
        while lower < upper {
            let previous = text.index(before: upper)
            let character = text[previous]
            guard
                trailingQuotationWrappers.contains(character)
                    || trailingSentencePunctuation.contains(character)
            else { break }
            upper = previous
        }
        guard lower < upper,
            isSafeLexicalBoundary(text[lower]),
            isSafeLexicalBoundary(text[text.index(before: upper)]),
            let lowerUTF8 = lower.samePosition(in: text.utf8),
            let upperUTF8 = upper.samePosition(in: text.utf8)
        else {
            return nil
        }

        let lowerOffset = text.utf8.distance(from: text.utf8.startIndex, to: lowerUTF8)
        let upperOffset = text.utf8.distance(from: text.utf8.startIndex, to: upperUTF8)
        return lowerOffset..<upperOffset
    }

    private static func isDelimiterOnly(_ text: Substring) -> Bool {
        text.allSatisfy(isInterWordDelimiter)
    }

    private static func isInterWordDelimiter(_ character: Character) -> Bool {
        !character.isLetter && !character.isNumber
    }

    private static func isSafeLexicalBoundary(_ character: Character) -> Bool {
        if character.isLetter || character.isNumber || technicalPunctuation.contains(character) {
            return true
        }

        var containsTechnicalSymbol = false
        for scalar in character.unicodeScalars {
            switch scalar.properties.generalCategory {
            case .mathSymbol, .currencySymbol:
                containsTechnicalSymbol = true
            case .nonspacingMark, .spacingMark, .enclosingMark:
                continue
            default:
                return false
            }
        }
        return containsTechnicalSymbol
    }

    private static let leadingQuotationWrappers: Set<Character> = [
        "\"", "“", "«", "‹", "「", "『",
    ]

    private static let trailingQuotationWrappers: Set<Character> = [
        "\"", "”", "»", "›", "」", "』",
    ]

    private static let trailingSentencePunctuation: Set<Character> = [
        ".", ",", ";", "…", "。", "、", "；",
    ]

    /// Punctuation that Unicode does not classify as a symbol but that commonly belongs to a
    /// technical token. Unicode math and currency categories cover operators and currency signs
    /// such as `+`, `=`, and `$` without admitting unrelated symbols such as emoji or trademarks.
    private static let technicalPunctuation: Set<Character> = [
        "#", ".", "@", "%", "&", "*", "/", "\\", "_", "-", "`", "'", "’", "^",
    ]

    /// Build set of normalized vocabulary terms for guard checks
    func buildVocabularyNormalizedSet() -> Set<String> {
        var normalizedSet = Set<String>()
        for term in vocabulary.terms {
            let normalized = Self.normalizeForSimilarity(term.text)
            if !normalized.isEmpty {
                normalizedSet.insert(normalized)
            }
            // Also add aliases if present
            if let aliases = term.aliases {
                for alias in aliases {
                    let normalizedAlias = Self.normalizeForSimilarity(alias)
                    if !normalizedAlias.isEmpty {
                        normalizedSet.insert(normalizedAlias)
                    }
                }
            }
        }
        return normalizedSet
    }
}

// MARK: - Token Word Boundary Utilities

/// Check if a token string indicates a word boundary.
///
/// SentencePiece and TDT tokenizers use prefixes to indicate word starts:
/// - `▁` (U+2581 LOWER ONE EIGHTH BLOCK) - SentencePiece convention
/// - ` ` (space) - TDT/some tokenizer formats
///
/// - Parameter token: The token string to check
/// - Returns: True if the token starts a new word
public func isWordBoundary(_ token: String) -> Bool {
    token.hasPrefix(ASRConstants.sentencePieceWordBoundary) || token.hasPrefix(" ")
}

/// Strip word boundary prefix from a token.
///
/// Removes the leading `▁` or space character if present.
///
/// - Parameter token: The token string to process
/// - Returns: Token with word boundary prefix removed
public func stripWordBoundaryPrefix(_ token: String) -> String {
    if token.hasPrefix(ASRConstants.sentencePieceWordBoundary) || token.hasPrefix(" ") {
        return String(token.dropFirst())
    }
    return token
}
