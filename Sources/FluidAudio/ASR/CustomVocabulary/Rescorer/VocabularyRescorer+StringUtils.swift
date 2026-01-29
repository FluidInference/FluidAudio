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
        let raw: String
        let normalized: String
        let wordCount: Int
    }

    /// Build all normalized forms (canonical + aliases) for a vocabulary term
    func buildNormalizedForms(for term: CustomVocabularyTerm) -> [NormalizedForm] {
        var rawForms: [String] = [term.text]
        let termLower = term.textLowercased

        // Look up canonical term in vocabulary to get ALL aliases
        for vocabTerm in vocabulary.terms where vocabTerm.textLowercased == termLower {
            if let aliases = vocabTerm.aliases {
                rawForms.append(contentsOf: aliases)
            }
        }
        // Also add aliases from the term itself
        if let aliases = term.aliases {
            rawForms.append(contentsOf: aliases)
        }

        var seen = Set<String>()
        var forms: [NormalizedForm] = []

        for raw in rawForms {
            let normalized = Self.normalizeForSimilarity(raw)
            guard !normalized.isEmpty else { continue }
            guard !seen.contains(normalized) else { continue }
            seen.insert(normalized)

            let wordCount = normalized.split(separator: " ").count
            forms.append(NormalizedForm(raw: raw, normalized: normalized, wordCount: wordCount))
        }

        return forms
    }

    // MARK: - Similarity Thresholds

    /// Determine required similarity threshold based on span length and word length
    /// Note: Using permissive thresholds to avoid rejecting valid matches
    func requiredSimilarity(minSimilarity: Float, spanLength: Int, normalizedText: String) -> Float {
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
