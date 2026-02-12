import Foundation

/// A custom pronunciation dictionary for TTS synthesis.
///
/// Custom lexicons allow domain-specific pronunciation overrides that take precedence
/// over all built-in dictionaries and grapheme-to-phoneme conversion.
///
public struct TtsCustomLexicon: Sendable {
    /// Word-to-phoneme mappings. Keys are matched case-sensitively.
    public let entries: [String: [String]]

    /// Case-insensitive entries for fallback matching.
    public let lowercaseEntries: [String: [String]]

    /// Normalized entries for robust fallback matching (e.g., collapsing apostrophe variants).
    private let normalizedEntries: [String: [String]]

    /// Creates a custom lexicon from a dictionary of word-to-phoneme mappings.
    ///
    /// - Parameter entries: Dictionary where keys are words and values are arrays of phoneme tokens.
    public init(entries: [String: [String]]) {
        self.entries = entries

        struct Candidate {
            let key: String
            let phonemes: [String]
            let isCanonical: Bool
        }

        func selectPreferredPhonemes(from candidates: [Candidate]) -> [String] {
            guard !candidates.isEmpty else { return [] }

            let canonical = candidates.filter { $0.isCanonical }
            let pool = canonical.isEmpty ? candidates : canonical

            guard let chosen = pool.min(by: { $0.key < $1.key }) else { return [] }
            return chosen.phonemes
        }

        var lowercaseBuckets: [String: [Candidate]] = [:]
        lowercaseBuckets.reserveCapacity(entries.count)

        var normalizedBuckets: [String: [Candidate]] = [:]
        normalizedBuckets.reserveCapacity(entries.count)

        for (word, phonemes) in entries {
            let lower = word.lowercased()
            lowercaseBuckets[lower, default: []].append(
                Candidate(key: word, phonemes: phonemes, isCanonical: word == lower)
            )

            let normalized = Self.normalizeForLookup(word)
            guard !normalized.isEmpty else { continue }
            normalizedBuckets[normalized, default: []].append(
                Candidate(key: word, phonemes: phonemes, isCanonical: word == normalized)
            )
        }

        var lowercase: [String: [String]] = [:]
        lowercase.reserveCapacity(lowercaseBuckets.count)
        for (key, candidates) in lowercaseBuckets {
            lowercase[key] = selectPreferredPhonemes(from: candidates)
        }
        self.lowercaseEntries = lowercase

        var normalized: [String: [String]] = [:]
        normalized.reserveCapacity(normalizedBuckets.count)
        for (key, candidates) in normalizedBuckets {
            normalized[key] = selectPreferredPhonemes(from: candidates)
        }
        self.normalizedEntries = normalized
    }

    /// Loads a custom lexicon from a file URL.
    ///
    /// - Parameter url: File URL containing the lexicon in line-based format.
    /// - Returns: Parsed custom lexicon.
    /// - Throws: If the file cannot be read or parsed.
    public static func load(from url: URL) throws -> TtsCustomLexicon {
        let content = try String(contentsOf: url, encoding: .utf8)
        return try parse(content)
    }

    /// Parses a custom lexicon from a string.
    ///
    /// Format: One entry per line as `word=phonemes`.
    /// - The phoneme string is interpreted as compact IPA and split by Swift grapheme cluster (`Character`).
    /// - Any whitespace becomes a `" "` token (word separator), allowing multi-word expansions like
    ///   `UN=junˈITᵻd nˈAʃənz`.
    /// - Whitespace is reserved for word separation; do not space-separate individual phoneme tokens.
    /// Lines starting with `#` are comments. Empty lines are ignored.
    ///
    /// - Parameter content: String containing the lexicon entries.
    /// - Returns: Parsed custom lexicon.
    /// - Throws: `TTSError.processingFailed` if the format is invalid.
    public static func parse(_ content: String) throws -> TtsCustomLexicon {
        var entries: [String: [String]] = [:]

        let lines = content.components(separatedBy: .newlines)

        for (lineNumber, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }

            guard let separatorIndex = trimmed.firstIndex(of: "=") else {
                throw TTSError.processingFailed(
                    "Invalid lexicon format at line \(lineNumber + 1): missing '=' separator"
                )
            }

            let word = String(trimmed[..<separatorIndex])
                .trimmingCharacters(in: .whitespaces)
            let phonemeString = String(trimmed[trimmed.index(after: separatorIndex)...])
                .trimmingCharacters(in: .whitespaces)

            guard !word.isEmpty else {
                throw TTSError.processingFailed(
                    "Invalid lexicon format at line \(lineNumber + 1): empty word"
                )
            }

            guard !phonemeString.isEmpty else {
                throw TTSError.processingFailed(
                    "Invalid lexicon format at line \(lineNumber + 1): empty phonemes for '\(word)'"
                )
            }

            let phonemes = parsePhonemes(phonemeString)
            guard !phonemes.isEmpty else {
                throw TTSError.processingFailed(
                    "Invalid lexicon format at line \(lineNumber + 1): no valid phonemes for '\(word)'"
                )
            }

            entries[word] = phonemes
        }

        return TtsCustomLexicon(entries: entries)
    }

    /// Looks up phonemes for a word, trying case-sensitive match first, then case-insensitive.
    /// If still not found, falls back to normalized lookup
    /// (lowercased and stripped to letters/digits/apostrophes).
    ///
    /// - Parameter word: The word to look up.
    /// - Returns: Array of phoneme tokens, or nil if not found.
    public func phonemes(for word: String) -> [String]? {
        if let exact = entries[word] {
            return exact
        }
        if let folded = lowercaseEntries[word.lowercased()] {
            return folded
        }

        let normalized = Self.normalizeForLookup(word)
        guard !normalized.isEmpty else { return nil }
        return normalizedEntries[normalized]
    }

    /// Whether this lexicon contains any entries.
    public var isEmpty: Bool {
        entries.isEmpty
    }

    /// Number of entries in the lexicon.
    public var count: Int {
        entries.count
    }

    // MARK: - Private

    /// Parses a phoneme string into individual phoneme tokens.
    ///
    /// - **Compact IPA**: `kəkˈɔɹO` → ["k", "ə", "k", "ˈ", "ɔ", "ɹ", "O"]
    /// - **With word separators**: `junˈITᵻd nˈAʃənz` → ["j", "u", "n", "ˈ", "I", "T", "ᵻ", "d", " ", …]
    ///
    /// Each Swift Character (grapheme cluster) becomes one token, correctly handling combining diacritics.
    /// Runs of whitespace are collapsed into a single `" "` separator token.
    private static func parsePhonemes(_ value: String) -> [String] {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return [] }

        var tokens: [String] = []
        tokens.reserveCapacity(trimmed.count)

        var pendingWordSeparator = false
        for character in trimmed {
            if character.isWhitespace {
                pendingWordSeparator = true
                continue
            }

            if pendingWordSeparator, !tokens.isEmpty {
                tokens.append(" ")
                pendingWordSeparator = false
            }

            tokens.append(String(character))
        }

        return tokens
    }

    private static let apostropheCharacters: Set<Character> = ["'", "’", "ʼ", "‛", "‵", "′"]
    private static let canonicalApostrophe: Character = "'"

    private static func normalizeForLookup(_ word: String) -> String {
        let lowered = word.lowercased()
        let allowedSet = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))

        var normalized = ""
        normalized.reserveCapacity(lowered.count)

        for character in lowered {
            if apostropheCharacters.contains(character) {
                normalized.append(canonicalApostrophe)
                continue
            }

            for scalar in character.unicodeScalars where allowedSet.contains(scalar) {
                normalized.unicodeScalars.append(scalar)
            }
        }

        return normalized
    }
}

// MARK: - Convenience Extensions

extension TtsCustomLexicon {
    /// Creates an empty custom lexicon.
    public static var empty: TtsCustomLexicon {
        TtsCustomLexicon(entries: [:])
    }

    /// Merges this lexicon with another, with the other taking precedence for conflicts.
    public func merged(with other: TtsCustomLexicon) -> TtsCustomLexicon {
        var combined = entries
        for (word, phonemes) in other.entries {
            combined[word] = phonemes
        }
        return TtsCustomLexicon(entries: combined)
    }
}
