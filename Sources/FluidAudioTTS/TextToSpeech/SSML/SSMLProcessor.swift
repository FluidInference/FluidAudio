import Foundation

/// Main SSML processing orchestrator
/// Parses and processes SSML tags, returning cleaned text with phonetic overrides
public enum SSMLProcessor {

    /// Process SSML tags in text
    /// - Parameter text: Input text potentially containing SSML tags
    /// - Returns: Processed result with cleaned text and phonetic overrides
    public static func process(_ text: String) -> SSMLProcessingResult {
        // Quick check: if no angle brackets, skip SSML processing entirely
        guard text.contains("<") else {
            return SSMLProcessingResult(text: text, phoneticOverrides: [])
        }

        var workingText = text
        var phoneticOverrides: [TtsPhoneticOverride] = []

        // Parse all SSML tags (returned in reverse document order for safe replacement)
        let tags = SSMLTagParser.parse(text)

        // Process tags in reverse order (from end to start) to preserve indices
        for tag in tags {
            switch tag.type {
            case .phoneme(_, let ph, let content):
                // Calculate word index BEFORE replacement
                let wordIndex = countWordsBeforeIndex(in: workingText, index: tag.range.lowerBound)

                // Replace tag with content
                workingText.replaceSubrange(tag.range, with: content)

                // Create phonetic override
                let tokens = tokenizePhonemes(ph)
                let scalarTokens = ph.unicodeScalars.map { String($0) }

                phoneticOverrides.append(
                    TtsPhoneticOverride(
                        wordIndex: wordIndex,
                        tokens: tokens,
                        scalarTokens: scalarTokens,
                        raw: ph,
                        word: content.trimmingCharacters(in: .whitespaces)
                    ))

            case .sub(let alias, _):
                // Replace tag with alias text
                workingText.replaceSubrange(tag.range, with: alias)

            case .sayAs(let interpretAs, let format, let content):
                // Interpret content and replace
                let interpreted = SayAsInterpreter.interpret(
                    content: content,
                    interpretAs: interpretAs,
                    format: format
                )
                workingText.replaceSubrange(tag.range, with: interpreted)
            }
        }

        // Sort overrides by word index (since we processed in reverse)
        let sortedOverrides = phoneticOverrides.sorted { $0.wordIndex < $1.wordIndex }

        return SSMLProcessingResult(text: workingText, phoneticOverrides: sortedOverrides)
    }

    // MARK: - Private Helpers

    /// Count completed words before a given index in the text
    /// This matches the word counting logic in TtsTextPreprocessor
    private static func countWordsBeforeIndex(in text: String, index: String.Index) -> Int {
        let prefix = String(text[..<index])
        var wordCount = 0
        var inWord = false

        for char in prefix {
            if FluidAudioTTS.isWordCharacter(char) {
                if !inWord {
                    inWord = true
                }
            } else if inWord {
                wordCount += 1
                inWord = false
            }
        }

        // If we ended in a word, it's a partial word - don't count it
        // The word index points to the word that will contain our phoneme
        return wordCount
    }

    /// Tokenize phoneme string into tokens
    /// Handles both space-separated and continuous IPA strings
    private static func tokenizePhonemes(_ ph: String) -> [String] {
        let trimmed = ph.trimmingCharacters(in: .whitespaces)

        // If contains spaces, split on spaces
        if trimmed.contains(" ") {
            return trimmed.split(separator: " ").map { String($0) }
        }

        // Otherwise, return as single-element array
        // The downstream PhonemeMapper will handle individual character mapping
        return [trimmed]
    }
}
