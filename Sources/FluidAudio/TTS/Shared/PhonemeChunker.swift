import Foundation

/// Splits a phoneme / IPA string into chunks that each fit a model's input
/// cap, preferring to break at whitespace or pause punctuation so words and
/// prosody cues stay intact.
///
/// Frontend-agnostic — both ``KokoroAneManager`` and (in a follow-up) the
/// StyleTTS2 high-level text API can share it so long multi-sentence input
/// is handled by the manager instead of every downstream caller writing its
/// own chunker (issue #712).
///
/// Chunking operates on the already-resolved phoneme string, so it runs
/// *after* text normalization / G2P — decimals, abbreviations, and quote
/// delimiters are never split at the raw-text layer.
enum PhonemeChunker {

    /// Pause / prosody punctuation that marks a natural break point. These
    /// match the tokens TTS vocabularies encode as their own pause cues, so
    /// breaking right after one keeps clause boundaries with the preceding
    /// chunk.
    static let defaultBoundaryPunctuation: Set<Character> = [
        ",", ".", ";", ":", "!", "?", "…", "—",
    ]

    /// Split `phonemes` into chunks of at most `maxLength` characters each.
    ///
    /// Each break is taken at the latest whitespace or pause-punctuation
    /// boundary within the `maxLength` window, so chunks stay as full as
    /// possible without splitting a word and punctuation stays attached to
    /// the preceding chunk. A run longer than `maxLength` with no boundary
    /// is hard-split at the cap (rare for real phoneme strings). Leading and
    /// trailing whitespace is trimmed from every chunk.
    ///
    /// - Returns: `[phonemes]` (trimmed) when the input already fits,
    ///   `[]` for blank input, and otherwise the ordered chunks. Length is
    ///   counted in `Character`s to match the cap TTS vocabularies enforce.
    static func chunk(
        _ phonemes: String,
        maxLength: Int,
        boundaryPunctuation: Set<Character> = defaultBoundaryPunctuation
    ) -> [String] {
        precondition(maxLength > 0, "maxLength must be positive")

        let characters = Array(phonemes)
        let count = characters.count
        if count == 0 { return [] }

        // Fast path: already within the cap (the common case).
        if count <= maxLength {
            let trimmed = phonemes.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? [] : [trimmed]
        }

        func isBoundary(_ character: Character) -> Bool {
            character.isWhitespace || boundaryPunctuation.contains(character)
        }

        var chunks: [String] = []
        var start = 0

        // Skip any leading whitespace so the first window is fully usable.
        while start < count, characters[start].isWhitespace { start += 1 }

        while count - start > maxLength {
            let windowEnd = start + maxLength  // exclusive upper bound

            // Latest boundary in [start, windowEnd): break right after it.
            var breakAt = -1
            var index = windowEnd - 1
            while index > start {
                if isBoundary(characters[index]) {
                    breakAt = index + 1
                    break
                }
                index -= 1
            }
            // No usable boundary in the window → hard-split at the cap.
            if breakAt <= start { breakAt = windowEnd }

            appendChunk(characters[start..<breakAt], to: &chunks)

            start = breakAt
            while start < count, characters[start].isWhitespace { start += 1 }
        }

        if start < count {
            appendChunk(characters[start..<count], to: &chunks)
        }
        return chunks
    }

    private static func appendChunk(_ slice: ArraySlice<Character>, to chunks: inout [String]) {
        let text = String(slice).trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty { chunks.append(text) }
    }
}
