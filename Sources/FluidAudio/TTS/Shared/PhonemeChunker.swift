#if TTS
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
        "/",  // Kokoro Mandarin word separator
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
    /// - Parameter countsUnicodeScalars: measure length in Unicode scalars
    ///   instead of `Character`s, for vocabs that encode combining marks as
    ///   their own token (KokoroAne: `ɑ̃` is two symbols).
    /// - Returns: `[phonemes]` (trimmed) when the input already fits,
    ///   `[]` for blank input, and otherwise the ordered chunks. Length is
    ///   counted in `Character`s by default to match the cap TTS vocabularies
    ///   enforce.
    static func chunk(
        _ phonemes: String,
        maxLength: Int,
        boundaryPunctuation: Set<Character> = defaultBoundaryPunctuation,
        countsUnicodeScalars: Bool = false
    ) -> [String] {
        let characters: [Character] =
            countsUnicodeScalars ? phonemes.unicodeScalars.map(Character.init) : Array(phonemes)
        return chunkRanges(characters, maxLength: maxLength, boundaryPunctuation: boundaryPunctuation)
            .map { String(characters[$0]) }
    }

    /// ``chunk(_:maxLength:boundaryPunctuation:countsUnicodeScalars:)`` as
    /// ranges into `characters`, so callers can map positions from the full
    /// string into each chunk.
    static func chunkRanges(
        _ characters: [Character],
        maxLength: Int,
        boundaryPunctuation: Set<Character> = defaultBoundaryPunctuation
    ) -> [Range<Int>] {
        precondition(maxLength > 0, "maxLength must be positive")

        let count = characters.count
        var ranges: [Range<Int>] = []

        // Chunks are trimmed like `trimmingCharacters(in: .whitespacesAndNewlines)`,
        // which also strips U+200B. A closure, not the method reference
        // `allSatisfy(CharacterSet.whitespacesAndNewlines.contains)`: Swift 6.4 `-O`
        // inverts that reference once the module holds a second one, and release
        // builds then trimmed the first and last word of every chunk.
        let trimmedScalars = CharacterSet.whitespacesAndNewlines
        func isTrimmed(_ character: Character) -> Bool {
            character.unicodeScalars.allSatisfy { trimmedScalars.contains($0) }
        }
        func appendChunk(_ range: Range<Int>) {
            var lower = range.lowerBound
            var upper = range.upperBound
            while lower < upper, isTrimmed(characters[lower]) { lower += 1 }
            while upper > lower, isTrimmed(characters[upper - 1]) { upper -= 1 }
            if lower < upper { ranges.append(lower..<upper) }
        }

        // Fast path: already within the cap (the common case).
        if count <= maxLength {
            appendChunk(0..<count)
            return ranges
        }

        func isBoundary(_ character: Character) -> Bool {
            character.isWhitespace || boundaryPunctuation.contains(character)
        }

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

            appendChunk(start..<breakAt)

            start = breakAt
            while start < count, characters[start].isWhitespace { start += 1 }
        }

        if start < count {
            appendChunk(start..<count)
        }
        return ranges
    }
}
#endif
