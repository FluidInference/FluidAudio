#if TTS
import Foundation

/// One token from the English phonemize walk, and the phoneme scalars it emitted.
///
/// `surface` counts extended grapheme clusters in the string passed to phonemize,
/// the same unit as a speech span. Letter names, hyphen splits, and a glued
/// possessive clitic stay inside `phonemes`; callers do not split that string
/// on spaces. The separator space `parts.joined(separator: " ")` inserts sits
/// on the previous kept segment's `scalars`.
public struct KokoroAneSpokenSegment: Sendable, Equatable {
    /// Character offsets into the phonemize argument.
    public var surface: Range<Int>
    /// The piece appended for this token, including internal spaces.
    /// Empty when the token was skipped.
    public var phonemes: String
    /// Unicode scalars into the joined phoneme string. `nil` when the token was skipped.
    public var scalars: Range<Int>?

    public init(surface: Range<Int>, phonemes: String, scalars: Range<Int>? = nil) {
        self.surface = surface
        self.phonemes = phonemes
        self.scalars = scalars
    }
}

/// Phonemes from one English phonemize call, plus the token that produced each piece.
public struct KokoroAneEnglishSegmentation: Sendable, Equatable {
    public var phonemes: String
    public var segments: [KokoroAneSpokenSegment]

    public init(phonemes: String, segments: [KokoroAneSpokenSegment]) {
        self.phonemes = phonemes
        self.segments = segments
    }
}
#endif
