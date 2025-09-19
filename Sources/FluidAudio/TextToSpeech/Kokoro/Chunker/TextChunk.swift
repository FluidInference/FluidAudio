import Foundation

/// A chunk of text prepared for synthesis.
/// - words: The original words in this chunk
/// - phonemes: Flat phoneme sequence with single-space separators between words
/// - totalFrames: Reserved for legacy/frame-aware modes (unused here)
/// - pauseAfterMs: Silence to insert after this chunk (punctuation/paragraph driven)
struct TextChunk {
    let words: [String]
    let phonemes: [String]
    let totalFrames: Float
    let pauseAfterMs: Int
    let isForcedSplit: Bool
    let trailingPunctuation: String?
    let isSentenceBoundary: Bool
}
