import XCTest

@testable import FluidAudio

/// Tests for StyleTTS2's high-level long-text auto-chunking (issue #712).
///
/// The full `synthesize(text:)` path needs the CoreML chain + reference
/// audio, so these cover the model-free correctness claim: chunking the
/// phoneme string at `maxPhonemeChunkChars` keeps every chunk within the
/// largest bert/sampler bucket once encoded. The chunk-splitting mechanics
/// themselves live in `PhonemeChunkerTests`.
final class StyleTTS2ChunkingTests: XCTestCase {

    func testMaxPhonemeChunkCharsStaysUnderLargestBucket() {
        let largestBucket = StyleTTS2Constants.bucketTokenSizes.max()!
        // One pad token + one token per kept char ≤ bucket capacity.
        XCTAssertEqual(StyleTTS2Constants.maxPhonemeChunkChars, largestBucket - 1)
    }

    func testEveryChunkEncodesWithinTheLargestBucket() {
        let largestBucket = StyleTTS2Constants.bucketTokenSizes.max()!
        // A long phoneme-like string of encodable chars + word boundaries.
        let longPhonemes = Array(repeating: "həlo wɝld", count: 80).joined(separator: " ")
        XCTAssertGreaterThan(longPhonemes.count, largestBucket)

        let chunks = PhonemeChunker.chunk(
            longPhonemes, maxLength: StyleTTS2Constants.maxPhonemeChunkChars)
        XCTAssertGreaterThan(chunks.count, 1)

        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.count, StyleTTS2Constants.maxPhonemeChunkChars)
            // The encoded sequence (pad + tokens) must fit the bucket so
            // `resolveBucket` never throws `noBucketAvailable`.
            let tokenCount = StyleTTS2TextCleaner.encode(chunk).count
            XCTAssertLessThanOrEqual(tokenCount, largestBucket)
        }
    }

    func testShortTextProducesAtMostOneChunk() {
        let chunks = PhonemeChunker.chunk(
            "həlo wɝld", maxLength: StyleTTS2Constants.maxPhonemeChunkChars)
        XCTAssertEqual(chunks, ["həlo wɝld"])
    }
}
