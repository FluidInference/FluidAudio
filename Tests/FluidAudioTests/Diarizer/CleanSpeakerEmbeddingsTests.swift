import XCTest

@testable import FluidAudio

/// Tests for `SpeakerUtilities.cleanEmbeddingsBySpeaker`
final class CleanSpeakerEmbeddingsTests: XCTestCase {

    private func embedding(pattern: Int) -> [Float] {
        (0..<256).map { sin(Float($0 + pattern * 100) * 0.1) }
    }

    /// `base` with a little of `other` mixed in: the same voice in a different segment.
    private func variant(of base: Int, toward other: Int, by amount: Float) -> [Float] {
        let a = VDSPOperations.l2Normalize(embedding(pattern: base))
        let b = VDSPOperations.l2Normalize(embedding(pattern: other))
        return zip(a, b).map { $0 * (1 - amount) + $1 * amount }
    }

    private func segment(_ speaker: String, _ start: Float, _ end: Float, _ embedding: [Float]) -> TimedSpeakerSegment {
        TimedSpeakerSegment(
            speakerId: speaker, embedding: embedding, startTimeSeconds: start, endTimeSeconds: end, qualityScore: 1)
    }

    private func distance(_ result: [String: [Float]], _ speaker: String, _ pattern: Int) -> Float {
        SpeakerUtilities.cosineDistance(result[speaker] ?? [], embedding(pattern: pattern))
    }

    func testSkipsSegmentsThatOverlapAnotherSpeaker() {
        let blended = zip(embedding(pattern: 1), embedding(pattern: 2)).map { $0 + $1 }
        let segments = [
            segment("A", 0, 5, embedding(pattern: 1)),
            segment("A", 10, 15, blended),
            segment("B", 12, 16, embedding(pattern: 2)),
        ]
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        XCTAssertEqual(distance(result, "A", 1), 0, accuracy: 1e-4)
        XCTAssertNil(result["B"], "B's only segment overlaps A")
    }

    func testToleratesSmallOverlapAtTurnBoundaries() {
        let segments = [
            segment("A", 0, 5, embedding(pattern: 1)),
            segment("B", 4.9, 10, embedding(pattern: 2)),
        ]
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        XCTAssertNotNil(result["A"])
        XCTAssertNotNil(result["B"])
    }

    func testSkipsShortSegments() {
        let segments = [
            segment("A", 0, 5, embedding(pattern: 1)),
            segment("A", 6, 7, embedding(pattern: 3)),
        ]
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        XCTAssertEqual(distance(result, "A", 1), 0, accuracy: 1e-4)
    }

    func testDropsOddSegmentOut() {
        let segments = [
            segment("A", 0, 3, variant(of: 1, toward: 4, by: 0.1)),
            segment("A", 4, 7, variant(of: 1, toward: 5, by: 0.1)),
            segment("A", 8, 11, embedding(pattern: 1)),
            segment("A", 12, 15, embedding(pattern: 2)),
        ]
        XCTAssertGreaterThan(
            SpeakerUtilities.cosineDistance(embedding(pattern: 1), embedding(pattern: 2)), 0.7,
            "fixture: pattern 2 must read as another voice")
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        XCTAssertLessThan(distance(result, "A", 1), 0.05)
        XCTAssertGreaterThan(distance(result, "A", 2), 0.7)
    }

    func testKeepsAllSegmentsWhenNoneStandsOut() {
        let segments = [
            segment("A", 0, 3, variant(of: 1, toward: 4, by: 0.1)),
            segment("A", 4, 7, variant(of: 1, toward: 5, by: 0.1)),
            segment("A", 8, 11, embedding(pattern: 1)),
        ]
        let expected = VDSPOperations.l2Normalize(
            segments.map { VDSPOperations.l2Normalize($0.embedding) }.reduce([Float](repeating: 0, count: 256)) {
                zip($0, $1).map { $0 + $1 }
            })
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        XCTAssertEqual(SpeakerUtilities.cosineDistance(result["A"] ?? [], expected), 0, accuracy: 1e-4)
    }

    func testOmitsSpeakersWithTooLittleCleanSpeech() {
        let segments = [
            segment("A", 0, 2, embedding(pattern: 1)),
            segment("A", 3, 5, embedding(pattern: 1)),
            segment("B", 6, 12, embedding(pattern: 2)),
        ]
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments, minTotalDuration: 5)
        XCTAssertNil(result["A"])
        XCTAssertNotNil(result["B"])
    }

    func testWeightsByDuration() {
        let segments = [
            segment("A", 0, 8, embedding(pattern: 1)),
            segment("A", 10, 12, embedding(pattern: 2)),
        ]
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        XCTAssertLessThan(distance(result, "A", 1), distance(result, "A", 2))
    }

    func testResultIsNormalized() {
        let segments = [segment("A", 0, 6, embedding(pattern: 1).map { $0 * 5 })]
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        let norm = (result["A"] ?? []).reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1, accuracy: 1e-4)
    }

    func testIgnoresSegmentsWithoutEmbeddings() {
        let segments = [
            segment("A", 0, 6, []),
            segment("A", 7, 13, embedding(pattern: 1)),
        ]
        let result = SpeakerUtilities.cleanEmbeddingsBySpeaker(segments)
        XCTAssertEqual(distance(result, "A", 1), 0, accuracy: 1e-4)
        XCTAssertTrue(SpeakerUtilities.cleanEmbeddingsBySpeaker([]).isEmpty)
    }
}
