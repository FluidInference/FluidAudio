import XCTest

@testable import FluidAudio

/// Tests for `SpeakerUtilities.cleanEmbeddingsBySpeaker`
final class CleanSpeakerEmbeddingsTests: XCTestCase {

    private func embedding(pattern: Int) -> [Float] {
        (0..<256).map { sin(Float($0 + pattern * 100) * 0.1) }
    }

    /// `base` with a little of `other` mixed in: the same voice in a different chunk.
    private func variant(of base: Int, toward other: Int, by amount: Float) -> [Float] {
        let a = VDSPOperations.l2Normalize(embedding(pattern: base))
        let b = VDSPOperations.l2Normalize(embedding(pattern: other))
        return zip(a, b).map { $0 * (1 - amount) + $1 * amount }
    }

    private func segment(_ speaker: String, _ start: Float, _ end: Float) -> TimedSpeakerSegment {
        TimedSpeakerSegment(
            speakerId: speaker, embedding: [], startTimeSeconds: start, endTimeSeconds: end, qualityScore: 1)
    }

    private func chunk(
        _ speaker: String, _ index: Int, _ start: Double, _ end: Double, _ embedding: [Float]
    )
        -> ChunkEmbedding
    {
        ChunkEmbedding(
            speakerId: speaker, chunkIndex: index, speakerIndex: 0, startTimeSeconds: start, endTimeSeconds: end,
            embedding256: embedding)
    }

    private func distance(_ result: [String: [Float]], _ speaker: String, _ pattern: Int) -> Float {
        SpeakerUtilities.cosineDistance(result[speaker] ?? [], embedding(pattern: pattern))
    }

    func testNeedsChunkEmbeddings() {
        let result = DiarizationResult(segments: [segment("A", 0, 10)])
        XCTAssertTrue(SpeakerUtilities.cleanEmbeddingsBySpeaker(result).isEmpty)
    }

    func testUsesChunkEmbeddingsNotSegmentEmbeddings() {
        let segments = [
            TimedSpeakerSegment(
                speakerId: "A", embedding: embedding(pattern: 3), startTimeSeconds: 0, endTimeSeconds: 10,
                qualityScore: 1)
        ]
        let result = DiarizationResult(
            segments: segments, chunkEmbeddings: [chunk("A", 0, 0, 10, embedding(pattern: 1))])
        XCTAssertEqual(distance(SpeakerUtilities.cleanEmbeddingsBySpeaker(result), "A", 1), 0, accuracy: 1e-4)
    }

    func testSkipsChunksWithoutEnoughCleanSpeech() {
        // Chunk 1: A speaks 12-18 but B talks over 13-18, leaving 1 s of clean speech for A.
        let blended = zip(embedding(pattern: 1), embedding(pattern: 2)).map { $0 + $1 }
        let result = DiarizationResult(
            segments: [segment("A", 0, 6), segment("A", 12, 18), segment("B", 13, 18)],
            chunkEmbeddings: [
                chunk("A", 0, 0, 6, embedding(pattern: 1)),
                chunk("A", 1, 12, 18, blended),
            ])
        let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(result)
        XCTAssertEqual(distance(embeddings, "A", 1), 0, accuracy: 1e-4)
    }

    func testCleanSpeechCountsOnlyTheSpeakersOwnTurnsInsideTheSpan() {
        // The span covers 0-10, but A only speaks 0-3 in it: 3 s, under the 4 s minimum.
        let result = DiarizationResult(
            segments: [segment("A", 0, 3), segment("B", 3, 10)],
            chunkEmbeddings: [chunk("A", 0, 0, 10, embedding(pattern: 1))])
        XCTAssertNil(SpeakerUtilities.cleanEmbeddingsBySpeaker(result)["A"])
    }

    func testDropsOddChunkOut() {
        XCTAssertGreaterThan(
            SpeakerUtilities.cosineDistance(embedding(pattern: 1), embedding(pattern: 2)), 0.7,
            "fixture: pattern 2 must read as another voice")
        let result = DiarizationResult(
            segments: [segment("A", 0, 3), segment("A", 10, 13), segment("A", 20, 23), segment("A", 30, 33)],
            chunkEmbeddings: [
                chunk("A", 0, 0, 3, variant(of: 1, toward: 4, by: 0.1)),
                chunk("A", 1, 10, 13, variant(of: 1, toward: 5, by: 0.1)),
                chunk("A", 2, 20, 23, embedding(pattern: 1)),
                chunk("A", 3, 30, 33, embedding(pattern: 2)),
            ])
        let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(result)
        XCTAssertLessThan(distance(embeddings, "A", 1), 0.05)
    }

    func testLongOddChunkCantOutvoteShortOnes() {
        // Two 2.5 s chunks of A's voice and one 10 s chunk of another voice: a duration-weighted
        // outlier check would keep the long chunk and drop the short ones.
        let result = DiarizationResult(
            segments: [segment("A", 0, 2.5), segment("A", 10, 12.5), segment("A", 20, 30)],
            chunkEmbeddings: [
                chunk("A", 0, 0, 2.5, embedding(pattern: 1)),
                chunk("A", 1, 10, 12.5, variant(of: 1, toward: 5, by: 0.1)),
                chunk("A", 2, 20, 30, embedding(pattern: 4)),
            ])
        XCTAssertGreaterThan(
            SpeakerUtilities.cosineDistance(embedding(pattern: 1), embedding(pattern: 4)), 0.7,
            "fixture: pattern 4 must read as another voice")
        let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(result)
        XCTAssertLessThan(distance(embeddings, "A", 1), 0.05)
    }

    func testKeepsAllChunksWhenNoneStandsOut() {
        let vectors = [variant(of: 1, toward: 4, by: 0.1), variant(of: 1, toward: 5, by: 0.1), embedding(pattern: 1)]
        let result = DiarizationResult(
            segments: [segment("A", 0, 3), segment("A", 10, 13), segment("A", 20, 23)],
            chunkEmbeddings: vectors.enumerated().map { i, v in
                chunk("A", i, Double(i * 10), Double(i * 10 + 3), v)
            })
        let expected = VDSPOperations.l2Normalize(
            vectors.map { VDSPOperations.l2Normalize($0) }.reduce([Float](repeating: 0, count: 256)) {
                zip($0, $1).map { $0 + $1 }
            })
        let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(result)
        XCTAssertEqual(SpeakerUtilities.cosineDistance(embeddings["A"] ?? [], expected), 0, accuracy: 1e-4)
    }

    func testWeightsByCleanSpeech() {
        let result = DiarizationResult(
            segments: [segment("A", 0, 8), segment("A", 10, 12)],
            chunkEmbeddings: [
                chunk("A", 0, 0, 8, embedding(pattern: 1)),
                chunk("A", 1, 10, 12, embedding(pattern: 2)),
            ])
        let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(result)
        XCTAssertLessThan(distance(embeddings, "A", 1), distance(embeddings, "A", 2))
    }

    func testOmitsSpeakersWithTooLittleCleanSpeech() {
        let result = DiarizationResult(
            segments: [segment("A", 0, 2), segment("A", 3, 5), segment("B", 6, 12)],
            chunkEmbeddings: [
                chunk("A", 0, 0, 2, embedding(pattern: 1)),
                chunk("A", 1, 3, 5, embedding(pattern: 1)),
                chunk("B", 1, 6, 12, embedding(pattern: 2)),
            ])
        let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(result, minTotalCleanSpeech: 5)
        XCTAssertNil(embeddings["A"])
        XCTAssertNotNil(embeddings["B"])
    }

    func testSkipsUnassignedChunksAndNormalizes() {
        let result = DiarizationResult(
            segments: [segment("A", 0, 6)],
            chunkEmbeddings: [
                chunk("", 0, 0, 6, embedding(pattern: 2)),
                chunk("A", 0, 0, 6, embedding(pattern: 1).map { $0 * 5 }),
            ])
        let embeddings = SpeakerUtilities.cleanEmbeddingsBySpeaker(result)
        XCTAssertEqual(embeddings.count, 1)
        let norm = (embeddings["A"] ?? []).reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1, accuracy: 1e-4)
        XCTAssertEqual(distance(embeddings, "A", 1), 0, accuracy: 1e-4)
    }
}
