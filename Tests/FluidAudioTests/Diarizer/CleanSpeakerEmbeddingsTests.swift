import XCTest

@testable import FluidAudio

/// Tests for `SpeakerUtilities.cleanEmbeddingsBySpeaker`. Real-audio checks are in
/// `CleanSpeakerEmbeddingsAMITests`.
final class CleanSpeakerEmbeddingsTests: XCTestCase {

    /// A 256-dim vector from a few components, so similarities are exact.
    private func vec(_ parts: [Int: Float]) -> [Float] {
        var v = [Float](repeating: 0, count: 256)
        for (i, x) in parts { v[i] = x }
        return v
    }

    // Voices: a and b are different people; a1, a2 are a in other chunks; b1 is b.
    private lazy var a = vec([0: 1])
    private lazy var a1 = vec([0: 0.95, 1: 0.2, 2: 0.1])
    private lazy var a2 = vec([0: 0.9, 1: 0.1, 3: 0.2])
    private lazy var b = vec([1: 1])
    private lazy var b1 = vec([0: 0.1, 1: 0.95, 3: 0.1])
    private lazy var c = vec([2: 1])

    private func segment(_ speaker: String, _ start: Float, _ end: Float) -> TimedSpeakerSegment {
        TimedSpeakerSegment(
            speakerId: speaker, embedding: [], startTimeSeconds: start, endTimeSeconds: end, qualityScore: 1)
    }

    private func chunk(_ speaker: String, _ start: Double, _ end: Double, _ embedding: [Float]) -> ChunkEmbedding {
        ChunkEmbedding(
            speakerId: speaker, chunkIndex: Int(start), speakerIndex: 0, startTimeSeconds: start,
            endTimeSeconds: end, embedding256: embedding)
    }

    /// One chunk per segment, each the only speech in its span.
    private func result(_ speaker: String, _ chunks: [(Double, Double, [Float])]) -> DiarizationResult {
        DiarizationResult(
            segments: chunks.map { segment(speaker, Float($0.0), Float($0.1)) },
            chunkEmbeddings: chunks.map { chunk(speaker, $0.0, $0.1, $0.2) })
    }

    private func clean(_ result: DiarizationResult) -> [String: [Float]] {
        SpeakerUtilities.cleanEmbeddingsBySpeaker(result)
    }

    /// For tests of other rules that use a single chunk.
    private func cleanAllowingOneChunk(_ result: DiarizationResult) -> [String: [Float]] {
        SpeakerUtilities.cleanEmbeddingsBySpeaker(result, minChunks: 1)
    }

    private func similarity(_ x: [Float]?, _ y: [Float]) -> Float {
        guard let x else { return -2 }
        return 1 - SpeakerUtilities.cosineDistance(x, y)
    }

    // MARK: - Input

    func testNeedsChunkEmbeddings() {
        XCTAssertTrue(clean(DiarizationResult(segments: [segment("A", 0, 10)])).isEmpty)
        XCTAssertTrue(clean(DiarizationResult(segments: [], chunkEmbeddings: [])).isEmpty)
    }

    func testUsesChunkEmbeddingsNotSegmentEmbeddings() {
        let segments = [
            TimedSpeakerSegment(speakerId: "A", embedding: b, startTimeSeconds: 0, endTimeSeconds: 10, qualityScore: 1)
        ]
        let result = DiarizationResult(segments: segments, chunkEmbeddings: [chunk("A", 0, 10, a)])
        XCTAssertGreaterThan(similarity(cleanAllowingOneChunk(result)["A"], a), 0.9999)
    }

    func testSkipsUnassignedAndInvalidChunks() {
        var nan = a
        nan[5] = .nan
        let result = DiarizationResult(
            segments: [segment("A", 0, 30)],
            chunkEmbeddings: [
                chunk("", 0, 10, b),
                chunk("A", 0, 10, a),
                chunk("A", 10, 20, nan),
                chunk("A", 20, 30, vec([:])),
            ])
        let embeddings = cleanAllowingOneChunk(result)
        XCTAssertEqual(Array(embeddings.keys), ["A"])
        XCTAssertGreaterThan(similarity(embeddings["A"], a), 0.9999)
    }

    func testSkipsSpeakerWithMixedEmbeddingSizes() {
        let result = DiarizationResult(
            segments: [segment("A", 0, 20)],
            chunkEmbeddings: [chunk("A", 0, 10, a), chunk("A", 10, 20, [1, 0, 0])])
        XCTAssertNil(clean(result)["A"])
    }

    // MARK: - Clean speech per chunk

    func testOtherSpeakersTimeIsSubtracted() {
        // A talks 0-10, B talks over 1-9.5: A has 1.5 s clean in this chunk.
        let result = DiarizationResult(
            segments: [segment("A", 0, 10), segment("B", 1, 9.5), segment("A", 20, 25)],
            chunkEmbeddings: [chunk("A", 0, 10, b), chunk("A", 20, 25, a)])
        // Both count (1.5 s meets the minimum); the 1.5 s chunk weighs 1.5 against 5.
        let mixed = clean(result)["A"]
        XCTAssertNil(mixed, "two chunks that disagree")
        let shorter = DiarizationResult(
            segments: [segment("A", 0, 10), segment("B", 1, 9.6), segment("A", 20, 25)],
            chunkEmbeddings: [chunk("A", 0, 10, b), chunk("A", 20, 25, a)])
        XCTAssertGreaterThan(
            similarity(cleanAllowingOneChunk(shorter)["A"], a), 0.9999, "1.4 s clean: below the minimum, skipped")
    }

    func testOnlyTheSpeakersOwnTurnsInsideTheSpanCount() {
        let result = DiarizationResult(
            segments: [segment("A", 0, 3), segment("B", 3, 10), segment("A", 12, 20)],
            chunkEmbeddings: [chunk("A", 0, 10, a)])
        XCTAssertNil(cleanAllowingOneChunk(result)["A"], "3 s of A inside 0-10 s: under the 4 s total")
    }

    func testOwnOverlappingTurnsCountOnce() {
        let result = DiarizationResult(
            segments: [segment("A", 0, 3), segment("A", 2, 4.5)],
            chunkEmbeddings: [chunk("A", 0, 10, a)])
        XCTAssertGreaterThan(
            similarity(cleanAllowingOneChunk(result)["A"], a), 0.9999, "4.5 s, not 5.5 s, but over 4 s")
        let short = DiarizationResult(
            segments: [segment("A", 0, 3), segment("A", 1, 3.9)],
            chunkEmbeddings: [chunk("A", 0, 10, a)])
        XCTAssertNil(cleanAllowingOneChunk(short)["A"], "3.9 s once merged")
    }

    func testChunkMinimumBoundary() {
        let atMinimum = DiarizationResult(
            segments: [segment("A", 0, 1.5), segment("A", 10, 14)],
            chunkEmbeddings: [chunk("A", 0, 1.5, a1), chunk("A", 10, 14, a)])
        XCTAssertGreaterThan(SpeakerUtilities.cleanEmbeddingsBySpeaker(atMinimum, minTotalCleanSpeech: 5).count, 0)
        let below = DiarizationResult(
            segments: [segment("A", 0, 1.4), segment("A", 10, 14)],
            chunkEmbeddings: [chunk("A", 0, 1.4, a1), chunk("A", 10, 14, a)])
        XCTAssertTrue(SpeakerUtilities.cleanEmbeddingsBySpeaker(below, minTotalCleanSpeech: 5).isEmpty)
    }

    // MARK: - Agreement

    func testOneChunkNeedsCorroborationByDefault() {
        XCTAssertNil(clean(result("A", [(0, 8, a)]))["A"], "one chunk, however long, has nothing to back it up")
        XCTAssertNotNil(clean(result("A", [(0, 4, a), (5, 8, a1)]))["A"], "two that agree")
        XCTAssertNotNil(cleanAllowingOneChunk(result("A", [(0, 4, a)]))["A"], "exactly the 4 s minimum")
        XCTAssertNil(cleanAllowingOneChunk(result("A", [(0, 3.9, a)]))["A"])
    }

    func testTwoChunksMustAgree() {
        XCTAssertNotNil(clean(result("A", [(0, 3, a), (5, 8, a1)]))["A"])
        XCTAssertNil(clean(result("A", [(0, 3, a), (5, 8, b)]))["A"], "can't tell which is the speaker's voice")
    }

    func testOddChunkOutIsDropped() {
        let embedding = clean(result("A", [(0, 3, a), (4, 7, a1), (8, 11, a2), (12, 14, b)]))["A"]
        XCTAssertGreaterThan(similarity(embedding, a), 0.95)
        XCTAssertLessThan(similarity(embedding, b), 0.25)
    }

    func testLongOddChunkCantStandForTheSpeaker() {
        // Picking the medoid by seconds would make b the voice and drop a's chunks.
        // a holds 18 of 25 s (72%), well clear of the 60% gate; b is the longest single chunk.
        let embedding = clean(result("A", [(0, 6, a), (10, 16, a1), (20, 26, a2), (30, 37, b)]))["A"]
        XCTAssertGreaterThan(similarity(embedding, a), 0.95)
        XCTAssertLessThan(similarity(embedding, b), 0.25)
    }

    func testSpeakerMostlyInAnotherVoiceGetsNone() {
        XCTAssertNil(clean(result("A", [(0, 2.5, a), (10, 12.5, a1), (20, 30, b)]))["A"])
    }

    func testEvenTwoVoiceClusterGetsNone() {
        XCTAssertNil(clean(result("A", [(0, 4, a), (5, 9, a1), (10, 14, b), (15, 19, b1)]))["A"])
    }

    func testMajorityVoiceWinsWithoutPullFromTheMinority() {
        let merged = clean(result("A", [(0, 4, a), (5, 9, a1), (10, 14, a2), (15, 18, b), (20, 21.5, b1)]))["A"]
        let pure = clean(result("A", [(0, 4, a), (5, 9, a1), (10, 14, a2)]))["A"]
        XCTAssertNotNil(merged)
        XCTAssertEqual(merged, pure, "the minority's chunks don't move it at all")
    }

    func testConsensusBoundary() {
        let r = result("A", [(0, 3, a), (4, 7, a1), (8, 10, b), (11, 13, c)])  // kept 6 s of 10 s
        XCTAssertNotNil(SpeakerUtilities.cleanEmbeddingsBySpeaker(r, minConsensus: 0.6)["A"])
        XCTAssertNil(SpeakerUtilities.cleanEmbeddingsBySpeaker(r, minConsensus: 0.61)["A"])
    }

    func testKeptSpeechMustReachTheMinimum() {
        let r = result("A", [(0, 1.5, a), (2, 3.5, a1), (4, 7, b)])
        XCTAssertNil(
            SpeakerUtilities.cleanEmbeddingsBySpeaker(r, minConsensus: 0)["A"],
            "6 s of clean speech, only 3 s in the speaker's voice")
    }

    func testDistanceThresholdIsRespected() {
        let near = vec([0: 0.6, 1: 0.8])  // distance 0.4 from a
        let r = result("A", [(0, 3, a), (4, 7, near), (8, 11, a)])
        let loose = SpeakerUtilities.cleanEmbeddingsBySpeaker(r, maxDistanceToMedoid: 0.45)["A"]
        let strict = SpeakerUtilities.cleanEmbeddingsBySpeaker(r, maxDistanceToMedoid: 0.35)["A"]
        XCTAssertGreaterThan(similarity(loose, near), similarity(strict, near))
        XCTAssertGreaterThan(similarity(strict, a), 0.9999, "near left out")
    }

    // MARK: - Output

    func testWeightedByCleanSpeech() {
        let embedding = clean(result("A", [(0, 8, a), (10, 12, a1)]))["A"]
        XCTAssertGreaterThan(similarity(embedding, a), similarity(embedding, a1))
        let expected = VDSPOperations.l2Normalize(
            zip(VDSPOperations.l2Normalize(a), VDSPOperations.l2Normalize(a1)).map { 8 * $0 + 2 * $1 })
        XCTAssertGreaterThan(similarity(embedding, expected), 0.9999)
    }

    func testResultIsNormalized() {
        let embedding = cleanAllowingOneChunk(result("A", [(0, 6, a.map { $0 * 7 })]))["A"] ?? []
        XCTAssertEqual(embedding.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1, accuracy: 1e-4)
    }

    func testSpeakersAreIndependentAndOrderDoesNotMatter() {
        let chunks = [
            chunk("A", 0, 3, a), chunk("B", 3, 6, b), chunk("A", 6, 9, a1), chunk("B", 9, 12, b1),
            chunk("A", 12, 14, a2),
        ]
        let segments = chunks.map { segment($0.speakerId, Float($0.startTimeSeconds), Float($0.endTimeSeconds)) }
        let forward = clean(DiarizationResult(segments: segments, chunkEmbeddings: chunks))
        let reversed = clean(DiarizationResult(segments: segments.reversed(), chunkEmbeddings: chunks.reversed()))
        XCTAssertGreaterThan(similarity(forward["A"], a), 0.95)
        XCTAssertGreaterThan(similarity(forward["B"], b), 0.95)
        for id in ["A", "B"] {
            XCTAssertGreaterThan(similarity(forward[id], reversed[id] ?? []), 0.99999)
        }
    }

    // MARK: - Overlapping windows (OfflineDiarizerManager)

    /// Windows of 3 s every 0.6 s (80% overlap) over 3 s of speech.
    private func overlapping(_ speechEnd: Double, _ embedding: [Float]) -> DiarizationResult {
        let starts = stride(from: 0.0, to: speechEnd, by: 0.6)
        return DiarizationResult(
            segments: [segment("A", 0, Float(speechEnd))],
            chunkEmbeddings: starts.map { chunk("A", $0, $0 + 3, embedding) })
    }

    func testOverlappingWindowsCountEachSecondOnce() {
        // Per chunk, 3 + 2.4 + 1.8 + ... = 9 s; as audio it's 3 s, under the 4 s floor.
        XCTAssertNil(clean(overlapping(3, a))["A"])
        XCTAssertNotNil(clean(overlapping(5, a))["A"], "5 s of audio clears it")
    }

    func testOverlappingWindowsShareTheirWeight() {
        // c1 and c2 cover the same 0-6 s, c3 covers 6-8 s alone: weights 3, 3, 2, not 6, 6, 2.
        let r = DiarizationResult(
            segments: [segment("A", 0, 8)],
            chunkEmbeddings: [chunk("A", 0, 6, a), chunk("A", 0, 6, a1), chunk("A", 6, 8, a2)])
        let ua = VDSPOperations.l2Normalize(a)
        let ua1 = VDSPOperations.l2Normalize(a1)
        let ua2 = VDSPOperations.l2Normalize(a2)
        var weighted = [Float](repeating: 0, count: ua.count)
        for i in weighted.indices {
            weighted[i] = 3 * ua[i] + 3 * ua1[i] + 2 * ua2[i]
        }
        let shared = VDSPOperations.l2Normalize(weighted)
        XCTAssertGreaterThan(similarity(clean(r)["A"], shared), 0.99999)
        XCTAssertEqual(SpeakerUtilities.sharedWeights([[(0, 6)], [(0, 6)], [(6, 8)]]), [3, 3, 2])
        XCTAssertEqual(SpeakerUtilities.sharedWeights([[(0, 2), (4, 6)], [(1, 5)]]), [3, 3])
        XCTAssertEqual(SpeakerUtilities.sharedWeights([]), [])
    }

    func testShortChunksStillCountTowardConsensus() {
        // 6 s of a in three usable chunks, and 6 s of another voice in six 1 s chunks (each too short to
        // use): the kept chunks cover half the speaker's clean speech, under 60%.
        var chunks = [chunk("A", 0, 2, a), chunk("A", 2, 4, a1), chunk("A", 4, 6, a2)]
        chunks += (0..<6).map { chunk("A", Double(10 + $0), Double(11 + $0), b) }
        let r = DiarizationResult(segments: [segment("A", 0, 6), segment("A", 10, 16)], chunkEmbeddings: chunks)
        XCTAssertNil(clean(r)["A"])
        XCTAssertNotNil(SpeakerUtilities.cleanEmbeddingsBySpeaker(r, minConsensus: 0.5)["A"])
    }
}
