import XCTest

@testable import FluidAudio

/// Tests for the public two-phase API (`prepare(audio:)` + `cluster(_:)`) added alongside
/// the single-call `process(audio:)`.
///
/// Both tests run real model inference, so they gate on local model availability the same
/// way `OfflineDiarizerManagerProgressTests` does and skip cleanly when models are absent.
@available(macOS 14.0, iOS 17.0, *)
final class OfflineDiarizerTwoPhaseTests: XCTestCase {

    // MARK: - process == prepare + cluster

    /// `process(audio:)` is documented as exactly `prepare(audio:)` followed by one
    /// `cluster(_:)`. Verify the two call paths produce bit-identical results.
    func testProcessMatchesPrepareThenCluster() async throws {
        try requireOfflineDiarizerModels()

        let manager = OfflineDiarizerManager()
        let audio = try DiarizationTestFixtures.fixtureAudio(sampleRate: 16_000)

        let prepared = try await manager.prepare(audio: audio)
        let twoPhase = try manager.cluster(prepared)
        let singleCall = try await manager.process(audio: audio)

        assertBitIdentical(singleCall, twoPhase, context: "process vs prepare+cluster")
    }

    // MARK: - cluster determinism

    /// `cluster(_:)` is documented as safe to call any number of times on the same
    /// `PreparedDiarization`. Verify repeated calls are deterministic (identical outputs).
    func testClusterIsDeterministicAcrossRepeatedCalls() async throws {
        try requireOfflineDiarizerModels()

        let manager = OfflineDiarizerManager()
        let audio = try DiarizationTestFixtures.fixtureAudio(sampleRate: 16_000)

        let prepared = try await manager.prepare(audio: audio)
        let first = try manager.cluster(prepared)

        for run in 2...3 {
            let repeated = try manager.cluster(prepared)
            assertBitIdentical(first, repeated, context: "cluster call #\(run)")
        }
    }

    // MARK: - Helpers

    /// Skips the test when the offline diarizer models are not cached locally
    /// (same gating convention as `OfflineDiarizerManagerProgressTests`, resolved
    /// against the repo cache folder `ModelHub` actually downloads into).
    private func requireOfflineDiarizerModels() throws {
        let repoDir = OfflineDiarizerModels.defaultModelsDirectory()
            .appendingPathComponent(Repo.diarizer.folderName, isDirectory: true)
        let allPresent = ModelNames.OfflineDiarizer.requiredModels.allSatisfy {
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        guard allPresent else {
            throw XCTSkip("Offline diarizer models not available")
        }
    }

    /// Asserts two diarization results are bit-identical in every deterministic field.
    ///
    /// Excluded by design: `TimedSpeakerSegment.id` (a fresh `UUID` per value) and
    /// `timings` (wall-clock measurements).
    private func assertBitIdentical(
        _ lhs: DiarizationResult,
        _ rhs: DiarizationResult,
        context: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            lhs.segments.count, rhs.segments.count,
            "[\(context)] segment count", file: file, line: line)

        for (index, (lhsSegment, rhsSegment)) in zip(lhs.segments, rhs.segments).enumerated() {
            XCTAssertEqual(
                lhsSegment.speakerId, rhsSegment.speakerId,
                "[\(context)] segments[\(index)].speakerId", file: file, line: line)
            XCTAssertEqual(
                lhsSegment.startTimeSeconds, rhsSegment.startTimeSeconds,
                "[\(context)] segments[\(index)].startTimeSeconds", file: file, line: line)
            XCTAssertEqual(
                lhsSegment.endTimeSeconds, rhsSegment.endTimeSeconds,
                "[\(context)] segments[\(index)].endTimeSeconds", file: file, line: line)
            XCTAssertEqual(
                lhsSegment.qualityScore, rhsSegment.qualityScore,
                "[\(context)] segments[\(index)].qualityScore", file: file, line: line)
            XCTAssertEqual(
                lhsSegment.embedding, rhsSegment.embedding,
                "[\(context)] segments[\(index)].embedding", file: file, line: line)
        }

        XCTAssertEqual(
            lhs.speakerDatabase, rhs.speakerDatabase,
            "[\(context)] speakerDatabase", file: file, line: line)

        XCTAssertEqual(
            lhs.chunkEmbeddings == nil, rhs.chunkEmbeddings == nil,
            "[\(context)] chunkEmbeddings presence", file: file, line: line)
        if let lhsChunks = lhs.chunkEmbeddings, let rhsChunks = rhs.chunkEmbeddings {
            XCTAssertEqual(
                lhsChunks.count, rhsChunks.count,
                "[\(context)] chunkEmbeddings count", file: file, line: line)
            for (index, (lhsChunk, rhsChunk)) in zip(lhsChunks, rhsChunks).enumerated() {
                XCTAssertEqual(
                    lhsChunk.speakerId, rhsChunk.speakerId,
                    "[\(context)] chunkEmbeddings[\(index)].speakerId", file: file, line: line)
                XCTAssertEqual(
                    lhsChunk.embedding256, rhsChunk.embedding256,
                    "[\(context)] chunkEmbeddings[\(index)].embedding256", file: file, line: line)
                XCTAssertEqual(
                    lhsChunk.rho128, rhsChunk.rho128,
                    "[\(context)] chunkEmbeddings[\(index)].rho128", file: file, line: line)
            }
        }
    }
}
