import XCTest

@testable import FluidAudio

final class ZeroVoteReembedderTests: XCTestCase {

    // Orthogonal unit centroids: cosine(embedding, centroid) is trivially predictable.
    private let centroids: [[Double]] = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    // MARK: - Run detection

    func testDetectsRunBoundedByVotedFrames() {
        // Frames 0–1 voted, 2–6 speech-active with zero votes, 7 voted.
        let speakerCount = [1, 1, 1, 1, 1, 1, 1, 1]
        let sums: [[Double]] = [
            [0.9, 0], [0.8, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0.7, 0],
        ]

        let runs = ZeroVoteReembedder.detectRuns(
            speakerCountPerFrame: speakerCount,
            activationSums: sums,
            frameDuration: 0.1,
            minDurationSeconds: 0.4
        )

        XCTAssertEqual(runs, [2..<7])
    }

    func testOverlapFramesAreNeverDetectedAsZeroVoteRuns() {
        // Frames 2-6 have zero votes but 2+ active speakers (overlap): re-embedding would
        // collapse the overlap to one speaker, so they must keep the existing tie-break.
        let speakerCount = [1, 1, 2, 2, 2, 2, 2, 1]
        let sums: [[Double]] = [
            [0.9, 0], [0.8, 0],
            [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            [0.7, 0],
        ]

        let runs = ZeroVoteReembedder.detectRuns(
            speakerCountPerFrame: speakerCount,
            activationSums: sums,
            frameDuration: 0.1,
            minDurationSeconds: 0.4
        )

        XCTAssertEqual(runs, [])
    }

    func testDetectsRunBoundedByNonSpeechGaps() {
        // Non-speech frames (speakerCount 0) delimit the run even with zero sums everywhere.
        let speakerCount = [0, 1, 1, 1, 1, 1, 0, 0]
        let sums = [[Double]](repeating: [0, 0], count: 8)

        let runs = ZeroVoteReembedder.detectRuns(
            speakerCountPerFrame: speakerCount,
            activationSums: sums,
            frameDuration: 0.1,
            minDurationSeconds: 0.4
        )

        XCTAssertEqual(runs, [1..<6])
    }

    func testDropsRunsShorterThanMinDuration() {
        // 3 zero-vote frames at 0.1 s each = 0.3 s < 0.4 s minimum.
        let speakerCount = [1, 1, 1, 1, 1]
        let sums: [[Double]] = [[0.9, 0], [0, 0], [0, 0], [0, 0], [0.9, 0]]

        let runs = ZeroVoteReembedder.detectRuns(
            speakerCountPerFrame: speakerCount,
            activationSums: sums,
            frameDuration: 0.1,
            minDurationSeconds: 0.4
        )

        XCTAssertTrue(runs.isEmpty)
    }

    func testDetectsMultipleMaximalRuns() {
        let speakerCount = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        var sums = [[Double]](repeating: [0, 0], count: 14)
        sums[0] = [0.9, 0]
        sums[5] = [0, 0.8]
        sums[6] = [0, 0.8]
        // Zero-vote runs: 1..<5 (0.4 s) and 7..<14 (0.7 s).

        let runs = ZeroVoteReembedder.detectRuns(
            speakerCountPerFrame: speakerCount,
            activationSums: sums,
            frameDuration: 0.1,
            minDurationSeconds: 0.4
        )

        XCTAssertEqual(runs, [1..<5, 7..<14])
    }

    func testRunExtendingToTimelineEndIsDetected() {
        let speakerCount = [1, 1, 1, 1, 1, 1]
        let sums: [[Double]] = [[0.9, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

        let runs = ZeroVoteReembedder.detectRuns(
            speakerCountPerFrame: speakerCount,
            activationSums: sums,
            frameDuration: 0.1,
            minDurationSeconds: 0.4
        )

        XCTAssertEqual(runs, [1..<6])
    }

    func testEmptyTimelineAndInvalidFrameDurationYieldNoRuns() {
        XCTAssertTrue(
            ZeroVoteReembedder.detectRuns(
                speakerCountPerFrame: [],
                activationSums: [],
                frameDuration: 0.1,
                minDurationSeconds: 0.4
            ).isEmpty
        )
        XCTAssertTrue(
            ZeroVoteReembedder.detectRuns(
                speakerCountPerFrame: [1, 1, 1, 1, 1],
                activationSums: [[Double]](repeating: [0], count: 5),
                frameDuration: 0,
                minDurationSeconds: 0.4
            ).isEmpty
        )
    }

    // MARK: - Assignment

    func testAssignsToClosestCentroidRegardlessOfMargin() {
        // Nearly tied with cluster 0, but cluster 1 is best by a sliver — there is no
        // incumbent to defend, so the best centroid wins outright.
        let assignment = ZeroVoteReembedder.assignment(
            embedding: [0.70, 0.71, 0.0],
            centroids: centroids
        )

        XCTAssertEqual(assignment?.cluster, 1)
        XCTAssertEqual(assignment?.cosines.count, 3)
    }

    func testExactTieResolvesToLowestClusterIndex() {
        let assignment = ZeroVoteReembedder.assignment(
            embedding: [1, 1, 0],
            centroids: centroids
        )

        XCTAssertEqual(assignment?.cluster, 0)
    }

    func testNonFiniteEmbeddingReturnsNil() {
        XCTAssertNil(
            ZeroVoteReembedder.assignment(
                embedding: [Double.nan, 0.5, 0.5],
                centroids: centroids
            )
        )
        XCTAssertNil(
            ZeroVoteReembedder.assignment(
                embedding: [Double.infinity, 0, 0],
                centroids: centroids
            )
        )
    }

    func testEmptyInputsAndDimensionMismatchReturnNil() {
        XCTAssertNil(ZeroVoteReembedder.assignment(embedding: [], centroids: centroids))
        XCTAssertNil(ZeroVoteReembedder.assignment(embedding: [1, 0, 0], centroids: []))
        XCTAssertNil(
            ZeroVoteReembedder.assignment(embedding: [1, 0], centroids: centroids)
        )
    }

    // MARK: - Config surface

    func testZeroVoteReembedDisabledByDefault() {
        let config = OfflineDiarizerConfig.default
        XCTAssertFalse(config.zeroVoteReembed.enabled)
        XCTAssertEqual(config.zeroVoteReembed.minDurationSeconds, 0.4)
    }

    func testValidateRejectsNegativeMinDuration() {
        var config = OfflineDiarizerConfig.default
        config.zeroVoteReembed.minDurationSeconds = -0.1
        XCTAssertThrowsError(try config.validate())
    }

    // MARK: - Reconstruction integration (synthetic frames, injected embedder)

    /// One chunk, two local speakers. Speaker 0 is voted to cluster 0 on both flanks;
    /// speaker 1 is speech-active in the middle but carries assignment −2 (no embedding),
    /// producing a zero-vote run that reconstruction must re-embed and emit as its own
    /// S2 segment with boundaries on the frame-run edges.
    private func makeZeroVoteScenario() -> (
        config: OfflineDiarizerConfig,
        segmentation: SegmentationOutput,
        hardClusters: [[Int]],
        centroids: [[Double]]
    ) {
        var config = OfflineDiarizerConfig(
            minSegmentDuration: 0.1,
            minGapDuration: 0.05,
            segmentationMinDurationOn: 0.0,
            segmentationMinDurationOff: 0.0
        )
        config.zeroVoteReembed = OfflineDiarizerConfig.ZeroVoteReembed(
            enabled: true,
            minDurationSeconds: 0.4
        )

        var weights = [[Float]]()
        for frame in 0..<30 {
            let middle = (10..<20).contains(frame)
            weights.append(middle ? [0, 1] : [1, 0])
        }
        let segmentation = SegmentationOutput(
            logProbs: [[[0]]],
            speakerWeights: [weights],
            numChunks: 1,
            numFrames: 30,
            numSpeakers: 2,
            chunkOffsets: [0],
            frameDuration: 0.1
        )

        return (config, segmentation, [[0, -2]], [[1, 0, 0], [0, 1, 0]])
    }

    func testReembeddedRunBecomesItsOwnSegmentOnFrameBoundaries() {
        let scenario = makeZeroVoteScenario()
        var embeddedSpans: [(Double, Double)] = []

        let segments = OfflineReconstruction(config: scenario.config).buildSegments(
            segmentation: scenario.segmentation,
            hardClusters: scenario.hardClusters,
            centroids: scenario.centroids,
            spanEmbedder: { start, end in
                embeddedSpans.append((start, end))
                return [0.1, 0.9, 0.0]  // Clearly centroid 1.
            }
        )

        XCTAssertEqual(embeddedSpans.count, 1)
        XCTAssertEqual(embeddedSpans.first?.0 ?? -1, 1.0, accuracy: 1e-6)
        XCTAssertEqual(embeddedSpans.first?.1 ?? -1, 2.0, accuracy: 1e-6)

        let shape = segments.map { ($0.speakerId, $0.startTimeSeconds, $0.endTimeSeconds) }
        XCTAssertEqual(shape.map { $0.0 }, ["S1", "S2", "S1"])
        XCTAssertEqual(Double(shape[1].1), 1.0, accuracy: 1e-3)
        XCTAssertEqual(Double(shape[1].2), 2.0, accuracy: 1e-3)
    }

    func testDisabledConfigNeverInvokesEmbedder() {
        var scenario = makeZeroVoteScenario()
        scenario.config.zeroVoteReembed.enabled = false
        var invocationCount = 0

        _ = OfflineReconstruction(config: scenario.config).buildSegments(
            segmentation: scenario.segmentation,
            hardClusters: scenario.hardClusters,
            centroids: scenario.centroids,
            spanEmbedder: { _, _ in
                invocationCount += 1
                return [0.1, 0.9, 0.0]
            }
        )

        XCTAssertEqual(invocationCount, 0)
    }

    func testFailedEmbeddingFallsBackToTieBreakBehavior() {
        let scenario = makeZeroVoteScenario()

        let withFailingEmbedder = OfflineReconstruction(config: scenario.config).buildSegments(
            segmentation: scenario.segmentation,
            hardClusters: scenario.hardClusters,
            centroids: scenario.centroids,
            spanEmbedder: { _, _ in nil }
        )

        var disabledConfig = scenario.config
        disabledConfig.zeroVoteReembed.enabled = false
        let withoutPass = OfflineReconstruction(config: disabledConfig).buildSegments(
            segmentation: scenario.segmentation,
            hardClusters: scenario.hardClusters,
            centroids: scenario.centroids
        )

        XCTAssertEqual(
            withFailingEmbedder.map { "\($0.speakerId)|\($0.startTimeSeconds)|\($0.endTimeSeconds)" },
            withoutPass.map { "\($0.speakerId)|\($0.startTimeSeconds)|\($0.endTimeSeconds)" }
        )
    }
}
