import XCTest

@testable import FluidAudio

/// `PostProcessing.exclusiveSegments`: off by default, so overlapping speech stays in the output
/// like pyannote community-1's `speaker_diarization` (#1049); on, one speaker at a time.
final class OfflineExclusiveSegmentsTests: XCTestCase {

    func testDefaultsKeepOverlappingSpeech() {
        XCTAssertFalse(OfflineDiarizerConfig.PostProcessing.community.exclusiveSegments)
        XCTAssertFalse(OfflineDiarizerConfig.PostProcessing(minGapDurationSeconds: 0.1).exclusiveSegments)
        XCTAssertFalse(OfflineDiarizerConfig.default.exclusiveSegments)
        XCTAssertFalse(OfflineDiarizerConfig(clusteringThreshold: 0.6).exclusiveSegments)
    }

    /// One chunk, two local speakers on clusters 0 and 1: speaker 0 talks over frames 0..<20,
    /// speaker 1 over frames 10..<30, so both are active from 1.0 s to 2.0 s.
    private func buildOverlapSegments(exclusive: Bool) -> [(String, Double, Double)] {
        var config = OfflineDiarizerConfig(
            minSegmentDuration: 0.1,
            minGapDuration: 0.05,
            segmentationMinDurationOn: 0.0,
            segmentationMinDurationOff: 0.0
        )
        config.exclusiveSegments = exclusive

        let weights: [[Float]] = (0..<30).map { frame in
            [frame < 20 ? 1 : 0, frame >= 10 ? 1 : 0]
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

        return OfflineReconstruction(config: config).buildSegments(
            segmentation: segmentation,
            hardClusters: [[0, 1]],
            centroids: [[1, 0, 0], [0, 1, 0]]
        )
        .map { ($0.speakerId, Double($0.startTimeSeconds), Double($0.endTimeSeconds)) }
    }

    func testOverlapIsKeptByDefault() {
        let segments = buildOverlapSegments(exclusive: false)

        XCTAssertEqual(segments.map(\.0), ["S1", "S2"])
        XCTAssertEqual(segments[0].1, 0.0, accuracy: 1e-3)
        XCTAssertEqual(segments[0].2, 2.0, accuracy: 1e-3)
        XCTAssertEqual(segments[1].1, 1.0, accuracy: 1e-3)
        XCTAssertEqual(segments[1].2, 3.0, accuracy: 1e-3)
    }

    func testExclusiveSegmentsTrimTheLaterSpeaker() {
        let segments = buildOverlapSegments(exclusive: true)

        XCTAssertEqual(segments.map(\.0), ["S1", "S2"])
        XCTAssertEqual(segments[0].1, 0.0, accuracy: 1e-3)
        XCTAssertEqual(segments[0].2, 2.0, accuracy: 1e-3)
        XCTAssertEqual(segments[1].1, 2.0, accuracy: 1e-3)
        XCTAssertEqual(segments[1].2, 3.0, accuracy: 1e-3)
    }
}
