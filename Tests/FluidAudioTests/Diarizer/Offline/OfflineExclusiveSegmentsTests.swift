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

    /// One chunk of 30 frames (0.1 s each), two local speakers on clusters 0 and 1: speaker 0
    /// talks over `speaker0`, speaker 1 over `speaker1`. `exclusive` nil keeps the config default.
    private func buildOverlapSegments(
        speaker0: Range<Int> = 0..<20,
        speaker1: Range<Int> = 10..<30,
        minSegmentDuration: Double = 0.1,
        hardClusters: [[Int]] = [[0, 1]],
        exclusive: Bool?
    ) -> [(String, Double, Double)] {
        var config = OfflineDiarizerConfig(
            minSegmentDuration: minSegmentDuration,
            minGapDuration: 0.05,
            segmentationMinDurationOn: 0.0,
            segmentationMinDurationOff: 0.0
        )
        if let exclusive {
            config.exclusiveSegments = exclusive
        }

        let weights: [[Float]] = (0..<30).map { frame in
            [speaker0.contains(frame) ? 1 : 0, speaker1.contains(frame) ? 1 : 0]
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
            hardClusters: hardClusters,
            centroids: [[1, 0, 0], [0, 1, 0]]
        )
        .map { ($0.speakerId, Double($0.startTimeSeconds), Double($0.endTimeSeconds)) }
    }

    func testOverlapIsKeptByDefault() {
        // Both speakers are active from 1.0 s to 2.0 s.
        let segments = buildOverlapSegments(exclusive: nil)

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

    func testExclusiveSegmentsDropAShortRemainder() {
        // Speaker 1 runs 1.0-2.8 s; trimmed to start at 2.5 s, 0.3 s is left, under the 0.5 s minimum.
        let shape = { (exclusive: Bool) in
            self.buildOverlapSegments(
                speaker0: 0..<25, speaker1: 10..<28, minSegmentDuration: 0.5, exclusive: exclusive
            ).map(\.0)
        }

        XCTAssertEqual(shape(false), ["S1", "S2"])
        XCTAssertEqual(shape(true), ["S1"])
    }

    func testOverlapKeepsItsSecondSpeakerWithoutVotes() {
        // Speaker 1 is active but got no embedding (cluster -2). The overlap frames still count
        // two speakers, and like pyannote the second slot is filled rather than left as a miss.
        let segments = buildOverlapSegments(hardClusters: [[0, -2]], exclusive: nil)

        XCTAssertEqual(segments.map(\.0), ["S1", "S2"])
        XCTAssertEqual(segments[1].1, 1.0, accuracy: 1e-3)
        XCTAssertEqual(segments[1].2, 2.0, accuracy: 1e-3)
    }

    func testSegmentsThatStartAndEndTogetherKeepAFixedOrder() {
        // Both accumulators close in the same frame and come out of a dictionary in
        // per-process order; the speaker breaks the tie.
        let segments = buildOverlapSegments(speaker0: 0..<10, speaker1: 0..<10, exclusive: nil)

        XCTAssertEqual(segments.map(\.0), ["S1", "S2"])
    }
}
