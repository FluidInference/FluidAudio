import Foundation
import XCTest

@testable import FluidAudio

final class SortformerTypesTests: XCTestCase {

    // MARK: - SortformerConfig Computed Properties

    func testDefaultConfigComputedProperties() {
        let config = SortformerConfig.default

        // chunkMelFrames = (chunkLen + leftContext + rightContext) * subsampling
        // = (6 + 1 + 7) * 8 = 112
        XCTAssertEqual(config.chunkMelFrames, 112)

        // coreFrames = chunkLen * subsampling = 6 * 8 = 48
        XCTAssertEqual(config.coreFrames, 48)

        // frameDurationSeconds = subsampling * melStride / sampleRate = 8 * 160 / 16000 = 0.08
        XCTAssertEqual(config.frameDurationSeconds, 0.08, accuracy: 1e-6)
    }

    func testConfigClampsChunkLenToMinimumOne() {
        let config = SortformerConfig(chunkLen: 0)
        XCTAssertEqual(config.chunkLen, 1, "chunkLen should be clamped to at least 1")
    }

    func testConfigClampsNegativeChunkLen() {
        let config = SortformerConfig(chunkLen: -5)
        XCTAssertEqual(config.chunkLen, 1, "Negative chunkLen should be clamped to 1")
    }

    func testSpkcacheLenConstraint() {
        // spkcacheLen must be >= (1 + silFramesPerSpk) * numSpeakers = (1 + 3) * 4 = 16
        let config = SortformerConfig(spkcacheLen: 1)
        XCTAssertGreaterThanOrEqual(config.spkcacheLen, 16)
    }

    func testSpkcacheUpdatePeriodConstraint() {
        // spkcacheUpdatePeriod = max(min(updatePeriod, fifoLen + chunkLen), chunkLen)
        // With defaults: max(min(31, 40 + 6), 6) = max(min(31, 46), 6) = max(31, 6) = 31
        let config = SortformerConfig.default
        XCTAssertEqual(config.spkcacheUpdatePeriod, 31)
    }

    func testSpkcacheUpdatePeriodClampedToChunkLen() {
        // If updatePeriod < chunkLen, it gets clamped up
        let config = SortformerConfig(chunkLen: 10, spkcacheUpdatePeriod: 2)
        XCTAssertGreaterThanOrEqual(config.spkcacheUpdatePeriod, config.chunkLen)
    }

    // MARK: - Compatibility

    func testConfigIsCompatibleWithSameShape() {
        let a = SortformerConfig.default
        let b = SortformerConfig.default
        XCTAssertTrue(a.isCompatible(with: b))
    }

    func testConfigIncompatibleWithDifferentShape() {
        let a = SortformerConfig.default
        let b = SortformerConfig.nvidiaHighLatency
        XCTAssertFalse(a.isCompatible(with: b))
    }

    // MARK: - SortformerPostProcessingConfig

    func testPostProcessingConfigDefaultValues() {
        let config = SortformerPostProcessingConfig.default
        XCTAssertEqual(config.onsetThreshold, 0.5)
        XCTAssertEqual(config.offsetThreshold, 0.5)
        XCTAssertEqual(config.onsetPadFrames, 0)
        XCTAssertEqual(config.offsetPadFrames, 0)
    }

    func testPostProcessingConfigFrameToSecondsConversion() {
        var config = SortformerPostProcessingConfig(onsetPadFrames: 5)
        // onsetPadSeconds = 5 * 0.08 = 0.4
        XCTAssertEqual(config.onsetPadSeconds, 0.4, accuracy: 1e-5)

        // Set via seconds, verify frames
        config.onsetPadSeconds = 0.16
        XCTAssertEqual(config.onsetPadFrames, 2, "0.16 / 0.08 = 2 frames")
    }

    func testPostProcessingConfigMinDurationConversion() {
        var config = SortformerPostProcessingConfig(minFramesOn: 10)
        XCTAssertEqual(config.minDurationOn, 0.8, accuracy: 1e-5, "10 * 0.08 = 0.8s")

        config.minDurationOn = 0.24
        XCTAssertEqual(config.minFramesOn, 3, "0.24 / 0.08 = 3 frames")
    }

    // MARK: - SortformerStreamingState

    func testStreamingStateInitializesEmpty() {
        let config = SortformerConfig.default
        let state = SortformerStreamingState(config: config)

        XCTAssertEqual(state.spkcacheLength, 0)
        XCTAssertEqual(state.fifoLength, 0)
        XCTAssertEqual(state.silenceFrameCount, 0)
        XCTAssertNil(state.spkcachePreds)
        XCTAssertNil(state.fifoPreds)
        XCTAssertEqual(state.meanSilenceEmbedding.count, config.preEncoderDims)
    }

    func testStreamingStateCleanup() {
        let config = SortformerConfig.default
        var state = SortformerStreamingState(config: config)

        // Simulate some data
        state.fifo = [Float](repeating: 1, count: 100)
        state.fifoLength = 10
        state.spkcache = [Float](repeating: 1, count: 100)
        state.spkcacheLength = 5

        state.cleanup()

        XCTAssertTrue(state.fifo.isEmpty)
        XCTAssertTrue(state.spkcache.isEmpty)
        XCTAssertEqual(state.fifoLength, 0)
        XCTAssertEqual(state.spkcacheLength, 0)
        XCTAssertNil(state.fifoPreds)
        XCTAssertNil(state.spkcachePreds)
    }

    // MARK: - StreamingUpdateResult

    func testStreamingUpdateResultFrameCounts() {
        let result = StreamingUpdateResult(
            confirmed: [Float](repeating: 0.5, count: 24),
            tentative: [Float](repeating: 0.3, count: 28),
            numSpeakers: 4
        )

        XCTAssertEqual(result.confirmedFrameCount, 6, "24 / 4 = 6 frames")
        XCTAssertEqual(result.tentativeFrameCount, 7, "28 / 4 = 7 frames")
    }

    // MARK: - SortformerChunkResult

    func testChunkResultGetSpeakerPrediction() {
        // 2 frames, 4 speakers: [f0s0, f0s1, f0s2, f0s3, f1s0, f1s1, f1s2, f1s3]
        let predictions: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        let result = SortformerChunkResult(
            startFrame: 0,
            speakerPredictions: predictions,
            frameCount: 2
        )

        XCTAssertEqual(result.getSpeakerPrediction(speaker: 0, frame: 0), 0.1, accuracy: 1e-5)
        XCTAssertEqual(result.getSpeakerPrediction(speaker: 3, frame: 1), 0.8, accuracy: 1e-5)
        XCTAssertEqual(result.getSpeakerPrediction(speaker: 0, frame: 5), 0.0, "Out of bounds returns 0")
    }

    func testChunkResultTentativeStartFrame() {
        let result = SortformerChunkResult(
            startFrame: 10,
            speakerPredictions: [Float](repeating: 0, count: 24),
            frameCount: 6
        )
        XCTAssertEqual(result.tentativeStartFrame, 16, "10 + 6 = 16")
    }

    // MARK: - SortformerError

    func testSortformerErrorDescriptions() {
        let errors: [SortformerError] = [
            .notInitialized,
            .modelLoadFailed("test"),
            .preprocessorFailed("test"),
            .inferenceFailed("test"),
            .invalidAudioData,
            .invalidState("test"),
            .configurationError("test"),
            .insufficientChunkLength("test"),
            .insufficientPredsLength("test"),
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription, "\(error) should have a description")
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }
}
