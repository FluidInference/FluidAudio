import Foundation
import XCTest

@testable import FluidAudio

final class SortformerStateUpdaterTests: XCTestCase {

    // MARK: - Sigmoid

    func testApplySigmoidKnownValues() {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)

        let result = updater.applySigmoid([0.0])
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 0.5, accuracy: 1e-5, "sigmoid(0) should be 0.5")
    }

    func testApplySigmoidLargePositive() {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)

        let result = updater.applySigmoid([100.0])
        XCTAssertEqual(result[0], 1.0, accuracy: 1e-5, "sigmoid(100) should be ~1.0")
    }

    func testApplySigmoidLargeNegative() {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)

        let result = updater.applySigmoid([-100.0])
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-5, "sigmoid(-100) should be ~0.0")
    }

    func testApplySigmoidSymmetry() {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)

        let result = updater.applySigmoid([2.0, -2.0])
        XCTAssertEqual(result[0] + result[1], 1.0, accuracy: 1e-5, "sigmoid(x) + sigmoid(-x) = 1")
    }

    func testApplySigmoidMultipleValues() {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)

        let input: [Float] = [-10.0, -1.0, 0.0, 1.0, 10.0]
        let result = updater.applySigmoid(input)

        XCTAssertEqual(result.count, 5)
        // Should be monotonically increasing
        for i in 1..<result.count {
            XCTAssertGreaterThan(result[i], result[i - 1], "Sigmoid should be monotonically increasing")
        }

        // All values in [0, 1]
        for value in result {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }
    }

    func testApplySigmoidEmpty() {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)

        let result = updater.applySigmoid([])
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Streaming Update Edge Cases

    func testStreamingUpdateThrowsOnInsufficientPreds() throws {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)
        var state = SortformerStreamingState(config: config)

        // Provide enough chunk data but insufficient predictions
        let chunkFrames = config.coreFrames + config.chunkLeftContext + config.chunkRightContext
        let chunk = [Float](repeating: 0, count: chunkFrames * config.preEncoderDims)
        let preds = [Float](repeating: 0, count: 1)  // Way too short

        XCTAssertThrowsError(
            try updater.streamingUpdate(
                state: &state,
                chunk: chunk,
                preds: preds,
                leftContext: config.chunkLeftContext,
                rightContext: config.chunkRightContext
            )
        ) { error in
            guard let sortformerError = error as? SortformerError else {
                XCTFail("Expected SortformerError, got \(type(of: error))")
                return
            }
            if case .insufficientPredsLength = sortformerError {
                // Expected
            } else {
                XCTFail("Expected insufficientPredsLength, got \(sortformerError)")
            }
        }
    }

    func testStreamingUpdateBasicFlow() throws {
        let config = SortformerConfig.default
        let updater = SortformerStateUpdater(config: config)
        var state = SortformerStreamingState(config: config)

        // Build valid-sized inputs
        let totalChunkFrames = config.coreFrames + config.chunkLeftContext + config.chunkRightContext
        let chunk = [Float](repeating: 0, count: totalChunkFrames * config.preEncoderDims)
        let predsFrames = totalChunkFrames  // spkcache=0, fifo=0, so preds matches chunk frames
        let preds = [Float](repeating: 0, count: predsFrames * config.numSpeakers)

        let result = try updater.streamingUpdate(
            state: &state,
            chunk: chunk,
            preds: preds,
            leftContext: config.chunkLeftContext,
            rightContext: config.chunkRightContext
        )

        XCTAssertEqual(
            result.confirmed.count, config.coreFrames * config.numSpeakers,
            "Confirmed should have coreFrames * numSpeakers elements"
        )
    }
}
