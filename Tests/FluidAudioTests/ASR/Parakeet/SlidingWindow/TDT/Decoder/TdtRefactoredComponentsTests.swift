import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Tests for refactored TDT decoder components.
final class TdtRefactoredComponentsTests: XCTestCase {

    // MARK: - TdtFrameNavigation Tests

    func testCalculateInitialTimeIndicesFirstChunk() {
        // First chunk with no timeJump
        let result = TdtFrameNavigation.calculateInitialTimeIndices(
            timeJump: nil,
            contextFrameAdjustment: 0
        )
        XCTAssertEqual(result, 0, "First chunk should start at 0")
    }

    func testCalculateInitialTimeIndicesFirstChunkWithContext() {
        // First chunk with context adjustment
        let result = TdtFrameNavigation.calculateInitialTimeIndices(
            timeJump: nil,
            contextFrameAdjustment: 5
        )
        XCTAssertEqual(result, 5, "First chunk should start at context adjustment")
    }

    func testCalculateInitialTimeIndicesStreamingContinuation() {
        // Normal streaming continuation
        let result = TdtFrameNavigation.calculateInitialTimeIndices(
            timeJump: 10,
            contextFrameAdjustment: -5
        )
        XCTAssertEqual(result, 5, "Should sum timeJump and context adjustment")
    }

    func testCalculateInitialTimeIndicesSpecialOverlapCase() {
        // Special case: decoder finished exactly at boundary with overlap
        let result = TdtFrameNavigation.calculateInitialTimeIndices(
            timeJump: 0,
            contextFrameAdjustment: 0
        )
        XCTAssertEqual(
            result,
            ASRConstants.standardOverlapFrames,
            "Should skip standard overlap frames"
        )
    }

    func testCalculateInitialTimeIndicesNegativeResult() {
        // Should clamp negative results to 0
        let result = TdtFrameNavigation.calculateInitialTimeIndices(
            timeJump: -10,
            contextFrameAdjustment: 5
        )
        XCTAssertEqual(result, 0, "Should clamp negative results to 0")
    }

    func testInitializeNavigationState() {
        let (effectiveLength, safeIndices, lastTimestep, activeMask) =
            TdtFrameNavigation.initializeNavigationState(
                timeIndices: 10,
                encoderSequenceLength: 100,
                actualAudioFrames: 80
            )

        XCTAssertEqual(effectiveLength, 80, "Should use minimum of encoder and audio frames")
        XCTAssertEqual(safeIndices, 10, "Safe indices should be clamped to valid range")
        XCTAssertEqual(lastTimestep, 79, "Last timestep is effectiveLength - 1")
        XCTAssertTrue(activeMask, "Active mask should be true when timeIndices < effectiveLength")
    }

    func testInitializeNavigationStateOutOfBounds() {
        let (_, safeIndices, _, activeMask) = TdtFrameNavigation.initializeNavigationState(
            timeIndices: 100,
            encoderSequenceLength: 80,
            actualAudioFrames: 80
        )

        XCTAssertEqual(safeIndices, 79, "Should clamp to effectiveLength - 1")
        XCTAssertFalse(activeMask, "Active mask should be false when timeIndices >= effectiveLength")
    }

    func testCalculateFinalTimeJumpLastChunk() {
        let result = TdtFrameNavigation.calculateFinalTimeJump(
            currentTimeIndices: 100,
            effectiveSequenceLength: 80,
            isLastChunk: true
        )
        XCTAssertNil(result, "Last chunk should return nil")
    }

    func testCalculateFinalTimeJumpStreamingChunk() {
        let result = TdtFrameNavigation.calculateFinalTimeJump(
            currentTimeIndices: 100,
            effectiveSequenceLength: 80,
            isLastChunk: false
        )
        XCTAssertEqual(result, 20, "Should return offset from chunk boundary")
    }

    func testCalculateFinalTimeJumpNegativeOffset() {
        let result = TdtFrameNavigation.calculateFinalTimeJump(
            currentTimeIndices: 50,
            effectiveSequenceLength: 80,
            isLastChunk: false
        )
        XCTAssertEqual(result, -30, "Should handle negative offsets")
    }

    // MARK: - TdtDurationMapping Tests

    func testMapDurationBinValidIndices() throws {
        let v3Bins = [1, 2, 3, 4, 5]
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(0, durationBins: v3Bins), 1)
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(1, durationBins: v3Bins), 2)
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(2, durationBins: v3Bins), 3)
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(3, durationBins: v3Bins), 4)
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(4, durationBins: v3Bins), 5)
    }

    func testMapDurationBinCustomMapping() throws {
        let customBins = [1, 1, 2, 3, 5, 8]
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(0, durationBins: customBins), 1)
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(3, durationBins: customBins), 3)
        XCTAssertEqual(try TdtDurationMapping.mapDurationBin(5, durationBins: customBins), 8)
    }

    func testMapDurationBinOutOfRange() {
        let v3Bins = [1, 2, 3, 4, 5]
        XCTAssertThrowsError(try TdtDurationMapping.mapDurationBin(5, durationBins: v3Bins)) { error in
            guard let asrError = error as? ASRError else {
                XCTFail("Expected ASRError")
                return
            }
            if case .processingFailed(let message) = asrError {
                XCTAssertTrue(message.contains("Duration bin index out of range"))
            } else {
                XCTFail("Expected processingFailed error")
            }
        }
    }

    func testMapDurationBinNegativeIndex() {
        let v3Bins = [1, 2, 3, 4, 5]
        XCTAssertThrowsError(try TdtDurationMapping.mapDurationBin(-1, durationBins: v3Bins))
    }

    func testClampProbabilityValidRange() {
        XCTAssertEqual(TdtDurationMapping.clampProbability(0.5), 0.5, accuracy: 0.0001)
        XCTAssertEqual(TdtDurationMapping.clampProbability(0.0), 0.0, accuracy: 0.0001)
        XCTAssertEqual(TdtDurationMapping.clampProbability(1.0), 1.0, accuracy: 0.0001)
    }

    func testClampProbabilityBelowRange() {
        XCTAssertEqual(TdtDurationMapping.clampProbability(-0.5), 0.0, accuracy: 0.0001)
        XCTAssertEqual(TdtDurationMapping.clampProbability(-100.0), 0.0, accuracy: 0.0001)
    }

    func testClampProbabilityAboveRange() {
        XCTAssertEqual(TdtDurationMapping.clampProbability(1.5), 1.0, accuracy: 0.0001)
        XCTAssertEqual(TdtDurationMapping.clampProbability(100.0), 1.0, accuracy: 0.0001)
    }

    func testClampProbabilityNonFinite() {
        XCTAssertEqual(TdtDurationMapping.clampProbability(.nan), 0.0, accuracy: 0.0001)
        XCTAssertEqual(TdtDurationMapping.clampProbability(.infinity), 0.0, accuracy: 0.0001)
        XCTAssertEqual(TdtDurationMapping.clampProbability(-.infinity), 0.0, accuracy: 0.0001)
    }

    // MARK: - TdtJointDecision Tests

    func testJointDecisionCreation() {
        let decision = TdtJointDecision(
            token: 42,
            probability: 0.95,
            durationBin: 3
        )

        XCTAssertEqual(decision.token, 42)
        XCTAssertEqual(decision.probability, 0.95, accuracy: 0.0001)
        XCTAssertEqual(decision.durationBin, 3)
    }

    func testJointDecisionWithNegativeValues() {
        let decision = TdtJointDecision(
            token: -1,
            probability: 0.0,
            durationBin: 0
        )

        XCTAssertEqual(decision.token, -1)
        XCTAssertEqual(decision.probability, 0.0, accuracy: 0.0001)
        XCTAssertEqual(decision.durationBin, 0)
    }

    // MARK: - TdtJointInputProvider Tests

    func testJointInputProviderFeatureNames() throws {
        let encoderArray = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)
        let decoderArray = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)

        let provider = ReusableJointInputProvider(
            encoderStep: encoderArray,
            decoderStep: decoderArray
        )

        XCTAssertEqual(provider.featureNames, ["encoder_step", "decoder_step"])
    }

    func testJointInputProviderFeatureValues() throws {
        let encoderArray = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)
        let decoderArray = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)

        let provider = ReusableJointInputProvider(
            encoderStep: encoderArray,
            decoderStep: decoderArray
        )

        let encoderFeature = provider.featureValue(for: "encoder_step")
        let decoderFeature = provider.featureValue(for: "decoder_step")

        XCTAssertNotNil(encoderFeature)
        XCTAssertNotNil(decoderFeature)
        XCTAssertIdentical(encoderFeature?.multiArrayValue, encoderArray)
        XCTAssertIdentical(decoderFeature?.multiArrayValue, decoderArray)
    }

    func testJointInputProviderInvalidFeatureName() throws {
        let encoderArray = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)
        let decoderArray = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)

        let provider = ReusableJointInputProvider(
            encoderStep: encoderArray,
            decoderStep: decoderArray
        )

        let invalidFeature = provider.featureValue(for: "invalid_feature")
        XCTAssertNil(invalidFeature, "Should return nil for invalid feature name")
    }

    // MARK: - TdtModelInference Tests

    func testNormalizeDecoderProjectionAlreadyNormalized() throws {
        // Input already in [1, 640, 1] format
        let input = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)
        for i in 0..<640 {
            input[[0, i, 0] as [NSNumber]] = NSNumber(value: Float(i))
        }

        let inference = TdtModelInference()
        let normalized = try inference.normalizeDecoderProjection(input)

        XCTAssertEqual(normalized.shape.map { $0.intValue }, [1, 640, 1])

        // Verify data was copied correctly
        for i in 0..<640 {
            let expected = Float(i)
            let actual = normalized[[0, i, 0] as [NSNumber]].floatValue
            XCTAssertEqual(actual, expected, accuracy: 0.0001)
        }
    }

    func testNormalizeDecoderProjectionTranspose() throws {
        // Input in [1, 1, 640] format (needs transpose)
        let input = try MLMultiArray(shape: [1, 1, 640], dataType: .float32)
        for i in 0..<640 {
            input[[0, 0, i] as [NSNumber]] = NSNumber(value: Float(i))
        }

        let inference = TdtModelInference()
        let normalized = try inference.normalizeDecoderProjection(input)

        XCTAssertEqual(normalized.shape.map { $0.intValue }, [1, 640, 1])

        // Verify data was copied correctly
        for i in 0..<640 {
            let expected = Float(i)
            let actual = normalized[[0, i, 0] as [NSNumber]].floatValue
            XCTAssertEqual(actual, expected, accuracy: 0.0001)
        }
    }

    func testNormalizeDecoderProjectionWithDestination() throws {
        // Input in [1, 1, 640] format
        let input = try MLMultiArray(shape: [1, 1, 640], dataType: .float32)
        for i in 0..<640 {
            input[[0, 0, i] as [NSNumber]] = NSNumber(value: Float(i * 2))
        }

        // Pre-allocate destination
        let destination = try MLMultiArray(shape: [1, 640, 1], dataType: .float32)

        let inference = TdtModelInference()
        let normalized = try inference.normalizeDecoderProjection(input, into: destination)

        XCTAssertIdentical(normalized, destination, "Should reuse destination array")

        // Verify data was copied correctly
        for i in 0..<640 {
            let expected = Float(i * 2)
            let actual = normalized[[0, i, 0] as [NSNumber]].floatValue
            XCTAssertEqual(actual, expected, accuracy: 0.0001)
        }
    }

    func testNormalizeDecoderProjectionInvalidRank() throws {
        // Input with wrong rank
        let input = try MLMultiArray(shape: [640], dataType: .float32)

        let inference = TdtModelInference()
        XCTAssertThrowsError(try inference.normalizeDecoderProjection(input)) { error in
            guard let asrError = error as? ASRError else {
                XCTFail("Expected ASRError")
                return
            }
            if case .processingFailed(let message) = asrError {
                XCTAssertTrue(message.contains("Invalid decoder projection rank"))
            } else {
                XCTFail("Expected processingFailed error")
            }
        }
    }

    func testNormalizeDecoderProjectionInvalidBatchSize() throws {
        // Input with batch size != 1
        let input = try MLMultiArray(shape: [2, 640, 1], dataType: .float32)

        let inference = TdtModelInference()
        XCTAssertThrowsError(try inference.normalizeDecoderProjection(input)) { error in
            guard let asrError = error as? ASRError else {
                XCTFail("Expected ASRError")
                return
            }
            if case .processingFailed(let message) = asrError {
                XCTAssertTrue(message.contains("Unsupported decoder batch dimension"))
            } else {
                XCTFail("Expected processingFailed error")
            }
        }
    }

    func testNormalizeDecoderProjectionInvalidHiddenSize() throws {
        // Input with wrong hidden size
        let input = try MLMultiArray(shape: [1, 128, 1], dataType: .float32)

        let inference = TdtModelInference()
        XCTAssertThrowsError(try inference.normalizeDecoderProjection(input)) { error in
            guard let asrError = error as? ASRError else {
                XCTFail("Expected ASRError")
                return
            }
            if case .processingFailed(let message) = asrError {
                XCTAssertTrue(message.contains("Decoder projection hidden size mismatch"))
            } else {
                XCTFail("Expected processingFailed error")
            }
        }
    }

    func testNormalizeDecoderProjectionInvalidTimeAxis() throws {
        // Input with time axis != 1
        let input = try MLMultiArray(shape: [1, 640, 2], dataType: .float32)

        let inference = TdtModelInference()
        XCTAssertThrowsError(try inference.normalizeDecoderProjection(input)) { error in
            guard let asrError = error as? ASRError else {
                XCTFail("Expected ASRError")
                return
            }
            if case .processingFailed(let message) = asrError {
                XCTAssertTrue(message.contains("Decoder projection time axis must be 1"))
            } else {
                XCTFail("Expected processingFailed error")
            }
        }
    }
}
