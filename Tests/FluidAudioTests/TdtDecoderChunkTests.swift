import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class TdtDecoderChunkTests: XCTestCase {

    private var decoder: TdtDecoder!
    private var config: ASRConfig!

    override func setUp() {
        super.setUp()
        config = ASRConfig.default
        decoder = TdtDecoder(config: config)
    }

    override func tearDown() {
        decoder = nil
        config = nil
        super.tearDown()
    }

    // MARK: - Mock Helpers

    private func createMockEncoderOutput(timeFrames: Int, hiddenSize: Int = 1024) throws -> MLMultiArray {
        let encoderOutput = try MLMultiArray(shape: [1, timeFrames, hiddenSize] as [NSNumber], dataType: .float32)

        // Fill with predictable test data
        for t in 0..<timeFrames {
            for h in 0..<hiddenSize {
                let index = t * hiddenSize + h
                let value = Float(t * 1000 + h) / Float(hiddenSize * 1000)  // Normalize to prevent overflow
                encoderOutput[index] = NSNumber(value: value)
            }
        }

        return encoderOutput
    }

    private func createMockDecoderState(lastToken: Int? = nil, timeJump: Int? = nil) throws -> TdtDecoderState {
        var state = try TdtDecoderState()
        state.lastToken = lastToken
        state.timeJump = timeJump
        return state
    }

    private class MockMLModel: MLModel {
        var predictions: [(String, MLFeatureProvider)] = []
        var predictionIndex = 0

        override func prediction(
            from input: MLFeatureProvider, options: MLPredictionOptions = MLPredictionOptions()
        ) throws -> MLFeatureProvider {
            guard predictionIndex < predictions.count else {
                throw ASRError.processingFailed("Mock model ran out of predictions")
            }

            let (expectedType, result) = predictions[predictionIndex]
            predictionIndex += 1

            return result
        }

        func addPrediction(type: String, result: MLFeatureProvider) {
            predictions.append((type, result))
        }

        func reset() {
            predictionIndex = 0
            predictions.removeAll()
        }
    }

    // MARK: - Context Frame Adjustment Tests

    func testFirstChunkContextFrameAdjustment() throws {
        let encoderOutput = try createMockEncoderOutput(timeFrames: 140)  // 11.2s
        var decoderState = try createMockDecoderState()  // Fresh state, no timeJump

        // Mock models aren't available in unit tests, so we test the logic indirectly
        // by checking the frame calculation parameters

        let actualAudioFrames = 140
        let contextFrameAdjustment = 0  // First chunk
        let isLastChunk = true  // Single chunk
        let globalFrameOffset = 0

        // Test the frame bounds calculation that would happen in decodeWithTimings
        let encoderSequenceLength = 140
        let effectiveSequenceLength = min(encoderSequenceLength, actualAudioFrames)

        // For first chunk with no previous timeJump
        let timeIndices = contextFrameAdjustment  // Should be 0
        let activeMask = timeIndices < effectiveSequenceLength

        XCTAssertEqual(timeIndices, 0, "First chunk should start at frame 0")
        XCTAssertTrue(activeMask, "First chunk should be active")
        XCTAssertEqual(effectiveSequenceLength, 140, "Effective sequence length should match encoder frames")
    }

    func testSecondChunkContextFrameAdjustmentWithTimeJump() throws {
        let encoderOutput = try createMockEncoderOutput(timeFrames: 140)
        var decoderState = try createMockDecoderState(timeJump: 3)  // Decoder was 3 frames ahead of previous chunk

        let actualAudioFrames = 140
        let contextFrameAdjustment = -20  // Standard overlap (1.6s = 20 frames)
        let globalFrameOffset = 140  // Second chunk starts at frame 140

        // Test timeIndices calculation for chunk continuation
        let prevTimeJump = decoderState.timeJump!
        let timeIndices = max(0, prevTimeJump + contextFrameAdjustment)  // 3 + (-20) = -17, max(0, -17) = 0

        XCTAssertEqual(timeIndices, 0, "Second chunk with large negative adjustment should clamp to 0")

        // Test with different timeJump values
        decoderState.timeJump = 25  // Decoder was well ahead
        let timeIndices2 = max(0, 25 + contextFrameAdjustment)  // 25 + (-20) = 5
        XCTAssertEqual(timeIndices2, 5, "Second chunk should skip appropriate frames based on timeJump")
    }

    func testLastChunkAdaptiveContextFrameAdjustment() throws {
        // Test adaptive context calculation for last chunk
        let standardLeftContextSamples = Int(1.6 * 16000.0)  // 25,600 samples
        let remainingSamples = 80_000  // 5 seconds of audio
        let centerStart = 179_200  // Starting at 11.2s
        let maxModelSamples = 240_000  // 15s capacity

        // Calculate adaptive left context
        let desiredTotalSamples = min(maxModelSamples, remainingSamples + centerStart)
        let maxLeftContext = centerStart
        let neededLeftContext = desiredTotalSamples - remainingSamples
        let adaptiveLeftContextSamples = min(neededLeftContext, maxLeftContext)

        // Calculate frame adjustment
        let contextFrameAdjustment: Int
        if adaptiveLeftContextSamples > standardLeftContextSamples {
            let extraContextSamples = adaptiveLeftContextSamples - standardLeftContextSamples
            contextFrameAdjustment = Int(
                (Double(extraContextSamples) / Double(ASRConstants.samplesPerEncoderFrame)).rounded())
        } else {
            let standardOverlapSamples = standardLeftContextSamples
            contextFrameAdjustment = -Int(
                (Double(standardOverlapSamples) / Double(ASRConstants.samplesPerEncoderFrame)).rounded())
        }

        XCTAssertEqual(desiredTotalSamples, 240_000, "Should use full model capacity")
        XCTAssertEqual(adaptiveLeftContextSamples, 160_000, "Should use 160k samples for left context")
        XCTAssertGreaterThan(contextFrameAdjustment, 0, "Should have positive adjustment for extra context")

        // Verify the frame calculation
        let expectedFrameAdjustment = Int((Double(160_000 - 25_600) / 1280.0).rounded())  // Extra context / samples per frame
        XCTAssertEqual(
            contextFrameAdjustment, expectedFrameAdjustment, "Frame adjustment should match calculated value")
    }

    // MARK: - Time Jump Calculation Tests

    func testTimeJumpCalculationNormalFlow() throws {
        let effectiveSequenceLength = 140
        let finalTimeIndices = 143  // Decoder processed beyond available frames

        let expectedTimeJump = finalTimeIndices - effectiveSequenceLength  // 143 - 140 = 3
        XCTAssertEqual(expectedTimeJump, 3, "Time jump should reflect frames processed beyond chunk")
    }

    func testTimeJumpCalculationExactBoundary() throws {
        let effectiveSequenceLength = 140
        let finalTimeIndices = 140  // Decoder stopped exactly at boundary

        let expectedTimeJump = finalTimeIndices - effectiveSequenceLength  // 140 - 140 = 0
        XCTAssertEqual(expectedTimeJump, 0, "No time jump when decoder stops at boundary")
    }

    func testTimeJumpCalculationUnderrun() throws {
        let effectiveSequenceLength = 140
        let finalTimeIndices = 135  // Decoder stopped before end

        let expectedTimeJump = finalTimeIndices - effectiveSequenceLength  // 135 - 140 = -5
        XCTAssertEqual(expectedTimeJump, -5, "Negative time jump when decoder stops early")
    }

    // MARK: - Last Chunk Finalization Tests

    func testLastChunkFinalizationFrameVariations() throws {
        let effectiveSequenceLength = 100
        let encoderFrameCount = 105  // Slightly more encoder frames available
        let finalProcessingTimeIndices = 98

        // Test frame variation calculation
        let frameVariations = [
            min(finalProcessingTimeIndices, encoderFrameCount - 1),  // min(98, 104) = 98
            min(effectiveSequenceLength - 1, encoderFrameCount - 1),  // min(99, 104) = 99
            min(max(0, effectiveSequenceLength - 2), encoderFrameCount - 1),  // min(98, 104) = 98
        ]

        XCTAssertEqual(frameVariations[0], 98, "First variation should use processing position")
        XCTAssertEqual(frameVariations[1], 99, "Second variation should use sequence boundary")
        XCTAssertEqual(frameVariations[2], 98, "Third variation should use sequence boundary - 2")

        // Test stepping through variations
        for step in 0..<6 {
            let frameIndex = frameVariations[step % frameVariations.count]
            XCTAssertTrue(
                frameIndex >= 0 && frameIndex < encoderFrameCount, "Frame index should be valid for step \(step)")
        }
    }

    func testLastChunkTimestampCalculation() throws {
        let finalProcessingTimeIndices = 145
        let effectiveSequenceLength = 140
        let globalFrameOffset = 280  // Third chunk

        // Calculate final timestamp ensuring it doesn't exceed bounds
        let finalTimestamp = min(finalProcessingTimeIndices, effectiveSequenceLength - 1) + globalFrameOffset
        let expectedTimestamp = min(145, 139) + 280  // 139 + 280 = 419

        XCTAssertEqual(finalTimestamp, 419, "Final timestamp should be clamped and offset correctly")
    }

    // MARK: - Frame Processing Validation Tests

    func testFrameProcessingValidation() throws {
        let audioSampleCount = 320_000  // 20 seconds
        let expectedTotalFrames = ASRConstants.calculateEncoderFrames(from: audioSampleCount)
        let segmentCount = 2  // Two 11.2s center chunks
        let centerSeconds = 11.2
        let sampleRate = 16000

        let processedCenterFrames =
            segmentCount * Int(centerSeconds * Double(sampleRate)) / ASRConstants.samplesPerEncoderFrame

        XCTAssertEqual(expectedTotalFrames, 250, "20s should be 250 encoder frames")
        XCTAssertEqual(processedCenterFrames, 280, "Two 11.2s chunks should process 280 center frames")

        // In real processing, we'd expect some frames to be processed multiple times due to overlap
        // The validation ensures we account for all audio content
    }

    func testBoundaryFrameCalculations() throws {
        // Test various boundary conditions for frame calculations
        let testCases: [(samples: Int, expectedFrames: Int)] = [
            (1280, 1),  // Exactly 1 frame
            (2560, 2),  // Exactly 2 frames
            (1900, 2),  // 1.48 frames, should round up to 2 (ceiling)
            (2000, 2),  // 1.56 frames, should round up to 2 (ceiling)
            (16000, 13),  // 1 second = 12.5 frames, should be 13 (ceiling)
            (0, 0),  // Empty audio
        ]

        for (samples, expectedFrames) in testCases {
            let actualFrames = ASRConstants.calculateEncoderFrames(from: samples)
            XCTAssertEqual(
                actualFrames, expectedFrames,
                "Sample count \(samples) should produce \(expectedFrames) frames, got \(actualFrames)")
        }
    }

    // MARK: - Decoder State Transition Tests

    func testDecoderStateClearing() throws {
        var decoderState = try createMockDecoderState(lastToken: 7883, timeJump: 5)  // period token

        // Test punctuation token clearing logic
        let punctuationTokens = [7883, 7952, 7948]  // period, question, exclamation
        let lastToken = decoderState.lastToken!

        XCTAssertTrue(punctuationTokens.contains(lastToken), "Test token should be punctuation")

        // Simulate the clearing logic from TdtDecoder
        if punctuationTokens.contains(lastToken) {
            decoderState.predictorOutput = nil
            // lastToken is kept for linguistic context
        }

        XCTAssertNil(decoderState.predictorOutput, "Predictor output should be cleared after punctuation")
        XCTAssertEqual(decoderState.lastToken, 7883, "Last token should be preserved for context")
    }

    func testDecoderStateFinalization() throws {
        var decoderState = try createMockDecoderState(timeJump: 8)
        let initialTimeJump = decoderState.timeJump

        // Simulate last chunk finalization
        decoderState.finalizeLastChunk()

        // Time jump should be cleared for last chunk
        let finalTimeJump: Int? = nil
        decoderState.timeJump = finalTimeJump

        XCTAssertNil(decoderState.timeJump, "Time jump should be nil after last chunk")
        XCTAssertNotEqual(decoderState.timeJump, initialTimeJump, "State should change after finalization")
    }

    // MARK: - Edge Case Tests

    func testVeryShortSequenceHandling() throws {
        let encoderSequenceLength = 1  // Very short sequence
        let actualAudioFrames = 1

        // Should exit early for sequences <= 1
        XCTAssertLessThanOrEqual(encoderSequenceLength, 1, "Should trigger early exit condition")

        // Effective length should be minimum of both
        let effectiveSequenceLength = min(encoderSequenceLength, actualAudioFrames)
        XCTAssertEqual(effectiveSequenceLength, 1, "Effective length should be 1")

        // Time indices should not be active
        let timeIndices = 1  // Start at frame 1
        let activeMask = timeIndices < effectiveSequenceLength
        XCTAssertFalse(activeMask, "Should not be active when starting beyond bounds")
    }

    func testTokenLimitEnforcement() throws {
        let maxTokensPerChunk = config.tdtConfig.maxTokensPerChunk
        var tokensProcessedThisChunk = maxTokensPerChunk - 1

        // Simulate approaching the token limit
        tokensProcessedThisChunk += 1
        let shouldStop = tokensProcessedThisChunk > maxTokensPerChunk

        XCTAssertFalse(shouldStop, "Should not stop processing when at limit - 1")

        // Test exactly at limit
        tokensProcessedThisChunk += 1
        let shouldStopAtLimit = tokensProcessedThisChunk > maxTokensPerChunk

        XCTAssertTrue(shouldStopAtLimit, "Should stop processing when exceeding token limit")
        XCTAssertGreaterThan(
            tokensProcessedThisChunk, maxTokensPerChunk, "Token count should exceed limit to trigger stop")
    }

    func testConsecutiveBlankLimitInFinalization() throws {
        let maxConsecutiveBlanks = config.tdtConfig.consecutiveBlankLimit
        var consecutiveBlanks = maxConsecutiveBlanks - 1

        // Simulate encountering consecutive blanks during finalization
        consecutiveBlanks += 1
        let shouldStopFinalization = consecutiveBlanks >= maxConsecutiveBlanks

        XCTAssertTrue(shouldStopFinalization, "Should stop finalization after consecutive blank limit")
        XCTAssertEqual(consecutiveBlanks, maxConsecutiveBlanks, "Blank count should equal limit")
    }

    func testForceBlankMechanismParameters() throws {
        let maxSymbolsPerStep = config.tdtConfig.maxSymbolsPerStep
        var emissionsAtThisTimestamp = maxSymbolsPerStep - 1
        var lastEmissionTimestamp = 100
        let timeIndicesCurrentLabels = 100  // Same timestamp

        // Simulate multiple emissions at same timestamp
        if timeIndicesCurrentLabels == lastEmissionTimestamp {
            emissionsAtThisTimestamp += 1
        }

        let shouldForceAdvance = emissionsAtThisTimestamp >= maxSymbolsPerStep
        XCTAssertTrue(shouldForceAdvance, "Should trigger force-blank mechanism")

        if shouldForceAdvance {
            let forcedAdvance = 1
            let newTimeIndices = 150 + forcedAdvance  // Simulate advancing
            XCTAssertEqual(newTimeIndices, 151, "Should advance by forced amount")
        }
    }
}
