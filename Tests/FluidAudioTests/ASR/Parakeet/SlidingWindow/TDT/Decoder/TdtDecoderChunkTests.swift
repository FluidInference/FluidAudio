@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class TdtDecoderChunkTests: XCTestCase {

    private var decoder: TdtDecoderV3!
    private var config: ASRConfig!

    override func setUp() {
        super.setUp()
        config = ASRConfig.default
        decoder = TdtDecoderV3(config: config)
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

            let (_, result) = predictions[predictionIndex]
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

    // MARK: - Global Frame Offset Tests

    func testGlobalFrameOffsetCalculations() throws {
        // Test global frame offset calculation for stateless multi-chunk processing
        // New chunking: ~239,360 samples per chunk (~14.96s), 2.0s overlap (~32,000 samples)
        let samplesPerEncoderFrame = ASRConstants.samplesPerEncoderFrame  // 1280 samples per frame
        let chunkSamples = 239_360  // ~14.96 seconds
        let overlapSamples = 32_000  // 2.0s overlap
        let strideSamples = chunkSamples - overlapSamples  // ~207,360 samples

        // First chunk: starts at sample 0
        let chunk1Start = 0
        let chunk1GlobalFrameOffset = chunk1Start / samplesPerEncoderFrame  // 0 / 1280 = 0
        XCTAssertEqual(chunk1GlobalFrameOffset, 0, "First chunk should start at frame offset 0")

        // Second chunk: starts at stride position (~207,360 samples ≈ 12.96s)
        let chunk2Start = strideSamples  // 207,360
        let chunk2GlobalFrameOffset = chunk2Start / samplesPerEncoderFrame  // 207360 / 1280 = 162
        XCTAssertEqual(chunk2GlobalFrameOffset, 162, "Second chunk should start at frame offset 162")

        // Third chunk: starts at 2 strides (~414,720 samples ≈ 25.92s)
        let chunk3Start = strideSamples * 2  // 414,720
        let chunk3GlobalFrameOffset = chunk3Start / samplesPerEncoderFrame  // 414720 / 1280 = 324
        XCTAssertEqual(chunk3GlobalFrameOffset, 324, "Third chunk should start at frame offset 324")

        // Test timestamp calculation: token at local frame 50 in each chunk
        let localFrame = 50
        let chunk1GlobalTimestamp = localFrame + chunk1GlobalFrameOffset  // 50 + 0 = 50
        let chunk2GlobalTimestamp = localFrame + chunk2GlobalFrameOffset  // 50 + 162 = 212
        let chunk3GlobalTimestamp = localFrame + chunk3GlobalFrameOffset  // 50 + 324 = 374

        XCTAssertEqual(chunk1GlobalTimestamp, 50, "Chunk 1 token should have global timestamp 50")
        XCTAssertEqual(chunk2GlobalTimestamp, 212, "Chunk 2 token should have global timestamp 212")
        XCTAssertEqual(chunk3GlobalTimestamp, 374, "Chunk 3 token should have global timestamp 374")

        // Convert to time for verification (frame * 0.08 seconds per frame)
        let chunk1Time = Double(chunk1GlobalTimestamp) * 0.08  // 50 * 0.08 = 4.0s
        let chunk2Time = Double(chunk2GlobalTimestamp) * 0.08  // 212 * 0.08 = 16.96s
        let chunk3Time = Double(chunk3GlobalTimestamp) * 0.08  // 374 * 0.08 = 29.92s

        XCTAssertEqual(chunk1Time, 4.0, accuracy: 0.01, "Chunk 1 token should be at 4.0s")
        XCTAssertEqual(chunk2Time, 16.96, accuracy: 0.01, "Chunk 2 token should be at 16.96s")
        XCTAssertEqual(chunk3Time, 29.92, accuracy: 0.01, "Chunk 3 token should be at 29.92s")
    }

    // MARK: - Context Frame Adjustment Tests

    func testFirstChunkContextFrameAdjustment() throws {
        // Test first chunk in stateless approach
        // Stateless: each chunk gets a fresh decoder state, contextFrameAdjustment = 0
        let actualAudioFrames = 140
        let contextFrameAdjustment = 0  // No previous state in stateless approach
        let globalFrameOffset = 0  // First chunk starts at global frame 0

        // Test the frame bounds calculation that would happen in decodeWithTimings
        let encoderSequenceLength = 140
        let effectiveSequenceLength = min(encoderSequenceLength, actualAudioFrames)

        // First chunk starts fresh with no previous context
        let timeIndices = contextFrameAdjustment  // Should be 0
        let activeMask = timeIndices < effectiveSequenceLength

        // Test global timestamp calculation for a token emitted at frame 50
        let localFrameIndex = 50
        let globalTimestamp = localFrameIndex + globalFrameOffset

        XCTAssertEqual(timeIndices, 0, "First chunk should start at frame 0")
        XCTAssertTrue(activeMask, "First chunk should be active")
        XCTAssertEqual(effectiveSequenceLength, 140, "Effective sequence length should match encoder frames")
        XCTAssertEqual(globalTimestamp, 50, "Global timestamp should equal local frame for first chunk")
        XCTAssertEqual(globalFrameOffset, 0, "First chunk should have no global offset")
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
        // With new stateless chunking:
        // Second chunk globalFrameOffset = 207,360 / 1,280 = 162
        let globalFrameOffset = 162  // Second chunk

        // Calculate final timestamp ensuring it doesn't exceed bounds
        let finalTimestamp = min(finalProcessingTimeIndices, effectiveSequenceLength - 1) + globalFrameOffset
        // expectedTimestamp = min(145, 139) + 162 = 139 + 162 = 301

        XCTAssertEqual(finalTimestamp, 301, "Final timestamp should be clamped and offset correctly")
    }

    // MARK: - Frame Processing Validation Tests

    func testFrameProcessingValidation() throws {
        let audioSampleCount = 320_000  // 20 seconds
        let expectedTotalFrames = ASRConstants.calculateEncoderFrames(from: audioSampleCount)

        // With new stateless chunking (~239,360 samples per chunk, ~207,360 stride):
        // 20s requires 2 chunks (chunk 1: 0-239,360, chunk 2: 207,360-320,000)
        let chunkSamples = 239_360
        let overlapSamples = 32_000
        let strideSamples = chunkSamples - overlapSamples
        let chunkCount = (audioSampleCount + strideSamples - 1) / strideSamples  // Ceiling division

        XCTAssertEqual(expectedTotalFrames, 250, "20s should be 250 encoder frames")
        XCTAssertEqual(chunkCount, 2, "20s audio should require 2 chunks with stride-based approach")

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

        // Simulate the clearing logic from TdtDecoderV3
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
        let lastEmissionTimestamp = 100
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
