import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class TdtDecoderTests: XCTestCase {

    var decoder: TdtDecoder!
    var config: ASRConfig!
    var decoderConfig: TdtDecoderConfig!

    override func setUp() {
        super.setUp()
        config = ASRConfig.default
        decoderConfig = TdtDecoderConfig.default
        decoder = TdtDecoder(config: config, decoderConfig: decoderConfig)
    }

    override func tearDown() {
        decoder = nil
        config = nil
        decoderConfig = nil
        super.tearDown()
    }

    // MARK: - Extract Encoder Time Step Tests

    func testExtractEncoderTimeStep() throws {

        // Create encoder output: [batch=1, sequence=5, hidden=4]
        let encoderOutput = try MLMultiArray(shape: [1, 5, 4], dataType: .float32)

        // Fill with  data: time * 10 + hidden
        for t in 0..<5 {
            for h in 0..<4 {
                let index = t * 4 + h
                encoderOutput[index] = NSNumber(value: Float(t * 10 + h))
            }
        }

        // Extract time step 2
        let timeStep = try decoder.extractEncoderTimeStep(encoderOutput, timeIndex: 2)

        XCTAssertEqual(timeStep.shape, [1, 1, 4] as [NSNumber])

        // Verify extracted values
        for h in 0..<4 {
            let expectedValue = Float(2 * 10 + h)
            XCTAssertEqual(
                timeStep[h].floatValue, expectedValue, accuracy: 0.0001,
                "Mismatch at hidden index \(h)")
        }
    }

    func testExtractEncoderTimeStepBoundaries() throws {

        let encoderOutput = try MLMultiArray(shape: [1, 3, 2], dataType: .float32)

        // Fill with sequential values
        for i in 0..<encoderOutput.count {
            encoderOutput[i] = NSNumber(value: Float(i))
        }

        // Test first time step
        let firstStep = try decoder.extractEncoderTimeStep(encoderOutput, timeIndex: 0)
        XCTAssertEqual(firstStep[0].floatValue, 0.0, accuracy: 0.0001)
        XCTAssertEqual(firstStep[1].floatValue, 1.0, accuracy: 0.0001)

        // Test last time step
        let lastStep = try decoder.extractEncoderTimeStep(encoderOutput, timeIndex: 2)
        XCTAssertEqual(lastStep[0].floatValue, 4.0, accuracy: 0.0001)
        XCTAssertEqual(lastStep[1].floatValue, 5.0, accuracy: 0.0001)

        // Test out of bounds
        XCTAssertThrowsError(
            try decoder.extractEncoderTimeStep(encoderOutput, timeIndex: 3)
        ) { error in
            guard case ASRError.processingFailed(let message) = error else {
                XCTFail("Expected processingFailed error")
                return
            }
            XCTAssertTrue(message.contains("out of bounds"))
        }
    }

    // MARK: - Calculate Next Time Index Tests

    func testCalculateNextTimeIndex() {

        // Test normal skip in long sequence
        var nextIdx = decoder.calculateNextTimeIndex(currentIdx: 5, skip: 3, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 8)

        // Test capped skip in long sequence
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 5, skip: 10, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 9)  // Capped at 4

        // Test skip at sequence boundary
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 98, skip: 5, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 100)  // Should not exceed sequence length

        // Test short sequence behavior
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 2, skip: 5, sequenceLength: 8)
        XCTAssertEqual(nextIdx, 4)  // Limited to 2 for short sequences

        // Test zero skip
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 5, skip: 0, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 5)  // No movement
    }

    // MARK: - Prepare Decoder Input Tests

    func testPrepareDecoderInput() throws {

        let token = 42
        let hiddenState = try MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        let cellState = try MLMultiArray(shape: [2, 1, 640], dataType: .float32)

        let input = try decoder.prepareDecoderInput(
            targetToken: token,
            hiddenState: hiddenState,
            cellState: cellState
        )

        // Verify all features are present
        XCTAssertNotNil(input.featureValue(for: "targets"))
        XCTAssertNotNil(input.featureValue(for: "target_lengths"))
        XCTAssertNotNil(input.featureValue(for: "h_in"))
        XCTAssertNotNil(input.featureValue(for: "c_in"))

        // Verify target token
        guard let targets = input.featureValue(for: "targets")?.multiArrayValue else {
            XCTFail("Missing targets")
            return
        }
        XCTAssertEqual(targets[0].intValue, token)
    }

    // MARK: - Prepare Joint Input Tests

    func testPrepareJointInput() throws {

        // Create encoder output
        let encoderOutput = try MLMultiArray(shape: [1, 1, 256], dataType: .float32)

        // Create mock decoder output
        let decoderOutputArray = try MLMultiArray(shape: [1, 1, 128], dataType: .float32)
        let decoderOutput = try MLDictionaryFeatureProvider(dictionary: [
            "decoder_output": MLFeatureValue(multiArray: decoderOutputArray)
        ])

        let jointInput = try decoder.prepareJointInput(
            encoderOutput: encoderOutput,
            decoderOutput: decoderOutput,
            timeIndex: 0
        )

        // Verify both inputs are present
        XCTAssertNotNil(jointInput.featureValue(for: "encoder_outputs"))
        XCTAssertNotNil(jointInput.featureValue(for: "decoder_outputs"))

        // Verify shapes
        guard let encoderFeature = jointInput.featureValue(for: "encoder_outputs")?.multiArrayValue else {
            XCTFail("Missing encoder_outputs")
            return
        }
        XCTAssertEqual(encoderFeature.shape, encoderOutput.shape)

        guard let decoderFeature = jointInput.featureValue(for: "decoder_outputs")?.multiArrayValue else {
            XCTFail("Missing decoder_outputs")
            return
        }
        XCTAssertEqual(decoderFeature.shape, decoderOutputArray.shape)
    }

    // MARK: - Predict Token and Duration Tests

    func testPredictTokenAndDuration() throws {

        // Create logits for 10 tokens + 5 durations
        let logits = try MLMultiArray(shape: [15], dataType: .float32)

        // Set token logits (make token 5 the highest)
        for i in 0..<10 {
            logits[i] = NSNumber(value: Float(i == 5 ? 0.9 : 0.1))
        }

        // Set duration logits (make duration 2 the highest)
        for i in 0..<5 {
            logits[10 + i] = NSNumber(value: Float(i == 2 ? 0.8 : 0.2))
        }

        let (token, score, duration) = try decoder.predictTokenAndDuration(logits)

        XCTAssertEqual(token, 5)
        XCTAssertEqual(score, 0.9, accuracy: 0.0001)
        XCTAssertEqual(duration, 2)  // durations[2] = 2
    }

    // MARK: - Update Hypothesis Tests

    func testUpdateHypothesis() throws {

        var hypothesis = TdtHypothesis()
        let newState = try DecoderState()

        decoder.updateHypothesis(
            &hypothesis,
            token: 42,
            score: 0.95,
            duration: 3,
            timeIdx: 10,
            decoderState: newState
        )

        XCTAssertEqual(hypothesis.ySequence, [42])
        XCTAssertEqual(hypothesis.score, 0.95, accuracy: 0.0001)
        XCTAssertEqual(hypothesis.timestamps, [10])
        XCTAssertEqual(hypothesis.lastToken, 42)
        XCTAssertNotNil(hypothesis.decState)

        // Test with includeTokenDuration
        if config.tdtConfig.includeTokenDuration {
            XCTAssertEqual(hypothesis.tokenDurations, [3])
        }

        // Add another token
        decoder.updateHypothesis(
            &hypothesis,
            token: 100,
            score: 0.85,
            duration: 1,
            timeIdx: 13,
            decoderState: newState
        )

        XCTAssertEqual(hypothesis.ySequence, [42, 100])
        XCTAssertEqual(hypothesis.score, 1.8, accuracy: 0.0001)
        XCTAssertEqual(hypothesis.timestamps, [10, 13])
        XCTAssertEqual(hypothesis.lastToken, 100)
    }

    // MARK: - TdtConfig Tests

    func testTdtConfigDefaultValues() {
        let config = TdtConfig.default

        XCTAssertEqual(config.durations, [0, 1, 2, 3, 4])
        XCTAssertTrue(config.includeTokenDuration)
        XCTAssertNil(config.maxSymbolsPerStep)

        // Test new configurable thresholds with defaults
        XCTAssertEqual(config.veryShortSequenceThreshold, 10)
        XCTAssertEqual(config.shortSequenceThreshold, 30)
        XCTAssertEqual(config.veryShortSequenceMaxSkip, 2)
        XCTAssertEqual(config.blankTokenMaxSkip, 2)
        XCTAssertEqual(config.nonBlankTokenMaxSkip, 4)
    }

    func testTdtConfigCustomInitialization() {
        let customDurations = [0, 2, 4, 6, 8]
        let config = TdtConfig(
            durations: customDurations,
            includeTokenDuration: false,
            maxSymbolsPerStep: 5,
            veryShortSequenceThreshold: 15,
            shortSequenceThreshold: 40,
            veryShortSequenceMaxSkip: 3,
            blankTokenMaxSkip: 3,
            nonBlankTokenMaxSkip: 5
        )

        XCTAssertEqual(config.durations, customDurations)
        XCTAssertFalse(config.includeTokenDuration)
        XCTAssertEqual(config.maxSymbolsPerStep, 5)
        XCTAssertEqual(config.veryShortSequenceThreshold, 15)
        XCTAssertEqual(config.shortSequenceThreshold, 40)
        XCTAssertEqual(config.veryShortSequenceMaxSkip, 3)
        XCTAssertEqual(config.blankTokenMaxSkip, 3)
        XCTAssertEqual(config.nonBlankTokenMaxSkip, 5)
    }

    func testTdtConfigPresetConfigurations() {
        // Test conservative preset
        let conservative = TdtConfig.conservative
        XCTAssertEqual(conservative.veryShortSequenceThreshold, 15)
        XCTAssertEqual(conservative.shortSequenceThreshold, 50)
        XCTAssertEqual(conservative.veryShortSequenceMaxSkip, 1)
        XCTAssertEqual(conservative.blankTokenMaxSkip, 1)
        XCTAssertEqual(conservative.nonBlankTokenMaxSkip, 2)

        // Test balanced preset (should match default)
        let balanced = TdtConfig.balanced
        XCTAssertEqual(balanced.veryShortSequenceThreshold, 10)
        XCTAssertEqual(balanced.shortSequenceThreshold, 30)
        XCTAssertEqual(balanced.veryShortSequenceMaxSkip, 2)
        XCTAssertEqual(balanced.blankTokenMaxSkip, 2)
        XCTAssertEqual(balanced.nonBlankTokenMaxSkip, 4)

        // Test aggressive preset
        let aggressive = TdtConfig.aggressive
        XCTAssertEqual(aggressive.veryShortSequenceThreshold, 5)
        XCTAssertEqual(aggressive.shortSequenceThreshold, 20)
        XCTAssertEqual(aggressive.veryShortSequenceMaxSkip, 3)
        XCTAssertEqual(aggressive.blankTokenMaxSkip, 3)
        XCTAssertEqual(aggressive.nonBlankTokenMaxSkip, 6)

        // Test conversational preset
        let conversational = TdtConfig.conversational
        XCTAssertEqual(conversational.veryShortSequenceThreshold, 12)
        XCTAssertEqual(conversational.shortSequenceThreshold, 40)
        XCTAssertEqual(conversational.veryShortSequenceMaxSkip, 2)
        XCTAssertEqual(conversational.blankTokenMaxSkip, 2)
        XCTAssertEqual(conversational.nonBlankTokenMaxSkip, 3)

        // Test longForm preset
        let longForm = TdtConfig.longForm
        XCTAssertEqual(longForm.veryShortSequenceThreshold, 8)
        XCTAssertEqual(longForm.shortSequenceThreshold, 25)
        XCTAssertEqual(longForm.veryShortSequenceMaxSkip, 2)
        XCTAssertEqual(longForm.blankTokenMaxSkip, 3)
        XCTAssertEqual(longForm.nonBlankTokenMaxSkip, 5)
    }

    // MARK: - TdtDecoderConfig Tests

    func testTdtDecoderConfigDefaultValues() {
        let config = TdtDecoderConfig.default

        XCTAssertEqual(config.vocabSize, 8192)
        XCTAssertEqual(config.blankTokenId, 8192)
        XCTAssertEqual(config.startOfSequenceId, 8192)
        XCTAssertEqual(config.encoderHiddenSize, 1024)
        XCTAssertEqual(config.totalOutputTokens, 8193)
        XCTAssertEqual(config.durationTokens, 5)
    }

    func testTdtDecoderConfigConsistency() {
        let config = TdtDecoderConfig.default

        // Verify blank token and start-of-sequence are the same (RNNT convention)
        XCTAssertEqual(config.blankTokenId, config.startOfSequenceId)

        // Verify blank token is outside vocabulary range
        XCTAssertEqual(config.blankTokenId, config.vocabSize)

        // Verify total tokens = vocab + blank token
        XCTAssertEqual(config.totalOutputTokens, config.vocabSize + 1)
    }

    // MARK: - Encoder Output Shape Tests (Transpose Fix)

    func testEncoderOutputTransposedShape() throws {
        // Test that decoder expects [batch, time, hidden] format (post-transpose)
        let batchSize = 1
        let timeSteps = 10
        let hiddenSize = 1024

        // Create encoder output in the correct [batch, time, hidden] format
        let encoderOutput = try MLMultiArray(
            shape: [batchSize, timeSteps, hiddenSize] as [NSNumber],
            dataType: .float32
        )

        // Fill with test data
        for t in 0..<timeSteps {
            for h in 0..<hiddenSize {
                let index = t * hiddenSize + h
                encoderOutput[index] = NSNumber(value: Float(t) * 0.01 + Float(h) * 0.0001)
            }
        }

        // This should not throw an error with the correct shape
        XCTAssertNoThrow(
            try performEncoderValidation(encoderOutput, expectedHiddenSize: hiddenSize)
        )
    }

    func testEncoderOutputWrongHiddenSize() throws {
        // Test validation of incorrect hidden size
        let encoderOutput = try MLMultiArray(
            shape: [1, 10, 512] as [NSNumber],  // Wrong hidden size (512 instead of 1024)
            dataType: .float32
        )

        XCTAssertThrowsError(
            try performEncoderValidation(encoderOutput, expectedHiddenSize: 1024)
        ) { error in
            guard case ASRError.processingFailed(let message) = error else {
                XCTFail("Expected processingFailed error")
                return
            }
            XCTAssertTrue(message.contains("Invalid encoder frame size"))
            XCTAssertTrue(message.contains("512"))
            XCTAssertTrue(message.contains("1024"))
        }
    }

    // MARK: - Hypothesis State Management Tests

    func testHypothesisInitialization() {
        let hypothesis = TdtHypothesis()

        XCTAssertEqual(hypothesis.score, 0.0)
        XCTAssertTrue(hypothesis.ySequence.isEmpty)
        XCTAssertNil(hypothesis.decState)
        XCTAssertTrue(hypothesis.timestamps.isEmpty)
        XCTAssertTrue(hypothesis.tokenDurations.isEmpty)
        XCTAssertNil(hypothesis.lastToken)
    }

    func testHypothesisLastTokenTracking() throws {
        var hypothesis = TdtHypothesis()
        let decoderState = try DecoderState()

        // Update hypothesis with a non-blank token
        decoder.updateHypothesis(
            &hypothesis,
            token: 42,
            score: 0.9,
            duration: 2,
            timeIdx: 5,
            decoderState: decoderState
        )

        XCTAssertEqual(hypothesis.lastToken, 42)

        // Update with another non-blank token
        decoder.updateHypothesis(
            &hypothesis,
            token: 100,
            score: 0.8,
            duration: 1,
            timeIdx: 7,
            decoderState: decoderState
        )

        XCTAssertEqual(hypothesis.lastToken, 100)
    }

    // MARK: - Duration Logic Tests

    func testDurationLogicForShortSequences() {
        // Test duration capping for very short sequences (< 30 frames)
        let actualSkip = decoder.calculateNextTimeIndex(
            currentIdx: 5,
            skip: 4,
            sequenceLength: 25  // Short sequence
        )

        // For sequences < 30, duration should be limited
        XCTAssertLessThanOrEqual(actualSkip - 5, 4)
    }

    func testDurationLogicForVeryShortSequences() {
        // Test duration capping for very short sequences (< 10 frames)
        let actualSkip = decoder.calculateNextTimeIndex(
            currentIdx: 2,
            skip: 5,
            sequenceLength: 8  // Very short sequence
        )

        // For sequences < 10, skip should be capped at 2
        XCTAssertEqual(actualSkip, 4)  // 2 + 2 (capped)
    }

    func testDurationLogicWithConservativePreset() {
        // Create decoder with conservative config
        let conservativeConfig = ASRConfig(tdtConfig: TdtConfig.conservative)
        let conservativeDecoder = TdtDecoder(config: conservativeConfig, decoderConfig: decoderConfig)

        // Test with a sequence length that would be considered "very short" in conservative mode
        // Conservative has veryShortSequenceThreshold=15
        let actualSkip = conservativeDecoder.calculateNextTimeIndex(
            currentIdx: 5,
            skip: 5,
            sequenceLength: 14  // Less than 15 (conservative threshold)
        )

        // Should be capped at conservative's veryShortSequenceMaxSkip=1
        XCTAssertEqual(actualSkip, 6)  // 5 + 1 (capped at 1)
    }

    func testDurationLogicWithAggressivePreset() {
        // Create decoder with aggressive config
        let aggressiveConfig = ASRConfig(tdtConfig: TdtConfig.aggressive)
        let aggressiveDecoder = TdtDecoder(config: aggressiveConfig, decoderConfig: decoderConfig)

        // Test with a normal sequence (> 20 frames for aggressive)
        let actualSkip = aggressiveDecoder.calculateNextTimeIndex(
            currentIdx: 10,
            skip: 8,
            sequenceLength: 100
        )

        // Should allow up to 6 frames skip (aggressive's nonBlankTokenMaxSkip)
        XCTAssertEqual(actualSkip, 16)  // 10 + 6 (capped at 6)
    }

    func testDurationLogicWithCustomThresholds() {
        // Create decoder with completely custom thresholds
        let customTdtConfig = TdtConfig(
            veryShortSequenceThreshold: 20,
            shortSequenceThreshold: 60,
            veryShortSequenceMaxSkip: 1,
            blankTokenMaxSkip: 5,
            nonBlankTokenMaxSkip: 7
        )
        let customConfig = ASRConfig(tdtConfig: customTdtConfig)
        let customDecoder = TdtDecoder(config: customConfig, decoderConfig: decoderConfig)

        // Test very short sequence behavior (< 20 frames)
        var actualSkip = customDecoder.calculateNextTimeIndex(
            currentIdx: 5,
            skip: 10,
            sequenceLength: 15
        )
        XCTAssertEqual(actualSkip, 6)  // 5 + 1 (capped at veryShortSequenceMaxSkip=1)

        // Test normal sequence behavior (>= 60 frames)
        actualSkip = customDecoder.calculateNextTimeIndex(
            currentIdx: 30,
            skip: 10,
            sequenceLength: 100
        )
        XCTAssertEqual(actualSkip, 37)  // 30 + 7 (capped at nonBlankTokenMaxSkip=7)
    }

    // MARK: - Helper Methods

    private func performEncoderValidation(_ encoderOutput: MLMultiArray, expectedHiddenSize: Int) throws {
        let shape = encoderOutput.shape
        guard shape.count >= 3 else {
            throw ASRError.processingFailed("Invalid encoder output shape: \(shape)")
        }

        let hiddenSize = shape[2].intValue
        guard hiddenSize == expectedHiddenSize else {
            throw ASRError.processingFailed(
                "Invalid encoder frame size: \(hiddenSize), expected \(expectedHiddenSize)")
        }
    }
}
