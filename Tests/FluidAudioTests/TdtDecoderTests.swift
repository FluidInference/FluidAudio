import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class TdtDecoderTests: XCTestCase {

    var decoder: TdtDecoder!
    var config: ASRConfig!

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

        let (token, score, duration) = try decoder.predictTokenAndDuration(
            logits, durationBins: config.tdtConfig.durationBins)

        XCTAssertEqual(token, 5)
        XCTAssertEqual(score, 0.9, accuracy: 0.0001)
        XCTAssertEqual(duration, 2)  // durations[2] = 2
    }

    // MARK: - Update Hypothesis Tests

    func testUpdateHypothesis() throws {

        let newState = try TdtDecoderState()
        var hypothesis = TdtHypothesis(decState: newState)

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

    // MARK: - Softmax Tests

    func testSoftmaxBasicComputation() throws {
        let logits: [Float] = [1.0, 2.0, 3.0]
        let result = decoder.softmax(logits)

        XCTAssertEqual(result.count, 3)

        // Verify it's a valid probability distribution (sums to 1.0)
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.0001)

        // Verify ordering is preserved (highest logit = highest probability)
        XCTAssertTrue(result[2] > result[1])
        XCTAssertTrue(result[1] > result[0])

        // Verify specific values for known input
        let expected: [Float] = [0.09003, 0.24473, 0.66524]
        for i in 0..<3 {
            XCTAssertEqual(result[i], expected[i], accuracy: 0.001)
        }
    }

    func testSoftmaxNumericalStability() throws {
        // Test with large values that could cause overflow without numerical stability
        let largeLogits: [Float] = [1000.0, 1001.0, 1002.0]
        let result = decoder.softmax(largeLogits)

        XCTAssertEqual(result.count, 3)

        // Should still sum to 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.0001)

        // Should not contain NaN or infinite values
        for prob in result {
            XCTAssertFalse(prob.isNaN, "Softmax result contains NaN")
            XCTAssertFalse(prob.isInfinite, "Softmax result contains infinite value")
            XCTAssertTrue(prob >= 0.0, "Softmax result contains negative probability")
        }

        // Ordering should still be preserved
        XCTAssertTrue(result[2] > result[1])
        XCTAssertTrue(result[1] > result[0])
    }

    func testSoftmaxEmptyArray() throws {
        let emptyLogits: [Float] = []
        let result = decoder.softmax(emptyLogits)

        XCTAssertTrue(result.isEmpty)
    }

    func testSoftmaxSingleElement() throws {
        let singleLogit: [Float] = [5.0]
        let result = decoder.softmax(singleLogit)

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 1.0, accuracy: 0.0001)
    }

    func testSoftmaxIdenticalValues() throws {
        // When all logits are identical, probabilities should be uniform
        let identicalLogits: [Float] = [2.0, 2.0, 2.0, 2.0]
        let result = decoder.softmax(identicalLogits)

        XCTAssertEqual(result.count, 4)

        // Should sum to 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.0001)

        // All probabilities should be equal (0.25 each)
        for prob in result {
            XCTAssertEqual(prob, 0.25, accuracy: 0.0001)
        }
    }

    func testSoftmaxNegativeValues() throws {
        let negativeLogits: [Float] = [-1.0, -2.0, -3.0]
        let result = decoder.softmax(negativeLogits)

        XCTAssertEqual(result.count, 3)

        // Should sum to 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.0001)

        // All probabilities should be positive
        for prob in result {
            XCTAssertTrue(prob > 0.0, "Probability should be positive")
        }

        // Higher logit should have higher probability
        XCTAssertTrue(result[0] > result[1])
        XCTAssertTrue(result[1] > result[2])
    }

    func testSoftmaxLargeArray() throws {
        // Test with larger array similar to vocabulary size
        let vocabSize = 1000
        var logits = [Float](repeating: 0.1, count: vocabSize)

        // Make a few tokens have higher probabilities
        logits[42] = 5.0
        logits[100] = 4.0
        logits[500] = 3.0

        let result = decoder.softmax(logits)

        XCTAssertEqual(result.count, vocabSize)

        // Should sum to 1.0
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)

        // The highest logit should have the highest probability
        let maxIndex = result.enumerated().max(by: { $0.element < $1.element })?.offset
        XCTAssertEqual(maxIndex, 42)

        // Verify ordering of our special tokens
        XCTAssertTrue(result[42] > result[100])
        XCTAssertTrue(result[100] > result[500])

        // All probabilities should be non-negative
        for prob in result {
            XCTAssertTrue(prob >= 0.0, "All probabilities should be non-negative")
        }
    }

    func testSoftmaxAccelerateOptimization() throws {
        // Test that the Accelerate framework implementation produces correct results
        let logits: [Float] = [0.5, 1.5, 2.5, 3.5, 4.5]
        let result = decoder.softmax(logits)

        // Manually calculate expected softmax for comparison
        let maxLogit = logits.max()!
        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        let expected = expValues.map { $0 / sumExp }

        XCTAssertEqual(result.count, expected.count)

        for i in 0..<result.count {
            XCTAssertEqual(
                result[i], expected[i], accuracy: 0.0001,
                "Accelerate implementation should match manual calculation at index \(i)")
        }

        // Verify probability distribution properties
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.0001)
    }
}
