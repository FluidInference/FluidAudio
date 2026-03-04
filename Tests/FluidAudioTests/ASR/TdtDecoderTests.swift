@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class TdtDecoderV3HelperTests: XCTestCase {

    var decoder: TdtDecoderV3!
    var config: ASRConfig!

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
        let stateShape: [NSNumber] = [
            NSNumber(value: 2),
            NSNumber(value: 1),
            NSNumber(value: ASRConstants.decoderHiddenSize),
        ]

        let hiddenState = try MLMultiArray(shape: stateShape, dataType: .float32)
        let cellState = try MLMultiArray(shape: stateShape, dataType: .float32)

        let input = try decoder.prepareDecoderInput(
            targetToken: token,
            hiddenState: hiddenState,
            cellState: cellState
        )

        // Verify all features are present
        XCTAssertNotNil(input.featureValue(for: "targets"))
        XCTAssertNotNil(input.featureValue(for: "target_length"))
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
        let encoderOutput = try MLMultiArray(
            shape: [1, NSNumber(value: ASRConstants.encoderHiddenSize), 1],
            dataType: .float32
        )

        // Create mock decoder output
        let decoderOutputArray = try MLMultiArray(
            shape: [1, NSNumber(value: ASRConstants.decoderHiddenSize), 1],
            dataType: .float32
        )
        let decoderOutput = try MLDictionaryFeatureProvider(dictionary: [
            "decoder": MLFeatureValue(multiArray: decoderOutputArray)
        ])

        let jointInput = try decoder.prepareJointInput(
            encoderOutput: encoderOutput,
            decoderOutput: decoderOutput,
            timeIndex: 0
        )

        // Verify both inputs are present
        XCTAssertNotNil(jointInput.featureValue(for: "encoder_step"))
        XCTAssertNotNil(jointInput.featureValue(for: "decoder_step"))

        // Verify shapes
        guard let encoderFeature = jointInput.featureValue(for: "encoder_step")?.multiArrayValue else {
            XCTFail("Missing encoder_step")
            return
        }
        XCTAssertEqual(encoderFeature.shape, encoderOutput.shape)

        guard let decoderFeature = jointInput.featureValue(for: "decoder_step")?.multiArrayValue else {
            XCTFail("Missing decoder_step")
            return
        }
        XCTAssertEqual(decoderFeature.shape, [1, NSNumber(value: ASRConstants.decoderHiddenSize), 1])
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

}
