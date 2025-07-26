import Foundation
import CoreML
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
    
    func ExtractEncoderTimeStep() throws {
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
            XCTAssertEqual(timeStep[h].floatValue, expectedValue, accuracy: 0.0001,
                          "Mismatch at hidden index \(h)")
        }
    }
    
    func ExtractEncoderTimeStepBoundaries() throws {
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
    
    // MARK: - Split Logits Tests
    
    func SplitLogits() throws {
        // TDT config has 5 durations by default: [0, 1, 2, 3, 4]
        let vocabSize = 1025
        let durationCount = config.tdtConfig.durations.count
        let totalSize = vocabSize + durationCount
        
        let logits = try MLMultiArray(shape: [totalSize as NSNumber], dataType: .float32)
        
        // Fill vocab logits with values 0-1024
        for i in 0..<vocabSize {
            logits[i] = NSNumber(value: Float(i))
        }
        
        // Fill duration logits with values 1000-1004
        for i in 0..<durationCount {
            logits[vocabSize + i] = NSNumber(value: Float(1000 + i))
        }
        
        let (tokenLogits, durationLogits) = try decoder.splitLogits(logits)
        
        XCTAssertEqual(tokenLogits.count, vocabSize)
        XCTAssertEqual(durationLogits.count, durationCount)
        
        // Verify token logits
        XCTAssertEqual(tokenLogits[0], 0.0, accuracy: 0.0001)
        XCTAssertEqual(tokenLogits[1024], 1024.0, accuracy: 0.0001)
        
        // Verify duration logits
        XCTAssertEqual(durationLogits[0], 1000.0, accuracy: 0.0001)
        XCTAssertEqual(durationLogits[4], 1004.0, accuracy: 0.0001)
    }
    
    func SplitLogitsEdgeCases() throws {
        // Test with minimum size
        let minLogits = try MLMultiArray(shape: [5], dataType: .float32)
        for i in 0..<5 {
            minLogits[i] = NSNumber(value: Float(i))
        }
        
        let (tokenLogits, durationLogits) = try decoder.splitLogits(minLogits)
        XCTAssertEqual(tokenLogits.count, 0)
        XCTAssertEqual(durationLogits.count, 5)
        
        // Test error case with too small array
        let tooSmall = try MLMultiArray(shape: [3], dataType: .float32)
        XCTAssertThrowsError(try decoder.splitLogits(tooSmall))
    }
    
    // MARK: - Argmax Tests
    
    func Argmax() {
        // Test normal case
        let values: [Float] = [0.1, 0.5, 0.9, 0.3, 0.7]
        let maxIndex = decoder.argmax(values)
        XCTAssertEqual(maxIndex, 2)
        
        // Test with negative values
        let negativeValues: [Float] = [-0.5, -0.1, -0.3, -0.8]
        let negMaxIndex = decoder.argmax(negativeValues)
        XCTAssertEqual(negMaxIndex, 1)
        
        // Test with all equal values
        let equalValues: [Float] = [0.5, 0.5, 0.5, 0.5]
        let equalMaxIndex = decoder.argmax(equalValues)
        XCTAssertEqual(equalMaxIndex, 0) // Should return first occurrence
        
        // Test empty array
        let emptyValues: [Float] = []
        let emptyMaxIndex = decoder.argmax(emptyValues)
        XCTAssertEqual(emptyMaxIndex, 0)
        
        // Test single element
        let singleValue: [Float] = [42.0]
        let singleMaxIndex = decoder.argmax(singleValue)
        XCTAssertEqual(singleMaxIndex, 0)
    }
    
    // MARK: - Process Duration Logits Tests
    
    func ProcessDurationLogits() throws {
        // Test with clear winner
        let logits: [Float] = [0.1, 0.2, 0.9, 0.3, 0.1]
        let (index, skip) = try decoder.processDurationLogits(logits)
        
        XCTAssertEqual(index, 2)
        XCTAssertEqual(skip, config.tdtConfig.durations[2]) // Should be 2
    }
    
    func ProcessDurationLogitsEdgeCases() throws {
        // Test with first duration selected (no skip)
        let noSkipLogits: [Float] = [0.9, 0.1, 0.1, 0.1, 0.1]
        let (index, skip) = try decoder.processDurationLogits(noSkipLogits)
        
        XCTAssertEqual(index, 0)
        XCTAssertEqual(skip, 0) // No skip
        
        // Test with last duration selected
        let maxSkipLogits: [Float] = [0.1, 0.1, 0.1, 0.1, 0.9]
        let (maxIndex, maxSkip) = try decoder.processDurationLogits(maxSkipLogits)
        
        XCTAssertEqual(maxIndex, 4)
        XCTAssertEqual(maxSkip, 4) // Maximum skip
    }
    
    // MARK: - Calculate Next Time Index Tests
    
    func CalculateNextTimeIndex() {
        // Test normal skip in long sequence
        var nextIdx = decoder.calculateNextTimeIndex(currentIdx: 5, skip: 3, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 8)
        
        // Test capped skip in long sequence
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 5, skip: 10, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 9) // Capped at 4
        
        // Test skip at sequence boundary
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 98, skip: 5, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 100) // Should not exceed sequence length
        
        // Test short sequence behavior
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 2, skip: 5, sequenceLength: 8)
        XCTAssertEqual(nextIdx, 4) // Limited to 2 for short sequences
        
        // Test zero skip
        nextIdx = decoder.calculateNextTimeIndex(currentIdx: 5, skip: 0, sequenceLength: 100)
        XCTAssertEqual(nextIdx, 5) // No movement
    }
    
    // MARK: - Prepare Decoder Input Tests
    
    func PrepareDecoderInput() throws {
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
    
    func PrepareJointInput() throws {
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
    
    func PredictTokenAndDuration() throws {
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
        XCTAssertEqual(duration, 2) // durations[2] = 2
    }
    
    // MARK: - Update Hypothesis Tests
    
    func testUpdateHypothesis() {
        var hypothesis = TdtHypothesis()
        let newState = DecoderState()
        
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