import Foundation
import CoreML
import XCTest
@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class AsrManagerTests: XCTestCase {
    
    var manager: AsrManager!
    
    override func setUp() {
        super.setUp()
        manager = AsrManager()
    }
    
    override func tearDown() {
        manager = nil
        super.tearDown()
    }
    
    // MARK: - MLMultiArray Creation Tests
    
    func testCreateScalarArray() throws {
        // Test Int32 scalar array
        let intArray = try manager.createScalarArray(value: 42)
        XCTAssertEqual(intArray.shape, [1] as [NSNumber])
        XCTAssertEqual(intArray.dataType, .int32)
        XCTAssertEqual(intArray[0].intValue, 42)
        
        // Test with custom shape
        let customShape = try manager.createScalarArray(value: 100, shape: [1, 1])
        XCTAssertEqual(customShape.shape, [1, 1] as [NSNumber])
        XCTAssertEqual(customShape[0].intValue, 100)
        
        // Test boundary values
        let maxInt = try manager.createScalarArray(value: Int(Int32.max))
        XCTAssertEqual(maxInt[0].intValue, Int(Int32.max))
        
        let zero = try manager.createScalarArray(value: 0)
        XCTAssertEqual(zero[0].intValue, 0)
        
        let negative = try manager.createScalarArray(value: -100)
        XCTAssertEqual(negative[0].intValue, -100)
    }
    
    // MARK: - Mel Spectrogram Input Tests
    
    func testPrepareMelSpectrogramInput() throws {
        // Test normal audio samples
        let audioSamples: [Float] = [0.1, -0.2, 0.3, -0.4, 0.5]
        let input = try manager.prepareMelSpectrogramInput(audioSamples)
        
        // Verify audio_signal feature
        guard let audioSignal = input.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(audioSignal.shape, [1, 5] as [NSNumber])
        XCTAssertEqual(audioSignal.dataType, .float32)
        
        // Verify values
        for i in 0..<audioSamples.count {
            XCTAssertEqual(audioSignal[i].floatValue, audioSamples[i], accuracy: 0.0001)
        }
        
        // Verify audio_length feature
        guard let audioLength = input.featureValue(for: "audio_length")?.multiArrayValue else {
            XCTFail("Missing audio_length feature")
            return
        }
        XCTAssertEqual(audioLength.shape, [1] as [NSNumber])
        XCTAssertEqual(audioLength[0].intValue, 5)
    }
    
    func testPrepareMelSpectrogramInputEdgeCases() throws {
        // Test empty audio
        let emptyAudio: [Float] = []
        let emptyInput = try manager.prepareMelSpectrogramInput(emptyAudio)
        guard let emptySignal = emptyInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(emptySignal.shape, [1, 0] as [NSNumber])
        
        // Test single sample
        let singleSample: [Float] = [0.5]
        let singleInput = try manager.prepareMelSpectrogramInput(singleSample)
        guard let singleSignal = singleInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(singleSignal.shape, [1, 1] as [NSNumber])
        XCTAssertEqual(singleSignal[0].floatValue, 0.5, accuracy: 0.0001)
        
        // Test large audio
        let largeAudio = Array(repeating: Float(0.1), count: 16000)
        let largeInput = try manager.prepareMelSpectrogramInput(largeAudio)
        guard let largeLength = largeInput.featureValue(for: "audio_length")?.multiArrayValue else {
            XCTFail("Missing audio_length feature")
            return
        }
        XCTAssertEqual(largeLength[0].intValue, 16000)
    }
    
    // MARK: - Encoder Output Transpose Tests
    
    func testTransposeEncoderOutput() throws {
        // Create a test encoder output: [batch=1, features=3, time=4]
        let encoderOutput = try MLMultiArray(shape: [1, 3, 4], dataType: .float32)
        
        // Fill with test data: feature * 10 + time
        for f in 0..<3 {
            for t in 0..<4 {
                let index = f * 4 + t
                encoderOutput[index] = NSNumber(value: Float(f * 10 + t))
            }
        }
        
        // Transpose to [batch=1, time=4, features=3]
        let transposed = try manager.transposeEncoderOutput(encoderOutput)
        
        XCTAssertEqual(transposed.shape, [1, 4, 3] as [NSNumber])
        
        // Verify transposition
        for t in 0..<4 {
            for f in 0..<3 {
                let expectedValue = Float(f * 10 + t)
                let actualValue = transposed[t * 3 + f].floatValue
                XCTAssertEqual(actualValue, expectedValue, accuracy: 0.0001,
                              "Mismatch at time=\(t), feature=\(f)")
            }
        }
    }
    
    func testTransposeEncoderOutputLargeArray() throws {
        // Test with larger dimensions
        let encoderOutput = try MLMultiArray(shape: [1, 256, 100], dataType: .float32)
        
        // Fill with sequential values
        for i in 0..<encoderOutput.count {
            encoderOutput[i] = NSNumber(value: Float(i))
        }
        
        let transposed = try manager.transposeEncoderOutput(encoderOutput)
        XCTAssertEqual(transposed.shape, [1, 100, 256] as [NSNumber])
        
        // Spot check a few values
        // Original: feature=0, time=0 should be at index 0
        // Transposed: time=0, feature=0 should be at index 0
        XCTAssertEqual(transposed[0].floatValue, 0.0, accuracy: 0.0001)
        
        // Original: feature=1, time=0 should be at index 100
        // Transposed: time=0, feature=1 should be at index 1
        XCTAssertEqual(transposed[1].floatValue, 100.0, accuracy: 0.0001)
    }
    
    // MARK: - Decoder Input Preparation Tests
    
    func testPrepareDecoderInput() throws {
        let targetToken = 42
        let hiddenState = try MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        let cellState = try MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        
        // Fill states with test values
        for i in 0..<hiddenState.count {
            hiddenState[i] = NSNumber(value: Float(i) * 0.01)
            cellState[i] = NSNumber(value: Float(i) * 0.02)
        }
        
        let input = try manager.prepareDecoderInput(
            targetToken: targetToken,
            hiddenState: hiddenState,
            cellState: cellState
        )
        
        // Verify targets
        guard let targets = input.featureValue(for: "targets")?.multiArrayValue else {
            XCTFail("Missing targets feature")
            return
        }
        XCTAssertEqual(targets.shape, [1, 1] as [NSNumber])
        XCTAssertEqual(targets[0].intValue, 42)
        
        // Verify target_lengths
        guard let targetLengths = input.featureValue(for: "target_lengths")?.multiArrayValue else {
            XCTFail("Missing target_lengths feature")
            return
        }
        XCTAssertEqual(targetLengths.shape, [1] as [NSNumber])
        XCTAssertEqual(targetLengths[0].intValue, 1)
        
        // Verify states are passed through
        XCTAssertNotNil(input.featureValue(for: "h_in")?.multiArrayValue)
        XCTAssertNotNil(input.featureValue(for: "c_in")?.multiArrayValue)
    }
    
    // MARK: - Feature Extraction Tests
    
    func testExtractFeatureValue() throws {
        // Create mock feature provider
        let mockArray = try MLMultiArray(shape: [1, 5], dataType: .float32)
        let featureValue = MLFeatureValue(multiArray: mockArray)
        let mockProvider = try MLDictionaryFeatureProvider(dictionary: ["test_feature": featureValue])
        
        // Test successful extraction
        let extracted = try manager.extractFeatureValue(
            from: mockProvider,
            key: "test_feature",
            errorMessage: "Test error"
        )
        XCTAssertEqual(extracted.shape, mockArray.shape)
        
        // Test missing key
        XCTAssertThrowsError(
            try manager.extractFeatureValue(
                from: mockProvider,
                key: "missing_key",
                errorMessage: "Key not found"
            )
        ) { error in
            guard case ASRError.processingFailed(let message) = error else {
                XCTFail("Expected processingFailed error")
                return
            }
            XCTAssertEqual(message, "Key not found")
        }
    }
    
    func testExtractFeatureValues() throws {
        // Create mock feature provider with multiple features
        let array1 = try MLMultiArray(shape: [1, 3], dataType: .float32)
        let array2 = try MLMultiArray(shape: [2, 4], dataType: .float32)
        
        let mockProvider = try MLDictionaryFeatureProvider(dictionary: [
            "feature1": MLFeatureValue(multiArray: array1),
            "feature2": MLFeatureValue(multiArray: array2)
        ])
        
        let keys: [(key: String, errorSuffix: String)] = [
            ("feature1", "feature 1"),
            ("feature2", "feature 2")
        ]
        
        let results = try manager.extractFeatureValues(from: mockProvider, keys: keys)
        
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results["feature1"]?.shape, [1, 3] as [NSNumber])
        XCTAssertEqual(results["feature2"]?.shape, [2, 4] as [NSNumber])
    }
    
    // MARK: - Token Conversion Tests
    
    func testConvertTokensWithExistingTimings() {
        // Test with mock vocabulary
        let mockVocab: [Int: String] = [
            0: "▁hello",
            1: "▁world",
            2: "▁",
            3: "",
            4: "▁test",
            5: "ing"
        ]
        
        // Temporarily replace vocabulary
        #if DEBUG
        manager.setVocabularyForTesting(mockVocab)
        #else
        manager.vocabulary = mockVocab
        #endif
        
        let tokenIds = [0, 1, 2, 4, 5]
        let timings = [
            TokenTiming(token: "hello", tokenId: 0, startTime: 0.0, endTime: 0.5, confidence: 0.9),
            TokenTiming(token: "world", tokenId: 1, startTime: 0.5, endTime: 1.0, confidence: 0.8),
            TokenTiming(token: "", tokenId: 2, startTime: 1.0, endTime: 1.1, confidence: 0.5),
            TokenTiming(token: "test", tokenId: 4, startTime: 1.1, endTime: 1.5, confidence: 0.9),
            TokenTiming(token: "ing", tokenId: 5, startTime: 1.5, endTime: 2.0, confidence: 0.85)
        ]
        
        let (text, adjustedTimings) = manager.convertTokensWithExistingTimings(tokenIds, timings: timings)
        
        XCTAssertEqual(text, "hello world testing")
        XCTAssertEqual(adjustedTimings.count, 4) // Empty token should be filtered
        XCTAssertEqual(adjustedTimings[0].token, "hello")
        XCTAssertEqual(adjustedTimings[1].token, "world")
        XCTAssertEqual(adjustedTimings[2].token, "test")
        XCTAssertEqual(adjustedTimings[3].token, "ing")
    }
    
    func testConvertTokensEdgeCases() {
        // Test empty tokens
        let (emptyText, emptyTimings) = manager.convertTokensWithExistingTimings([], timings: [])
        XCTAssertEqual(emptyText, "")
        XCTAssertEqual(emptyTimings.count, 0)
        
        // Test timing array handling with mismatched lengths
        let testVocab: [Int: String] = [
            100: "test",
            101: "▁word"
        ]
        #if DEBUG
        manager.setVocabularyForTesting(testVocab)
        #else
        manager.vocabulary = testVocab
        #endif
        
        // More tokens than timings
        let (text1, timings1) = manager.convertTokensWithExistingTimings([100, 101], timings: [
            TokenTiming(token: "test", tokenId: 100, startTime: 0.0, endTime: 1.0, confidence: 0.9)
        ])
        XCTAssertEqual(text1, "test word")
        XCTAssertEqual(timings1.count, 1) // Only one timing available
        
        // Test with unknown token IDs (not in vocabulary)
        let (text2, timings2) = manager.convertTokensWithExistingTimings([999, 1000], timings: [])
        XCTAssertEqual(text2, "") // Unknown tokens produce empty string
        XCTAssertEqual(timings2.count, 0)
    }
    
    // MARK: - Encoder Input Preparation Tests
    
    func testPrepareEncoderInput() throws {
        // Create mock mel-spectrogram output
        let melArray = try MLMultiArray(shape: [1, 80, 100], dataType: .float32)
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: 100)
        
        let melOutput = try MLDictionaryFeatureProvider(dictionary: [
            "melspectogram": MLFeatureValue(multiArray: melArray),
            "melspectogram_length": MLFeatureValue(multiArray: lengthArray)
        ])
        
        let encoderInput = try manager.prepareEncoderInput(melOutput)
        
        // Verify audio_signal is passed through
        guard let audioSignal = encoderInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(audioSignal.shape, melArray.shape)
        
        // Verify length is passed through
        guard let length = encoderInput.featureValue(for: "length")?.multiArrayValue else {
            XCTFail("Missing length feature")
            return
        }
        XCTAssertEqual(length[0].intValue, 100)
    }
}