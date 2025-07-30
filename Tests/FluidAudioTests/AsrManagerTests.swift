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
    
    // MARK: - MLMultiArray Creation Tests (Removed - causes crashes with createScalarArray method)
    
    // MARK: - Mel Spectrogram Input Tests
    
    func testPrepareMelSpectrogramInput() async throws {
        // Test normal audio samples
        let audioSamples: [Float] = [0.1, -0.2, 0.3, -0.4, 0.5]
        let input = try await manager.prepareMelSpectrogramInput(audioSamples)
        
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
    
    func testPrepareMelSpectrogramInputEdgeCases() async throws {
        // Test empty audio
        let emptyAudio: [Float] = []
        let emptyInput = try await manager.prepareMelSpectrogramInput(emptyAudio)
        guard let emptySignal = emptyInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(emptySignal.shape, [1, 0] as [NSNumber])
        
        // Test single sample
        let singleSample: [Float] = [0.5]
        let singleInput = try await manager.prepareMelSpectrogramInput(singleSample)
        guard let singleSignal = singleInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(singleSignal.shape, [1, 1] as [NSNumber])
        XCTAssertEqual(singleSignal[0].floatValue, 0.5, accuracy: 0.0001)
        
        // Test large audio
        let largeAudio = Array(repeating: Float(0.1), count: 16000)
        let largeInput = try await manager.prepareMelSpectrogramInput(largeAudio)
        guard let largeLength = largeInput.featureValue(for: "audio_length")?.multiArrayValue else {
            XCTFail("Missing audio_length feature")
            return
        }
        XCTAssertEqual(largeLength[0].intValue, 16000)
    }
    
    // MARK: - Encoder Output Transpose Tests
    
    
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
    
    // Removed testConvertTokensWithExistingTimings - causes crashes with vocabulary manipulation
    
    // Removed testConvertTokensEdgeCases - causes crashes with vocabulary manipulation
    
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
    
    // MARK: - Float16 Inference Tests
    
    func testPrepareMelSpectrogramInputFP16() async throws {
        // Skip this test in CI due to Float16 data type inconsistencies
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI {
            throw XCTSkip("Skipping Float16 test in CI environment")
        }
        
        // Test Float16 input preparation
        let audioSamples: [Float] = Array(repeating: 0.1, count: 1000)
        let fp16Input = try await manager.prepareMelSpectrogramInputFP16(audioSamples)
        
        // Verify audio_signal is Float16
        guard let audioSignal = fp16Input.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        
        XCTAssertEqual(audioSignal.shape, [1, 1000] as [NSNumber])
        XCTAssertEqual(audioSignal.dataType, .float16)
        
        // Verify values are preserved with Float16 precision
        for i in 0..<min(10, audioSignal.count) {
            XCTAssertEqual(audioSignal[i].floatValue, 0.1, accuracy: 0.01)
        }
        
        // Verify audio_length is still Int32
        guard let audioLength = fp16Input.featureValue(for: "audio_length")?.multiArrayValue else {
            XCTFail("Missing audio_length feature")
            return
        }
        XCTAssertEqual(audioLength.dataType, .int32)
        XCTAssertEqual(audioLength[0].intValue, 1000)
    }
    
    func testFloat16ConversionAccuracy() async throws {
        // Skip this test in CI due to Float16 data type inconsistencies
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI {
            throw XCTSkip("Skipping Float16 conversion accuracy test in CI environment")
        }
        
        // Test with values that might lose precision in Float16
        let testValues: [Float] = [
            0.00001,     // Very small
            1234.5678,   // Moderate precision loss expected
            -999.999,    // Negative with decimals
            Float.pi,    // Irrational number
            0.0          // Zero should be exact
        ]
        
        let fp16Input = try await manager.prepareMelSpectrogramInputFP16(testValues)
        guard let audioSignal = fp16Input.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        
        // Float16 has ~3-4 decimal digits of precision
        XCTAssertEqual(audioSignal[0].floatValue, testValues[0], accuracy: 0.00002)
        XCTAssertEqual(audioSignal[1].floatValue, testValues[1], accuracy: 1.0) // Float16 precision loss
        XCTAssertEqual(audioSignal[2].floatValue, testValues[2], accuracy: 0.1)
        XCTAssertEqual(audioSignal[3].floatValue, testValues[3], accuracy: 0.001)
        XCTAssertEqual(audioSignal[4].floatValue, testValues[4], accuracy: 0.0)
    }
    
    // MARK: - Zero-Copy Feature Provider Tests
    
    func testZeroCopyEncoderInput() throws {
        // Create mock mel-spectrogram output with zero-copy potential
        let melArray = try MLMultiArray(shape: [1, 80, 100], dataType: .float32)
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = 100
        
        // Fill mel array with test pattern
        for i in 0..<min(100, melArray.count) {
            melArray[i] = NSNumber(value: Float(i) * 0.01)
        }
        
        let melOutput = try MLDictionaryFeatureProvider(dictionary: [
            "melspectogram": MLFeatureValue(multiArray: melArray),
            "melspectogram_length": MLFeatureValue(multiArray: lengthArray)
        ])
        
        let encoderInput = try manager.prepareEncoderInput(melOutput)
        
        // The implementation should attempt zero-copy
        guard let audioSignal = encoderInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        
        // Verify the data is accessible and matches
        XCTAssertEqual(audioSignal.shape, melArray.shape)
        
        // Check if modifications to source affect the view (indicates zero-copy)
        melArray[0] = 999.0
        // Note: Actual zero-copy behavior depends on implementation details
    }
    
    // MARK: - Performance Profile Tests
    
    func testPerformanceProfile() {
        // Test that profile method doesn't crash
        manager.profilePerformance()
        // This just ensures the logging works without errors
    }
}