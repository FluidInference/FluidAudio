import CoreML
import Foundation
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

    // MARK: - Mel Encoder Input Tests

    func testPrepareMelEncoderInput() async throws {
        // Test normal audio samples
        let audioSamples: [Float] = [0.1, -0.2, 0.3, -0.4, 0.5]
        let input = try await manager.prepareMelEncoderInput(audioSamples)

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

    func testPrepareMelEncoderInputEdgeCases() async throws {
        // Test empty audio
        let emptyAudio: [Float] = []
        let emptyInput = try await manager.prepareMelEncoderInput(emptyAudio)
        guard let emptySignal = emptyInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(emptySignal.shape, [1, 0] as [NSNumber])

        // Test single sample
        let singleSample: [Float] = [0.5]
        let singleInput = try await manager.prepareMelEncoderInput(singleSample)
        guard let singleSignal = singleInput.featureValue(for: "audio_signal")?.multiArrayValue else {
            XCTFail("Missing audio_signal feature")
            return
        }
        XCTAssertEqual(singleSignal.shape, [1, 1] as [NSNumber])
        XCTAssertEqual(singleSignal[0].floatValue, 0.5, accuracy: 0.0001)

        // Test large audio
        let largeAudio = Array(repeating: Float(0.1), count: 16000)
        let largeInput = try await manager.prepareMelEncoderInput(largeAudio)
        guard let largeLength = largeInput.featureValue(for: "audio_length")?.multiArrayValue else {
            XCTFail("Missing audio_length feature")
            return
        }
        XCTAssertEqual(largeLength[0].intValue, 16000)
    }

    // MARK: - Encoder Output Transpose Tests

    // MARK: - Decoder Input Preparation Tests

    func testDecoderStateInitialization() async throws {
        // Since prepareDecoderInput is now private and requires models to be loaded,
        // we test that the manager correctly handles the not initialized state

        // Test that resetDecoderState throws notInitialized when models aren't loaded
        do {
            try await manager.resetDecoderState()
            XCTFail("Expected notInitialized error")
        } catch ASRError.notInitialized {
            // Expected behavior when models aren't loaded
            XCTAssertTrue(true, "Correctly threw notInitialized error")
        } catch {
            XCTFail("Expected notInitialized error, got: \(error)")
        }
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
            "feature2": MLFeatureValue(multiArray: array2),
        ])

        let keys: [(key: String, errorSuffix: String)] = [
            ("feature1", "feature 1"),
            ("feature2", "feature 2"),
        ]

        let results = try manager.extractFeatureValues(from: mockProvider, keys: keys)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results["feature1"]?.shape, [1, 3] as [NSNumber])
        XCTAssertEqual(results["feature2"]?.shape, [2, 4] as [NSNumber])
    }

    func testNormalizeEncoderOutputTransposesHiddenAndTime() throws {
        let hidden = ASRConstants.encoderHiddenSize
        let time = 3
        let encoderOutput = try MLMultiArray(
            shape: [1, NSNumber(value: hidden), NSNumber(value: time)],
            dataType: .float32
        )

        let srcPtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
        for idx in 0..<encoderOutput.count {
            srcPtr[idx] = Float(idx)
        }

        let normalized = try manager.normalizeEncoderOutput(encoderOutput)

        XCTAssertEqual(normalized.shape, [1, NSNumber(value: time), NSNumber(value: hidden)])

        let strides = normalized.strides.map { $0.intValue }
        func valueAt(_ b: Int, _ t: Int, _ h: Int) -> Float {
            let index = b * strides[0] + t * strides[1] + h * strides[2]
            return normalized[index].floatValue
        }

        XCTAssertEqual(valueAt(0, 0, 0), 0)
        XCTAssertEqual(valueAt(0, 0, 1), Float(time))
        XCTAssertEqual(valueAt(0, 1, 0), 1)
        XCTAssertEqual(valueAt(0, 1, 1), Float(time + 1))
    }

    func testNormalizeEncoderOutputNoOpWhenAlreadyTransposed() throws {
        let hidden = ASRConstants.encoderHiddenSize
        let time = 2
        let encoderOutput = try MLMultiArray(
            shape: [1, NSNumber(value: time), NSNumber(value: hidden)],
            dataType: .float32
        )

        let normalized = try manager.normalizeEncoderOutput(encoderOutput)

        // Identity path should return the same instance for efficiency
        XCTAssertTrue(normalized === encoderOutput)
    }

    // MARK: - Token Conversion Tests

    // Removed testConvertTokensWithExistingTimings - causes crashes with vocabulary manipulation

    // Removed testConvertTokensEdgeCases - causes crashes with vocabulary manipulation

}
