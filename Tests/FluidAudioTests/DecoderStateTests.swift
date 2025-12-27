@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class TdtDecoderStateTests: XCTestCase {

    private let decoderStateShape: [NSNumber] = [
        NSNumber(value: 2),
        NSNumber(value: 1),
        NSNumber(value: ASRConstants.decoderHiddenSize),
    ]

    // MARK: - Initialization Tests

    func testDecoderStateInitialization() throws {
        let state = try TdtDecoderState()

        // Check shapes
        XCTAssertEqual(state.hiddenState.shape, decoderStateShape)
        XCTAssertEqual(state.cellState.shape, decoderStateShape)

        // Check data types
        XCTAssertEqual(state.hiddenState.dataType, .float32)
        XCTAssertEqual(state.cellState.dataType, .float32)

        // Check all values are initialized to zero
        for i in 0..<state.hiddenState.count {
            XCTAssertEqual(
                state.hiddenState[i].floatValue, 0.0, accuracy: 0.0001,
                "Hidden state at index \(i) should be zero")
            XCTAssertEqual(
                state.cellState[i].floatValue, 0.0, accuracy: 0.0001,
                "Cell state at index \(i) should be zero")
        }
    }

    // MARK: - Copy Constructor Tests

    func testDecoderStateCopyConstructor() throws {
        // Create original state with test values
        let originalState = try TdtDecoderState()

        // Fill with test data
        for i in 0..<originalState.hiddenState.count {
            originalState.hiddenState[i] = NSNumber(value: Float(i) * 0.01)
            originalState.cellState[i] = NSNumber(value: Float(i) * 0.02)
        }

        // Create copy
        let copiedState = try TdtDecoderState(from: originalState)

        // Verify shapes match
        XCTAssertEqual(copiedState.hiddenState.shape, originalState.hiddenState.shape)
        XCTAssertEqual(copiedState.cellState.shape, originalState.cellState.shape)

        // Verify values are copied
        for i in 0..<originalState.hiddenState.count {
            XCTAssertEqual(
                copiedState.hiddenState[i].floatValue,
                originalState.hiddenState[i].floatValue,
                accuracy: 0.0001,
                "Hidden state mismatch at index \(i)")
            XCTAssertEqual(
                copiedState.cellState[i].floatValue,
                originalState.cellState[i].floatValue,
                accuracy: 0.0001,
                "Cell state mismatch at index \(i)")
        }

        // Verify it's a deep copy by modifying original
        originalState.hiddenState[0] = NSNumber(value: 999.0)
        XCTAssertNotEqual(
            copiedState.hiddenState[0].floatValue, 999.0,
            "Copy should be independent of original")
    }

    // MARK: - Update Tests

    func testDecoderStateUpdate() throws {
        var state = try TdtDecoderState()

        // Create mock decoder output
        let newHiddenState = try MLMultiArray(shape: decoderStateShape, dataType: .float32)
        let newCellState = try MLMultiArray(shape: decoderStateShape, dataType: .float32)

        // Fill with test values
        for i in 0..<newHiddenState.count {
            newHiddenState[i] = NSNumber(value: Float(i) * 0.1)
            newCellState[i] = NSNumber(value: Float(i) * 0.2)
        }

        let mockOutput = try MLDictionaryFeatureProvider(dictionary: [
            "h_out": MLFeatureValue(multiArray: newHiddenState),
            "c_out": MLFeatureValue(multiArray: newCellState),
        ])

        // Update state
        state.update(from: mockOutput)

        // Verify state now references the new arrays
        XCTAssertTrue(state.hiddenState === newHiddenState)
        XCTAssertTrue(state.cellState === newCellState)
    }

    func testDecoderStateUpdateWithMissingFeatures() throws {
        var state = try TdtDecoderState()

        // Store original references
        let originalHidden = state.hiddenState
        let originalCell = state.cellState

        // Create output with missing features
        let emptyOutput = try MLDictionaryFeatureProvider(dictionary: [:])

        // Update should retain original states when features are missing
        state.update(from: emptyOutput)

        XCTAssertTrue(state.hiddenState === originalHidden)
        XCTAssertTrue(state.cellState === originalCell)
    }

    func testDecoderStateUpdatePartialFeatures() throws {
        var state = try TdtDecoderState()

        let originalHidden = state.hiddenState

        // Create new cell state only
        let newCellState = try MLMultiArray(shape: decoderStateShape, dataType: .float32)
        for i in 0..<newCellState.count {
            newCellState[i] = NSNumber(value: Float(i) * 0.3)
        }

        let partialOutput = try MLDictionaryFeatureProvider(dictionary: [
            "c_out": MLFeatureValue(multiArray: newCellState)
        ])

        state.update(from: partialOutput)

        // Hidden state should be unchanged (same reference)
        XCTAssertTrue(state.hiddenState === originalHidden)
        // Cell state should be the new array
        XCTAssertTrue(state.cellState === newCellState)
    }

    // MARK: - MLMultiArray Extension Tests

    func testMLMultiArrayResetData() throws {
        let array = try MLMultiArray(shape: [2, 3, 4], dataType: .float32)

        // Fill with non-zero values
        for i in 0..<array.count {
            array[i] = NSNumber(value: Float(i) + 1.0)
        }

        // Reset to zero
        array.resetData(to: 0)

        // Verify all values are zero
        for i in 0..<array.count {
            XCTAssertEqual(
                array[i].floatValue, 0.0, accuracy: 0.0001,
                "Value at index \(i) should be zero")
        }

        // Reset to different value
        array.resetData(to: NSNumber(value: 5.5))

        // Verify all values are 5.5
        for i in 0..<array.count {
            XCTAssertEqual(
                array[i].floatValue, 5.5, accuracy: 0.0001,
                "Value at index \(i) should be 5.5")
        }
    }

    func testMLMultiArrayCopyData() throws {
        let source = try MLMultiArray(shape: [2, 2, 3], dataType: .float32)
        let destination = try MLMultiArray(shape: [2, 2, 3], dataType: .float32)

        // Fill source with test data
        for i in 0..<source.count {
            source[i] = NSNumber(value: Float(i) * 2.0)
        }

        // Fill destination with different data
        for i in 0..<destination.count {
            destination[i] = NSNumber(value: Float(i) * 10.0)
        }

        // Copy data
        destination.copyData(from: source)

        // Verify all values are copied
        for i in 0..<source.count {
            XCTAssertEqual(
                destination[i].floatValue, source[i].floatValue,
                accuracy: 0.0001,
                "Value at index \(i) should match source")
        }
    }

    func testMLMultiArrayCopyDataDifferentValues() throws {
        let source = try MLMultiArray(shape: [10], dataType: .float32)
        let destination = try MLMultiArray(shape: [10], dataType: .float32)

        // Test with various values including edge cases
        let testValues: [Float] = [
            0.0, -1.0, 1.0, Float.pi, -Float.pi,
            1e-6, -1e-6, 1e6, -1e6, 42.42,
        ]

        for (i, value) in testValues.enumerated() {
            source[i] = NSNumber(value: value)
        }

        destination.copyData(from: source)

        for (i, expectedValue) in testValues.enumerated() {
            XCTAssertEqual(
                destination[i].floatValue, expectedValue,
                accuracy: 0.0001,
                "Value at index \(i) should be \(expectedValue)")
        }
    }

    // MARK: - State Size Tests

    func testDecoderStateMemorySize() throws {
        let state = try TdtDecoderState()

        // Each state has shape [2, 1, decoderHiddenSize] with float32
        let expectedElements = 2 * 1 * ASRConstants.decoderHiddenSize
        XCTAssertEqual(state.hiddenState.count, expectedElements)
        XCTAssertEqual(state.cellState.count, expectedElements)

        // Total memory: 2 arrays * 1280 elements * 4 bytes = 10,240 bytes
        let totalElements = state.hiddenState.count + state.cellState.count
        XCTAssertEqual(totalElements, expectedElements * 2)
    }

    // MARK: - Reset Tests

    func testDecoderStateReset() throws {
        var state = try TdtDecoderState()

        // Set some test values
        state.hiddenState[0] = NSNumber(value: 42.0)
        state.cellState[0] = NSNumber(value: 24.0)
        state.lastToken = 123

        // Create a mock predictorOutput
        let mockPredictorOutput = try MLMultiArray(shape: [1, 1, 256], dataType: .float32)
        mockPredictorOutput[0] = NSNumber(value: 99.0)
        state.predictorOutput = mockPredictorOutput

        // Verify initial state is set
        XCTAssertEqual(state.hiddenState[0].floatValue, 42.0, accuracy: 0.0001)
        XCTAssertEqual(state.cellState[0].floatValue, 24.0, accuracy: 0.0001)
        XCTAssertEqual(state.lastToken, 123)
        XCTAssertNotNil(state.predictorOutput)
        XCTAssertEqual(state.predictorOutput?[0].floatValue ?? 0, 99.0, accuracy: 0.0001)

        // Reset the state
        state.reset()

        // Verify everything is reset
        XCTAssertEqual(state.hiddenState[0].floatValue, 0.0, accuracy: 0.0001)
        XCTAssertEqual(state.cellState[0].floatValue, 0.0, accuracy: 0.0001)
        XCTAssertNil(state.lastToken)
        XCTAssertNil(state.predictorOutput)

        // Check that all elements are zero
        for i in 0..<state.hiddenState.count {
            XCTAssertEqual(
                state.hiddenState[i].floatValue, 0.0, accuracy: 0.0001,
                "Hidden state at index \(i) should be zero after reset")
            XCTAssertEqual(
                state.cellState[i].floatValue, 0.0, accuracy: 0.0001,
                "Cell state at index \(i) should be zero after reset")
        }
    }

    // MARK: - Thread Safety Tests

    @MainActor
    func testDecoderStateConcurrentAccess() throws {
        let state = try TdtDecoderState()
        let iterations = 100
        let expectation = self.expectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = 2

        // Concurrent reads should be safe
        DispatchQueue.global().async {
            for _ in 0..<iterations {
                _ = state.hiddenState[0]
                _ = state.cellState[0]
            }
            expectation.fulfill()
        }

        DispatchQueue.global().async {
            for _ in 0..<iterations {
                _ = state.hiddenState[state.hiddenState.count - 1]
                _ = state.cellState[state.cellState.count - 1]
            }
            expectation.fulfill()
        }

        waitForExpectations(timeout: 5.0)
    }
}
