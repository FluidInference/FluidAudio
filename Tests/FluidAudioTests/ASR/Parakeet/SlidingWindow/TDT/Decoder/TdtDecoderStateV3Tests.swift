import Accelerate
@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class TdtDecoderStateV3Tests: XCTestCase {

    private let decoderStateShape: [NSNumber] = [2, 1, NSNumber(value: ASRConstants.decoderHiddenSize)]

    // MARK: - Initialization Tests

    func testDefaultInitialization() throws {
        let state = try TdtDecoderState()

        // Verify shapes
        XCTAssertEqual(state.hiddenState.shape, decoderStateShape)
        XCTAssertEqual(state.cellState.shape, decoderStateShape)
        XCTAssertEqual(state.hiddenState.dataType, .float32)
        XCTAssertEqual(state.cellState.dataType, .float32)

        // Verify initial values
        XCTAssertNil(state.lastToken)
        XCTAssertNil(state.predictorOutput)
        XCTAssertNil(state.timeJump)

        // Verify arrays are zeroed
        verifyArrayIsZero(state.hiddenState)
        verifyArrayIsZero(state.cellState)
    }

    func testCopyInitialization() throws {
        // Create original state with some data
        var originalState = try TdtDecoderState()
        originalState.lastToken = 42
        originalState.timeJump = 5

        // Fill arrays with test data
        fillArrayWithTestData(originalState.hiddenState, multiplier: 1.0)
        fillArrayWithTestData(originalState.cellState, multiplier: 2.0)

        // Create copy
        let copiedState = try TdtDecoderState(from: originalState)

        // Verify copied state
        XCTAssertEqual(copiedState.lastToken, originalState.lastToken)
        XCTAssertEqual(copiedState.timeJump, originalState.timeJump)

        // Verify arrays were copied correctly
        verifyArraysEqual(copiedState.hiddenState, originalState.hiddenState)
        verifyArraysEqual(copiedState.cellState, originalState.cellState)
    }

    // MARK: - State Management Tests

    func testReset() throws {
        var state = try TdtDecoderState()

        // Set some values
        state.lastToken = 123
        state.timeJump = -3
        fillArrayWithTestData(state.hiddenState, multiplier: 3.0)
        fillArrayWithTestData(state.cellState, multiplier: 4.0)

        // Reset state
        state.reset()

        // Verify everything is reset
        XCTAssertNil(state.lastToken)
        XCTAssertNil(state.predictorOutput)
        XCTAssertNil(state.timeJump)
        verifyArrayIsZero(state.hiddenState)
        verifyArrayIsZero(state.cellState)
    }

    func testUpdateFromDecoderOutput() throws {
        var state = try TdtDecoderState()

        // Create mock decoder output
        let newHidden = try createTestArray(shape: decoderStateShape, multiplier: 5.0)
        let newCell = try createTestArray(shape: decoderStateShape, multiplier: 6.0)

        let mockOutput = try MLDictionaryFeatureProvider(dictionary: [
            "h_out": MLFeatureValue(multiArray: newHidden),
            "c_out": MLFeatureValue(multiArray: newCell),
        ])

        // Update state
        state.update(from: mockOutput)

        // Verify arrays were updated
        verifyArraysEqual(state.hiddenState, newHidden)
        verifyArraysEqual(state.cellState, newCell)
    }

    func testUpdateFromIncompleteDecoderOutput() throws {
        var state = try TdtDecoderState()
        let originalHidden = try MLMultiArray(shape: decoderStateShape, dataType: .float32)
        let originalCell = try MLMultiArray(shape: decoderStateShape, dataType: .float32)

        // Fill with initial test data
        fillArrayWithTestData(originalHidden, multiplier: 1.0)
        fillArrayWithTestData(originalCell, multiplier: 2.0)
        state.hiddenState.copyData(from: originalHidden)
        state.cellState.copyData(from: originalCell)

        // Create output missing one state
        let newHidden = try createTestArray(shape: decoderStateShape, multiplier: 10.0)
        let mockOutput = try MLDictionaryFeatureProvider(dictionary: [
            "h_out": MLFeatureValue(multiArray: newHidden)
            // Missing c_out
        ])

        // Update state
        state.update(from: mockOutput)

        // Hidden should be updated, cell should remain unchanged
        verifyArraysEqual(state.hiddenState, newHidden)
        verifyArraysEqual(state.cellState, originalCell)
    }

    // MARK: - Token Management Tests

    func testLastTokenManagement() throws {
        var state = try TdtDecoderState()

        // Initially nil
        XCTAssertNil(state.lastToken)

        // Set token
        state.lastToken = 999
        XCTAssertEqual(state.lastToken, 999)

        // Update token
        state.lastToken = 1234
        XCTAssertEqual(state.lastToken, 1234)

        // Reset should clear it
        state.reset()
        XCTAssertNil(state.lastToken)
    }

    func testTimeJumpManagement() throws {
        var state = try TdtDecoderState()

        // Initially nil
        XCTAssertNil(state.timeJump)

        // Set positive jump
        state.timeJump = 10
        XCTAssertEqual(state.timeJump, 10)

        // Set negative jump
        state.timeJump = -5
        XCTAssertEqual(state.timeJump, -5)

        // Set zero jump
        state.timeJump = 0
        XCTAssertEqual(state.timeJump, 0)

        // Reset should clear it
        state.reset()
        XCTAssertNil(state.timeJump)
    }

    // MARK: - MLMultiArray Extension Tests

    func testMLMultiArrayResetData() throws {
        let array = try MLMultiArray(shape: [10, 5], dataType: .float32)

        // Fill with random data
        for i in 0..<array.count {
            array[i] = NSNumber(value: Float(i))
        }

        // Reset to zeros
        array.resetData(to: 0.0)
        verifyArrayIsZero(array)

        // Reset to different value
        array.resetData(to: 3.14)
        verifyArrayHasValue(array, value: 3.14)
    }

    func testMLMultiArrayResetDataNonFloat() throws {
        let array = try MLMultiArray(shape: [5, 3], dataType: .int32)

        // Fill with test data
        for i in 0..<array.count {
            array[i] = NSNumber(value: i * 2)
        }

        // Reset to zeros
        array.resetData(to: 0)

        // Verify all zeros
        for i in 0..<array.count {
            XCTAssertEqual(array[i].intValue, 0, "Array should be reset to zero at index \(i)")
        }
    }

    func testMLMultiArrayCopyData() throws {
        let sourceArray = try MLMultiArray(shape: [3, 4], dataType: .float32)
        let destArray = try MLMultiArray(shape: [3, 4], dataType: .float32)

        // Fill source with test data
        fillArrayWithTestData(sourceArray, multiplier: 7.0)

        // Copy data
        destArray.copyData(from: sourceArray)

        // Verify copy
        verifyArraysEqual(destArray, sourceArray)
    }

    func testMLMultiArrayCopyDataNonFloat() throws {
        let sourceArray = try MLMultiArray(shape: [2, 3], dataType: .int32)
        let destArray = try MLMultiArray(shape: [2, 3], dataType: .int32)

        // Fill source with test data
        for i in 0..<sourceArray.count {
            sourceArray[i] = NSNumber(value: i * 10)
        }

        // Copy data
        destArray.copyData(from: sourceArray)

        // Verify copy
        for i in 0..<sourceArray.count {
            XCTAssertEqual(
                destArray[i].intValue, sourceArray[i].intValue,
                "Arrays should be equal at index \(i)")
        }
    }

    // MARK: - Error Handling Tests

    func testInitializationDoesNotThrow() {
        XCTAssertNoThrow(try TdtDecoderState())
    }

    // MARK: - Performance Tests

    func testInitializationPerformance() {
        measure {
            for _ in 0..<100 {
                do {
                    _ = try TdtDecoderState()
                } catch {
                    XCTFail("Initialization failed: \(error)")
                }
            }
        }
    }

    func testResetPerformance() throws {
        var state = try TdtDecoderState()

        measure {
            for _ in 0..<1000 {
                state.reset()
            }
        }
    }

    func testCopyPerformance() throws {
        let originalState = try TdtDecoderState()
        fillArrayWithTestData(originalState.hiddenState, multiplier: 1.0)
        fillArrayWithTestData(originalState.cellState, multiplier: 2.0)

        measure {
            for _ in 0..<100 {
                do {
                    _ = try TdtDecoderState(from: originalState)
                } catch {
                    XCTFail("Copy failed: \(error)")
                }
            }
        }
    }

    func testArrayResetPerformance() throws {
        let array = try MLMultiArray(shape: decoderStateShape, dataType: .float32)

        measure {
            for _ in 0..<1000 {
                array.resetData(to: 0.0)
            }
        }
    }

    func testArrayCopyPerformance() throws {
        let sourceArray = try MLMultiArray(shape: decoderStateShape, dataType: .float32)
        let destArray = try MLMultiArray(shape: decoderStateShape, dataType: .float32)

        measure {
            for _ in 0..<1000 {
                destArray.copyData(from: sourceArray)
            }
        }
    }

    // MARK: - Memory Tests

    func testLargeStateManagement() throws {
        // Test with multiple states to ensure no memory leaks
        var states: [TdtDecoderState] = []

        for i in 0..<50 {
            var state = try TdtDecoderState()
            state.lastToken = i
            state.timeJump = i * 2
            states.append(state)
        }

        // Verify all states are independent
        for (index, state) in states.enumerated() {
            XCTAssertEqual(state.lastToken, index)
            XCTAssertEqual(state.timeJump, index * 2)
        }
    }

    // MARK: - Helper Methods

    private func verifyArrayIsZero(_ array: MLMultiArray) {
        for i in 0..<array.count {
            XCTAssertEqual(
                array[i].floatValue, 0.0, accuracy: 0.0001,
                "Array should be zero at index \(i)")
        }
    }

    private func verifyArrayHasValue(_ array: MLMultiArray, value: Float) {
        for i in 0..<array.count {
            XCTAssertEqual(
                array[i].floatValue, value, accuracy: 0.0001,
                "Array should have value \(value) at index \(i)")
        }
    }

    private func verifyArraysEqual(_ array1: MLMultiArray, _ array2: MLMultiArray) {
        XCTAssertEqual(array1.count, array2.count, "Arrays should have same count")

        for i in 0..<array1.count {
            XCTAssertEqual(
                array1[i].floatValue, array2[i].floatValue, accuracy: 0.0001,
                "Arrays should be equal at index \(i)")
        }
    }

    private func fillArrayWithTestData(_ array: MLMultiArray, multiplier: Float) {
        for i in 0..<array.count {
            array[i] = NSNumber(value: Float(i) * multiplier)
        }
    }

    private func createTestArray(shape: [NSNumber], multiplier: Float) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        fillArrayWithTestData(array, multiplier: multiplier)
        return array
    }
}
