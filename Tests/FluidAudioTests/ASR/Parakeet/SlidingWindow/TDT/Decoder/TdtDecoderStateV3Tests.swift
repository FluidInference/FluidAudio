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

    func testMLMultiArrayResetDataInt32Value() throws {
        let array = try MLMultiArray(shape: [4, 6], dataType: .int32)

        array.resetData(to: 7)

        verifyArrayHasValue(array, value: 7)
    }

    func testMLMultiArrayResetDataFloat64Value() throws {
        let array = try MLMultiArray(shape: [3, 4], dataType: .float64)

        array.resetData(to: 2.25)
        verifyArrayHasValue(array, value: 2.25)

        array.resetData(to: 0)
        verifyArrayIsZero(array)
    }

    func testMLMultiArrayResetDataFloat16Value() throws {
        let array = try MLMultiArray(shape: [3, 5], dataType: .float16)

        array.resetData(to: 1.5)
        verifyArrayHasValue(array, value: 1.5)

        array.resetData(to: 0)
        verifyArrayIsZero(array)
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

    func testMLMultiArrayResetDataClearsEveryElementOfPaddedStrides() throws {
        // Aligned arrays are zero-cleared on allocation, so the storage is poisoned first;
        // otherwise a fill that skips elements would still look clean.
        let array = try ANEMemoryUtils.createAlignedArray(shape: [10, 10], dataType: .float32)
        array.withUnsafeMutableBytes { bytes, _ in
            bytes.bindMemory(to: Float.self).update(repeating: .nan)
        }

        array.resetData(to: 0)

        verifyArrayIsZero(array)
    }

    func testMLMultiArrayResetDataClearsTheWholeContiguousStorage() throws {
        let array = try ANEMemoryUtils.createAlignedArray(shape: [1, 64], dataType: .float32)
        array.withUnsafeMutableBytes { bytes, _ in
            bytes.bindMemory(to: Float.self).update(repeating: .nan)
        }

        array.resetData(to: 0)

        array.withUnsafeBytes { bytes in
            XCTAssertTrue(bytes.bindMemory(to: Float.self).allSatisfy { $0 == 0 }, "Every stored value should be zero")
        }
    }

    func testMLMultiArrayResetDataLargeArrayWithinBudget() throws {
        // A per-element reset of 240000 samples costs tens of milliseconds; the bulk path is well
        // under a millisecond. Timed locally only: the parallel CI job shares its machine.
        try XCTSkipIf(ProcessInfo.processInfo.environment["CI"] != nil, "Timing budgets run locally only")
        let shape: [NSNumber] = [1, NSNumber(value: ASRConstants.maxModelSamples)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let clock = ContinuousClock()
        var best = Double.infinity

        for _ in 0..<5 {
            array[0] = NSNumber(value: Float(1))
            let elapsed = clock.measure {
                array.resetData(to: 0)
            }
            best = min(best, elapsed / .milliseconds(1))
        }

        XCTAssertEqual(array[0].floatValue, 0)
        XCTAssertLessThan(best, 5, "resetData took \(best) ms for \(ASRConstants.maxModelSamples) elements")
    }

    func testMLMultiArrayResetDataStaysInsideTheLogicalElements() throws {
        // A padded layout reports a byte span past its last element: shape [2, 10] with strides
        // [16, 1] ends at element 26 while the span covers 32. Storage beyond the last element can
        // belong to someone else, so the reset must not touch it.
        let elements = 32
        let storage = UnsafeMutablePointer<Float>.allocate(capacity: elements)
        defer { storage.deallocate() }
        storage.initialize(repeating: 1, count: elements)
        let sentinel: Float = 12345
        for i in 26..<elements {
            storage[i] = sentinel
        }
        let view = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(storage), shape: [2, 10], dataType: .float32,
            strides: [16, 1], deallocator: nil)

        view.resetData(to: 0)

        verifyArrayIsZero(view)
        for i in 26..<elements {
            XCTAssertEqual(storage[i], sentinel, "Storage past the last element was written at \(i)")
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

    func testMLMultiArrayCopyDataLargeArrayWithinBudget() throws {
        // The decoder state is snapshotted before every inference, so the copy must be a bulk
        // transfer. A per-element copy of 240000 samples costs tens of milliseconds. Timed locally
        // only: the parallel CI job shares its machine.
        try XCTSkipIf(ProcessInfo.processInfo.environment["CI"] != nil, "Timing budgets run locally only")
        let shape: [NSNumber] = [1, NSNumber(value: ASRConstants.maxModelSamples)]
        let sourceArray = try MLMultiArray(shape: shape, dataType: .float32)
        let destArray = try MLMultiArray(shape: shape, dataType: .float32)
        sourceArray[sourceArray.count - 1] = NSNumber(value: Float(3))
        let clock = ContinuousClock()
        var best = Double.infinity

        for _ in 0..<5 {
            let elapsed = clock.measure {
                destArray.copyData(from: sourceArray)
            }
            best = min(best, elapsed / .milliseconds(1))
        }

        XCTAssertEqual(destArray[destArray.count - 1].floatValue, 3)
        XCTAssertLessThan(best, 5, "copyData took \(best) ms for \(ASRConstants.maxModelSamples) elements")
    }

    func testMLMultiArrayCopyDataAcrossStrideLayouts() throws {
        // A plain array and an ANE-aligned array of the same shape have different strides;
        // the copy must still land every element.
        let shape: [NSNumber] = [10, 10]
        let sourceArray = try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: .float32)
        let destArray = try MLMultiArray(shape: shape, dataType: .float32)
        XCTAssertNotEqual(sourceArray.strides, destArray.strides)

        fillArrayWithTestData(sourceArray, multiplier: 1.5)

        destArray.copyData(from: sourceArray)

        verifyArraysEqual(destArray, sourceArray)
    }

    func testMLMultiArrayCopyDataBetweenOverlappingViews() throws {
        // Two zero-copy views of one allocation, offset by 16 elements, overlap on 48 of their 64
        // elements; the copy must behave as if the source were read completely first.
        let backing = try ANEMemoryUtils.createAlignedArray(shape: [1, 96], dataType: .float32)
        for i in 0..<backing.count {
            backing[i] = NSNumber(value: Float(i))
        }
        let strides: [NSNumber] = [64, 1]
        let sourceView = try ANEMemoryUtils.createZeroCopyView(
            from: backing, offset: 0, shape: [1, 64], strides: strides)
        let destinationView = try ANEMemoryUtils.createZeroCopyView(
            from: backing, offset: 16, shape: [1, 64], strides: strides)

        destinationView.copyData(from: sourceView)

        verifyArraysEqual(destinationView, try createTestArray(shape: [1, 64], multiplier: 1))
    }

    func testMLMultiArrayCopyDataFromItselfLeavesValues() throws {
        let array = try createTestArray(shape: decoderStateShape, multiplier: 0.5)
        let expected = try createTestArray(shape: decoderStateShape, multiplier: 0.5)

        array.copyData(from: array)

        verifyArraysEqual(array, expected)
    }

    func testMLMultiArrayCopyDataStaysInsideTheLogicalElements() throws {
        // Same padded layout as the reset test: the copy must land every element and leave the
        // destination's storage past its last element alone.
        let elements = 32
        let sourceStorage = UnsafeMutablePointer<Float>.allocate(capacity: elements)
        let destinationStorage = UnsafeMutablePointer<Float>.allocate(capacity: elements)
        defer {
            sourceStorage.deallocate()
            destinationStorage.deallocate()
        }
        sourceStorage.initialize(repeating: 1, count: elements)
        let sentinel: Float = 12345
        destinationStorage.initialize(repeating: sentinel, count: elements)
        let sourceView = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(sourceStorage), shape: [2, 10], dataType: .float32,
            strides: [16, 1], deallocator: nil)
        let destinationView = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(destinationStorage), shape: [2, 10], dataType: .float32,
            strides: [16, 1], deallocator: nil)
        fillArrayWithTestData(sourceView, multiplier: 3)

        destinationView.copyData(from: sourceView)

        verifyArraysEqual(destinationView, sourceView)
        for i in 26..<elements {
            XCTAssertEqual(destinationStorage[i], sentinel, "Storage past the last element was written at \(i)")
        }
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
