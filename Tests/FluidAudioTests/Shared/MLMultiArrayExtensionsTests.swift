@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class MLMultiArrayExtensionsTests: XCTestCase {

    func testResetClearsEveryElementOfPaddedStrides() throws {
        // An aligned [10, 10] array pads its rows to 16 elements, so the storage holds 160 slots
        // for 100 elements. A fill that walks `count` contiguous slots misses the last rows.
        let array = try ANEMemoryUtils.createAlignedArray(shape: [10, 10], dataType: .float32)
        for i in 0..<array.count {
            array[i] = NSNumber(value: Float(i) + 1)
        }

        array.reset(to: 0)

        for i in 0..<array.count {
            XCTAssertEqual(array[i].floatValue, 0, "Element \(i) should be zero")
        }
    }
}
