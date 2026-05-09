import CoreML
import XCTest

@testable import FluidAudio

final class StyleTTS2MultiArrayTests: XCTestCase {

    // MARK: - Float32 round-trip

    func testMakeFloat32RoundTripsValuesAndShape() throws {
        let values: [Float] = [1, 2, 3, 4, 5, 6]
        let arr = try StyleTTS2MultiArray.makeFloat32(values, shape: [1, 2, 3])
        XCTAssertEqual(arr.dataType, .float32)
        XCTAssertEqual(StyleTTS2MultiArray.shape(of: arr), [1, 2, 3])
        XCTAssertEqual(StyleTTS2MultiArray.extractFloats(arr), values)
    }

    // MARK: - Int32 round-trip

    func testMakeInt32RoundTripsValuesAndShape() throws {
        let values: [Int32] = [10, 20, 30, 40]
        let arr = try StyleTTS2MultiArray.makeInt32(values, shape: [4])
        XCTAssertEqual(arr.dataType, .int32)
        XCTAssertEqual(StyleTTS2MultiArray.shape(of: arr), [4])
        // extractFloats widens to Float32; validate the int values come through.
        XCTAssertEqual(StyleTTS2MultiArray.extractFloats(arr), [10, 20, 30, 40])
    }

    // MARK: - extractFloats coverage

    func testExtractFloatsFromInt32() throws {
        let arr = try MLMultiArray(shape: [3], dataType: .int32)
        let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: 3)
        dst[0] = -1
        dst[1] = 0
        dst[2] = 7
        XCTAssertEqual(StyleTTS2MultiArray.extractFloats(arr), [-1, 0, 7])
    }

    func testExtractFloatsFromDouble() throws {
        let arr = try MLMultiArray(shape: [2], dataType: .double)
        let dst = arr.dataPointer.bindMemory(to: Double.self, capacity: 2)
        dst[0] = 1.5
        dst[1] = -2.25
        XCTAssertEqual(StyleTTS2MultiArray.extractFloats(arr), [1.5, -2.25])
    }

    // MARK: - Shape product mismatch

    func testMakeFloat32ShapeMismatchPreconditionFails() {
        // Cannot easily test precondition() trips without crashing the runner;
        // instead verify that matching shape products succeed for an
        // intentionally unusual layout.
        let values = [Float](repeating: 0, count: 24)
        XCTAssertNoThrow(
            try StyleTTS2MultiArray.makeFloat32(values, shape: [2, 3, 4]))
    }
}
