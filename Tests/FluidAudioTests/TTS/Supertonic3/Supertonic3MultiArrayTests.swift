import CoreML
import XCTest

@testable import FluidAudio

final class Supertonic3MultiArrayTests: XCTestCase {

    // MARK: - Float32 round-trip

    func testMakeFloat32RoundTripsValuesAndShape() throws {
        let values: [Float] = [1, 2, 3, 4, 5, 6]
        let arr = try Supertonic3MultiArray.makeFloat32(values, shape: [1, 2, 3])
        XCTAssertEqual(arr.dataType, .float32)
        XCTAssertEqual(arr.shape.map { $0.intValue }, [1, 2, 3])
        XCTAssertEqual(Supertonic3MultiArray.extractFloats(arr), values)
    }

    // MARK: - Int32 round-trip

    func testMakeInt32RoundTripsValuesAndShape() throws {
        let values: [Int32] = [10, 20, 30, 40]
        let arr = try Supertonic3MultiArray.makeInt32(values, shape: [4])
        XCTAssertEqual(arr.dataType, .int32)
        XCTAssertEqual(arr.shape.map { $0.intValue }, [4])
        XCTAssertEqual(Supertonic3MultiArray.extractFloats(arr), [10, 20, 30, 40])
    }

    // MARK: - Shape mismatch

    func testMakeFloat32ThrowsOnShapeMismatch() {
        XCTAssertThrowsError(
            try Supertonic3MultiArray.makeFloat32([1, 2, 3], shape: [2, 2])
        ) { err in
            guard case Supertonic3Error.invalidTensorShape(let stage, _, _) = err else {
                XCTFail("expected invalidTensorShape, got \(err)")
                return
            }
            XCTAssertEqual(stage, "makeFloat32")
        }
    }

    func testMakeInt32ThrowsOnShapeMismatch() {
        XCTAssertThrowsError(
            try Supertonic3MultiArray.makeInt32([1, 2], shape: [3])
        ) { err in
            guard case Supertonic3Error.invalidTensorShape(let stage, _, _) = err else {
                XCTFail("expected invalidTensorShape, got \(err)")
                return
            }
            XCTAssertEqual(stage, "makeInt32")
        }
    }

    // MARK: - extractFloats coverage

    func testExtractFloatsFromInt32BackingStore() throws {
        let arr = try MLMultiArray(shape: [3], dataType: .int32)
        let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: 3)
        dst[0] = -1
        dst[1] = 0
        dst[2] = 7
        XCTAssertEqual(Supertonic3MultiArray.extractFloats(arr), [-1, 0, 7])
    }

    func testExtractFloatsFromDoubleBackingStore() throws {
        let arr = try MLMultiArray(shape: [3], dataType: .double)
        let dst = arr.dataPointer.bindMemory(to: Double.self, capacity: 3)
        dst[0] = 1.5
        dst[1] = -2.5
        dst[2] = 0
        XCTAssertEqual(Supertonic3MultiArray.extractFloats(arr), [1.5, -2.5, 0])
    }

    #if arch(arm64)
    func testExtractFloatsFromFloat16BackingStore() throws {
        let arr = try MLMultiArray(shape: [3], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: Float16.self, capacity: 3)
        dst[0] = Float16(1.0)
        dst[1] = Float16(-0.5)
        dst[2] = Float16(2.0)
        let out = Supertonic3MultiArray.extractFloats(arr)
        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(out[0], 1.0, accuracy: 1e-3)
        XCTAssertEqual(out[1], -0.5, accuracy: 1e-3)
        XCTAssertEqual(out[2], 2.0, accuracy: 1e-3)
    }
    #endif
}
