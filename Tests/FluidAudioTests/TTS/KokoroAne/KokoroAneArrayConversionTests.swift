import CoreML
import XCTest

@testable import FluidAudio

final class KokoroAneArrayConversionTests: XCTestCase {

    func testReadFloatsUsesLogicalOrderForStridedArrays() throws {
        var storage: [Float] = [
            1, 2, 3, -99,
            4, 5, 6, -99,
        ]

        try storage.withUnsafeMutableBufferPointer { buffer in
            let source = try makeStridedFloat32Array(buffer: buffer)

            XCTAssertEqual(KokoroAneArrays.readFloats(source), [1, 2, 3, 4, 5, 6])
        }
    }

    func testFloat32CopyUsesLogicalOrderForStridedArrays() throws {
        var storage: [Float] = [
            1, 2, 3, -99,
            4, 5, 6, -99,
        ]

        try storage.withUnsafeMutableBufferPointer { buffer in
            let source = try makeStridedFloat32Array(buffer: buffer)
            let copied = try KokoroAneArrays.float32Array(shape: [2, 3], from: source)

            XCTAssertEqual(KokoroAneArrays.readFloats(copied), [1, 2, 3, 4, 5, 6])
        }
    }

    func testFloat16CopyUsesLogicalOrderForStridedArrays() throws {
        var storage: [Float] = [
            1, 2, 3, -99,
            4, 5, 6, -99,
        ]

        try storage.withUnsafeMutableBufferPointer { buffer in
            let source = try makeStridedFloat32Array(buffer: buffer)
            let copied = try KokoroAneArrays.float16Array(shape: [2, 3], from: source)

            XCTAssertEqual(KokoroAneArrays.readFloats(copied), [1, 2, 3, 4, 5, 6])
        }
    }

    private func makeStridedFloat32Array(buffer: UnsafeMutableBufferPointer<Float>) throws -> MLMultiArray {
        try MLMultiArray(
            dataPointer: buffer.baseAddress!,
            shape: [2, 3],
            dataType: .float32,
            strides: [4, 1],
            deallocator: { _ in })
    }
}
