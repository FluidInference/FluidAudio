import CoreML
import XCTest

@testable import FluidAudio

final class KokoroAneArrayConversionTests: XCTestCase {

    /// Chain inputs carry a zeroed page past their end so a BNNS CPU kernel's
    /// small overread cannot fault on an unmapped page (macOS 27, T = 512).
    func testInputArraysCarryZeroedTailSlack() throws {
        // [1, 256, 20·512] fp16 is exactly 320 × 16 KB: the shape that faulted.
        let shape = [1, 256, 20 * 512]
        let total = shape.reduce(1, *)
        let arr = try KokoroAneArrays.float16Array(shape: shape, from: [Float](repeating: 1, count: total))
        XCTAssertEqual(arr.strides.map(\.intValue), [total, 20 * 512, 1])
        let bytes = arr.dataPointer.bindMemory(to: UInt8.self, capacity: total * 2 + KokoroAneArrays.tailSlackBytes)
        XCTAssertEqual(bytes[0], 0x00)  // fp16 1.0 = 0x3C00, little-endian
        XCTAssertEqual(bytes[1], 0x3C)
        let slack = UnsafeBufferPointer(start: bytes + total * 2, count: KokoroAneArrays.tailSlackBytes)
        XCTAssertTrue(slack.allSatisfy { $0 == 0 })
    }

    func testMakeArrayIsZeroedAndRowMajor() throws {
        let arr = try KokoroAneArrays.makeArray(shape: [2, 3, 4], dataType: .float32)
        XCTAssertEqual(arr.strides.map(\.intValue), [12, 4, 1])
        XCTAssertTrue((0..<arr.count).allSatisfy { arr[$0].floatValue == 0 })
        let ids = try KokoroAneArrays.int32Array(shape: [1, 3], from: [7, 8, 9])
        XCTAssertEqual((0..<3).map { ids[$0].int32Value }, [7, 8, 9])
    }

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
