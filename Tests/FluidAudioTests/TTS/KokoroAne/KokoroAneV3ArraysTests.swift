import CoreML
import XCTest

@testable import FluidAudio

final class KokoroAneV3ArraysTests: XCTestCase {

    /// Every v3 input carries a zeroed page past its end so the OS 27 BNNS
    /// overread reads zeros instead of stale bytes (#1037: Vocoder `x_source_1`
    /// at T = 233 came back all-NaN without it).
    func testBuildersCarryZeroedTailSlack() throws {
        let shape = [1, 128, 120 * 233 + 1]
        let total = shape.reduce(1, *)
        let source = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        for i in 0..<total { source[i] = 1 }
        let arrays: [(MLMultiArray, Int)] = [
            (try KokoroAneV3Arrays.float16Array(shape: shape, from: source), 2),
            (try KokoroAneV3Arrays.float16Array(shape: shape, from: [Float](repeating: 1, count: total)), 2),
            (try KokoroAneV3Arrays.float32Array(shape: shape, from: source), 4),
            (try KokoroAneV3Arrays.float32Array(shape: shape, from: [Float](repeating: 1, count: total)), 4),
            (try KokoroAneV3Arrays.int32Array(shape: [1, total], from: [Int32](repeating: 1, count: total)), 4),
            (
                try KokoroAneV3Pipeline.resizeTime(KokoroAneV3Arrays.float16Array(shape: shape, from: source), to: 400),
                2
            ),
        ]
        for (array, elementSize) in arrays {
            let count = array.count
            XCTAssertEqual(array[0].floatValue, 1)
            let bytes = array.dataPointer.bindMemory(
                to: UInt8.self, capacity: count * elementSize + KokoroAneArrays.tailSlackBytes)
            let slack = UnsafeBufferPointer(
                start: bytes + count * elementSize, count: KokoroAneArrays.tailSlackBytes)
            XCTAssertTrue(slack.allSatisfy { $0 == 0 }, "\(array.shape) \(array.dataType)")
        }
    }

    func testResizeTimeKeepsValuesAndZeroPadsTail() throws {
        let source = try KokoroAneV3Arrays.float16Array(shape: [1, 2, 3], from: [1, 2, 3, 4, 5, 6])
        let grown = try KokoroAneV3Pipeline.resizeTime(source, to: 5)
        XCTAssertEqual(KokoroAneV3Arrays.readFloats(grown), [1, 2, 3, 0, 0, 4, 5, 6, 0, 0])
        let cropped = try KokoroAneV3Pipeline.resizeTime(source, to: 2)
        XCTAssertEqual(KokoroAneV3Arrays.readFloats(cropped), [1, 2, 4, 5])
    }
}
