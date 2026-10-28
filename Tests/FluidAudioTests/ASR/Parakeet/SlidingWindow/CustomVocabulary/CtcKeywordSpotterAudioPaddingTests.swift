import CoreML
import XCTest

@testable import FluidAudio

/// `CtcKeywordSpotter.makePaddedAudioArray` must zero the tail past the copied clip.
/// `MLMultiArray(shape:dataType:)` does not, and the mel model reads that tail.
final class CtcKeywordSpotterAudioPaddingTests: XCTestCase {

    // MARK: - Short clip

    func testShortClipFloat32ZerosTheTail() throws {
        let maxSamples = 32
        let samples: [Float] = [0.25, -3, 2]
        XCTAssertTrue(samples[0].isFinite && samples[0] != 0)

        // Fresh allocations: a recycled dirty page shows up when the tail is left uninitialized.
        for _ in 0..<60 {
            for rank in [1, 2] {
                try assertPaddedAudio(samples, maxSamples: maxSamples, rank: rank, dataType: .float32)
            }
        }
    }

    /// Same contract for `.float16`. Values are exactly representable, and reads use
    /// `NSNumber.floatValue` only (macOS x86_64 has no Swift `Float16`).
    func testShortClipFloat16ZerosTheTail() throws {
        let maxSamples = 24
        let samples: [Float] = [1, 0.5, -1.25]
        XCTAssertTrue(samples[0].isFinite && samples[0] != 0)

        for _ in 0..<60 {
            for rank in [1, 2] {
                try assertPaddedAudio(samples, maxSamples: maxSamples, rank: rank, dataType: .float16)
            }
        }
    }

    // MARK: - Window boundaries

    func testInputMatchingWindowHasNoPadding() throws {
        let maxSamples = 16
        let samples = (0..<maxSamples).map { Float($0) + 0.5 }
        for rank in [1, 2] {
            try assertPaddedAudio(samples, maxSamples: maxSamples, rank: rank, dataType: .float32)
        }
    }

    func testInputLongerThanWindowKeepsLeadingPrefix() throws {
        let maxSamples = 16
        let samples = (0..<(maxSamples + 8)).map { Float($0) + 0.5 }
        for rank in [1, 2] {
            try assertPaddedAudio(samples, maxSamples: maxSamples, rank: rank, dataType: .float32)
        }
    }

    func testEmptyInputIsAllZeros() throws {
        let maxSamples = 20
        for rank in [1, 2] {
            try assertPaddedAudio([], maxSamples: maxSamples, rank: rank, dataType: .float32)
        }
    }

    // MARK: - Contract

    private func assertPaddedAudio(
        _ samples: [Float],
        maxSamples: Int,
        rank: Int,
        dataType: MLMultiArrayDataType,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let (array, clampedCount) = try CtcKeywordSpotter.makePaddedAudioArray(
            samples,
            maxSamples: maxSamples,
            rank: rank,
            dataType: dataType
        )
        let copiedCount = min(samples.count, maxSamples)

        XCTAssertEqual(clampedCount, copiedCount, file: file, line: line)
        XCTAssertEqual(array.count, maxSamples, file: file, line: line)
        XCTAssertEqual(array.dataType, dataType, file: file, line: line)
        let expectedShape = rank == 2 ? [1, maxSamples] : [maxSamples]
        XCTAssertEqual(array.shape.map(\.intValue), expectedShape, file: file, line: line)

        for index in 0..<array.count {
            let value = array[index].floatValue
            if index < clampedCount {
                XCTAssertEqual(value, samples[index], file: file, line: line)
                XCTAssertTrue(value.isFinite, file: file, line: line)
            } else {
                XCTAssertTrue(value.isFinite, "padding \(index) should be finite", file: file, line: line)
                XCTAssertEqual(value, Float(0), "padding \(index) should be 0", file: file, line: line)
            }
        }
    }
}
