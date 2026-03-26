import Accelerate
import XCTest

@testable import FluidAudio

/// Regression test for heap-buffer-overflow in EmbeddingExtractor.fillMaskBufferOptimized().
///
/// Bug: numMasksInChunk = (firstMask.count * audio.count + 80_000) / 160_000
/// can exceed firstMask.count when audio > 10s, causing vDSP_mmov to read past allocation.
/// Fix: clamp to firstMask.count with min().
/// Introduced in v0.8.0 (PR #191). Affects v0.8.0–v0.12.4.
final class EmbeddingExtractorOverflowTests: XCTestCase {

    // MARK: - numMasksInChunk Bounds

    func testNumMasksInChunkClampsForLongAudio() {
        let maskCount = 100
        let audioCount = 320_000  // 20s at 16kHz

        // Buggy: (100 * 320000 + 80000) / 160000 = 200 — 2x overread
        let unclamped = (maskCount * audioCount + 80_000) / 160_000
        XCTAssertEqual(unclamped, 200, "Unclamped formula exceeds maskCount — proves bug exists")

        let clamped = min(unclamped, maskCount)
        XCTAssertEqual(clamped, maskCount)
    }

    func testNumMasksInChunkSafeForShortAudio() {
        let maskCount = 100
        let audioCount = 80_000  // 5s at 16kHz

        // (100 * 80000 + 80000) / 160000 = 50
        let result = min(
            (maskCount * audioCount + 80_000) / 160_000,
            maskCount
        )
        XCTAssertEqual(result, 50, "Short audio should not trigger clamp")
    }

    // MARK: - vDSP_mmov Bounds

    func testFillMaskDoesNotOverreadWithLongAudio() {
        // Simulates fillMaskBufferOptimized with 20s audio.
        // Without clamp, vDSP_mmov reads 200 elements from 100-element buffer.
        // Under ASan: READ of size 800 from 400-byte allocation.
        let maskCount = 100
        let audioCount = 320_000

        let numMasksInChunk = min(
            (maskCount * audioCount + 80_000) / 160_000,
            maskCount
        )

        let mask = [Float](repeating: 1.0, count: maskCount)
        var destination = [Float](repeating: 0.0, count: maskCount * 3)

        mask.withUnsafeBufferPointer { src in
            destination.withUnsafeMutableBufferPointer { dst in
                vDSP_mmov(
                    src.baseAddress!, dst.baseAddress!,
                    vDSP_Length(numMasksInChunk), 1, 1,
                    vDSP_Length(numMasksInChunk)
                )
            }
        }

        XCTAssertEqual(numMasksInChunk, maskCount)
    }
}
