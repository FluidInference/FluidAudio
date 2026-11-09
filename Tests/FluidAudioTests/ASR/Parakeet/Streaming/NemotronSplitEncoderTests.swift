@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// The split 2240 ms encoder (#1116) keeps the single encoder's 24-layer cache
/// layout by slicing it per half and stacking the halves back.
final class NemotronSplitEncoderTests: XCTestCase {

    private func cache(layers: Int, rows: Int, width: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: layers), NSNumber(value: rows), NSNumber(value: width)], dataType: .float32)
        for i in 0..<array.count { array[i] = NSNumber(value: Float(i)) }
        return array
    }

    private func values(_ array: MLMultiArray) -> [Float] {
        (0..<array.count).map { array[$0].floatValue }
    }

    func testSliceAndStackRoundTrip() throws {
        let full = try cache(layers: 4, rows: 3, width: 5)
        let lower = try NemotronSplitEncoder.layers(full, 0..<2)
        let upper = try NemotronSplitEncoder.layers(full, 2..<4)
        XCTAssertEqual(lower.shape, [1, 2, 3, 5])
        XCTAssertEqual(values(lower), Array(values(full)[0..<30]))
        XCTAssertEqual(values(upper), Array(values(full)[30..<60]))
        XCTAssertEqual(values(try NemotronSplitEncoder.stackLayers(lower, upper)), values(full))
    }

    func testSliceHonoursPaddedStrides() throws {
        // Core ML outputs may pad the innermost rows; [1, 2, 2, 3] stored with a row stride of 4.
        let storage = UnsafeMutablePointer<Float>.allocate(capacity: 16)
        defer { storage.deallocate() }
        for i in 0..<16 { storage[i] = i % 4 == 3 ? -1 : Float(i) }
        let padded = try MLMultiArray(
            dataPointer: storage, shape: [1, 2, 2, 3], dataType: .float32, strides: [16, 8, 4, 1])
        let second = try NemotronSplitEncoder.layers(padded, 1..<2)
        XCTAssertEqual(values(second), [8, 9, 10, 12, 13, 14])
    }

    func testRejectsMismatchedCaches() throws {
        let a = try cache(layers: 2, rows: 3, width: 5)
        let b = try cache(layers: 2, rows: 4, width: 5)
        XCTAssertThrowsError(try NemotronSplitEncoder.stackLayers(a, b))
        XCTAssertThrowsError(try NemotronSplitEncoder.layers(a, 0..<3))
    }

    func testSplitIsPublishedForTheDefaultTierOnly() async {
        XCTAssertEqual(NemotronSplitEncoder.chunkMs, 2240)
        XCTAssertEqual(NemotronSplitEncoder.subdirectory, "encoder-split/2240ms")
        do {
            _ = try await StreamingNemotronMultilingualAsrManager.downloadSplitEncoder(chunkMs: 1120)
            XCTFail("1120 ms has no split encoder")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("2240"))
        }
    }
}
