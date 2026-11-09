@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// The opt-in split 2240 ms encoder (#1116). These tests use plain arrays and
/// filesystem fixtures; no CoreML model is loaded and nothing is downloaded.
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

    // MARK: - Cache layout (24-layer caches sliced per half, stacked back)

    func testSliceAndStackRoundTrip() throws {
        let full = try cache(layers: 4, rows: 3, width: 5)
        let lower = try NemotronSplitEncoder.layers(full, 0..<2)
        let upper = try NemotronSplitEncoder.layers(full, 2..<4)
        XCTAssertEqual(lower.shape, [1, 2, 3, 5])
        XCTAssertEqual(values(lower), Array(values(full)[0..<30]))
        XCTAssertEqual(values(upper), Array(values(full)[30..<60]))
        let stacked = try NemotronSplitEncoder.stackLayers(lower, upper)
        XCTAssertEqual(stacked.shape, [1, 4, 3, 5])
        XCTAssertEqual(values(stacked), values(full))
    }

    func testSliceAndStackHonourPaddedStrides() throws {
        // Core ML outputs may pad the innermost rows: [1, 2, 2, 3] stored with a row stride of 4.
        let storage = UnsafeMutablePointer<Float>.allocate(capacity: 16)
        defer { storage.deallocate() }
        for i in 0..<16 { storage[i] = i % 4 == 3 ? -1 : Float(i) }
        let padded = try MLMultiArray(
            dataPointer: storage, shape: [1, 2, 2, 3], dataType: .float32, strides: [16, 8, 4, 1])
        XCTAssertEqual(values(try NemotronSplitEncoder.layers(padded, 1..<2)), [8, 9, 10, 12, 13, 14])
        let unpadded: [Float] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        XCTAssertEqual(values(try NemotronSplitEncoder.stackLayers(padded, padded)), unpadded + unpadded)
    }

    func testRejectsMismatchedCaches() throws {
        let a = try cache(layers: 2, rows: 3, width: 5)
        let b = try cache(layers: 2, rows: 4, width: 5)
        XCTAssertThrowsError(try NemotronSplitEncoder.stackLayers(a, b))
        XCTAssertThrowsError(try NemotronSplitEncoder.layers(a, 0..<3))
    }

    // MARK: - Downloads

    func testSplitIsPublishedOutsideTheTierFolders() {
        XCTAssertEqual(StreamingNemotronMultilingualAsrManager.splitEncoderChunkMs, 2240)
        XCTAssertEqual(NemotronSplitEncoder.subdirectory, "encoder-split/2240ms")
        XCTAssertFalse(NemotronSplitEncoder.subdirectory.hasPrefix("latin/"))
        XCTAssertFalse(NemotronSplitEncoder.subdirectory.hasPrefix("multilingual/"))
    }

    func testIncludeEncoderFalseSkipsOnlyTheSingleEncoder() {
        let skip = StreamingNemotronMultilingualAsrManager.isSingleEncoderPath
        XCTAssertTrue(skip("latin/2240ms/encoder.mlmodelc"))
        XCTAssertTrue(skip("latin/2240ms/encoder.mlmodelc/weights/weight.bin"))
        XCTAssertFalse(skip("latin/2240ms/decoder_joint.mlmodelc/weights/weight.bin"))
        XCTAssertFalse(skip("latin/2240ms/metadata.json"))
        XCTAssertFalse(skip("encoder-split/2240ms/encoder_frontend.mlmodelc/coremldata.bin"))
    }

    func testCacheCheckRejectsAnInterruptedDownload() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("split-encoder-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        // Filesystem shape only (same fixtures as ModelCacheCompletenessTests).
        func bundle(_ name: String, partialWeights: Bool) throws {
            let weights = root.appendingPathComponent(name).appendingPathComponent("weights")
            try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
            try Data("x".utf8).write(to: root.appendingPathComponent(name).appendingPathComponent("coremldata.bin"))
            let weightFile = partialWeights ? "weight.bin.partial" : "weight.bin"
            try Data("w".utf8).write(to: weights.appendingPathComponent(weightFile))
        }
        XCTAssertFalse(NemotronSplitEncoder.isCached(at: root))
        for name in NemotronSplitEncoder.fileNames {
            try bundle(name, partialWeights: name == NemotronSplitEncoder.upperLayersFile)
        }
        // Every root coremldata.bin landed, but one weight file is still staged.
        XCTAssertFalse(NemotronSplitEncoder.isCached(at: root))
        try FileManager.default.removeItem(at: root.appendingPathComponent(NemotronSplitEncoder.upperLayersFile))
        try bundle(NemotronSplitEncoder.upperLayersFile, partialWeights: false)
        XCTAssertTrue(NemotronSplitEncoder.isCached(at: root))
    }
}
