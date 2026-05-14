import Foundation
import XCTest

@testable import FluidAudio

/// Pure-logic unit tests for `PocketTtsConstantsLoader`'s file-loading
/// behavior. The full `load(from:)` entry point needs a real SentencePiece
/// tokenizer file (and the rest of the language pack), so these tests
/// drive the smaller `loadBosBeforeVoiceIfPresent(in:)` helper instead.
final class PocketTtsConstantsLoaderTests: XCTestCase {

    private var tmpDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("FluidAudioPocketTtsLoaderTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tmpDir, FileManager.default.fileExists(atPath: tmpDir.path) {
            try FileManager.default.removeItem(at: tmpDir)
        }
        try super.tearDownWithError()
    }

    // MARK: - loadBosBeforeVoiceIfPresent

    func testBosBeforeVoiceReturnsNilWhenMissing() throws {
        // Missing file → nil (cloned-voice v1 prefill will fail later at
        // use time; this code path is fine for snapshot-only voices).
        let result = try PocketTtsConstantsLoader.loadBosBeforeVoiceIfPresent(in: tmpDir)
        XCTAssertNil(result, "Absent bos_before_voice.bin must yield nil")
    }

    func testBosBeforeVoiceLoadsExpectedFloats() throws {
        // Write a synthetic 1024-float file (every byte distinct so we
        // verify no truncation/padding).
        let dim = PocketTtsConstants.embeddingDim
        let expected: [Float] = (0..<dim).map { Float($0) * 0.001 }
        let data = expected.withUnsafeBufferPointer { buffer -> Data in
            Data(buffer: buffer)
        }
        let url = tmpDir.appendingPathComponent("bos_before_voice.bin")
        try data.write(to: url)

        let loaded = try PocketTtsConstantsLoader.loadBosBeforeVoiceIfPresent(in: tmpDir)
        XCTAssertNotNil(loaded)
        XCTAssertEqual(loaded?.count, dim)
        XCTAssertEqual(loaded ?? [], expected)
    }

    func testBosBeforeVoiceThrowsOnWrongSize() throws {
        // Truncated file (1023 floats instead of 1024) must be rejected,
        // not silently zero-padded.
        let bad: [Float] = Array(repeating: 0, count: PocketTtsConstants.embeddingDim - 1)
        let data = bad.withUnsafeBufferPointer { buffer -> Data in
            Data(buffer: buffer)
        }
        let url = tmpDir.appendingPathComponent("bos_before_voice.bin")
        try data.write(to: url)

        XCTAssertThrowsError(
            try PocketTtsConstantsLoader.loadBosBeforeVoiceIfPresent(in: tmpDir)
        ) { error in
            guard
                let loadError = error as? PocketTtsConstantsLoader.LoadError,
                case .invalidSize(let name, let expected, let actual) = loadError
            else {
                XCTFail("Expected LoadError.invalidSize, got \(error)")
                return
            }
            XCTAssertEqual(name, "bos_before_voice")
            XCTAssertEqual(expected, PocketTtsConstants.embeddingDim)
            XCTAssertEqual(actual, PocketTtsConstants.embeddingDim - 1)
        }
    }
}
