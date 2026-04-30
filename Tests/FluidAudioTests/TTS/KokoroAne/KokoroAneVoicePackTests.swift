import Foundation
import XCTest

@testable import FluidAudio

final class KokoroAneVoicePackTests: XCTestCase {

    private let rows = KokoroAneConstants.voicePackRows  // 510
    private let cols = KokoroAneConstants.voicePackCols  // 256

    /// Build a deterministic [510, 256] pack where each cell encodes its
    /// (row, col) so slice() correctness can be verified by inspection.
    private func makePack() throws -> KokoroAneVoicePack {
        var storage = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols {
                storage[r * cols + c] = Float(r) * 1000 + Float(c)
            }
        }
        return try KokoroAneVoicePack(storage: storage)
    }

    func testInitRejectsWrongSize() {
        XCTAssertThrowsError(try KokoroAneVoicePack(storage: [Float](repeating: 0, count: 10))) {
            error in
            guard case KokoroAneError.invalidVoicePack = error else {
                XCTFail("Expected invalidVoicePack, got \(error)")
                return
            }
        }
    }

    func testSliceReturnsCorrectColumnsForMidRow() throws {
        let pack = try makePack()
        let phonemeCount = 5  // → row 4
        let (styleS, styleTimbre) = pack.slice(for: phonemeCount)
        XCTAssertEqual(styleS.count, 128)
        XCTAssertEqual(styleTimbre.count, 128)

        // Row 4: cells encode 4000 + col. Timbre is cols [0..<128], style_s is [128..<256].
        XCTAssertEqual(styleTimbre.first, 4000.0)
        XCTAssertEqual(styleTimbre.last, 4127.0)
        XCTAssertEqual(styleS.first, 4128.0)
        XCTAssertEqual(styleS.last, 4255.0)
    }

    func testSliceClampsLowerBoundForZeroOrNegative() throws {
        let pack = try makePack()
        let (s0, t0) = pack.slice(for: 0)
        let (s1, t1) = pack.slice(for: -10)
        // Both should map to row 0.
        XCTAssertEqual(t0.first, 0.0)
        XCTAssertEqual(s0.first, 128.0)
        XCTAssertEqual(t1, t0)
        XCTAssertEqual(s1, s0)
    }

    func testSliceClampsUpperBoundForOverflow() throws {
        let pack = try makePack()
        let (s, t) = pack.slice(for: 9999)
        // Should clamp to row 509.
        XCTAssertEqual(t.first, 509_000.0)
        XCTAssertEqual(s.first, 509_000.0 + 128)
    }

    func testLoadFromBinaryRoundtrips() throws {
        let pack = try makePack()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kl-pack-\(UUID().uuidString).bin")
        let bytes = pack.storage.withUnsafeBufferPointer {
            Data(buffer: $0)
        }
        try bytes.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let loaded = try KokoroAneVoicePack.load(from: url)
        XCTAssertEqual(loaded.storage.count, rows * cols)
        XCTAssertEqual(loaded.storage[0], 0.0)
        XCTAssertEqual(loaded.storage[cols + 5], 1000.0 + 5)
        XCTAssertEqual(loaded.storage.last, 509_000.0 + 255)
    }

    func testLoadRejectsMisalignedFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kl-pack-bad-\(UUID().uuidString).bin")
        // 7 bytes — not a multiple of sizeof(Float)=4
        try Data([0, 1, 2, 3, 4, 5, 6]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try KokoroAneVoicePack.load(from: url)) { error in
            guard case KokoroAneError.invalidVoicePack = error else {
                XCTFail("Expected invalidVoicePack, got \(error)")
                return
            }
        }
    }

    func testLoadRejectsMissingFile() {
        let url = URL(fileURLWithPath: "/nonexistent/kl-pack-missing.bin")
        XCTAssertThrowsError(try KokoroAneVoicePack.load(from: url)) { error in
            guard case KokoroAneError.voicePackMissing = error else {
                XCTFail("Expected voicePackMissing, got \(error)")
                return
            }
        }
    }
}
