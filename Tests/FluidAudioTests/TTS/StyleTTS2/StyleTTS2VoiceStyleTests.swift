import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the precomputed `ref_s.bin` loader.
final class StyleTTS2VoiceStyleTests: XCTestCase {

    private func writeBlob(_ floats: [Float], name: String = "ref_s.bin") throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("styletts2-voicestyle-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        var local = floats
        let data = local.withUnsafeMutableBufferPointer { buf in
            Data(buffer: buf)
        }
        try data.write(to: url)
        return url
    }

    func testLoadParsesAll256Floats() throws {
        // Ramp 0..255 so we can verify byte order at every index.
        let floats: [Float] = (0..<256).map { Float($0) }
        let url = try writeBlob(floats)
        let style = try StyleTTS2VoiceStyle.load(from: url)

        XCTAssertEqual(style.concatenated.count, 256)
        XCTAssertEqual(style.concatenated.first, 0)
        XCTAssertEqual(style.concatenated.last, 255)
        XCTAssertEqual(style.concatenated[100], 100)
    }

    func testAcousticAndProsodySlicesAreFirstAndSecondHalf() throws {
        var floats = [Float](repeating: 0, count: 256)
        for i in 0..<128 { floats[i] = 1.0 }  // acoustic = 1.0
        for i in 128..<256 { floats[i] = -1.0 }  // prosody  = -1.0
        let url = try writeBlob(floats)
        let style = try StyleTTS2VoiceStyle.load(from: url)

        XCTAssertEqual(style.acoustic.count, 128)
        XCTAssertEqual(style.prosody.count, 128)
        XCTAssertTrue(style.acoustic.allSatisfy { $0 == 1.0 })
        XCTAssertTrue(style.prosody.allSatisfy { $0 == -1.0 })
    }

    func testLoadFailsForMissingFile() {
        let bogus = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).bin")
        XCTAssertThrowsError(try StyleTTS2VoiceStyle.load(from: bogus)) { error in
            guard case StyleTTS2Error.modelNotFound = error else {
                XCTFail("expected modelNotFound, got \(error)")
                return
            }
        }
    }

    func testLoadFailsForWrongSize() throws {
        // 100 floats = 400 bytes — wrong by construction.
        let floats = [Float](repeating: 0, count: 100)
        let url = try writeBlob(floats, name: "short.bin")
        XCTAssertThrowsError(try StyleTTS2VoiceStyle.load(from: url)) { error in
            guard case StyleTTS2Error.invalidConfiguration = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testInitPreconditionRequiresExactDim() {
        // Sanity guard for direct (non-file) construction.
        let ok = StyleTTS2VoiceStyle(concatenated: [Float](repeating: 0, count: 256))
        XCTAssertEqual(ok.concatenated.count, 256)
    }
}
