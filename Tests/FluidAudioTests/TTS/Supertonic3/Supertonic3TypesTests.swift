import XCTest

@testable import FluidAudio

final class Supertonic3TypesTests: XCTestCase {

    // MARK: - Supertonic3Config

    func testDefaultsMatchCompileTimeConstants() {
        let cfg = Supertonic3Config.defaults
        XCTAssertEqual(cfg.ae.sampleRate, Supertonic3Constants.sampleRate)
        XCTAssertEqual(cfg.ae.baseChunkSize, Supertonic3Constants.baseChunkSize)
        XCTAssertEqual(cfg.ttl.chunkCompressFactor, Supertonic3Constants.chunkCompressFactor)
        XCTAssertEqual(cfg.ttl.latentDim, Supertonic3Constants.latentDim)
    }

    func testConfigDecodesUpstreamSnakeCaseJSON() throws {
        let json = """
            {
              "ae":  { "sample_rate": 44100, "base_chunk_size": 512 },
              "ttl": { "chunk_compress_factor": 6, "latent_dim": 24 }
            }
            """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(Supertonic3Config.self, from: json)
        XCTAssertEqual(cfg.ae.sampleRate, 44_100)
        XCTAssertEqual(cfg.ae.baseChunkSize, 512)
        XCTAssertEqual(cfg.ttl.chunkCompressFactor, 6)
        XCTAssertEqual(cfg.ttl.latentDim, 24)
    }

    // MARK: - Supertonic3VoiceStyle

    func testVoiceStyleLoadsAndFlattensCorrectShapes() throws {
        // Construct the smallest valid voice-style JSON: `[1, 50, 256]` ttl
        // and `[1, 8, 16]` dp (matching Supertonic3Constants). Fill with a
        // deterministic ramp so we can verify flatten order is plane→row→col.
        let ttlTokens = Supertonic3Constants.ttlStyleTokens
        let ttlDim = Supertonic3Constants.ttlStyleDim
        let dpTokens = Supertonic3Constants.dpStyleTokens
        let dpDim = Supertonic3Constants.dpStyleDim
        let ttl: [[[Float]]] = [
            (0..<ttlTokens).map { row in
                (0..<ttlDim).map { col in Float(row * ttlDim + col) }
            }
        ]
        let dp: [[[Float]]] = [
            (0..<dpTokens).map { row in
                (0..<dpDim).map { col in Float(row * dpDim + col) }
            }
        ]
        let payload: [String: Any] = [
            "style_ttl": ["data": ttl, "dims": [1, ttlTokens, ttlDim], "type": "float32"],
            "style_dp": ["data": dp, "dims": [1, dpTokens, dpDim], "type": "float32"],
        ]
        let json = try JSONSerialization.data(withJSONObject: payload)
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("supertonic3_voice_style_\(UUID().uuidString).json")
        try json.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let style = try Supertonic3VoiceStyle.load(from: tmp)
        XCTAssertEqual(style.ttlDims, [1, ttlTokens, ttlDim])
        XCTAssertEqual(style.ttlValues.count, ttlTokens * ttlDim)
        XCTAssertEqual(style.ttlValues.first, 0)
        XCTAssertEqual(style.ttlValues.last, Float((ttlTokens - 1) * ttlDim + (ttlDim - 1)))
        XCTAssertEqual(style.dpDims, [1, dpTokens, dpDim])
        XCTAssertEqual(style.dpValues.count, dpTokens * dpDim)
        XCTAssertFalse(style.name.isEmpty)
    }

    func testVoiceStyleRejectsWrongTtlShape() throws {
        let json = """
            {
              "style_ttl": {
                "data": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
                "dims": [1, 2, 3],
                "type": "float32"
              },
              "style_dp": {
                "data": [[[7.0, 8.0]]],
                "dims": [1, 1, 2],
                "type": "float32"
              }
            }
            """.data(using: .utf8)!
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("supertonic3_voice_style_bad_\(UUID().uuidString).json")
        try json.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertThrowsError(try Supertonic3VoiceStyle.load(from: tmp)) { err in
            guard case Supertonic3Error.voiceStyleShapeMismatch(let component, _, _) = err else {
                XCTFail("expected voiceStyleShapeMismatch, got \(err)")
                return
            }
            XCTAssertEqual(component, "style_ttl")
        }
    }

    func testVoiceStyleLoadFromMissingFileThrows() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("supertonic3_does_not_exist_\(UUID().uuidString).json")
        XCTAssertThrowsError(try Supertonic3VoiceStyle.load(from: missing)) { err in
            guard case Supertonic3Error.voiceStyleLoadFailed = err else {
                XCTFail("expected voiceStyleLoadFailed, got \(err)")
                return
            }
        }
    }

    func testVoiceStyleLoadFromMalformedJSONThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("supertonic3_bad_\(UUID().uuidString).json")
        try Data("{ not valid json".utf8).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertThrowsError(try Supertonic3VoiceStyle.load(from: tmp)) { err in
            guard case Supertonic3Error.voiceStyleLoadFailed = err else {
                XCTFail("expected voiceStyleLoadFailed, got \(err)")
                return
            }
        }
    }

    // MARK: - Constants invariants

    func testCJKLanguagesMatchAvailableLanguagesSubset() {
        for code in Supertonic3Constants.cjkLanguages {
            XCTAssertTrue(
                Supertonic3Constants.availableLanguages.contains(code),
                "CJK code \(code) must appear in availableLanguages")
        }
    }

    func testMaxChunkLengthCJKIsTighterThanLatin() {
        XCTAssertLessThan(
            Supertonic3Constants.maxChunkLengthCJK,
            Supertonic3Constants.maxChunkLengthLatin)
    }
}
