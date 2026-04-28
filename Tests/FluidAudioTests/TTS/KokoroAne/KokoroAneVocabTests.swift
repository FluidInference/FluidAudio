import Foundation
import XCTest

@testable import FluidAudio

final class KokoroAneVocabTests: XCTestCase {

    // Minimal IPA-ish vocab for testing.
    private func makeVocab() -> KokoroAneVocab {
        let map: [Character: Int32] = [
            "h": 50, "ə": 16, "l": 53, "o": 156, "ʊ": 60,
            " ": 16, "w": 49, "ɹ": 156, "d": 32,
        ]
        return KokoroAneVocab(map: map)
    }

    func testEncodeWrapsWithBosEos() throws {
        let vocab = makeVocab()
        let ids = try vocab.encode("həloʊ")
        // BOS + 5 mapped chars + EOS = 7
        XCTAssertEqual(ids.count, 7)
        XCTAssertEqual(ids.first, KokoroAneConstants.bosTokenId)
        XCTAssertEqual(ids.last, KokoroAneConstants.eosTokenId)
        XCTAssertEqual(Array(ids[1..<6]), [50, 16, 53, 156, 60])
    }

    func testEncodeSilentlyDropsMissingPhonemes() throws {
        let vocab = makeVocab()
        // 'X' and '!' are absent; should be skipped (not throw, not unk).
        let ids = try vocab.encode("hX!o")
        XCTAssertEqual(ids.count, 4)  // BOS + h + o + EOS
        XCTAssertEqual(ids[0], KokoroAneConstants.bosTokenId)
        XCTAssertEqual(ids[1], 50)
        XCTAssertEqual(ids[2], 156)
        XCTAssertEqual(ids[3], KokoroAneConstants.eosTokenId)
    }

    func testEncodeEmptyStringYieldsBosEosOnly() throws {
        let vocab = makeVocab()
        let ids = try vocab.encode("")
        XCTAssertEqual(ids, [KokoroAneConstants.bosTokenId, KokoroAneConstants.eosTokenId])
    }

    func testEncodeRejectsOverlongSequence() {
        let vocab = makeVocab()
        let tooLong = String(repeating: "h", count: KokoroAneConstants.maxPhonemeLength + 1)
        XCTAssertThrowsError(try vocab.encode(tooLong)) { error in
            guard case KokoroAneError.phonemeSequenceTooLong(let n) = error else {
                XCTFail("Expected phonemeSequenceTooLong, got \(error)")
                return
            }
            XCTAssertEqual(n, KokoroAneConstants.maxPhonemeLength + 1)
        }
    }

    func testLoadFromJSONRoundtrips() throws {
        let json: [String: Int] = ["a": 1, "b": 2, "ʃ": 99]
        let data = try JSONSerialization.data(withJSONObject: json)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kl-vocab-\(UUID().uuidString).json")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let vocab = try KokoroAneVocab.load(from: url)
        XCTAssertEqual(vocab.map["a"], 1)
        XCTAssertEqual(vocab.map["b"], 2)
        XCTAssertEqual(vocab.map["ʃ"], 99)
        XCTAssertNil(vocab.map["z"])
    }

    func testLoadMissingFileThrows() {
        let url = URL(fileURLWithPath: "/nonexistent/kl-vocab-missing.json")
        XCTAssertThrowsError(try KokoroAneVocab.load(from: url)) { error in
            guard case KokoroAneError.vocabMissing = error else {
                XCTFail("Expected vocabMissing, got \(error)")
                return
            }
        }
    }
}
