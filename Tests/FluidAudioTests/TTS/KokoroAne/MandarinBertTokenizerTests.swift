import Foundation
import XCTest

@testable import FluidAudio

/// Network-free tests for the char-level WordPiece tokenizer used by
/// the g2pW polyphone disambiguator. The tokenizer only needs to feed
/// Hanzi targets through `bert-base-chinese`'s vocab, so the
/// implementation skips subword merges and these tests validate the
/// thin char-level path.
final class MandarinBertTokenizerTests: XCTestCase {

    /// Build a minimal vocab matching the BERT layout: special tokens
    /// at fixed ids matching `bert-base-chinese`, then a handful of
    /// real Hanzi for the encode tests.
    private static func smallVocab() -> [String: Int32] {
        var v: [String: Int32] = [:]
        v["[PAD]"] = 0
        v["[UNK]"] = 100
        v["[CLS]"] = 101
        v["[SEP]"] = 102
        v["[MASK]"] = 103
        v["你"] = 200
        v["好"] = 201
        v["行"] = 202
        v["人"] = 203
        return v
    }

    func testInitRejectsMissingSpecialTokens() {
        var v = Self.smallVocab()
        v.removeValue(forKey: "[CLS]")
        XCTAssertThrowsError(try MandarinBertTokenizer(vocab: v)) { err in
            guard let e = err as? MandarinBertTokenizer.LoadError,
                case .missingSpecialToken(let name) = e
            else {
                XCTFail("Expected missingSpecialToken, got \(err)")
                return
            }
            XCTAssertEqual(name, "[CLS]")
        }
    }

    func testEncodesCharSequenceAndPositions() throws {
        let tok = try MandarinBertTokenizer(vocab: Self.smallVocab())
        let encoded = tok.encode(chars: ["你", "好"], maxLength: 8)
        XCTAssertEqual(
            encoded.inputIds,
            [101, 200, 201, 102, 0, 0, 0, 0])
        XCTAssertEqual(
            encoded.attentionMask,
            [1, 1, 1, 1, 0, 0, 0, 0])
        XCTAssertEqual(
            encoded.tokenTypeIds,
            [0, 0, 0, 0, 0, 0, 0, 0])
        // Char[0] lands at index 1 (right after [CLS]); Char[1] at 2.
        XCTAssertEqual(encoded.tokenPositionForChar, [1, 2])
    }

    func testUnknownCharFallsBackToUnk() throws {
        let tok = try MandarinBertTokenizer(vocab: Self.smallVocab())
        // 鵝 isn't in our miniature vocab — should encode as [UNK].
        let encoded = tok.encode(chars: ["鵝"], maxLength: 4)
        XCTAssertEqual(encoded.inputIds, [101, 100, 102, 0])
        XCTAssertEqual(encoded.attentionMask, [1, 1, 1, 0])
    }

    func testTruncationFromTheRight() throws {
        let tok = try MandarinBertTokenizer(vocab: Self.smallVocab())
        // maxLength=4 leaves room for 2 chars between [CLS] and [SEP].
        let encoded = tok.encode(
            chars: ["你", "好", "行", "人"], maxLength: 4)
        XCTAssertEqual(encoded.inputIds, [101, 200, 201, 102])
        XCTAssertEqual(encoded.attentionMask, [1, 1, 1, 1])
        // Only the first two chars get a tracked position; the
        // remainder are dropped, so the position list is shorter than
        // the original input.
        XCTAssertEqual(encoded.tokenPositionForChar, [1, 2])
    }

    func testMaxLengthPreconditionEnforced() throws {
        // maxLength == 1 leaves no room for [CLS] + [SEP]; the encode
        // helper requires at least 2.
        // (This is checked by precondition, not throw, so we only
        // exercise the boundary value of 2 here.)
        let tok = try MandarinBertTokenizer(vocab: Self.smallVocab())
        let encoded = tok.encode(chars: [], maxLength: 2)
        XCTAssertEqual(encoded.inputIds, [101, 102])
        XCTAssertEqual(encoded.attentionMask, [1, 1])
        XCTAssertEqual(encoded.tokenPositionForChar, [])
    }

    func testLoadFromTextFile() throws {
        // `vocab.txt` semantics: line index = token id. Synthesise a
        // tiny file in a temp dir and round-trip through `load`.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("MandarinBertTokenizerTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let url = tmpDir.appendingPathComponent("vocab.txt")
        // Pad with placeholder tokens so [CLS]/[SEP]/[PAD]/[UNK] land
        // at the canonical bert-base-chinese ids (0/100/101/102) — the
        // test only needs the special tokens to resolve, not match
        // upstream id-to-id.
        var lines: [String] = ["[PAD]"]
        for i in 1...99 { lines.append("[reserved_\(i)]") }
        lines.append("[UNK]")
        lines.append("[CLS]")
        lines.append("[SEP]")
        lines.append("你")
        lines.append("好")
        try lines.joined(separator: "\n").write(
            to: url, atomically: true, encoding: .utf8)

        let tok = try MandarinBertTokenizer.load(vocabURL: url)
        XCTAssertEqual(tok.padId, 0)
        XCTAssertEqual(tok.unkId, 100)
        XCTAssertEqual(tok.clsId, 101)
        XCTAssertEqual(tok.sepId, 102)
        let encoded = tok.encode(chars: ["你", "好"], maxLength: 4)
        XCTAssertEqual(encoded.inputIds, [101, 103, 104, 102])
    }

    func testTrailingNewlineDoesNotAddBlankToken() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("MandarinBertTokenizerTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let url = tmpDir.appendingPathComponent("vocab.txt")
        var lines: [String] = ["[PAD]"]
        for i in 1...99 { lines.append("x_\(i)") }
        lines.append("[UNK]")
        lines.append("[CLS]")
        lines.append("[SEP]")
        // Leave a trailing newline — the loader should drop the empty
        // final token rather than seed `vocab[""]`.
        let content = lines.joined(separator: "\n") + "\n"
        try content.write(to: url, atomically: true, encoding: .utf8)

        let tok = try MandarinBertTokenizer.load(vocabURL: url)
        XCTAssertNil(tok.vocab[""])
        XCTAssertEqual(tok.sepId, 102)
    }
}
