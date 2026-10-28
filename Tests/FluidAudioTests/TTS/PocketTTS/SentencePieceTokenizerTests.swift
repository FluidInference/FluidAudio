import Foundation
import XCTest

@testable import FluidAudio

/// Builds SentencePiece `.model` bytes from a piece list, so tokenizer-dependent
/// tests run without the downloaded language pack.
enum SentencePieceTestModel {

    static func data(
        pieces: [(piece: String, score: Float, type: SentencePieceProto.PieceType)],
        byteFallback: Bool = false
    ) -> Data {
        var bytes: [UInt8] = []
        for entry in pieces {
            var body: [UInt8] = []
            let text = Array(entry.piece.utf8)
            body.append(contentsOf: tag(fieldNumber: 1, wireType: 2))
            body.append(contentsOf: varint(UInt64(text.count)))
            body.append(contentsOf: text)
            body.append(contentsOf: tag(fieldNumber: 2, wireType: 5))
            var score = entry.score
            body.append(contentsOf: withUnsafeBytes(of: &score) { Array($0) })
            if entry.type != .normal {
                body.append(contentsOf: tag(fieldNumber: 3, wireType: 0))
                body.append(contentsOf: varint(UInt64(entry.type.rawValue)))
            }

            bytes.append(contentsOf: tag(fieldNumber: 1, wireType: 2))
            bytes.append(contentsOf: varint(UInt64(body.count)))
            bytes.append(contentsOf: body)
        }
        if byteFallback {
            // ModelProto.trainer_spec (field 2) holding TrainerSpec.byte_fallback (field 35).
            let spec = tag(fieldNumber: 35, wireType: 0) + varint(1)
            bytes.append(contentsOf: tag(fieldNumber: 2, wireType: 2))
            bytes.append(contentsOf: varint(UInt64(spec.count)))
            bytes.append(contentsOf: spec)
        }
        return Data(bytes)
    }

    /// Every piece scores -1 and is typed the way a trained model types it.
    static func tokenizer(pieces: [String], byteFallback: Bool = false) throws -> SentencePieceTokenizer {
        try SentencePieceTokenizer(
            modelData: data(
                pieces: pieces.map { ($0, Float(-1), pieceType(of: $0)) },
                byteFallback: byteFallback))
    }

    /// `<unk>` is the unknown piece, `<s>`, `</s>` and `<pad>` are control
    /// pieces, `<0xHH>` are byte pieces, and the rest are normal.
    static func pieceType(of piece: String) -> SentencePieceProto.PieceType {
        switch piece {
        case "<unk>": return .unknown
        case "<s>", "</s>", "<pad>": return .control
        default: return piece.hasPrefix("<0x") ? .byte : .normal
        }
    }

    private static func varint(_ value: UInt64) -> [UInt8] {
        var result: [UInt8] = []
        var remaining = value
        while remaining > 0x7F {
            result.append(UInt8(remaining & 0x7F) | 0x80)
            remaining >>= 7
        }
        result.append(UInt8(remaining))
        return result
    }

    private static func tag(fieldNumber: Int, wireType: Int) -> [UInt8] {
        varint(UInt64((fieldNumber << 3) | wireType))
    }
}

final class SentencePieceTokenizerTests: XCTestCase {

    private let words = ["<unk>", "\u{2581}", "\u{2581}hello", "\u{2581}world", "world"]
    private let letters = ["h", "e", "l", "o", "w", "r", "d"]

    private func id(_ piece: String, in vocab: [String]) throws -> Int {
        try XCTUnwrap(vocab.firstIndex(of: piece), "piece \(piece) missing from test vocab")
    }

    func testKnownTextSegmentsIntoWholePieces() throws {
        let vocab = words + letters
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
        XCTAssertEqual(
            tokenizer.encode("hello world"),
            [try id("\u{2581}hello", in: vocab), try id("\u{2581}world", in: vocab)])
    }

    func testUnknownScalarDoesNotBreakSegmentationAroundIt() throws {
        // A newline has no piece. It must cost its own byte piece and leave
        // "hello" and "world" as whole pieces, not push the entire input to
        // one token per character.
        let vocab = words + letters + ["<0x0A>"]
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab, byteFallback: true)
        XCTAssertEqual(
            tokenizer.encode("hello\nworld"),
            [
                try id("\u{2581}hello", in: vocab),
                try id("<0x0A>", in: vocab),
                try id("world", in: vocab),
            ])
    }

    func testUnknownScalarBecomesItsUtf8BytesInOrder() throws {
        // U+1F600 is F0 9F 98 80 in UTF-8.
        let vocab = words + letters + ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab, byteFallback: true)
        XCTAssertEqual(
            tokenizer.encode("hello \u{1F600}"),
            [
                try id("\u{2581}hello", in: vocab),
                try id("\u{2581}", in: vocab),
                try id("<0xF0>", in: vocab),
                try id("<0x9F>", in: vocab),
                try id("<0x98>", in: vocab),
                try id("<0x80>", in: vocab),
            ])
    }

    func testByteFallbackSpellsOutEveryUnknownScalar() throws {
        // Only the unknown piece merges a run; byte pieces keep every scalar.
        let vocab = words + letters + ["<0x0A>"]
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab, byteFallback: true)
        XCTAssertEqual(
            tokenizer.encode("hello\n\nworld"),
            [
                try id("\u{2581}hello", in: vocab),
                try id("<0x0A>", in: vocab),
                try id("<0x0A>", in: vocab),
                try id("world", in: vocab),
            ])
    }

    func testUnknownScalarBecomesUnknownPieceWithoutByteFallback() throws {
        let vocab = words + letters
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
        XCTAssertEqual(
            tokenizer.encode("hello\nworld"),
            [
                try id("\u{2581}hello", in: vocab),
                try id("<unk>", in: vocab),
                try id("world", in: vocab),
            ])
    }

    func testRunOfUnknownScalarsIsOneUnknownPiece() throws {
        let vocab = words + letters
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
        XCTAssertEqual(
            tokenizer.encode("hello\n\u{1F600}\nworld"),
            [
                try id("\u{2581}hello", in: vocab),
                try id("<unk>", in: vocab),
                try id("world", in: vocab),
            ])
    }

    func testBytePiecesAreUnusedWithoutByteFallback() throws {
        let vocab = words + letters + ["<0x0A>"]
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
        XCTAssertEqual(
            tokenizer.encode("hello\nworld"),
            [
                try id("\u{2581}hello", in: vocab),
                try id("<unk>", in: vocab),
                try id("world", in: vocab),
            ])
    }

    func testControlPieceDoesNotMatchInputText() throws {
        // "<s>" in the input is three unknown characters, not the control piece.
        let vocab = words + letters + ["<s>"]
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
        XCTAssertEqual(
            tokenizer.encode("hello <s>"),
            [
                try id("\u{2581}hello", in: vocab),
                try id("\u{2581}", in: vocab),
                try id("<unk>", in: vocab),
            ])
    }

    func testUserDefinedPieceScoresByLengthNotStoredScore() throws {
        // Stored at -10, "a" still scores 0 as a user-defined piece, so
        // "▁" + "a" + "b" (-2) beats the normal "▁ab" (-5).
        let tokenizer = try SentencePieceTokenizer(
            modelData: SentencePieceTestModel.data(pieces: [
                ("<unk>", 0, .unknown),
                ("\u{2581}", -1, .normal),
                ("\u{2581}ab", -5, .normal),
                ("a", -10, .userDefined),
                ("b", -1, .normal),
            ]))
        XCTAssertEqual(tokenizer.encode("ab"), [1, 3, 4])
    }

    func testTokenCountIsAdditiveAcrossAnUnknownScalar() throws {
        let vocab = words + letters + ["<0x0A>"]
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab, byteFallback: true)
        let clean = tokenizer.encode("hello world").count
        XCTAssertEqual(tokenizer.encode("hello world\n").count, clean + 1)
    }
}
