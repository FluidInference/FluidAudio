import Foundation
import XCTest

@testable import FluidAudio

/// Builds SentencePiece `.model` bytes from a piece list, so tokenizer-dependent
/// tests run without the downloaded language pack.
enum SentencePieceTestModel {

    static func data(pieces: [(piece: String, score: Float)]) -> Data {
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

            bytes.append(contentsOf: tag(fieldNumber: 1, wireType: 2))
            bytes.append(contentsOf: varint(UInt64(body.count)))
            bytes.append(contentsOf: body)
        }
        return Data(bytes)
    }

    static func tokenizer(pieces: [String]) throws -> SentencePieceTokenizer {
        try SentencePieceTokenizer(modelData: data(pieces: pieces.map { ($0, Float(-1)) }))
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
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
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
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
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

    func testUnknownScalarIsDroppedWithoutBytePieces() throws {
        let vocab = words + letters
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
        XCTAssertEqual(
            tokenizer.encode("hello\nworld"),
            [try id("\u{2581}hello", in: vocab), try id("world", in: vocab)])
    }

    func testTokenCountIsAdditiveAcrossAnUnknownScalar() throws {
        let vocab = words + letters + ["<0x0A>"]
        let tokenizer = try SentencePieceTestModel.tokenizer(pieces: vocab)
        let clean = tokenizer.encode("hello world").count
        XCTAssertEqual(tokenizer.encode("hello world\n").count, clean + 1)
    }
}
