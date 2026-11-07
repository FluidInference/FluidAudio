import Foundation
import XCTest

@testable import FluidAudio

final class SentencePieceProtoTests: XCTestCase {

    // MARK: - Protobuf Encoding Helpers

    private func makeVarint(_ value: UInt64) -> [UInt8] {
        var result: [UInt8] = []
        var v = value
        while v > 0x7F {
            result.append(UInt8(v & 0x7F) | 0x80)
            v >>= 7
        }
        result.append(UInt8(v))
        return result
    }

    private func makeTag(fieldNumber: Int, wireType: Int) -> [UInt8] {
        return makeVarint(UInt64((fieldNumber << 3) | wireType))
    }

    private func makeFloat32Bytes(_ value: Float) -> [UInt8] {
        var v = value
        return withUnsafeBytes(of: &v) { Array($0) }
    }

    /// Build a SentencePiece sub-message with string (field 1), optional score
    /// (field 2) and optional type (field 3)
    private func makePieceMessage(string: String, score: Float? = nil, type: Int? = nil) -> [UInt8] {
        var body: [UInt8] = []

        // field 1, wire type 2 (string)
        let stringBytes = Array(string.utf8)
        body.append(contentsOf: makeTag(fieldNumber: 1, wireType: 2))
        body.append(contentsOf: makeVarint(UInt64(stringBytes.count)))
        body.append(contentsOf: stringBytes)

        // field 2, wire type 5 (float32)
        if let score = score {
            body.append(contentsOf: makeTag(fieldNumber: 2, wireType: 5))
            body.append(contentsOf: makeFloat32Bytes(score))
        }

        // field 3, wire type 0 (enum)
        if let type = type {
            body.append(contentsOf: makeTag(fieldNumber: 3, wireType: 0))
            body.append(contentsOf: makeVarint(UInt64(type)))
        }

        return body
    }

    /// Wrap piece messages as top-level field 1 of ModelProto, and an optional
    /// TrainerSpec body as top-level field 2
    private func wrapInModelProto(pieces: [[UInt8]], trainerSpec: [UInt8]? = nil) -> Data {
        var data: [UInt8] = []
        for piece in pieces {
            // Top-level field 1, wire type 2 (length-delimited)
            data.append(contentsOf: makeTag(fieldNumber: 1, wireType: 2))
            data.append(contentsOf: makeVarint(UInt64(piece.count)))
            data.append(contentsOf: piece)
        }
        if let trainerSpec = trainerSpec {
            data.append(contentsOf: makeTag(fieldNumber: 2, wireType: 2))
            data.append(contentsOf: makeVarint(UInt64(trainerSpec.count)))
            data.append(contentsOf: trainerSpec)
        }
        return Data(data)
    }

    // MARK: - Parse Tests

    func testParseEmptyDataReturnsEmptyArray() throws {
        let pieces = try SentencePieceProto.parse(Data())
        XCTAssertTrue(pieces.isEmpty)
    }

    func testParseSinglePiece() throws {
        let pieceMsg = makePieceMessage(string: "hello", score: 1.5)
        let data = wrapInModelProto(pieces: [pieceMsg])

        let pieces = try SentencePieceProto.parse(data)
        XCTAssertEqual(pieces.count, 1)
        XCTAssertEqual(pieces[0].piece, "hello")
        XCTAssertEqual(pieces[0].score, 1.5, accuracy: 1e-5)
    }

    func testParseMultiplePieces() throws {
        let pieces = [
            makePieceMessage(string: "hello", score: -1.0),
            makePieceMessage(string: "world", score: -2.0),
            makePieceMessage(string: "foo", score: -0.5),
        ]
        let data = wrapInModelProto(pieces: pieces)

        let parsed = try SentencePieceProto.parse(data)
        XCTAssertEqual(parsed.count, 3)
        XCTAssertEqual(parsed[0].piece, "hello")
        XCTAssertEqual(parsed[1].piece, "world")
        XCTAssertEqual(parsed[2].piece, "foo")
        XCTAssertEqual(parsed[0].score, -1.0, accuracy: 1e-5)
        XCTAssertEqual(parsed[1].score, -2.0, accuracy: 1e-5)
        XCTAssertEqual(parsed[2].score, -0.5, accuracy: 1e-5)
    }

    func testParseMissingScoreDefaultsToZero() throws {
        let pieceMsg = makePieceMessage(string: "test", score: nil)
        let data = wrapInModelProto(pieces: [pieceMsg])

        let pieces = try SentencePieceProto.parse(data)
        XCTAssertEqual(pieces.count, 1)
        XCTAssertEqual(pieces[0].piece, "test")
        XCTAssertEqual(pieces[0].score, 0, accuracy: 1e-5)
    }

    func testParsePieceType() throws {
        let data = wrapInModelProto(pieces: [
            makePieceMessage(string: "<unk>", score: 0, type: 2),
            makePieceMessage(string: "<0x0A>", score: 0, type: 6),
            makePieceMessage(string: "hello", score: -1),
        ])

        let pieces = try SentencePieceProto.parse(data)
        XCTAssertEqual(pieces.map(\.type), [.unknown, .byte, .normal])
    }

    func testParseByteFallbackDefaultsToFalse() throws {
        let data = wrapInModelProto(pieces: [makePieceMessage(string: "hello", score: -1)])
        XCTAssertFalse(try SentencePieceProto.parseModel(data).byteFallback)
    }

    func testParseByteFallbackSkipsOtherTrainerSpecFields() throws {
        var spec: [UInt8] = []
        // input (field 1, string)
        spec += makeTag(fieldNumber: 1, wireType: 2) + makeVarint(4) + Array("data".utf8)
        // model_type (field 3, enum)
        spec += makeTag(fieldNumber: 3, wireType: 0) + makeVarint(1)
        // character_coverage (field 10, float)
        spec += makeTag(fieldNumber: 10, wireType: 5) + makeFloat32Bytes(0.9995)
        // byte_fallback (field 35, bool)
        spec += makeTag(fieldNumber: 35, wireType: 0) + makeVarint(1)
        let data = wrapInModelProto(
            pieces: [makePieceMessage(string: "hello", score: -1)], trainerSpec: spec)

        let model = try SentencePieceProto.parseModel(data)
        XCTAssertTrue(model.byteFallback)
        XCTAssertEqual(model.pieces.map(\.piece), ["hello"])
    }

    func testParseInvalidWireTypeThrows() {
        // Wire type 3 is deprecated/invalid for our purposes
        var data: [UInt8] = []
        data.append(contentsOf: makeTag(fieldNumber: 1, wireType: 3))
        XCTAssertThrowsError(try SentencePieceProto.parse(Data(data)))
    }

    // MARK: - PocketTTSError

    func testPocketTTSErrorDescriptions() {
        let errors: [PocketTTSError] = [
            .downloadFailed("network"),
            .corruptedModel("encoder"),
            .modelNotFound("decoder"),
            .processingFailed("timeout"),
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription)
            XCTAssertFalse(error.errorDescription!.isEmpty)
        }
    }

    func testPocketTTSErrorContainsDetail() {
        let error = PocketTTSError.modelNotFound("myModel")
        XCTAssertTrue(
            error.errorDescription!.contains("myModel"),
            "Error description should contain the model name"
        )
    }
}
