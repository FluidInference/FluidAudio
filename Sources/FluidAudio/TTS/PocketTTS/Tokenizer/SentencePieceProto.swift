#if TTS
import Foundation

/// Minimal protobuf parser for SentencePiece `.model` files.
///
/// Extracts the vocabulary pieces (string, score, type) and
/// `TrainerSpec.byte_fallback` from the `ModelProto` message, ignoring the
/// rest of the trainer spec and the normalizer spec.
///
/// Wire format reference:
/// - Tag = (field_number << 3) | wire_type
/// - Wire type 0 = varint, 2 = length-delimited, 5 = 32-bit fixed
enum SentencePieceProto {

    /// `SentencePiece.Type` from `sentencepiece_model.proto`.
    enum PieceType: Int, Sendable {
        case normal = 1
        case unknown = 2
        case control = 3
        case userDefined = 4
        case unused = 5
        case byte = 6
    }

    struct Piece: Sendable {
        let piece: String
        let score: Float
        let type: PieceType
    }

    /// The parts of `ModelProto` the tokenizer uses.
    struct Model: Sendable {
        let pieces: [Piece]
        /// `TrainerSpec.byte_fallback`: a scalar no piece covers becomes its
        /// UTF-8 byte pieces instead of the unknown piece.
        let byteFallback: Bool
    }

    enum ParseError: Error {
        case invalidData
        case unexpectedEnd
        case invalidUtf8
    }

    /// Parse a SentencePiece `.model` file and return the vocabulary pieces.
    static func parse(_ data: Data) throws -> [Piece] {
        try parseModel(data).pieces
    }

    /// Parse a SentencePiece `.model` file.
    static func parseModel(_ data: Data) throws -> Model {
        var pieces: [Piece] = []
        var byteFallback = false
        var offset = 0
        let bytes = Array(data)
        let count = bytes.count

        while offset < count {
            let (fieldNumber, wireType) = try readTag(bytes: bytes, count: count, offset: &offset)

            switch wireType {
            case 0:
                // Varint — skip
                _ = try readVarint(bytes: bytes, count: count, offset: &offset)
            case 1:
                // 64-bit fixed — skip 8 bytes
                offset += 8
                guard offset <= count else { throw ParseError.unexpectedEnd }
            case 2:
                // Length-delimited
                let length = try readVarint(bytes: bytes, count: count, offset: &offset)
                let end = offset + Int(length)
                guard end <= count else { throw ParseError.unexpectedEnd }

                if fieldNumber == 1 {
                    // Top-level field 1 = repeated SentencePiece message
                    let piece = try parsePiece(bytes: bytes, start: offset, end: end)
                    pieces.append(piece)
                } else if fieldNumber == 2 {
                    // Top-level field 2 = TrainerSpec message
                    byteFallback = try parseByteFallback(bytes: bytes, start: offset, end: end)
                }
                // Skip to end of this field regardless
                offset = end
            case 5:
                // 32-bit fixed — skip 4 bytes
                offset += 4
                guard offset <= count else { throw ParseError.unexpectedEnd }
            default:
                throw ParseError.invalidData
            }
        }

        return Model(pieces: pieces, byteFallback: byteFallback)
    }

    // MARK: - Private

    private static func parsePiece(bytes: [UInt8], start: Int, end: Int) throws -> Piece {
        var offset = start
        var piece: String?
        var score: Float = 0
        var type: PieceType = .normal

        while offset < end {
            let (fieldNumber, wireType) = try readTag(bytes: bytes, count: end, offset: &offset)

            switch wireType {
            case 0:
                let value = try readVarint(bytes: bytes, count: end, offset: &offset)
                if fieldNumber == 3 {
                    // SentencePiece.type (enum). Like protobuf, read a value
                    // this enum doesn't know as the default.
                    type = PieceType(rawValue: Int(value)) ?? .normal
                }
            case 1:
                offset += 8
                guard offset <= end else { throw ParseError.unexpectedEnd }
            case 2:
                let length = try readVarint(bytes: bytes, count: end, offset: &offset)
                let fieldEnd = offset + Int(length)
                guard fieldEnd <= end else { throw ParseError.unexpectedEnd }

                if fieldNumber == 1 {
                    // SentencePiece.piece (string)
                    let slice = bytes[offset..<fieldEnd]
                    guard let str = String(bytes: slice, encoding: .utf8) else {
                        throw ParseError.invalidUtf8
                    }
                    piece = str
                }
                offset = fieldEnd
            case 5:
                if fieldNumber == 2 {
                    // SentencePiece.score (float)
                    guard offset + 4 <= end else { throw ParseError.unexpectedEnd }
                    score = readFloat32(bytes: bytes, offset: offset)
                }
                offset += 4
                guard offset <= end else { throw ParseError.unexpectedEnd }
            default:
                throw ParseError.invalidData
            }
        }

        return Piece(piece: piece ?? "", score: score, type: type)
    }

    /// `TrainerSpec.byte_fallback` (field 35, bool). False when absent, the
    /// proto default.
    private static func parseByteFallback(bytes: [UInt8], start: Int, end: Int) throws -> Bool {
        var offset = start
        var byteFallback = false

        while offset < end {
            let (fieldNumber, wireType) = try readTag(bytes: bytes, count: end, offset: &offset)

            switch wireType {
            case 0:
                let value = try readVarint(bytes: bytes, count: end, offset: &offset)
                if fieldNumber == 35 {
                    byteFallback = value != 0
                }
            case 1:
                offset += 8
            case 2:
                let length = try readVarint(bytes: bytes, count: end, offset: &offset)
                offset += Int(length)
            case 5:
                offset += 4
            default:
                throw ParseError.invalidData
            }
            guard offset <= end else { throw ParseError.unexpectedEnd }
        }

        return byteFallback
    }

    private static func readTag(
        bytes: [UInt8], count: Int, offset: inout Int
    ) throws -> (fieldNumber: Int, wireType: Int) {
        let tag = try readVarint(bytes: bytes, count: count, offset: &offset)
        let wireType = Int(tag & 0x07)
        let fieldNumber = Int(tag >> 3)
        return (fieldNumber, wireType)
    }

    private static func readVarint(
        bytes: [UInt8], count: Int, offset: inout Int
    ) throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0

        while offset < count {
            let byte = bytes[offset]
            offset += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 {
                return result
            }
            shift += 7
            if shift >= 64 { throw ParseError.invalidData }
        }

        throw ParseError.unexpectedEnd
    }

    private static func readFloat32(bytes: [UInt8], offset: Int) -> Float {
        var value: Float = 0
        withUnsafeMutableBytes(of: &value) { ptr in
            ptr[0] = bytes[offset]
            ptr[1] = bytes[offset + 1]
            ptr[2] = bytes[offset + 2]
            ptr[3] = bytes[offset + 3]
        }
        return value
    }
}
#endif
