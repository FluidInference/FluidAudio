#if TTS
import Foundation

/// Minimal SentencePiece unigram tokenizer for PocketTTS.
///
/// Parses a `.model` protobuf to extract the vocabulary, then uses
/// Viterbi decoding to segment text into subword tokens.
public struct SentencePieceTokenizer: Sendable {

    /// Viterbi score per piece ID: the piece's log-probability, or for a
    /// user-defined piece SentencePiece's length bonus.
    private let scores: [Float]
    /// Lookup from piece string to token ID, over the pieces input text can
    /// match: normal and user-defined ones. Control, unknown, byte and unused
    /// pieces have no surface form, so `<s>` in the input is three characters.
    private let pieceToId: [String: Int]
    /// Maximum piece length in UTF-8 scalars for early termination.
    private let maxPieceLength: Int
    /// Byte pieces (`<0x00>`…`<0xFF>`) keyed by byte value. Empty unless the
    /// model was trained with byte fallback.
    private let byteToId: [UInt8: Int]
    /// The model's unknown piece, emitted for scalars no piece covers when the
    /// model has no byte fallback.
    private let unknownId: Int?
    /// Score of the edge that steps over one scalar no piece covers. Sits below
    /// every real piece so it is taken only where nothing else can be.
    private let unknownScore: Float

    /// The space replacement character used by SentencePiece.
    private static let spaceMarker: Character = "\u{2581}"
    /// Marks an unknown-scalar edge in the Viterbi lattice.
    private static let unknownPieceId = -1
    /// Matches SentencePiece's `kUnkPenalty`.
    private static let unknownPenalty: Float = 10

    public init(modelData: Data) throws {
        let model = try SentencePieceProto.parseModel(modelData)

        var scores: [Float] = []
        scores.reserveCapacity(model.pieces.count)
        var lookup: [String: Int] = [:]
        lookup.reserveCapacity(model.pieces.count)
        var maxLen = 0
        var minScore: Float?
        var unknownId: Int?
        var bytes: [UInt8: Int] = [:]
        for (index, entry) in model.pieces.enumerated() {
            switch entry.type {
            case .normal:
                lookup[entry.piece] = index
                maxLen = max(maxLen, entry.piece.unicodeScalars.count)
                minScore = min(minScore ?? entry.score, entry.score)
                scores.append(entry.score)
            case .userDefined:
                // SentencePiece's `GetUserDefinedScore`, on the UTF-8 length
                // its default encoder measures, so the piece is always taken.
                lookup[entry.piece] = index
                maxLen = max(maxLen, entry.piece.unicodeScalars.count)
                scores.append(0.1 * Float(entry.piece.utf8.count - 1))
            case .unknown:
                if unknownId == nil { unknownId = index }
                scores.append(entry.score)
            case .byte:
                if model.byteFallback, let value = Self.byteValue(of: entry.piece) {
                    bytes[value] = index
                }
                scores.append(entry.score)
            case .control, .unused:
                scores.append(entry.score)
            }
        }
        self.scores = scores
        self.pieceToId = lookup
        self.maxPieceLength = maxLen
        self.byteToId = bytes
        self.unknownId = unknownId
        self.unknownScore = (minScore ?? 0) - Self.unknownPenalty
    }

    /// Tokenize text into token IDs using Viterbi unigram decoding.
    ///
    /// Applies the standard SentencePiece normalization: replaces spaces
    /// with `\u{2581}` and prepends `\u{2581}` to the input.
    ///
    /// A scalar no piece covers costs only itself, and the text on either side
    /// of it still segments into whole pieces. As in SentencePiece, it becomes
    /// its UTF-8 byte pieces when the model was trained with byte fallback, and
    /// otherwise the model's unknown piece, one for a whole run of such scalars.
    public func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }

        // Normalize: prepend space marker, replace spaces with marker
        let normalized =
            String(Self.spaceMarker)
            + text.replacingOccurrences(
                of: " ", with: String(Self.spaceMarker))

        return viterbiDecode(normalized)
    }

    // MARK: - Viterbi Decoding

    /// Run Viterbi algorithm to find the highest-score segmentation.
    ///
    /// For each position in the string, finds the best-scoring
    /// vocabulary piece ending at that position.
    private func viterbiDecode(_ text: String) -> [Int] {
        let scalars = Array(text.unicodeScalars)
        let n = scalars.count
        guard n > 0 else { return [] }

        // bestScore[i] = best log-probability score for text[0..<i]
        // bestPiece[i] = (pieceId, startPosition) for the piece ending at i
        let negInf: Float = -.infinity
        var bestScore = [Float](repeating: negInf, count: n + 1)
        var bestPiece = [(pieceId: Int, start: Int)](repeating: (0, 0), count: n + 1)
        bestScore[0] = 0

        // Every position is reachable: position i + 1 can always be reached
        // from i through the unknown edge below.
        for i in 0..<n {
            var coversSingleScalar = false
            let maxLen = min(maxPieceLength, n - i)
            for length in 1...maxLen {
                let end = i + length
                // Build candidate substring from scalars
                let candidate = String(String.UnicodeScalarView(scalars[i..<end]))

                guard let pieceId = pieceToId[candidate] else { continue }
                if length == 1 {
                    coversSingleScalar = true
                }

                let newScore = bestScore[i] + scores[pieceId]
                if newScore > bestScore[end] {
                    bestScore[end] = newScore
                    bestPiece[end] = (pieceId: pieceId, start: i)
                }
            }

            // No piece is exactly this scalar, so give the lattice a way past
            // it. Without this edge one such scalar strands every later
            // position and the whole input loses its segmentation.
            if !coversSingleScalar {
                let newScore = bestScore[i] + unknownScore
                if newScore > bestScore[i + 1] {
                    bestScore[i + 1] = newScore
                    bestPiece[i + 1] = (pieceId: Self.unknownPieceId, start: i)
                }
            }
        }

        // Backtrack to collect token IDs. The list is built back to front, so
        // a scalar's byte pieces go in reversed too.
        var ids: [Int] = []
        var pos = n
        var inUnknownRun = false
        while pos > 0 {
            let (pieceId, start) = bestPiece[pos]
            if pieceId == Self.unknownPieceId {
                if let byteIds = byteFallbackIds(for: scalars[start]) {
                    ids.append(contentsOf: byteIds.reversed())
                    inUnknownRun = false
                } else {
                    // Adjacent unknown scalars share one unknown piece. A model
                    // with no unknown piece, which SentencePiece refuses to
                    // load, drops them.
                    if let unknownId, !inUnknownRun {
                        ids.append(unknownId)
                    }
                    inUnknownRun = true
                }
            } else {
                ids.append(pieceId)
                inUnknownRun = false
            }
            pos = start
        }

        ids.reverse()
        return ids
    }

    /// Byte-fallback token IDs for one scalar, in UTF-8 order. Nil when the
    /// model has no byte fallback or no byte piece for one of its bytes.
    private func byteFallbackIds(for scalar: Unicode.Scalar) -> [Int]? {
        guard !byteToId.isEmpty else { return nil }
        var ids: [Int] = []
        for byte in String(scalar).utf8 {
            guard let id = byteToId[byte] else { return nil }
            ids.append(id)
        }
        return ids
    }

    /// The byte a `<0xHH>` byte piece stands for.
    private static func byteValue(of piece: String) -> UInt8? {
        guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">") else { return nil }
        return UInt8(piece.dropFirst(3).dropLast(), radix: 16)
    }
}
#endif
