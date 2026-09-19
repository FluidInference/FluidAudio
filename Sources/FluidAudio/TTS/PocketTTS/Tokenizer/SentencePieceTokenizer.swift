import Foundation

/// Minimal SentencePiece unigram tokenizer for PocketTTS.
///
/// Parses a `.model` protobuf to extract the vocabulary, then uses
/// Viterbi decoding to segment text into subword tokens.
public struct SentencePieceTokenizer: Sendable {

    /// Vocabulary pieces with their log-probability scores.
    private let pieces: [SentencePieceProto.Piece]
    /// Lookup from piece string to token ID.
    private let pieceToId: [String: Int]
    /// Maximum piece length in UTF-8 scalars for early termination.
    private let maxPieceLength: Int
    /// Byte-fallback pieces (`<0x00>`…`<0xFF>`) keyed by byte value. Empty when
    /// the model was trained without byte fallback.
    private let byteToId: [UInt8: Int]
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
        let parsed = try SentencePieceProto.parse(modelData)
        self.pieces = parsed

        var lookup: [String: Int] = [:]
        lookup.reserveCapacity(parsed.count)
        var maxLen = 0
        var minScore: Float = 0
        for (index, entry) in parsed.enumerated() {
            lookup[entry.piece] = index
            maxLen = max(maxLen, entry.piece.unicodeScalars.count)
            minScore = min(minScore, entry.score)
        }
        self.pieceToId = lookup
        self.maxPieceLength = maxLen
        self.unknownScore = minScore - Self.unknownPenalty

        var bytes: [UInt8: Int] = [:]
        for value in UInt8.min...UInt8.max {
            if let id = lookup[String(format: "<0x%02X>", value)] {
                bytes[value] = id
            }
        }
        self.byteToId = bytes
    }

    /// Tokenize text into token IDs using Viterbi unigram decoding.
    ///
    /// Applies the standard SentencePiece normalization: replaces spaces
    /// with `\u{2581}` and prepends `\u{2581}` to the input.
    ///
    /// A scalar no piece covers costs only itself: it becomes its UTF-8
    /// byte-fallback pieces, or is dropped when the model has none, and the
    /// text on either side of it still segments into whole pieces.
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
                let piece = pieces[pieceId]
                if length == 1 {
                    coversSingleScalar = true
                }

                let newScore = bestScore[i] + piece.score
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
        while pos > 0 {
            let (pieceId, start) = bestPiece[pos]
            if pieceId == Self.unknownPieceId {
                ids.append(contentsOf: byteFallbackIds(for: scalars[start]).reversed())
            } else {
                ids.append(pieceId)
            }
            pos = start
        }

        ids.reverse()
        return ids
    }

    /// Byte-fallback token IDs for one scalar, in UTF-8 order. Empty when the
    /// model has no byte piece for one of its bytes.
    private func byteFallbackIds(for scalar: Unicode.Scalar) -> [Int] {
        var ids: [Int] = []
        for byte in String(scalar).utf8 {
            guard let id = byteToId[byte] else { return [] }
            ids.append(id)
        }
        return ids
    }
}
