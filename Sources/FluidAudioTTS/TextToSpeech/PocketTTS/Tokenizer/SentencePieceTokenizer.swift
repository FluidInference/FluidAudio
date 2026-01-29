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

    /// The space replacement character used by SentencePiece.
    private static let spaceMarker: Character = "\u{2581}"

    public init(modelData: Data) throws {
        let parsed = try SentencePieceProto.parse(modelData)
        self.pieces = parsed

        var lookup: [String: Int] = [:]
        lookup.reserveCapacity(parsed.count)
        var maxLen = 0
        for (index, entry) in parsed.enumerated() {
            lookup[entry.piece] = index
            maxLen = max(maxLen, entry.piece.unicodeScalars.count)
        }
        self.pieceToId = lookup
        self.maxPieceLength = maxLen
    }

    /// Tokenize text into token IDs using Viterbi unigram decoding.
    ///
    /// Applies the standard SentencePiece normalization: replaces spaces
    /// with `\u{2581}` and prepends `\u{2581}` to the input.
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

        // Build a string from scalars for substring matching
        // We work with Unicode scalar offsets for correctness
        for i in 0..<n {
            guard bestScore[i] > negInf else { continue }

            let maxLen = min(maxPieceLength, n - i)
            for length in 1...maxLen {
                let end = i + length
                // Build candidate substring from scalars
                let candidate = String(String.UnicodeScalarView(scalars[i..<end]))

                guard let pieceId = pieceToId[candidate] else { continue }
                let piece = pieces[pieceId]

                let newScore = bestScore[i] + piece.score
                if newScore > bestScore[end] {
                    bestScore[end] = newScore
                    bestPiece[end] = (pieceId: pieceId, start: i)
                }
            }
        }

        // Backtrack to collect token IDs
        guard bestScore[n] > negInf else {
            // Fallback: encode as individual characters
            return fallbackEncode(scalars)
        }

        var ids: [Int] = []
        var pos = n
        while pos > 0 {
            let (pieceId, start) = bestPiece[pos]
            ids.append(pieceId)
            pos = start
        }

        ids.reverse()
        return ids
    }

    /// Fallback: encode each character as a separate token.
    private func fallbackEncode(_ scalars: [Unicode.Scalar]) -> [Int] {
        var ids: [Int] = []
        for scalar in scalars {
            let char = String(scalar)
            if let id = pieceToId[char] {
                ids.append(id)
            }
            // Unknown characters are silently dropped
        }
        return ids
    }
}
