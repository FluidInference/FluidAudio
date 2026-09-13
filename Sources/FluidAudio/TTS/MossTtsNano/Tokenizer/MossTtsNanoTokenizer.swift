import Foundation

/// SentencePiece **BPE** tokenizer for MOSS-TTS-Nano (`tokenizer.model`, 16 384 pieces).
///
/// Reproduces `sentencepiece` inference for this model's spec: `nmt_nfkc`
/// normalization (NFKC, whitespace/format characters → single spaces, extra
/// whitespace removed), dummy prefix `▁`, greedy best-score pair merging over
/// NORMAL pieces, and byte fallback (`<0xNN>`) for characters outside the
/// vocabulary. The PocketTTS `SentencePieceTokenizer` is a *unigram* decoder and
/// cannot be reused for this model.
public struct MossTtsNanoTokenizer: Sendable {

    /// NORMAL pieces → id (the only pieces BPE may merge into or emit directly).
    private let pieceIds: [String: Int]
    /// Piece score by id (BPE models store `-rank`; higher merges first).
    private let scores: [Float]
    /// `<0x00>`…`<0xFF>` ids for byte fallback.
    private let byteIds: [Int]

    static let spaceMarker: Unicode.Scalar = "\u{2581}"

    public init(modelData: Data) throws {
        let pieces: [SentencePieceProto.Piece]
        do {
            pieces = try SentencePieceProto.parse(modelData)
        } catch {
            throw MossTtsNanoError.tokenizerLoadFailed("\(error)")
        }
        var pieceIds: [String: Int] = [:]
        pieceIds.reserveCapacity(pieces.count)
        var byteIds = [Int](repeating: -1, count: 256)
        for (id, entry) in pieces.enumerated() {
            switch entry.type {
            case 1:
                pieceIds[entry.piece] = id
            case 6:
                let body = entry.piece.dropFirst(3).dropLast()  // "<0x" … ">"
                if entry.piece.hasPrefix("<0x"), entry.piece.hasSuffix(">"),
                    let value = Int(body, radix: 16), value >= 0, value < 256
                {
                    byteIds[value] = id
                }
            default:
                break
            }
        }
        guard !pieceIds.isEmpty else {
            throw MossTtsNanoError.tokenizerLoadFailed("no NORMAL pieces in model")
        }
        guard !byteIds.contains(-1) else {
            throw MossTtsNanoError.tokenizerLoadFailed("incomplete byte-fallback table")
        }
        self.pieceIds = pieceIds
        self.scores = pieces.map(\.score)
        self.byteIds = byteIds
    }

    public init(modelURL: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: modelURL)
        } catch {
            throw MossTtsNanoError.tokenizerLoadFailed("\(modelURL.path): \(error)")
        }
        try self.init(modelData: data)
    }

    public var vocabularySize: Int { scores.count }

    /// Encode text to piece ids (no BOS/EOS — upstream uses `add_special_tokens=False`).
    public func encode(_ text: String) -> [Int] {
        let normalized = Self.normalize(text)
        guard !normalized.isEmpty else { return [] }

        var symbols = normalized.unicodeScalars.map { String($0) }
        while symbols.count > 1 {
            var bestScore = -Float.infinity
            var bestIndex = -1
            for i in 0..<(symbols.count - 1) {
                guard let id = pieceIds[symbols[i] + symbols[i + 1]] else { continue }
                let score = scores[id]
                if score > bestScore {
                    bestScore = score
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }

        var ids: [Int] = []
        ids.reserveCapacity(symbols.count)
        for symbol in symbols {
            if let id = pieceIds[symbol] {
                ids.append(id)
            } else {
                for byte in symbol.utf8 {
                    ids.append(byteIds[Int(byte)])
                }
            }
        }
        return ids
    }

    /// `nmt_nfkc` + `add_dummy_prefix` + `remove_extra_whitespaces` + `escape_whitespaces`.
    ///
    /// Whitespace and Unicode format characters (zero-width joiners, BOM) collapse to a
    /// single `▁`; other control characters are dropped; the result is stripped and
    /// prefixed with `▁`. Returns `""` for text with no printable content.
    static func normalize(_ text: String) -> String {
        let nfkc = text.precomposedStringWithCompatibilityMapping
        var out = String.UnicodeScalarView()
        var pendingSpace = false
        for scalar in nfkc.unicodeScalars {
            let props = scalar.properties
            if props.isWhitespace || props.generalCategory == .format {
                pendingSpace = true
                continue
            }
            if props.generalCategory == .control {
                continue
            }
            if pendingSpace && !out.isEmpty {
                out.append(Self.spaceMarker)
            }
            pendingSpace = false
            out.append(scalar)
        }
        guard !out.isEmpty else { return "" }
        var prefixed = String.UnicodeScalarView()
        prefixed.append(Self.spaceMarker)
        prefixed.append(contentsOf: out)
        return String(prefixed)
    }
}
