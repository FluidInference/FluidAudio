import Foundation
import os

/// SentencePiece-style BPE encoder for the mmBERT (Gemma vocabulary) tokenizer behind laya-multilingual.
///
/// Loads a HuggingFace `tokenizer.json` (256k vocabulary, 580k merges, byte fallback, added tokens)
/// and reproduces `tokenizers` encoding without special-token templates, which is how laya calls it.
/// Encoding only: laya never decodes.
public final class LayaTokenizer: Sendable {
    /// `<mask>` — one marker precedes every option in a laya sequence.
    public let maskTokenId: Int
    /// `<bos>` — laya's `[CLS]`.
    public let clsTokenId: Int
    /// `<eos>` — laya's `[SEP]`.
    public let sepTokenId: Int
    /// `<pad>` — fills the fixed-length model input.
    public let padTokenId: Int
    /// The literal mask token text, which laya strips from prompts so it can never appear inside them.
    public let maskToken: String

    /// Dictionary key that compares code points literally. Swift `String` equality applies Unicode
    /// canonical equivalence, which would merge distinct vocabulary entries such as U+4E86 and U+F9BA.
    struct ScalarKey: Hashable {
        let string: String

        init(_ string: String) { self.string = string }

        static func == (lhs: ScalarKey, rhs: ScalarKey) -> Bool {
            lhs.string.utf8.elementsEqual(rhs.string.utf8)
        }

        func hash(into hasher: inout Hasher) {
            var copy = string
            copy.withUTF8 { hasher.combine(bytes: UnsafeRawBufferPointer($0)) }
        }
    }

    private let vocab: [ScalarKey: Int]
    private let mergeRank: [ScalarKey: Int]
    private let byteTokens: [Int?]
    private let addedTokens: [ScalarKey: Int]
    /// Added tokens grouped by first scalar, longest first, so matching costs one lookup per position.
    private let addedTokensByFirstScalar: [UInt32: [AddedToken]]
    private let unknownTokenId: Int

    private struct AddedToken: Sendable {
        let scalars: [UInt32]
        let id: Int
        let lstrip: Bool
        let rstrip: Bool
    }
    /// Piece → ids memo. Prompts reuse the same words constantly; BPE merging is the dominant host cost.
    private let pieceCache = PieceCache()

    private static let spaceMarker: Unicode.Scalar = "\u{2581}"

    public init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        // Keep the vocabulary as NSDictionary: bridging to [String: Any] merges canonically
        // equivalent keys (U+4E86 vs U+F9BA) and silently drops vocabulary entries.
        guard let root = try JSONSerialization.jsonObject(with: data) as? NSDictionary,
            let model = root["model"] as? NSDictionary,
            let vocabAny = model["vocab"] as? NSDictionary,
            let mergesAny = model["merges"] as? [Any]
        else {
            throw LayaError.invalidAsset("tokenizer.json is missing model.vocab or model.merges")
        }
        guard model["type"] as? String == "BPE", model["byte_fallback"] as? Bool == true else {
            throw LayaError.invalidAsset("tokenizer.json must describe a byte-fallback BPE model")
        }

        var vocab = [ScalarKey: Int](minimumCapacity: vocabAny.count)
        for case (let token as NSString, let id as Int) in vocabAny {
            vocab[ScalarKey(String(token))] = id
        }
        self.vocab = vocab

        var mergeRank = [ScalarKey: Int](minimumCapacity: mergesAny.count)
        for (rank, entry) in mergesAny.enumerated() {
            if let pair = entry as? [String], pair.count == 2 {
                mergeRank[ScalarKey("\(pair[0]) \(pair[1])")] = rank
            } else if let text = entry as? String {
                mergeRank[ScalarKey(text)] = rank
            }
        }
        self.mergeRank = mergeRank

        var byteTokens = [Int?](repeating: nil, count: 256)
        for value in 0..<256 {
            byteTokens[value] = vocab[ScalarKey(String(format: "<0x%02X>", value))]
        }
        self.byteTokens = byteTokens

        var added = [ScalarKey: Int]()
        var byFirst: [UInt32: [AddedToken]] = [:]
        if let addedList = root["added_tokens"] as? [[String: Any]] {
            for entry in addedList {
                guard let content = entry["content"] as? String, let id = entry["id"] as? Int, !content.isEmpty
                else { continue }
                added[ScalarKey(content)] = id
                let scalars = content.unicodeScalars.map(\.value)
                let token = AddedToken(
                    scalars: scalars, id: id, lstrip: entry["lstrip"] as? Bool ?? false,
                    rstrip: entry["rstrip"] as? Bool ?? false)
                byFirst[scalars[0], default: []].append(token)
            }
        }
        // Longest content first so overlapping added tokens resolve like the Rust matcher.
        for key in byFirst.keys {
            byFirst[key]?.sort { $0.scalars.count > $1.scalars.count }
        }
        self.addedTokens = added
        self.addedTokensByFirstScalar = byFirst

        func requiredToken(_ content: String) throws -> Int {
            guard let id = added[ScalarKey(content)] ?? vocab[ScalarKey(content)] else {
                throw LayaError.invalidAsset("tokenizer.json has no \(content) token")
            }
            return id
        }
        self.maskToken = "<mask>"
        self.maskTokenId = try requiredToken(maskToken)
        self.clsTokenId = try requiredToken("<bos>")
        self.sepTokenId = try requiredToken("<eos>")
        self.padTokenId = try requiredToken("<pad>")
        self.unknownTokenId = try requiredToken("<unk>")
    }

    /// Encode text to token ids without any special-token template.
    ///
    /// Added tokens (`<mask>`, `<start_of_turn>`, newline runs, …) are matched first, longest match
    /// wins and `lstrip`/`rstrip` swallow adjacent whitespace like the Rust matcher; each remaining
    /// segment gets spaces replaced by `▁`, a leading `▁` prepended, and is split into `▁`-prefixed
    /// pieces that are BPE-merged with byte fallback for scalars outside the vocabulary.
    public func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        let scalars = Array(text.unicodeScalars)
        var ids: [Int] = []
        var segment: [Unicode.Scalar] = []
        var index = 0
        while index < scalars.count {
            if let candidates = addedTokensByFirstScalar[scalars[index].value],
                let token = candidates.first(where: { matches($0, in: scalars, at: index) })
            {
                if token.lstrip {
                    while let last = segment.last, last.properties.isWhitespace { segment.removeLast() }
                }
                encodeSegment(segment, into: &ids)
                segment.removeAll(keepingCapacity: true)
                ids.append(token.id)
                index += token.scalars.count
                if token.rstrip {
                    while index < scalars.count, scalars[index].properties.isWhitespace { index += 1 }
                }
                continue
            }
            segment.append(scalars[index])
            index += 1
        }
        encodeSegment(segment, into: &ids)
        return ids
    }

    private func matches(_ token: AddedToken, in scalars: [Unicode.Scalar], at index: Int) -> Bool {
        guard index + token.scalars.count <= scalars.count else { return false }
        for (offset, value) in token.scalars.enumerated() where scalars[index + offset].value != value {
            return false
        }
        return true
    }

    // MARK: - Metaspace + BPE

    private func encodeSegment(_ segment: [Unicode.Scalar], into ids: inout [Int]) {
        guard !segment.isEmpty else { return }
        var scalars: [Unicode.Scalar] = []
        scalars.reserveCapacity(segment.count + 1)
        for scalar in segment {
            scalars.append(scalar == " " ? Self.spaceMarker : scalar)
        }
        if scalars.first != Self.spaceMarker {
            scalars.insert(Self.spaceMarker, at: 0)
        }
        var piece: [Unicode.Scalar] = []
        for scalar in scalars {
            if scalar == Self.spaceMarker, !piece.isEmpty {
                ids.append(contentsOf: encodePiece(piece))
                piece.removeAll(keepingCapacity: true)
            }
            piece.append(scalar)
        }
        if !piece.isEmpty {
            ids.append(contentsOf: encodePiece(piece))
        }
    }

    private func encodePiece(_ scalars: [Unicode.Scalar]) -> [Int] {
        let key = scalars.map(\.value)
        if let cached = pieceCache.lookup(key) { return cached }
        let ids = bpe(scalars)
        pieceCache.store(key, ids)
        return ids
    }

    /// Bounded memo behind an unfair lock; cleared wholesale when full.
    private struct PieceCache: Sendable {
        private static let capacity = 8192
        private let entries = OSAllocatedUnfairLock<[[UInt32]: [Int]]>(initialState: [:])

        func lookup(_ key: [UInt32]) -> [Int]? {
            entries.withLock { $0[key] }
        }

        func store(_ key: [UInt32], _ ids: [Int]) {
            entries.withLock { table in
                if table.count >= Self.capacity { table.removeAll(keepingCapacity: true) }
                table[key] = ids
            }
        }
    }

    /// Byte-level fallback happens per scalar before merging, exactly like the Rust BPE model.
    private func bpe(_ scalars: [Unicode.Scalar]) -> [Int] {
        var symbols: [String] = []
        symbols.reserveCapacity(scalars.count)
        var unknownRun = false
        for scalar in scalars {
            let symbol = String(Character(scalar))
            if vocab[ScalarKey(symbol)] != nil {
                symbols.append(symbol)
                unknownRun = false
                continue
            }
            let bytes = Array(symbol.utf8)
            if bytes.allSatisfy({ byteTokens[Int($0)] != nil }) {
                for byte in bytes {
                    symbols.append(String(format: "<0x%02X>", byte))
                }
                unknownRun = false
            } else if !unknownRun {
                // fuse_unk: consecutive unknown scalars collapse into one <unk>.
                symbols.append("<unk>")
                unknownRun = true
            }
        }
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(symbols.count - 1) {
                if let rank = mergeRank[ScalarKey("\(symbols[index]) \(symbols[index + 1])")], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
        return symbols.map { vocab[ScalarKey($0)] ?? unknownTokenId }
    }
}
