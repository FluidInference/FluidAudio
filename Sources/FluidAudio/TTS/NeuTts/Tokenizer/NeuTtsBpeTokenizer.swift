import Foundation

/// Minimal byte-level BPE encoder for the NeuTTS-2E (Qwen2-style) tokenizer.
///
/// Loads a HuggingFace `tokenizer.json` (vocab + merges + pre-tokenizer split
/// regex + added tokens) and implements text→ids encoding only — decoding is
/// never needed because generated ids are compared/mapped numerically
/// (speech tokens occupy a contiguous id range).
///
/// Byte-level scheme: each pre-tokenized piece is UTF-8 encoded and every
/// byte mapped through the GPT-2 `bytes_to_unicode` table before BPE merges,
/// so vocabulary tokens are strings over that mapped alphabet.
final class NeuTtsBpeTokenizer: Sendable {

    enum TokenizerError: Error, LocalizedError {
        case malformedTokenizerJson(String)

        var errorDescription: String? {
            switch self {
            case .malformedTokenizerJson(let detail):
                return "Malformed tokenizer.json: \(detail)"
            }
        }
    }

    private let vocab: [String: Int]
    private let addedTokens: [String: Int]
    /// Merge pair "left right" → rank (lower merges first).
    private let mergeRank: [String: Int]
    private let splitRegex: NSRegularExpression
    /// GPT-2 byte→unicode-char mapping, indexed by byte value.
    private let byteChars: [String]

    /// Fallback pre-tokenizer split pattern (Qwen2), used when the JSON does
    /// not carry one. Single-digit \p{N} groups — Qwen splits digits singly.
    private static let qwenSplitPattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TokenizerError.malformedTokenizerJson("root is not an object")
        }
        guard let model = root["model"] as? [String: Any],
            let vocabAny = model["vocab"] as? [String: Any]
        else {
            throw TokenizerError.malformedTokenizerJson("missing model.vocab")
        }

        var vocab = [String: Int](minimumCapacity: vocabAny.count)
        for (token, id) in vocabAny {
            if let id = id as? Int { vocab[token] = id }
        }
        self.vocab = vocab

        // Merges appear either as "left right" strings or as [left, right] pairs.
        guard let mergesAny = model["merges"] as? [Any] else {
            throw TokenizerError.malformedTokenizerJson("missing model.merges")
        }
        var mergeRank = [String: Int](minimumCapacity: mergesAny.count)
        for (rank, entry) in mergesAny.enumerated() {
            if let s = entry as? String {
                mergeRank[s] = rank
            } else if let pair = entry as? [String], pair.count == 2 {
                mergeRank["\(pair[0]) \(pair[1])"] = rank
            }
        }
        self.mergeRank = mergeRank

        var added = [String: Int]()
        if let addedList = root["added_tokens"] as? [[String: Any]] {
            for entry in addedList {
                if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                    added[content] = id
                }
            }
        }
        self.addedTokens = added

        let pattern = Self.extractSplitPattern(root) ?? Self.qwenSplitPattern
        self.splitRegex = try NSRegularExpression(pattern: pattern)

        self.byteChars = Self.bytesToUnicode()
    }

    /// Id for a special/added token (e.g. `<|SPEECH_GENERATION_END|>`).
    func tokenId(_ token: String) -> Int? {
        addedTokens[token] ?? vocab[token]
    }

    /// Encode plain text (no added-token splitting — prompt text never
    /// legitimately contains special-token strings; mirrors upstream, which
    /// BPE-encodes the raw text in one pass).
    func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        let ns = text as NSString
        let matches = splitRegex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        for match in matches {
            let piece = ns.substring(with: match.range)
            for token in bpe(mapBytes(piece)) {
                if let id = vocab[token] {
                    ids.append(id)
                }
            }
        }
        return ids
    }

    // MARK: - BPE

    /// Map a text piece into the byte-level alphabet, one mapped char per
    /// UTF-8 byte.
    private func mapBytes(_ piece: String) -> [String] {
        piece.utf8.map { byteChars[Int($0)] }
    }

    /// Standard BPE: repeatedly merge the adjacent pair with the lowest rank.
    private func bpe(_ symbols: [String]) -> [String] {
        var parts = symbols
        while parts.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(parts.count - 1) {
                if let rank = mergeRank["\(parts[i]) \(parts[i + 1])"], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }
            parts[bestIndex] = parts[bestIndex] + parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }
        return parts
    }

    // MARK: - tokenizer.json helpers

    /// Pull the first Split-pretokenizer regex out of the JSON (handles both
    /// a bare Split and a Sequence of pretokenizers).
    private static func extractSplitPattern(_ root: [String: Any]) -> String? {
        guard let pre = root["pre_tokenizer"] as? [String: Any] else { return nil }
        func patternOf(_ node: [String: Any]) -> String? {
            guard node["type"] as? String == "Split",
                let pattern = node["pattern"] as? [String: Any],
                let regex = pattern["Regex"] as? String
            else { return nil }
            return regex
        }
        if let p = patternOf(pre) { return p }
        if let seq = pre["pretokenizers"] as? [[String: Any]] {
            for node in seq {
                if let p = patternOf(node) { return p }
            }
        }
        return nil
    }

    /// GPT-2 `bytes_to_unicode`: printable byte ranges map to themselves,
    /// everything else to U+0100 + running offset.
    private static func bytesToUnicode() -> [String] {
        var byteToScalar = [UInt32](repeating: 0, count: 256)
        var assigned = [Bool](repeating: false, count: 256)
        let keepRanges: [ClosedRange<UInt32>] = [33...126, 161...172, 174...255]
        for range in keepRanges {
            for b in range {
                byteToScalar[Int(b)] = b
                assigned[Int(b)] = true
            }
        }
        var offset: UInt32 = 0
        for b in 0..<256 where !assigned[b] {
            byteToScalar[b] = 256 + offset
            offset += 1
        }
        return byteToScalar.map { String(UnicodeScalar($0)!) }
    }
}
