import Foundation

/// GPT-2 byte-level BPE tokenizer for Chatterbox Nano/Turbo, loaded from the
/// upstream "slow"-format assets (`vocab.json` + `merges.txt` +
/// `added_tokens.json`).
///
/// Mirrors HF `AutoTokenizer` for this checkpoint:
///   1. `punc_norm` cleanup (applied by the caller via `puncNorm` — the
///      turbo variant, which differs from the multilingual one)
///   2. split out added tokens (the 20 paralinguistic tags `[laugh]`,
///      `[chuckle]`, … plus `<|endoftext|>`) — surrounding text is
///      tokenized independently, so a space before a tag becomes a
///      standalone `Ġ` token (id 220)
///   3. GPT-2 pre-tokenizer regex on the remaining segments
///   4. UTF-8 bytes → GPT-2 byte-unicode alphabet → BPE merges → vocab ids
///
/// No BOS/EOS wrapping — `inference_turbo` feeds the raw encoding.
final class ChatterboxNanoTokenizer: Sendable {

    private let vocab: [String: Int]
    /// Added tokens sorted longest-first for greedy left-to-right matching.
    private let addedTokens: [(content: String, id: Int)]
    /// Merge pair "left right" → rank (lower merges first).
    private let mergeRank: [String: Int]
    private let splitRegex: NSRegularExpression
    /// GPT-2 byte→unicode-char mapping, indexed by byte value.
    private let byteChars: [String]

    /// GPT-2 pre-tokenizer pattern (contractions, ` ?letters`, ` ?digits`,
    /// ` ?punctuation`, trailing/other whitespace).
    private static let gpt2SplitPattern =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

    init(vocabURL: URL, mergesURL: URL, addedTokensURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        guard let vocabAny = try JSONSerialization.jsonObject(with: vocabData) as? [String: Any]
        else {
            throw ChatterboxError.malformedAsset("\(vocabURL.lastPathComponent): not an object")
        }
        var vocab = [String: Int](minimumCapacity: vocabAny.count)
        for (token, id) in vocabAny {
            if let id = id as? Int { vocab[token] = id }
        }
        self.vocab = vocab

        // merges.txt: one "left right" pair per line; leading "#version" line.
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var mergeRank = [String: Int]()
        var rank = 0
        for line in mergesText.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#") { continue }
            mergeRank[String(line)] = rank
            rank += 1
        }
        self.mergeRank = mergeRank

        let addedData = try Data(contentsOf: addedTokensURL)
        guard let addedAny = try JSONSerialization.jsonObject(with: addedData) as? [String: Any]
        else {
            throw ChatterboxError.malformedAsset(
                "\(addedTokensURL.lastPathComponent): not an object")
        }
        var added: [(String, Int)] = []
        for (content, id) in addedAny {
            if let id = id as? Int { added.append((content, id)) }
        }
        self.addedTokens = added.sorted { $0.0.count > $1.0.count }

        self.splitRegex = try NSRegularExpression(pattern: Self.gpt2SplitPattern)
        self.byteChars = Self.bytesToUnicode()
    }

    /// Upstream `tts_turbo.punc_norm`: capitalization, whitespace collapse,
    /// LLM-punc replacement, and a trailing full stop. (The multilingual
    /// variant additionally rewrites `"..."`, `" - "`, `";"` and accepts CJK
    /// enders — the turbo variant does not.)
    static func puncNorm(_ text: String) -> String {
        if text.isEmpty { return "You need to add some text for me to talk." }
        var text = text
        if let first = text.first, first.isLowercase {
            text = first.uppercased() + text.dropFirst()
        }
        // Python str.split(): any whitespace (tabs, newlines, …), collapsed.
        text = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        let replacements: [(String, String)] = [
            ("…", ", "), (":", ","),
            ("—", "-"), ("–", "-"), (" ,", ","), ("\u{201C}", "\""), ("\u{201D}", "\""),
            ("\u{2018}", "'"), ("\u{2019}", "'"),
        ]
        for (old, new) in replacements {
            text = text.replacingOccurrences(of: old, with: new)
        }
        while text.hasSuffix(" ") { text = String(text.dropLast()) }
        let enders: Set<Character> = [".", "!", "?", "-", ","]
        if let last = text.last, !enders.contains(last) {
            text += "."
        }
        return text
    }

    /// text → token ids, mirroring `AutoTokenizer.__call__` (no specials).
    func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        for segment in splitAddedTokens(text) {
            switch segment {
            case .added(let id):
                ids.append(id)
            case .text(let piece):
                let ns = piece as NSString
                let matches = splitRegex.matches(
                    in: piece, range: NSRange(location: 0, length: ns.length))
                for match in matches {
                    for token in bpe(mapBytes(ns.substring(with: match.range))) {
                        if let id = vocab[token] { ids.append(id) }
                    }
                }
            }
        }
        return ids
    }

    // MARK: - Added-token splitting

    private enum Segment {
        case added(Int)
        case text(String)
    }

    private func splitAddedTokens(_ text: String) -> [Segment] {
        var segments: [Segment] = []
        var buffer = ""
        var index = text.startIndex
        outer: while index < text.endIndex {
            for (content, id) in addedTokens {
                if let end = text.index(
                    index, offsetBy: content.count, limitedBy: text.endIndex),
                    text[index..<end] == content
                {
                    if !buffer.isEmpty {
                        segments.append(.text(buffer))
                        buffer = ""
                    }
                    segments.append(.added(id))
                    index = end
                    continue outer
                }
            }
            buffer.append(text[index])
            index = text.index(after: index)
        }
        if !buffer.isEmpty { segments.append(.text(buffer)) }
        return segments
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
