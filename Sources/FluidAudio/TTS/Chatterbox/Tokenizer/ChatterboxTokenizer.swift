#if TTS
import Foundation

/// Grapheme BPE tokenizer for Chatterbox Multilingual, mirroring upstream
/// `MTLTokenizer` (chatterbox/models/tokenizers/tokenizer.py):
///
///   1. `punc_norm` cleanup (applied by the caller via `puncNorm`)
///   2. lowercase + NFKD normalization
///   3. prepend the `[lang]` added token, replace `" "` with `[SPACE]`
///   4. split out added tokens (`[en]`, `[SPACE]`, `[laughter]`, …)
///   5. HF `Whitespace` pre-tokenizer (`\w+|[^\w\s]+`) on the remainder
///   6. character-level BPE over 265 merges → vocab ids (2454 entries)
///
/// zh/ja/he/ko/ru additionally require language-specific text transforms
/// upstream (Cangjie codes, kana normalization, diacritization) that are not
/// ported; callers must reject those languages first.
final class ChatterboxTokenizer: Sendable {

    private let vocab: [String: Int]
    /// Added tokens sorted longest-first for greedy left-to-right matching.
    private let addedTokens: [(content: String, id: Int)]
    /// Merge pair "left right" → rank (lower merges first).
    private let mergeRank: [String: Int]
    private let unkId: Int
    private let splitRegex: NSRegularExpression

    init(tokenizerJsonURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJsonURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocabAny = model["vocab"] as? [String: Any]
        else {
            throw ChatterboxError.malformedAsset("tokenizer json missing model.vocab")
        }

        var vocab = [String: Int](minimumCapacity: vocabAny.count)
        for (token, id) in vocabAny {
            guard let id = id as? Int else {
                throw ChatterboxError.malformedAsset(
                    "tokenizer json: non-integer id for vocab entry '\(token)'")
            }
            vocab[token] = id
        }
        self.vocab = vocab

        var added: [(String, Int)] = []
        if let addedAny = root["added_tokens"] as? [[String: Any]] {
            for entry in addedAny {
                guard let content = entry["content"] as? String,
                    let id = entry["id"] as? Int
                else {
                    throw ChatterboxError.malformedAsset(
                        "tokenizer json: malformed added_tokens entry")
                }
                added.append((content, id))
            }
        }
        self.addedTokens = added.sorted { $0.0.unicodeScalars.count > $1.0.unicodeScalars.count }

        var mergeRank = [String: Int]()
        if let mergesAny = model["merges"] as? [Any] {
            for (rank, merge) in mergesAny.enumerated() {
                if let s = merge as? String {
                    mergeRank[s] = rank
                } else if let pair = merge as? [String], pair.count == 2 {
                    mergeRank["\(pair[0]) \(pair[1])"] = rank
                }
            }
        }
        self.mergeRank = mergeRank
        self.unkId = vocab["[UNK]"] ?? 1
        // HF `Whitespace` pre-tokenizer: `\w+|[^\w\s]+` (Unicode-aware).
        self.splitRegex = try NSRegularExpression(pattern: "[\\w]+|[^\\w\\s]+")
    }

    /// Every id this tokenizer can emit indexes `textEmb` with an unchecked
    /// row slice — reject out-of-range ids at load, before cache recovery
    /// is bypassed.
    func validate(embeddingRows: Int) throws {
        for (token, id) in vocab where id < 0 || id >= embeddingRows {
            throw ChatterboxError.malformedAsset(
                "tokenizer vocab id \(id) for '\(token)' outside 0..<\(embeddingRows)")
        }
        for (content, id) in addedTokens where id < 0 || id >= embeddingRows {
            throw ChatterboxError.malformedAsset(
                "tokenizer added-token id \(id) for '\(content)' outside 0..<\(embeddingRows)")
        }
        guard unkId >= 0 && unkId < embeddingRows else {
            throw ChatterboxError.malformedAsset("tokenizer [UNK] id \(unkId) out of range")
        }
    }

    /// Upstream `punc_norm`: capitalization, whitespace collapse, LLM-punc
    /// replacement, and a trailing full stop.
    static func puncNorm(_ text: String) -> String {
        if text.isEmpty { return "You need to add some text for me to talk." }
        // Python indexes, splits and compares code points, not grapheme clusters: a
        // combining mark right after a space or punctuation joins that cluster.
        var text = text
        if let first = text.unicodeScalars.first, first.properties.isLowercase {
            text = first.properties.uppercaseMapping + String(text.unicodeScalars.dropFirst())
        }
        // Python str.split(): any whitespace (tabs, newlines, …), collapsed.
        text = text.unicodeScalars.split(whereSeparator: Self.isPythonSpace)
            .map { String(String.UnicodeScalarView($0)) }
            .joined(separator: " ")
        let replacements: [(String, String)] = [
            ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
            ("—", "-"), ("–", "-"), (" ,", ","), ("\u{201C}", "\""), ("\u{201D}", "\""),
            ("\u{2018}", "'"), ("\u{2019}", "'"),
        ]
        for (old, new) in replacements {
            text = text.replacingOccurrences(of: old, with: new, options: .literal)
        }
        while text.unicodeScalars.last == " " { text.unicodeScalars.removeLast() }
        let enders: Set<Unicode.Scalar> = [".", "!", "?", "-", ",", "、", "，", "。", "？", "！"]
        if let last = text.unicodeScalars.last, !enders.contains(last) {
            text += "."
        } else if text.isEmpty {
            text = "."
        }
        return text
    }

    /// Python `str.isspace()`: White_Space plus the separators U+001C to U+001F.
    private static func isPythonSpace(_ scalar: Unicode.Scalar) -> Bool {
        scalar.properties.isWhitespace || (0x1C...0x1F).contains(scalar.value)
    }

    /// text → token ids, mirroring `MTLTokenizer.encode` (without the
    /// zh/ja/he/ko/ru language-specific transforms).
    func encode(_ text: String, languageId: String) -> [Int] {
        var txt = text.lowercased()
        txt = txt.decomposedStringWithCompatibilityMapping  // NFKD
        txt = "[\(languageId.lowercased())]" + txt
        txt = txt.replacingOccurrences(of: " ", with: "[SPACE]", options: .literal)

        var ids: [Int] = []
        for segment in splitAddedTokens(txt) {
            switch segment {
            case .added(let id):
                ids.append(id)
            case .text(let piece):
                for word in preTokenize(piece) {
                    ids.append(contentsOf: bpe(word))
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
        // Unicode scalars, NOT Characters: a combining mark after "]" joins its grapheme
        // cluster, so "[SPACE]" followed by U+0301 would not match as Characters.
        let scalars = Array(text.unicodeScalars)
        var segments: [Segment] = []
        var buffer = String.UnicodeScalarView()
        var index = 0
        outer: while index < scalars.count {
            if scalars[index] == "[" {
                for (content, id) in addedTokens where scalars[index...].starts(with: content.unicodeScalars) {
                    if !buffer.isEmpty {
                        segments.append(.text(String(buffer)))
                        buffer = String.UnicodeScalarView()
                    }
                    segments.append(.added(id))
                    index += content.unicodeScalars.count
                    continue outer
                }
            }
            buffer.append(scalars[index])
            index += 1
        }
        if !buffer.isEmpty { segments.append(.text(String(buffer))) }
        return segments
    }

    // MARK: - Pre-tokenizer + BPE

    private func preTokenize(_ piece: String) -> [String] {
        let ns = piece as NSString
        let matches = splitRegex.matches(
            in: piece, range: NSRange(location: 0, length: ns.length))
        return matches.map { ns.substring(with: $0.range) }
    }

    private func bpe(_ word: String) -> [Int] {
        // Unicode scalars, NOT Characters: after NFKD a decomposed pair like
        // "u" + U+0308 is one grapheme cluster, but the reference BPE (and
        // the shipped vocabulary) operate per scalar — clustering would send
        // every accented character to [UNK].
        var parts = word.unicodeScalars.map(String.init)
        guard !parts.isEmpty else { return [] }

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
            parts[bestIndex] += parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }
        return parts.map { vocab[$0] ?? unkId }
    }
}
#endif
