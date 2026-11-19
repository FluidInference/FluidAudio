#if arch(arm64)
import Foundation

/// Byte-level BPE tokenizer for the LFM3 `tokenizer.json` (HF tokenizers: Split(regex, Isolated) + ByteLevel
/// pre-tokenizer, BPE with `ignore_merges`, ByteLevel decoder, added tokens matched first). Token-exact against
/// Python `tokenizers` on the ASR / LiveASR / TTS prompts and transcripts.
final class LFM3AudioTokenizer {
    private let vocab: [String: Int32]
    private let idToToken: [Int32: String]
    private let ranks: [String: Int]
    private let added: [(content: String, id: Int32)]  // longest first
    private let addedIds: Set<Int32>
    private let specialIds: Set<Int32>
    private let regex: NSRegularExpression
    private let byteToChar: [Character]
    private let charToByte: [Character: UInt8]
    private let ignoreMerges: Bool
    private var cache: [String: [Int32]] = [:]

    init(url: URL) throws {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let rawVocab = model["vocab"] as? [String: Int]
        else {
            throw LFM3AudioError.invalidModel("\(url.lastPathComponent) is not a BPE tokenizer.json")
        }
        var vocab = [String: Int32](minimumCapacity: rawVocab.count + 200)
        var inverse = [Int32: String](minimumCapacity: rawVocab.count + 200)
        for (token, id) in rawVocab {
            vocab[token] = Int32(id)
            inverse[Int32(id)] = token
        }
        var ranks = [String: Int](minimumCapacity: 300_000)
        if let merges = model["merges"] as? [[String]] {
            for (rank, pair) in merges.enumerated() where pair.count == 2 { ranks[pair[0] + " " + pair[1]] = rank }
        } else if let merges = model["merges"] as? [String] {
            for (rank, pair) in merges.enumerated() { ranks[pair] = rank }
        }
        var added: [(String, Int32)] = []
        var special = Set<Int32>()
        for token in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = token["content"] as? String, let id = token["id"] as? Int else { continue }
            added.append((content, Int32(id)))
            vocab[content] = Int32(id)
            inverse[Int32(id)] = content
            if (token["special"] as? Bool) ?? false { special.insert(Int32(id)) }
        }
        var pattern =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
            + "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        if let pre = root["pre_tokenizer"] as? [String: Any], let steps = pre["pretokenizers"] as? [[String: Any]] {
            for step in steps where (step["type"] as? String) == "Split" {
                if let split = step["pattern"] as? [String: Any], let custom = split["Regex"] as? String {
                    pattern = custom
                }
            }
        }
        // GPT-2 bytes_to_unicode
        var bytes: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = bytes
        var extra = 0
        for b in 0..<256 where !bytes.contains(b) {
            bytes.append(b)
            chars.append(256 + extra)
            extra += 1
        }
        var byteToChar = [Character](repeating: " ", count: 256)
        var charToByte: [Character: UInt8] = [:]
        for (b, c) in zip(bytes, chars) {
            guard let scalar = Unicode.Scalar(UInt32(c)) else { continue }
            byteToChar[b] = Character(scalar)
            charToByte[Character(scalar)] = UInt8(b)
        }
        self.vocab = vocab
        self.idToToken = inverse
        self.ranks = ranks
        self.added = added.sorted { $0.0.count > $1.0.count }
        self.addedIds = Set(added.map { $0.1 })
        self.specialIds = special
        self.regex = try NSRegularExpression(pattern: pattern)
        self.byteToChar = byteToChar
        self.charToByte = charToByte
        self.ignoreMerges = (model["ignore_merges"] as? Bool) ?? false
    }

    var vocabularySize: Int { idToToken.count }

    func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            // earliest added-token occurrence (longest at that position)
            var best: (range: Range<Substring.Index>, id: Int32)?
            for (content, id) in added {
                if let range = rest.range(of: content), best.map({ range.lowerBound < $0.range.lowerBound }) ?? true {
                    best = (range, id)
                }
            }
            guard let match = best else {
                encodeOrdinary(String(rest), into: &out)
                break
            }
            if match.range.lowerBound > rest.startIndex {
                encodeOrdinary(String(rest[rest.startIndex..<match.range.lowerBound]), into: &out)
            }
            out.append(match.id)
            rest = rest[match.range.upperBound...]
        }
        return out
    }

    func decode(_ ids: [Int32], skipSpecial: Bool = true) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if skipSpecial && specialIds.contains(id) { continue }
            guard let token = idToToken[id] else { continue }
            if addedIds.contains(id) {
                bytes.append(contentsOf: Array(token.utf8))
                continue
            }
            for ch in token { if let b = charToByte[ch] { bytes.append(b) } }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodeOrdinary(_ text: String, into out: inout [Int32]) {
        let ns = text as NSString
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let piece = ns.substring(with: match.range)
            if let cached = cache[piece] {
                out.append(contentsOf: cached)
                continue
            }
            let mapped = String(piece.utf8.map { byteToChar[Int($0)] })
            let ids: [Int32]
            if ignoreMerges, let id = vocab[mapped] {
                ids = [id]
            } else {
                ids = bpe(mapped)
            }
            if cache.count < 50_000 { cache[piece] = ids }
            out.append(contentsOf: ids)
        }
    }

    private func bpe(_ word: String) -> [Int32] {
        var parts = word.map { String($0) }
        while parts.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(parts.count - 1) {
                if let rank = ranks[parts[i] + " " + parts[i + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }
            // merge every occurrence of the best pair, left to right (HF BPE per rank)
            let a = parts[bestIndex]
            let b = parts[bestIndex + 1]
            var merged: [String] = []
            merged.reserveCapacity(parts.count)
            var i = 0
            while i < parts.count {
                if i < parts.count - 1 && parts[i] == a && parts[i + 1] == b {
                    merged.append(a + b)
                    i += 2
                } else {
                    merged.append(parts[i])
                    i += 1
                }
            }
            parts = merged
        }
        return parts.map { vocab[$0] ?? 0 }
    }
}
#endif
