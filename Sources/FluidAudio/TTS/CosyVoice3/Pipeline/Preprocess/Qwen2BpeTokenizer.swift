import Foundation

/// Qwen2 byte-level BPE tokenizer. Mirrors
/// `transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer` on the slow
/// path used by CosyVoice3 (`AutoTokenizer.from_pretrained(...)` + runtime
/// `add_special_tokens(...)` as done in `CosyVoice3Tokenizer`).
///
/// Encoding pipeline:
///   1. Split input on registered special tokens (longest-match first). Special
///      chunks map 1:1 to their fixed ID.
///   2. Pretokenize non-special chunks with Qwen2's regex.
///   3. UTF-8 encode each match and remap bytes via the GPT-2 byte→unicode
///      shim (`ByteEncoder` below).
///   4. Apply BPE merges (lowest rank wins, all occurrences merged per pass).
///   5. Look up the resulting symbols in `vocab.json` to get token IDs.
///
/// Loader accepts the standard HuggingFace asset layout:
///   <dir>/vocab.json      — {"symbol": id, ...}
///   <dir>/merges.txt      — first line is a header or the first merge;
///                            subsequent lines are "A B" pairs, rank = line idx.
/// Special tokens are passed in separately (from a JSON map exported alongside
/// the CosyVoice3 fixtures — the runtime add_special_tokens list in Python is
/// not encoded in the HF assets).
public final class Qwen2BpeTokenizer {

    public enum Error: Swift.Error, LocalizedError {
        case fileNotFound(URL)
        case invalidJSON(String)
        case missingField(String)
        case regexCompileFailed

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let url): return "file not found: \(url.path)"
            case .invalidJSON(let m): return "invalid JSON: \(m)"
            case .missingField(let f): return "missing field: \(f)"
            case .regexCompileFailed: return "failed to compile pretokenize regex"
            }
        }
    }

    /// Qwen2 pretokenize regex (see `transformers` PRETOKENIZE_REGEX).
    /// Matches: contractions, letter words, single digits, punctuation runs,
    /// newline-led whitespace, trailing whitespace.
    public static let pretokenizePattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private let vocab: [String: Int32]
    private let mergeRanks: [String: Int]  // "firstSpace second" -> rank
    private let specialTokens: [String: Int32]
    private let specialPattern: NSRegularExpression?
    private let pretokenizeRegex: NSRegularExpression

    public init(
        vocab: [String: Int32],
        merges: [(String, String)],
        specialTokens: [String: Int32]
    ) throws {
        self.vocab = vocab
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (i, pair) in merges.enumerated() {
            ranks["\(pair.0) \(pair.1)"] = i
        }
        self.mergeRanks = ranks
        self.specialTokens = specialTokens

        if !specialTokens.isEmpty {
            // Longest-first so `<|endofprompt|>` wins over `<|end`.
            let ordered = specialTokens.keys.sorted { $0.count > $1.count }
            let alternation = ordered.map { NSRegularExpression.escapedPattern(for: $0) }
                .joined(separator: "|")
            self.specialPattern = try NSRegularExpression(pattern: alternation)
        } else {
            self.specialPattern = nil
        }

        do {
            self.pretokenizeRegex = try NSRegularExpression(pattern: Self.pretokenizePattern)
        } catch {
            throw Error.regexCompileFailed
        }
    }

    /// Load vocab.json + merges.txt from a directory and attach the runtime
    /// special-token map (must be supplied externally; Python `AutoTokenizer`
    /// adds these at import time via `add_special_tokens`).
    public static func load(
        directory: URL,
        specialTokens: [String: Int32]
    ) throws -> Qwen2BpeTokenizer {
        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw Error.fileNotFound(vocabURL)
        }
        guard FileManager.default.fileExists(atPath: mergesURL.path) else {
            throw Error.fileNotFound(mergesURL)
        }

        let vocabData = try Data(contentsOf: vocabURL)
        guard let raw = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            throw Error.invalidJSON("vocab.json is not {String: Int}")
        }
        var vocab: [String: Int32] = [:]
        vocab.reserveCapacity(raw.count)
        for (k, v) in raw { vocab[k] = Int32(v) }

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var merges: [(String, String)] = []
        merges.reserveCapacity(140_000)
        var isFirst = true
        for line in mergesText.split(separator: "\n", omittingEmptySubsequences: true) {
            if isFirst {
                isFirst = false
                // Typical merges.txt header: "#version: 0.2". Skip it.
                if line.hasPrefix("#") { continue }
            }
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            merges.append((String(parts[0]), String(parts[1])))
        }

        return try Qwen2BpeTokenizer(vocab: vocab, merges: merges, specialTokens: specialTokens)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        splitBySpecial(text) { chunk, isSpecial in
            if isSpecial {
                if let id = specialTokens[chunk] { out.append(id) }
                return
            }
            pretokenize(chunk) { piece in
                let mapped = ByteEncoder.encode(piece.utf8)
                let bpeTokens = bpe(mapped)
                for tok in bpeTokens {
                    if let id = vocab[tok] {
                        out.append(id)
                    } else if let id = specialTokens[tok] {
                        out.append(id)
                    }
                    // Unknown token: Qwen2 has no <unk>. Drop silently as
                    // upstream never produces one for valid UTF-8 input.
                }
            }
        }
        return out
    }

    // MARK: - Special token split

    private func splitBySpecial(_ text: String, _ handle: (String, Bool) -> Void) {
        guard let regex = specialPattern, !text.isEmpty else {
            if !text.isEmpty { handle(text, false) }
            return
        }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        var cursor = 0
        regex.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
            guard let m = match else { return }
            if m.range.location > cursor {
                let sub = ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
                if !sub.isEmpty { handle(sub, false) }
            }
            handle(ns.substring(with: m.range), true)
            cursor = m.range.location + m.range.length
        }
        if cursor < ns.length {
            let sub = ns.substring(with: NSRange(location: cursor, length: ns.length - cursor))
            if !sub.isEmpty { handle(sub, false) }
        }
    }

    // MARK: - Pretokenize

    private func pretokenize(_ text: String, _ handle: (String) -> Void) {
        guard !text.isEmpty else { return }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        pretokenizeRegex.enumerateMatches(in: text, options: [], range: range) { match, _, _ in
            guard let m = match else { return }
            if m.range.length > 0 {
                handle(ns.substring(with: m.range))
            }
        }
    }

    // MARK: - BPE

    /// Standard GPT-2 BPE: repeatedly merge the lowest-rank adjacent pair
    /// until no pair is mergeable, then return the final symbol list.
    private func bpe(_ text: String) -> [String] {
        if text.isEmpty { return [] }
        var symbols = text.map { String($0) }
        if symbols.count < 2 { return symbols }

        while true {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(symbols.count - 1) {
                let key = "\(symbols[i]) \(symbols[i + 1])"
                if let r = mergeRanks[key], r < bestRank {
                    bestRank = r
                    bestIndex = i
                }
            }
            if bestIndex < 0 { break }

            let first = symbols[bestIndex]
            let second = symbols[bestIndex + 1]
            var merged: [String] = []
            merged.reserveCapacity(symbols.count - 1)
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1 && symbols[i] == first && symbols[i + 1] == second {
                    merged.append(first + second)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
            if symbols.count < 2 { break }
        }
        return symbols
    }

    // MARK: - Byte encoder

    /// GPT-2 style reversible byte→unicode mapping used by Qwen2 BPE.
    ///
    /// Mirrors `transformers.models.qwen2.tokenization_qwen2.bytes_to_unicode`:
    /// - Printable ASCII, Latin-1 supplement (¡..¬), and (®..ÿ) map to themselves.
    /// - The 68 "unprintable" bytes are remapped to code points 256..323.
    ///
    /// After mapping, every byte of a UTF-8 string becomes a single-code-point
    /// unicode character that vocab/merges.txt expect.
    fileprivate enum ByteEncoder {

        /// byte (0..255) → single Unicode scalar.
        static let byteToUnicode: [Character] = {
            var map = [Character](repeating: Character(" "), count: 256)
            var printable = [Int]()
            printable.reserveCapacity(188)
            printable.append(contentsOf: Int(Character("!").asciiValue!)...Int(Character("~").asciiValue!))
            printable.append(contentsOf: 0xA1...0xAC)
            printable.append(contentsOf: 0xAE...0xFF)

            for b in printable {
                map[b] = Character(UnicodeScalar(b)!)
            }

            var extra = 0
            for b in 0..<256 {
                if !printable.contains(b) {
                    let scalar = UnicodeScalar(256 + extra)!
                    map[b] = Character(scalar)
                    extra += 1
                }
            }
            return map
        }()

        /// Encode a UTF-8 byte sequence as a string of mapped characters.
        static func encode(_ bytes: some Sequence<UInt8>) -> String {
            var out = ""
            for b in bytes {
                out.append(byteToUnicode[Int(b)])
            }
            return out
        }
    }
}
