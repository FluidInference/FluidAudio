import Foundation

/// User-supplied pronunciation overrides for the Mandarin
/// (``KokoroAneVariant/mandarin``) Kokoro-ANE pipeline.
///
/// Slots in **at the front** of ``MandarinG2P``'s segmentation cascade:
/// the orchestrator runs a longest-prefix match against the user lexicon
/// before falling back to the bundled `pinyin_phrases.bin` /
/// `pinyin_single.bin` dictionary. A user entry of equal length to a
/// dict entry wins.
///
/// Two entry forms are accepted:
///
/// * **Pinyin (digit form)** — e.g. `["zi4", "jie2", "tiao4", "dong4"]`
///   for `字节跳动`. Tokens are validated at registration time by
///   parsing through ``MandarinPinyinNormalizer/parseDigitForm(_:)`` and
///   re-encoding with ``MandarinBopomofoMap/encode(syllable:tone:)``;
///   bad tokens throw at load time. These tokens **participate in tone
///   sandhi** with surrounding context (3+3 chains, 不 / 一 rules) just
///   like dict entries.
///
/// * **Bopomofo escape** — tokens prefixed with `@`, e.g. `["@ㄈㄨ4"]`.
///   The `@` is stripped and the remainder is treated as already-encoded
///   final-form output (bopomofo + tone digit). Useful for OOV
///   characters that don't fit the pinyin model, or for callers porting
///   pronunciations from another zh-TTS. These tokens **bypass tone
///   sandhi** (already in final form).
///
/// Construction: `init(entries:)` (programmatic) or
/// `load(from:)` / `parse(_:)` (file-based, line format documented on
/// ``parse(_:)``).
public struct MandarinCustomLexicon: Sendable, Equatable {

    /// One pre-validated pronunciation token. The lexicon decomposes
    /// each user entry into an ordered list of these.
    public enum Token: Sendable, Equatable {
        /// Pinyin form (`"zi4"` → `Syllable(base: "zi", tone: 4)`).
        /// Participates in sandhi via ``MandarinG2P``'s syllable
        /// buffer.
        case syllable(MandarinPinyinNormalizer.Syllable)
        /// `@`-escape: pre-encoded bopomofo + tone digit emitted
        /// verbatim. Sandhi never sees these.
        case bopomofo(String)
    }

    /// Hanzi word (matched against ``MandarinG2P``'s segmenter input)
    /// → ordered token list. Equal-length matches resolve in dict order
    /// (deterministic via Swift's hashed dict iteration not being
    /// guaranteed — duplicates are rejected at parse time so this never
    /// matters in practice).
    public let entries: [String: [Token]]

    /// Longest entry key in characters. Pre-computed at init so
    /// ``longestMatch(in:from:)`` can bound its scan.
    public let maxKeyCharCount: Int

    public init(entries: [String: [Token]] = [:]) {
        self.entries = entries
        self.maxKeyCharCount = entries.keys.map { $0.count }.max() ?? 0
    }

    /// Validate raw user input. Each value list is parsed token-by-token;
    /// pinyin tokens go through
    /// ``MandarinPinyinNormalizer/parseDigitForm(_:)`` plus
    /// ``MandarinBopomofoMap/encode(syllable:tone:)`` to confirm they
    /// produce a valid bopomofo string. Bopomofo (`@`-prefixed) tokens
    /// are charset-checked against the v1.1-zh vocab character set.
    /// Throws on the first bad token.
    public init(entries raw: [String: [String]]) throws {
        var validated: [String: [Token]] = [:]
        validated.reserveCapacity(raw.count)
        for (word, tokens) in raw {
            guard !word.isEmpty else {
                throw KokoroAneError.inputProcessingFailed(
                    "MandarinCustomLexicon: empty word")
            }
            guard !tokens.isEmpty else {
                throw KokoroAneError.inputProcessingFailed(
                    "MandarinCustomLexicon: '\(word)' has no tokens")
            }
            var parsed: [Token] = []
            parsed.reserveCapacity(tokens.count)
            for raw in tokens {
                parsed.append(try Self.parseToken(raw, word: word))
            }
            validated[word] = parsed
        }
        self.entries = validated
        self.maxKeyCharCount = validated.keys.map { $0.count }.max() ?? 0
    }

    /// Empty lexicon — the default for ``MandarinG2P``.
    public static let empty = MandarinCustomLexicon(entries: [:] as [String: [Token]])

    public var count: Int { entries.count }
    public var isEmpty: Bool { entries.isEmpty }

    // MARK: - Lookup

    /// Longest-prefix match against `chars[start...]`. Returns the
    /// matched character count and the token list, or `nil` if no
    /// entry's key prefix-matches the substring.
    func longestMatch(
        in chars: [Character], from start: Int
    ) -> (length: Int, tokens: [Token])? {
        guard !entries.isEmpty, start < chars.count else { return nil }
        let remaining = chars.count - start
        let maxLen = min(maxKeyCharCount, remaining)
        guard maxLen >= 1 else { return nil }
        for len in stride(from: maxLen, through: 1, by: -1) {
            let candidate = String(chars[start..<(start + len)])
            if let tokens = entries[candidate] {
                return (len, tokens)
            }
        }
        return nil
    }

    // MARK: - Merge

    /// Combine two lexicons. Keys from `other` overwrite keys from
    /// `self` on collision (mirrors ``TtsCustomLexicon/merged(with:)``).
    public func merged(with other: MandarinCustomLexicon) -> MandarinCustomLexicon {
        var combined = entries
        for (k, v) in other.entries {
            combined[k] = v
        }
        return MandarinCustomLexicon(entries: combined)
    }

    // MARK: - File I/O

    /// Load a lexicon from a UTF-8 text file. See ``parse(_:)`` for the
    /// format spec.
    public static func load(from url: URL) throws -> MandarinCustomLexicon {
        let data = try Data(contentsOf: url)
        guard let content = String(data: data, encoding: .utf8) else {
            throw KokoroAneError.inputProcessingFailed(
                "MandarinCustomLexicon: file is not valid UTF-8 at \(url.path)")
        }
        return try parse(content)
    }

    /// Line format:
    ///
    /// ```
    /// # Comments start with '#'.
    /// # Blank lines are skipped.
    /// # First whitespace-run separates word from tokens.
    /// 字节跳动  zi4 jie2 tiao4 dong4
    /// 比亚迪    bi3 ya4 di2
    /// foo       @ㄈㄨ4
    /// ```
    ///
    /// Throws on duplicate words (last-wins is too easy to misread —
    /// callers must dedupe explicitly), empty token lists, or any token
    /// that fails validation.
    public static func parse(_ content: String) throws -> MandarinCustomLexicon {
        var raw: [String: [String]] = [:]
        for (lineIndex, rawLine) in content.split(
            separator: "\n", omittingEmptySubsequences: false
        ).enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let parts = line.split(
                whereSeparator: { $0.isWhitespace }
            ).map(String.init)
            guard parts.count >= 2 else {
                throw KokoroAneError.inputProcessingFailed(
                    "MandarinCustomLexicon: line \(lineIndex + 1) has no tokens: '\(line)'"
                )
            }
            let word = parts[0]
            let tokens = Array(parts.dropFirst())
            if raw[word] != nil {
                throw KokoroAneError.inputProcessingFailed(
                    "MandarinCustomLexicon: duplicate word '\(word)' on line \(lineIndex + 1)"
                )
            }
            raw[word] = tokens
        }
        return try MandarinCustomLexicon(entries: raw)
    }

    // MARK: - Token validation

    private static func parseToken(_ raw: String, word: String) throws -> Token {
        guard !raw.isEmpty else {
            throw KokoroAneError.inputProcessingFailed(
                "MandarinCustomLexicon: '\(word)' has an empty token")
        }
        if raw.hasPrefix("@") {
            let bopo = String(raw.dropFirst())
            guard !bopo.isEmpty else {
                throw KokoroAneError.inputProcessingFailed(
                    "MandarinCustomLexicon: '\(word)' has a bare '@' token")
            }
            try validateBopomofo(bopo, word: word)
            return .bopomofo(bopo)
        }
        guard let syl = MandarinPinyinNormalizer.parseDigitForm(raw) else {
            throw KokoroAneError.inputProcessingFailed(
                "MandarinCustomLexicon: '\(word)' has invalid pinyin token "
                    + "'\(raw)' (expected base+tone-digit, e.g. 'zi4')")
        }
        guard MandarinBopomofoMap.encode(syllable: syl.base, tone: syl.tone) != nil else {
            throw KokoroAneError.inputProcessingFailed(
                "MandarinCustomLexicon: '\(word)' token '\(raw)' parsed but "
                    + "MandarinBopomofoMap can't encode it")
        }
        return .syllable(syl)
    }

    /// Verify every character in a `@`-escape token is in the v1.1-zh
    /// vocab's emit-character set: bopomofo glyphs (initials + finals),
    /// the special hanzi tokens kokoro was trained on, tone digits 1-5,
    /// and the punctuation passthrough set.
    private static func validateBopomofo(_ bopo: String, word: String) throws {
        for ch in bopo {
            if Self.allowedBopomofoCharSet.contains(ch) { continue }
            throw KokoroAneError.inputProcessingFailed(
                "MandarinCustomLexicon: '\(word)' bopomofo token contains "
                    + "char '\(ch)' that is not in the v1.1-zh vocab")
        }
    }

    /// Pre-computed union of every legal output character from
    /// ``MandarinBopomofoMap`` plus tone digits 1-5 and allowed
    /// punctuation. Built lazily on first reference.
    private static let allowedBopomofoCharSet: Set<Character> = {
        var set = Set<Character>()
        for value in MandarinBopomofoMap.initialMap.values {
            for ch in value { set.insert(ch) }
        }
        for value in MandarinBopomofoMap.finalMap.values {
            for ch in value { set.insert(ch) }
        }
        for d in "12345" { set.insert(d) }
        for p in MandarinBopomofoMap.allowedPunctuation { set.insert(p) }
        return set
    }()
}
