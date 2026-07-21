import Foundation

/// English text → espeak-IPA phonemes for LuxTTS (phase 2).
///
/// espeak-ng (en-us) is what the model was trained on (via piper_phonemize
/// / EmiliaTokenizer), so this G2P reproduces espeak's *clause-level*
/// behavior from an offline-harvested lexicon instead of bundling espeak:
///
///   * per-word pronunciation variants probed from espeak in six carrier
///     contexts (mid / clause-final / before-only-unstressed ($strend2) /
///     before-vowel / before-$pause-word / clause-initial), stored sparsely
///     in `luxtts_en_us_lexicon.tsv.zz` (139k words, raw-DEFLATE)
///   * multi-word merge entries ("in the" -> ɪnðə, "did not" -> dɪdnˌɑːt)
///   * homograph verb/noun/past selection driven by espeak's
///     expect_verb/noun/past counters ($verbf/$nounf/$pastf words)
///   * position-dependent $pause behavior (liaison/flapping blocking)
///   * punctuation semantics (",;:" emit + space, ".!?" emit + no space,
///     "…" silent break, quotes = pause boundary, mid-token "." = "dot")
///
/// The algorithm mirrors `mobius/models/tts/zipvoice/coreml/g2p/
/// reference_g2p.py`, which is score-gated against the espeak oracle
/// (1,000-sentence corpus: 99.6% sentence exact match, 0.01% token edit
/// rate — see Documentation/TTS/LuxTts.md).
public struct LuxTtsG2p: Sendable {

    // MARK: - Data

    /// Sparse variant row; `nil` means "same as mid".
    struct Row: Sendable {
        let mid: String
        let final: String?
        let unstr: String?
        let vowel: String?
        let pause: String?
        let start: String?
        let rvar: String?
    }

    struct PhraseEntry: Sendable {
        let mid: String?
        let midcap: String?
        let final: String?
        let vowel: String?
        let start: String?
        let startcap: String?
        let rvar: String?
        let atEndOnly: Bool
        let flagWords: [String]  // constituent words, for counter updates

        var usableMidVariant: Bool {
            mid != nil || midcap != nil || vowel != nil || rvar != nil
        }
    }

    let entries: [String: Row]
    let phrases: [String: PhraseEntry]
    let maxPhraseLength: Int
    let homographs: [String: [String: String]]
    let verbf: Set<String>
    let verbsf: Set<String>
    let nounf: Set<String>
    let pastf: Set<String>
    let verbextend: Set<String>
    let pauseWords: Set<String>
    let allcapsWords: Set<String>
    let letters: [String: String]

    private static let clausePunct = Set(",.!?;:")
    private static let vowelScalars = Set("aeiouæɑɐɔəɛɜɪʊʌʉɒɚᵻ")
    private static let stressMarks = Set("ˈˌ")
    private static let voiceless = Set("ptkfθsʃ")
    private static let sibilant = Set("szʃʒ")

    // MARK: - Loading

    /// Load the bundled lexicon + aux tables.
    public init() throws {
        guard
            let lexiconURL = Bundle.module.url(
                forResource: "luxtts_en_us_lexicon.tsv", withExtension: "zz"),
            let auxURL = Bundle.module.url(
                forResource: "luxtts_en_us_g2p_aux", withExtension: "json")
        else {
            throw LuxTtsError.tokenizerFailed("bundled G2P resources missing")
        }
        try self.init(lexiconURL: lexiconURL, auxURL: auxURL)
    }

    init(lexiconURL: URL, auxURL: URL) throws {
        let compressed = try Data(contentsOf: lexiconURL)
        let tsvData: Data
        do {
            tsvData = try (compressed as NSData).decompressed(using: .zlib) as Data
        } catch {
            throw LuxTtsError.tokenizerFailed(
                "cannot decompress lexicon: \(error.localizedDescription)")
        }
        guard let tsv = String(data: tsvData, encoding: .utf8) else {
            throw LuxTtsError.tokenizerFailed("lexicon is not UTF-8")
        }

        var entries: [String: Row] = [:]
        entries.reserveCapacity(140_000)
        for line in tsv.split(separator: "\n", omittingEmptySubsequences: true) {
            let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
                .map(String.init)
            guard cols.count >= 2 else { continue }
            func col(_ i: Int) -> String? {
                i < cols.count && !cols[i].isEmpty ? cols[i] : nil
            }
            entries[cols[0]] = Row(
                mid: cols[1], final: col(2), unstr: col(3), vowel: col(4),
                pause: col(5), start: col(6), rvar: col(7))
        }
        self.entries = entries

        let auxData = try Data(contentsOf: auxURL)
        guard
            let aux = try JSONSerialization.jsonObject(with: auxData) as? [String: Any],
            let phrasesJson = aux["phrases"] as? [String: [String: Any]],
            let homographsJson = aux["homographs"] as? [String: [String: String]],
            let flagSets = aux["flag_sets"] as? [String: [String]],
            let letters = aux["letters"] as? [String: String]
        else {
            throw LuxTtsError.tokenizerFailed("malformed G2P aux JSON")
        }

        var phrases: [String: PhraseEntry] = [:]
        var maxLen = 1
        for (key, value) in phrasesJson {
            let flags = value["flags"] as? [String] ?? []
            phrases[key] = PhraseEntry(
                mid: value["mid"] as? String,
                midcap: value["midcap"] as? String,
                final: value["final"] as? String,
                vowel: value["vowel"] as? String,
                start: value["start"] as? String,
                startcap: value["startcap"] as? String,
                rvar: value["r"] as? String,
                atEndOnly: flags.contains("$atend"),
                flagWords: key.split(separator: " ").map(String.init))
            maxLen = max(maxLen, key.split(separator: " ").count)
        }
        self.phrases = phrases
        self.maxPhraseLength = maxLen
        self.homographs = homographsJson
        self.verbf = Set(flagSets["verbf"] ?? [])
        self.verbsf = Set(flagSets["verbsf"] ?? [])
        self.nounf = Set(flagSets["nounf"] ?? [])
        self.pastf = Set(flagSets["pastf"] ?? [])
        self.verbextend = Set(flagSets["verbextend"] ?? [])
        self.pauseWords = Set(flagSets["pause"] ?? []).union(Set(flagSets["brk"] ?? []))
        self.allcapsWords = Set(flagSets["allcaps"] ?? [])
            .subtracting(Set(flagSets["abbrev"] ?? []))
        self.letters = letters
    }

    // MARK: - Public API

    /// English text → espeak-IPA phoneme string (the `tokens.txt` scalar
    /// set). Includes the upstream ZipVoice text normalization.
    public func phonemize(text: String) -> String {
        phonemizeNormalized(LuxTtsEnglishNormalizer.normalize(text))
    }

    // MARK: - Tokenization

    private enum Item {
        case word(String)
        case hyph([String])
        case literal(String)
    }

    private struct RawToken {
        let text: String  // word, digit run, or single punctuation scalar
        let start: Int  // scalar offset
        let end: Int
        var isWordish: Bool {
            guard let c = text.unicodeScalars.first else { return false }
            return CharacterSet.letters.contains(c) || CharacterSet.decimalDigits.contains(c)
        }
    }

    private func rawTokens(_ text: String) -> [RawToken] {
        var tokens: [RawToken] = []
        let scalars = Array(text.unicodeScalars)
        var i = 0
        while i < scalars.count {
            let c = scalars[i]
            if CharacterSet.whitespacesAndNewlines.contains(c) {
                i += 1
                continue
            }
            let start = i
            if CharacterSet.letters.contains(c), c.isASCII {
                i += 1
                while i < scalars.count,
                    (CharacterSet.letters.contains(scalars[i]) && scalars[i].isASCII)
                        || scalars[i] == "'"
                {
                    i += 1
                }
            } else if CharacterSet.decimalDigits.contains(c) {
                i += 1
                while i < scalars.count, CharacterSet.decimalDigits.contains(scalars[i]) {
                    i += 1
                }
            } else {
                i += 1
            }
            tokens.append(
                RawToken(
                    text: String(String.UnicodeScalarView(scalars[start..<i])),
                    start: start, end: i))
        }
        return tokens
    }

    // MARK: - Sentence assembly

    func phonemizeNormalized(_ normalized: String) -> String {
        let raw = rawTokens(normalized)

        // hyphen chains: word(-word)+ with no spaces
        var tokens: [(item: Item?, punct: String?, start: Int, end: Int)] = []
        var i = 0
        while i < raw.count {
            let tok = raw[i]
            if tok.isWordish, isAlphaStart(tok.text),
                i + 2 < raw.count, raw[i + 1].text == "-",
                raw[i + 1].start == tok.end, raw[i + 2].start == raw[i + 1].end,
                isAlphaStart(raw[i + 2].text)
            {
                var parts = [tok.text]
                var j = i + 1
                var lastEnd = tok.end
                while j + 1 < raw.count, raw[j].text == "-", raw[j].start == lastEnd,
                    isAlphaStart(raw[j + 1].text), raw[j + 1].start == raw[j].end
                {
                    parts.append(raw[j + 1].text)
                    lastEnd = raw[j + 1].end
                    j += 2
                }
                tokens.append((.hyph(parts), nil, tok.start, lastEnd))
                i = j
                continue
            }
            if tok.isWordish {
                tokens.append((.word(tok.text), nil, tok.start, tok.end))
            } else {
                tokens.append((nil, tok.text, tok.start, tok.end))
            }
            i += 1
        }

        var pieces: [String] = []
        var clause: [Item] = []
        var breaks: [Bool] = []
        var clauseInitial = true
        var pendingBreak = false

        func flush() {
            defer {
                clause.removeAll()
                breaks.removeAll()
                pendingBreak = false
            }
            guard !clause.isEmpty else { return }
            let chunks = phonemizeClause(
                items: clause, clauseInitial: clauseInitial, breakBefore: breaks)
            let phon = chunks.filter { !$0.isEmpty }.joined(separator: " ")
            if !phon.isEmpty { pieces.append(phon) }
        }

        for (idx, entry) in tokens.enumerated() {
            if let item = entry.item {
                clause.append(item)
                breaks.append(pendingBreak)
                pendingBreak = false
                continue
            }
            guard var ch = entry.punct else { continue }
            if ch == "—" || ch == "–" { ch = ";" }

            let prevGlued = idx > 0 && tokens[idx - 1].end == entry.start
            let nextGlued = idx + 1 < tokens.count && tokens[idx + 1].start == entry.end
            let midToken = prevGlued && nextGlued

            if Self.clausePunct.contains(Character(ch)) || ch == "…" {
                if midToken {
                    // espeak reads glued "." as "dot", "!" as "exclamation";
                    // "," is dropped; "?" becomes a silent join
                    switch ch {
                    case ".":
                        clause.append(.literal("dˈɑːt"))
                        breaks.append(false)
                        continue
                    case "!":
                        clause.append(.literal("ˈɛkskləmˌeɪʃən"))
                        breaks.append(false)
                        continue
                    case ",":
                        continue
                    case "?":
                        flush()
                        clauseInitial = true
                        continue
                    default:
                        break
                    }
                }
                flush()
                if ch != "…" {
                    pieces.append(ch)
                    if ",;:".contains(ch) { pieces.append(" ") }
                }
                clauseInitial = true
            } else if "\"'()[]«»".contains(ch) {
                // quotes/parens: transparent, but a pause boundary
                if !clause.isEmpty { pendingBreak = true }
            }
            // dashes/other symbols: transparent
        }
        flush()

        var text = pieces.joined()
        while text.contains("  ") {
            text = text.replacingOccurrences(of: "  ", with: " ")
        }
        return text.trimmingCharacters(in: .whitespaces)
    }

    private func isAlphaStart(_ s: String) -> Bool {
        guard let c = s.unicodeScalars.first else { return false }
        return CharacterSet.letters.contains(c)
    }

    // MARK: - Clause translation

    private enum UnitKind {
        case word
        case hyph
        case literal
        case phrase(PhraseEntry)
    }

    private func phonemizeClause(
        items: [Item], clauseInitial: Bool, breakBefore: [Bool]
    ) -> [String] {
        let n = items.count
        guard n > 0 else { return [] }

        // unit segmentation: greedy phrase match over consecutive words,
        // consumed only when usable at its position
        var units: [(kind: UnitKind, start: Int, length: Int)] = []
        var i = 0
        while i < n {
            switch items[i] {
            case .hyph:
                units.append((.hyph, i, 1))
                i += 1
                continue
            case .literal:
                units.append((.literal, i, 1))
                i += 1
                continue
            case .word:
                break
            }
            var matched: (PhraseEntry, Int)? = nil
            let maxLen = min(maxPhraseLength, n - i)
            if maxLen >= 2 {
                for length in stride(from: maxLen, through: 2, by: -1) {
                    var words: [String] = []
                    var ok = true
                    for k in i..<(i + length) {
                        guard case .word(let w) = items[k] else {
                            ok = false
                            break
                        }
                        if k > i && breakBefore[k] {
                            ok = false
                            break
                        }
                        words.append(w.lowercased())
                    }
                    guard ok, let entry = phrases[words.joined(separator: " ")] else { continue }
                    let atEnd = i + length == n
                    let usable =
                        entry.usableMidVariant
                        || (atEnd && entry.final != nil)
                        || (clauseInitial && i == 0
                            && (entry.start != nil || entry.startcap != nil))
                    let atEndSatisfied = !entry.atEndOnly || atEnd || entry.usableMidVariant
                    if usable && atEndSatisfied {
                        matched = (entry, length)
                        break
                    }
                }
            }
            if let (entry, length) = matched {
                units.append((.phrase(entry), i, length))
                i += length
            } else {
                units.append((.word, i, 1))
                i += 1
            }
        }

        // pass 1 (left to right): homograph counters + $pause application
        var flags = [String?](repeating: nil, count: units.count)
        var pauseApplied = [Bool](repeating: false, count: units.count)
        var counters = Counters()
        var wordPos = 0
        for (u, unit) in units.enumerated() {
            if breakBefore[unit.start] { wordPos = 0 }
            if case .word = unit.kind, case .word(let token) = items[unit.start] {
                flags[u] = counters.homographKey(token: token)
                if pauseWords.contains(token.lowercased()), wordPos >= 2,
                    unit.start != n - 1
                {
                    pauseApplied[u] = true
                    wordPos = 0  // the pause word restarts the count as word 0
                }
            }
            for k in unit.start..<(unit.start + unit.length) {
                if case .word(let w) = items[k] {
                    updateCounters(&counters, token: w)
                }
                if !pauseApplied[u] { wordPos += 1 }
            }
        }

        // pass 2 (right to left): variant selection ($strend2 chains
        // resolve on the *selected* stress states of following words)
        var chunks = [String](repeating: "", count: units.count)
        var stressedAfter = false
        var nextFirstPhone: Character? = nil
        var nextIsBreak = false

        for u in stride(from: units.count - 1, through: 0, by: -1) {
            let unit = units[u]
            let atEnd = unit.start + unit.length == n
            let atStart = clauseInitial && unit.start == 0
            let chunk: String
            switch unit.kind {
            case .phrase(let entry):
                var tokens: [String] = []
                for k in unit.start..<(unit.start + unit.length) {
                    if case .word(let w) = items[k] { tokens.append(w) }
                }
                chunk = selectPhrase(
                    entry, tokens: tokens, atEnd: atEnd, atStart: atStart,
                    nextFirstPhone: nextFirstPhone, nextIsBreak: nextIsBreak)
            case .hyph:
                guard case .hyph(let parts) = items[unit.start] else { continue }
                chunk = selectHyph(
                    parts, flag: flags[u], atEnd: atEnd, atStart: atStart,
                    nextFirstPhone: nextFirstPhone, nextIsBreak: nextIsBreak,
                    stressedAfter: stressedAfter)
            case .literal:
                guard case .literal(let pron) = items[unit.start] else { continue }
                chunk = pron
            case .word:
                guard case .word(let token) = items[unit.start] else { continue }
                chunk = selectWord(
                    token, flag: flags[u], atEnd: atEnd, atStart: atStart,
                    nextFirstPhone: nextFirstPhone, nextIsBreak: nextIsBreak,
                    stressedAfter: stressedAfter)
            }
            chunks[u] = chunk
            if chunk.contains("ˈ") { stressedAfter = true }
            nextFirstPhone = firstPhone(chunk)
            nextIsBreak = pauseApplied[u] || breakBefore[unit.start]
        }
        return chunks
    }

    // MARK: - espeak expect_verb/noun/past counters (translateword.c)

    private struct Counters {
        var verb = 0
        var verbS = 0
        var noun = 0
        var past = 0

        func homographKey(token: String) -> String? {
            if verb > 0 || (verbS > 0 && token.lowercased().hasSuffix("s")) {
                return "verb"
            }
            if past > 0 { return "past" }
            if noun > 0 { return "noun" }
            return nil
        }
    }

    private static let contractionBase: [String: String] = [
        "won't": "will", "can't": "can", "shan't": "shall",
    ]

    /// The word whose espeak flags drive the counters; n't/'ll/'ve/'s
    /// contractions inherit the auxiliary's flags (doesn't -> does).
    private func flagWord(for token: String) -> String? {
        let lower = token.lowercased().replacingOccurrences(of: "’", with: "'")
        if lower == "i" && token != "I" {
            return nil  // lowercase i is the letter, not the pronoun
        }
        if let base = Self.contractionBase[lower] { return base }
        if lower.hasSuffix("n't") { return String(lower.dropLast(3)) }
        if lower.hasSuffix("'ll") { return "will" }
        if lower.hasSuffix("'ve") { return "have" }
        if lower.hasSuffix("'s") { return "is" }
        return lower
    }

    private func updateCounters(_ counters: inout Counters, token: String) {
        let word = flagWord(for: token)
        if let word {
            if pastf.contains(word) {
                counters.past = 3
                counters.verb = 0
                counters.noun = 0
            } else if verbf.contains(word) {
                counters.verb = 2
                counters.verbS = 0
                counters.noun = 0
            } else if verbsf.contains(word) {
                counters.verb = 0
                counters.verbS = 2
                counters.past = 0
                counters.noun = 0
            } else if nounf.contains(word) {
                counters.noun = 2
                counters.verb = 0
                counters.verbS = 0
                counters.past = 0
            }
        }
        if word == nil || !verbextend.contains(word!) {
            if counters.verb > 0 { counters.verb -= 1 }
            if counters.verbS > 0 { counters.verbS -= 1 }
            if counters.noun > 0 { counters.noun -= 1 }
            if counters.past > 0 { counters.past -= 1 }
        }
    }

    // MARK: - Variant selection

    /// Case-sensitive row first (I vs i, Polish vs polish), then lower-case.
    func lookup(_ word: String) -> Row? {
        let normalized = word.replacingOccurrences(of: "’", with: "'")
        return entries[normalized] ?? entries[normalized.lowercased()]
    }

    private func firstPhone(_ pron: String) -> Character? {
        pron.first { !Self.stressMarks.contains($0) }
    }

    private func selectWord(
        _ token: String, flag: String?, atEnd: Bool, atStart: Bool,
        nextFirstPhone: Character?, nextIsBreak: Bool, stressedAfter: Bool
    ) -> String {
        // all-caps tokens spell out unless espeak marks them $allcaps words
        if token.count >= 2, token == token.uppercased(),
            token.allSatisfy({ $0.isLetter }),
            !allcapsWords.contains(token.lowercased())
        {
            return spellLetters(token)
        }

        if let flag, let variants = homographs[token.lowercased()],
            let form = variants[flag]
        {
            return form
        }

        guard let row = lookup(token) else { return oovPron(token) }

        if atEnd { return row.final ?? row.mid }

        let nextIsVowel = nextFirstPhone.map { Self.vowelScalars.contains($0) } ?? false

        if let unstr = row.unstr, !stressedAfter, !nextIsBreak {
            // liaison/flap still applies on top of the $strend2 form
            var selected = unstr
            if let vowel = row.vowel, nextIsVowel, let last = row.mid.last {
                if vowel == row.mid + "ɹ", selected.hasSuffix(String(last)) {
                    selected += "ɹ"
                } else if vowel == String(row.mid.dropLast()) + "ɾ",
                    selected.hasSuffix(String(last))
                {
                    selected = String(selected.dropLast()) + "ɾ"
                }
            }
            return selected
        }
        if nextIsBreak { return row.pause ?? row.mid }
        if nextFirstPhone == "ɹ", let rvar = row.rvar { return rvar }
        if nextIsVowel, let vowel = row.vowel { return vowel }
        if atStart, let start = row.start { return start }
        return row.mid
    }

    private func selectPhrase(
        _ entry: PhraseEntry, tokens: [String], atEnd: Bool, atStart: Bool,
        nextFirstPhone: Character?, nextIsBreak: Bool
    ) -> String {
        let capitalized = tokens.first?.first?.isUppercase ?? false
        if atStart, capitalized, let startcap = entry.startcap { return startcap }
        if atStart, let start = entry.start { return start }
        if atEnd, let final = entry.final { return final }
        let nextIsVowel = nextFirstPhone.map { Self.vowelScalars.contains($0) } ?? false
        if !atEnd, !nextIsBreak {
            if nextFirstPhone == "ɹ", let rvar = entry.rvar { return rvar }
            if nextIsVowel, let vowel = entry.vowel { return vowel }
        }
        if capitalized, !atEnd, let midcap = entry.midcap { return midcap }
        if !atEnd, let mid = entry.mid { return mid }
        // fall back to word-by-word composition
        var chunks: [String] = []
        for (index, word) in tokens.enumerated() {
            guard let row = lookup(word) else {
                chunks.append(oovPron(word))
                continue
            }
            if index == tokens.count - 1 && atEnd {
                chunks.append(row.final ?? row.mid)
            } else {
                chunks.append(row.mid)
            }
        }
        return chunks.joined(separator: " ")
    }

    /// Hyphen chain: whole-token row if probed (twenty-six), else parts
    /// pronounced separately and concatenated without a space.
    private func selectHyph(
        _ parts: [String], flag: String?, atEnd: Bool, atStart: Bool,
        nextFirstPhone: Character?, nextIsBreak: Bool, stressedAfter: Bool
    ) -> String {
        let key = parts.joined(separator: "-")
        if lookup(key) != nil {
            return selectWord(
                key, flag: flag, atEnd: atEnd, atStart: atStart,
                nextFirstPhone: nextFirstPhone, nextIsBreak: nextIsBreak,
                stressedAfter: stressedAfter)
        }
        var out: [String] = []
        for (index, part) in parts.enumerated() {
            if index == parts.count - 1 {
                out.append(
                    selectWord(
                        part, flag: nil, atEnd: atEnd, atStart: false,
                        nextFirstPhone: nextFirstPhone, nextIsBreak: nextIsBreak,
                        stressedAfter: stressedAfter))
            } else if let row = lookup(part) {
                out.append(row.mid)
            } else {
                out.append(oovPron(part))
            }
        }
        return out.joined()
    }

    // MARK: - Fallbacks

    private func spellLetters(_ word: String) -> String {
        let letters = word.lowercased().filter { $0.isLetter }
        var out = ""
        for (index, letter) in letters.enumerated() {
            let name = self.letters[String(letter)] ?? ""
            let mark = index == letters.count - 1 ? "ˈ" : "ˌ"
            out += letterNameWithStress(name, mark: mark)
        }
        return out
    }

    private func letterNameWithStress(_ name: String, mark: String) -> String {
        let stripped = name.filter { !Self.stressMarks.contains($0) }
        if let index = stripped.firstIndex(where: { Self.vowelScalars.contains($0) }) {
            return String(stripped[..<index]) + mark + String(stripped[index...])
        }
        return mark + stripped
    }

    private func suffixS(_ mid: String) -> String {
        guard let last = mid.last else { return mid }
        if Self.sibilant.contains(last) { return mid + "ɪz" }
        if Self.voiceless.contains(last) { return mid + "s" }
        return mid + "z"
    }

    func oovPron(_ word: String) -> String {
        let lower = word.lowercased().replacingOccurrences(of: "’", with: "'")
        for suffix in ["'s", "s'"] where lower.hasSuffix(suffix) {
            if let base = lookup(String(lower.dropLast(suffix.count))) {
                return suffixS(base.mid)
            }
        }
        if lower.hasSuffix("s"), let base = lookup(String(lower.dropLast())) {
            return suffixS(base.mid)
        }
        if word.count >= 2, word == word.uppercased(), !word.contains("'") {
            return spellLetters(word)
        }
        // camelCase: split at case boundaries, parts space-joined,
        // single letters spoken as letter names (iPhone -> ˈaɪ fˈoʊn)
        let parts = camelParts(word)
        if parts.count > 1 {
            var out: [String] = []
            for part in parts {
                if part.count == 1, part.first!.isLetter {
                    out.append(
                        letterNameWithStress(
                            letters[part.lowercased()] ?? "", mark: "ˈ"))
                } else if let row = lookup(part) {
                    out.append(row.mid)
                } else {
                    out.append(oovPron(part))
                }
            }
            return out.joined(separator: " ")
        }
        return spellLetters(word)
    }

    /// `[A-Z]?[a-z']+|[A-Z]+(?![a-z])` — FluidAudio -> [Fluid, Audio].
    private func camelParts(_ word: String) -> [String] {
        var parts: [String] = []
        var current = ""
        let chars = Array(word)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c.isUppercase {
                // uppercase run: attach the last capital to a following
                // lowercase run (HTTPServer -> HTTP, Server)
                var run = ""
                while i < chars.count, chars[i].isUppercase {
                    run.append(chars[i])
                    i += 1
                }
                let nextIsLower = i < chars.count && (chars[i].isLowercase || chars[i] == "'")
                if nextIsLower {
                    if run.count > 1 { parts.append(String(run.dropLast())) }
                    current = String(run.last!)
                } else {
                    parts.append(run)
                    current = ""
                }
            } else {
                current.append(c)
                i += 1
                while i < chars.count, chars[i].isLowercase || chars[i] == "'" {
                    current.append(chars[i])
                    i += 1
                }
                parts.append(current)
                current = ""
            }
        }
        if !current.isEmpty { parts.append(current) }
        return parts.filter { !$0.isEmpty }
    }
}
