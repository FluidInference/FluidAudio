import Foundation

/// Hanzi → Bopomofo + tone-digit string for the Kokoro v1.1-zh
/// (`kokoro-82m-coreml/ANE-zh/`) acoustic chain.
///
/// Pipeline (mirrors `misaki/zh_frontend.py`):
///
///   1. **Number normalization** — `MandarinNumberNormalizer` verbalizes
///      Arabic numerals, dates, times, percentages, fractions, and
///      currency expressions into Hanzi the rest of the pipeline can
///      speak directly (`2025年5月3日` → `二零二五年五月三日`).
///   2. **Punctuation normalization** — fullwidth → ASCII for the
///      characters the v1.1-zh vocab actually carries (`，` → `,`,
///      `。` → `.`, …).
///   3. **Segmentation** — forward maximum-matching against
///      `MandarinPinyinDict.phrases`, falling back to single-char
///      lookups in `singles`. When a `MandarinJiebaHmm` is wired in,
///      runs of consecutive single-char fallbacks are first re-segmented
///      via the jieba B/M/E/S Viterbi to recover OOV proper-noun
///      boundaries (`特朗普`, `比特币`), and each recovered word is
///      retried against the phrase dict before falling back to per-char.
///   4. **Polyphone disambiguation (optional)** — when a g2pW model is
///      wired, every single-char Hanzi whose dict entry has multiple
///      candidate readings is handed to the BERT classifier with the
///      full sentence as context. The picked bopomofo overrides the
///      dict's first-listed reading and bypasses the rest of the
///      pipeline (sandhi included) since the classifier output already
///      encodes tone.
///   5. **Diacritic → digit** — each pinyin syllable is normalized to
///      `(base, tone)` via `MandarinPinyinNormalizer`.
///   6. **Erhua merge** — `MandarinErhua.merge` folds trailing `儿`
///      into the previous syllable so `小孩儿` emits a single
///      r-coloured token (`ㄒㄧㄠ3ㄏㄞㄦ2`).
///   7. **Tone sandhi** — 3+3 → 2+3, 不 / 一 contextual rules
///      (`MandarinToneSandhi`).
///   8. **Pinyin → Bopomofo** — `MandarinBopomofoMap.encode` produces
///      the final `<initial><final><digit>` string per syllable.
///   9. **Concatenation** — syllables joined with no separator,
///      punctuation interleaved verbatim. The output is fed straight
///      into `KokoroAneVocab.encode`.
///
/// User overrides:
///
///   * `customLexicon` slots in at the front of the segmentation
///     cascade (longest match wins, user beats dict at equal length).
///     Tokens may carry pre-parsed syllables or already-encoded
///     bopomofo and bypass the dict / g2pW lookups entirely.
///
/// Out of scope (deferred — a future PR can extend this without
/// breaking callers):
///
///   * POS-conditioned sandhi from `tone_sandhi.py`.
///   * English-letter normalization (`misaki.zh_normalization`).
public struct MandarinG2P: Sendable {

    private let dict: MandarinPinyinDict
    private let jiebaHmm: MandarinJiebaHmm?
    private let g2pw: MandarinG2pwModel?
    private static let logger = AppLogger(category: "MandarinG2P")

    /// User-supplied pronunciation overrides. When non-empty, slots in
    /// at the front of the segmentation cascade — longest match wins,
    /// user beats dict at equal length. Default ``MandarinCustomLexicon/empty``
    /// is a no-op.
    public var customLexicon: MandarinCustomLexicon = .empty

    public init(
        dict: MandarinPinyinDict,
        jiebaHmm: MandarinJiebaHmm? = nil,
        g2pw: MandarinG2pwModel? = nil
    ) {
        self.dict = dict
        self.jiebaHmm = jiebaHmm
        self.g2pw = g2pw
    }

    public init(dict: MandarinPinyinDict, customLexicon: MandarinCustomLexicon) {
        self.dict = dict
        self.jiebaHmm = nil
        self.g2pw = nil
        self.customLexicon = customLexicon
    }

    /// Convert text to a Bopomofo + tone-digit string ready for
    /// `KokoroAneVocab.encode`. Empty input is rejected with `throws`
    /// to match the existing English path's behaviour.
    public func phonemize(_ text: String) async throws -> String {
        let verbalized = MandarinNumberNormalizer.normalize(text)
        let normalized = Self.normalizeText(verbalized)
        guard !normalized.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }
        let normalizedChars = Array(normalized)
        let result = segment(chars: normalizedChars)
        var segments = result.segments

        // Polyphone disambiguation: when a g2pW model is wired and the
        // segmenter flagged candidate Hanzi, run the classifier and
        // splice in `.bopomofoOverride` cases. The classifier sees the
        // full normalized sentence so it can pick by context.
        if let g2pw, !result.polyphoneTargets.isEmpty {
            do {
                let picks = try await g2pw.disambiguate(
                    chars: normalizedChars,
                    targets: result.polyphoneTargets.map { $0.charPos }
                )
                for target in result.polyphoneTargets {
                    guard let bopomofo = picks[target.charPos] else { continue }
                    let digit = MandarinPolyphoneCatalog.toneDigitForm(bopomofo)
                    segments[target.segmentIdx] = .bopomofoOverride(digit)
                }
            } catch {
                Self.logger.warning(
                    "g2pW disambiguation failed (\(error.localizedDescription)) — "
                        + "falling back to dict pick")
            }
        }

        var output = ""
        var pendingSyllables: [MandarinPinyinNormalizer.Syllable] = []

        func flushPending() {
            guard !pendingSyllables.isEmpty else { return }
            // Order: erhua first (it shrinks the buffer), then sandhi
            // operates on the merged result so 3+3 promotion sees the
            // r-coloured syllable as a single tonal unit.
            MandarinErhua.merge(&pendingSyllables)
            MandarinToneSandhi.apply(&pendingSyllables)
            for syl in pendingSyllables {
                if let bo = MandarinBopomofoMap.encode(
                    syllable: syl.base, tone: syl.tone, erhua: syl.erhua)
                {
                    output.append(bo)
                } else {
                    Self.logger.warning(
                        "Mandarin G2P dropped untranslatable syllable '\(syl.base)\(syl.tone)'")
                }
            }
            pendingSyllables.removeAll(keepingCapacity: true)
        }

        for seg in segments {
            switch seg {
            case .pinyin(let list, _):
                for py in list {
                    pendingSyllables.append(MandarinPinyinNormalizer.normalize(py))
                }
            case .syllables(let list):
                // User-lexicon pinyin tokens — already in
                // (base, tone) form. They join the same syllable buffer
                // so sandhi runs across user/dict boundaries naturally
                // (e.g. user word ending in tone-3 followed by dict
                // word starting with tone-3 → 3+3 promotion).
                pendingSyllables.append(contentsOf: list)
            case .punctuation(let s):
                // Sandhi never crosses punctuation; emit accumulated
                // syllables first.
                flushPending()
                output.append(s)
            case .literal(let s):
                // ASCII letters / digits / unmapped Bopomofo: pass
                // through. KokoroAneVocab will encode what it can and
                // silently drop the rest.
                flushPending()
                output.append(s)
            case .bopomofoOverride(let s):
                // g2pW already encoded the bopomofo + tone — emit
                // verbatim and break the sandhi window so the next
                // syllable starts fresh. (Cross-syllable sandhi
                // through a g2pW pick is a documented limitation.)
                flushPending()
                output.append(s)
            }
        }
        flushPending()

        if output.isEmpty {
            throw KokoroAneError.inputProcessingFailed(
                "Mandarin G2P produced no phonemes for input '\(text)'")
        }
        return output
    }

    /// Quick predicate: should this string be routed through the
    /// Mandarin G2P pipeline (vs. treated as already-phonemised
    /// Bopomofo)?
    public static func looksLikeHanzi(_ text: String) -> Bool {
        for scalar in text.unicodeScalars {
            // Any Han ideograph the acoustic model can't speak without G2P.
            let v = scalar.value
            if (0x4E00...0x9FFF).contains(v)  // CJK Unified Ideographs (URO)
                || (0x3400...0x4DBF).contains(v)  // Extension A
                || (0xF900...0xFAFF).contains(v)  // Compatibility Ideographs
                || (0x20000...0x2FA1F).contains(v)  // Extensions B–F + Compat Supplement
                || (0x30000...0x3134F).contains(v)  // Extension G
                || v == 0x3007  // 〇 ideographic number zero
            {
                return true
            }
        }
        return false
    }

    // MARK: - Segmentation

    enum Segment {
        /// Diacritic-form pinyin syllables. `hanziCount` is the number
        /// of Hanzi consumed from the input — needed by the polyphone
        /// pass to know whether a segment is a single-char fallback
        /// (eligible for g2pW override) or a phrase match (which the
        /// dict already context-disambiguated).
        case pinyin([String], hanziCount: Int)
        /// Pre-parsed syllables (user-lexicon source).
        case syllables([MandarinPinyinNormalizer.Syllable])
        case punctuation(String)  // ASCII punctuation passthrough.
        case literal(String)  // Anything else (ASCII letters, digits,
        // already-phonemised bopomofo, etc.)
        /// Pre-encoded bopomofo + tone digit from a polyphone
        /// disambiguator. Bypasses normalization and sandhi.
        case bopomofoOverride(String)
    }

    /// Segmentation result: the segment list plus a side-channel of
    /// polyphone candidates. Each candidate is a single-char `.pinyin`
    /// segment whose underlying char has > 1 reading in the dict — the
    /// g2pW pass can override `segments[segmentIdx]` by char position.
    struct SegmentResult {
        let segments: [Segment]
        let polyphoneTargets: [PolyphoneTarget]
    }

    struct PolyphoneTarget {
        let segmentIdx: Int
        let charPos: Int
    }

    func segment(chars: [Character]) -> SegmentResult {
        var segments: [Segment] = []
        var polyphoneTargets: [PolyphoneTarget] = []
        var i = 0
        let upperBound = max(2, dict.maxPhraseCharCount)
        var literalBuffer = ""
        var hanziFallbackRun: [Character] = []
        var hanziFallbackStart = 0

        func flushLiteral() {
            if !literalBuffer.isEmpty {
                segments.append(.literal(literalBuffer))
                literalBuffer.removeAll(keepingCapacity: true)
            }
        }

        // Drain a buffered run of consecutive single-char hanzi (chars
        // the FMM phrase loop missed). When the jieba HMM is available
        // we ask it to re-segment the run first; each resulting word is
        // tried against the phrase dict before falling back to a
        // per-char lookup. This recovers boundaries on OOV proper nouns
        // like `特朗普` / `比特币` whose constituent chars individually
        // exist in the singles dict but whose compound only resolves as
        // a phrase. Polyphone targets are flagged on the per-char
        // fallback so g2pW sees them in the original sentence position.
        func flushHanziRun() {
            if hanziFallbackRun.isEmpty { return }
            let runStart = hanziFallbackStart
            defer { hanziFallbackRun.removeAll(keepingCapacity: true) }
            flushLiteral()
            let words =
                jiebaHmm?.segment(String(hanziFallbackRun))
                ?? hanziFallbackRun.map { String($0) }
            var offsetInRun = 0
            for word in words {
                let wordCharCount = word.count
                if wordCharCount >= 2, let pinyin = dict.phrases[word] {
                    segments.append(.pinyin(pinyin, hanziCount: wordCharCount))
                    offsetInRun += wordCharCount
                    continue
                }
                // Per-char fallback for either truly OOV words or single
                // chars from a `S` state. Mirrors the original loop's
                // single-char branch with the polyphone-target flag.
                for (offsetInWord, ch) in word.enumerated() {
                    let absPos = runStart + offsetInRun + offsetInWord
                    if let scalar = ch.unicodeScalars.first,
                        let pinyin = dict.singles[scalar.value],
                        !pinyin.isEmpty
                    {
                        if pinyin.count > 1 {
                            polyphoneTargets.append(
                                PolyphoneTarget(
                                    segmentIdx: segments.count, charPos: absPos))
                        }
                        segments.append(.pinyin([pinyin[0]], hanziCount: 1))
                    } else {
                        // Unknown char — fall through as literal so
                        // KokoroAneVocab can have a shot at it.
                        literalBuffer.append(ch)
                        flushLiteral()
                    }
                }
                offsetInRun += wordCharCount
            }
        }

        while i < chars.count {
            let ch = chars[i]
            // Pure ASCII punctuation passthrough.
            if let scalar = ch.unicodeScalars.first,
                MandarinBopomofoMap.allowedPunctuation.contains(ch) || scalar.value < 0x80
            {
                flushHanziRun()
                if MandarinBopomofoMap.allowedPunctuation.contains(ch) {
                    flushLiteral()
                    segments.append(.punctuation(String(ch)))
                } else {
                    // ASCII letter / digit / etc. — buffer for a single
                    // literal segment so the output stays compact.
                    literalBuffer.append(ch)
                }
                i += 1
                continue
            }

            // User lexicon takes priority over dict (longest match
            // wins, user beats dict at equal length). Allows
            // single-char overrides too (`maxKeyCharCount` may be 1).
            var matched = false
            if let hit = customLexicon.longestMatch(in: chars, from: i) {
                flushLiteral()
                emitLexiconHit(hit.tokens, into: &segments)
                i += hit.length
                continue
            }

            // Forward-max-match against phrases (only worth trying when
            // ≥ 2 hanzi remain).
            let remaining = chars.count - i
            if remaining > 1 {
                let maxLen = min(upperBound, remaining)
                if maxLen >= 2 {
                    for len in stride(from: maxLen, through: 2, by: -1) {
                        let candidate = String(chars[i..<(i + len)])
                        if let pinyin = dict.phrases[candidate] {
                            flushHanziRun()
                            flushLiteral()
                            segments.append(.pinyin(pinyin, hanziCount: len))
                            i += len
                            matched = true
                            break
                        }
                    }
                }
            }
            if matched { continue }

            // Buffer the char into the hanzi-fallback run. Whether the
            // singles dict knows it or not, we let `flushHanziRun`
            // decide — that keeps the HMM input contiguous, which is
            // what jieba expects (a "run" of unsegmented characters).
            if hanziFallbackRun.isEmpty {
                hanziFallbackStart = i
            }
            hanziFallbackRun.append(ch)
            i += 1
        }
        flushHanziRun()
        flushLiteral()
        return SegmentResult(segments: segments, polyphoneTargets: polyphoneTargets)
    }

    /// Emit a user-lexicon hit as a single segment when possible
    /// (consecutive run of one token kind), or as separate segments
    /// when the hit mixes pinyin + bopomofo tokens. Preserving order
    /// matters because sandhi accumulates across `.syllables` segments
    /// but resets at `.literal` segments (matching the existing
    /// punctuation/dict contract).
    private func emitLexiconHit(
        _ tokens: [MandarinCustomLexicon.Token],
        into segments: inout [Segment]
    ) {
        var pendingSyls: [MandarinPinyinNormalizer.Syllable] = []
        var pendingBopo = ""
        func flushSyls() {
            if !pendingSyls.isEmpty {
                segments.append(.syllables(pendingSyls))
                pendingSyls.removeAll(keepingCapacity: true)
            }
        }
        func flushBopo() {
            if !pendingBopo.isEmpty {
                segments.append(.literal(pendingBopo))
                pendingBopo.removeAll(keepingCapacity: true)
            }
        }
        for tok in tokens {
            switch tok {
            case .syllable(let s):
                flushBopo()
                pendingSyls.append(s)
            case .bopomofo(let s):
                flushSyls()
                pendingBopo.append(s)
            }
        }
        flushSyls()
        flushBopo()
    }

    // MARK: - Text normalization

    /// Map fullwidth punctuation to the ASCII forms the v1.1-zh vocab
    /// actually carries, and collapse whitespace to a single space.
    /// Anything not handled here falls through to the segmenter.
    static func normalizeText(_ text: String) -> String {
        var out = ""
        out.reserveCapacity(text.count)
        var lastWasSpace = false
        for ch in text {
            let mapped: Character?
            switch ch {
            case "，", "、": mapped = ","
            case "。": mapped = "."
            case "！": mapped = "!"
            case "？": mapped = "?"
            case "；": mapped = ";"
            case "：": mapped = ":"
            case "（": mapped = "("
            case "）": mapped = ")"
            case "／": mapped = "/"
            case "「", "『": mapped = "“"
            case "」", "』": mapped = "”"
            default: mapped = nil
            }
            if let m = mapped {
                out.append(m)
                lastWasSpace = false
                continue
            }
            if ch.isWhitespace {
                if !lastWasSpace && !out.isEmpty { out.append(" ") }
                lastWasSpace = true
                continue
            }
            out.append(ch)
            lastWasSpace = false
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
