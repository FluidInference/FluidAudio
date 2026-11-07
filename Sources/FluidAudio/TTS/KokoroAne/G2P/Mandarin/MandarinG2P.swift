#if TTS
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
///      characters the v1.1-zh vocab actually carries, with the spaces
///      misaki's `map_punctuation` adds (`，` → `, `, `。` → `. `,
///      `（` → ` (`, …).
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
///   6. **Erhua merge** — `MandarinErhua.merge` folds a word-final `儿`
///      into the previous syllable so `小孩儿` emits a single
///      r-coloured syllable (`ㄒ要3ㄏㄞR2`). Other characters read `er`
///      (`二`, `而`, `耳`, `尔`) keep their own syllable.
///   7. **Tone sandhi** — 3+3 → 2+3, 不 / 一 contextual rules
///      (`MandarinToneSandhi`).
///   8. **Pinyin → Bopomofo** — `MandarinBopomofoMap.encode` produces
///      the final `<initial><final><digit>` string per syllable.
///   9. **Concatenation** — syllables of one word joined with no
///      separator, adjacent words joined with misaki's `/`
///      (`MandarinWordBoundaries`), punctuation interleaved verbatim.
///      The output is fed straight into `KokoroAneVocab.encode`.
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
        try await phonemizeTracingPunctuation(text).phonemes
    }

    /// ``phonemize(_:)``, plus the mark as written for each punctuation
    /// mark the frontend rewrote (`、` → `,`), keyed by the rewritten
    /// mark's Unicode-scalar offset in `phonemes`.
    func phonemizeTracingPunctuation(
        _ text: String
    ) async throws -> (phonemes: String, punctuationSources: [Int: Character]) {
        let verbalized = MandarinNumberNormalizer.normalize(text)
        let traced = Self.normalizeTracingPunctuation(verbalized)
        guard !traced.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }
        let normalizedChars = traced.map(\.char)
        let result = segment(chars: normalizedChars, punctuationSources: traced.map(\.source))
        var segments = result.segments
        // From the dict readings, before g2pW, as misaki merges words on
        // pypinyin's.
        let slashBefore = Self.wordSeparators(segments: segments, wordStarts: result.wordStarts)

        // Polyphone disambiguation: when a g2pW model is wired and the
        // segmenter flagged candidate Hanzi, run the classifier and
        // splice in `.disambiguated` cases. The classifier sees the
        // full normalized sentence so it can pick by context.
        if let g2pw, !result.polyphoneTargets.isEmpty {
            do {
                let picks = try await g2pw.disambiguate(
                    chars: normalizedChars,
                    targets: result.polyphoneTargets.map { $0.charPos }
                )
                for target in result.polyphoneTargets {
                    guard let bopomofo = picks[target.charPos] else { continue }
                    guard let syllable = MandarinBopomofoMap.decode(bopomofo: bopomofo) else {
                        Self.logger.warning(
                            "g2pW label '\(bopomofo)' for '\(normalizedChars[target.charPos])' "
                                + "is not a known syllable — keeping dict pick")
                        continue
                    }
                    segments[target.segmentIdx] = .disambiguated(
                        syllable, hanzi: normalizedChars[target.charPos])
                }
            } catch {
                Self.logger.warning(
                    "g2pW disambiguation failed (\(error.localizedDescription)) — "
                        + "falling back to dict pick")
            }
        }

        var output = ""
        var punctuationSources: [Int: Character] = [:]
        var pendingSyllables: [MandarinPinyinNormalizer.Syllable] = []
        // Parallel to `pendingSyllables`: a `/` goes before this syllable.
        var pendingSlash: [Bool] = []
        // The Hanzi behind `pendingSyllables`, so the erhua pass can tell
        // `儿` from `二` / `而` / `耳` / `尔`, which all read `er`.
        var pendingHanzi: [Character] = []

        func appendPending(_ syllables: [MandarinPinyinNormalizer.Syllable], slash: Bool) {
            pendingSlash.append(contentsOf: syllables.indices.map { $0 == 0 && slash })
            pendingSyllables.append(contentsOf: syllables)
        }

        func flushPending() {
            defer { pendingHanzi.removeAll(keepingCapacity: true) }
            guard !pendingSyllables.isEmpty else { return }
            // Order: erhua first (it shrinks the buffer), then sandhi
            // operates on the merged result so 3+3 promotion sees the
            // r-coloured syllable as a single tonal unit.
            MandarinErhua.merge(&pendingSyllables, slashes: &pendingSlash)
            MandarinToneSandhi.apply(&pendingSyllables)
            var emitted = false
            var slash = false
            for (syl, slashBefore) in zip(pendingSyllables, pendingSlash) {
                slash = slash || slashBefore
                if let bo = MandarinBopomofoMap.encode(
                    syllable: syl.base, tone: syl.tone, erhua: syl.erhua)
                {
                    if slash && emitted { output.append("/") }
                    output.append(bo)
                    emitted = true
                    slash = false
                } else {
                    Self.logger.warning(
                        "Mandarin G2P dropped untranslatable syllable '\(syl.base)\(syl.tone)'")
                }
            }
            pendingSyllables.removeAll(keepingCapacity: true)
            pendingSlash.removeAll(keepingCapacity: true)
        }

        for (index, seg) in segments.enumerated() {
            let slash = slashBefore[index]
            switch seg {
            case .pinyin(let list, let word):
                var syllables = list.map(MandarinPinyinNormalizer.normalize)
                let chars = Array(word)
                // Readings line up one per character; a phrase entry that
                // doesn't leaves no character to check, so nothing folds.
                if chars.count == syllables.count, let last = syllables.indices.last,
                    syllables[last].base == "er",
                    MandarinErhua.isSuffix(word: chars, preceding: pendingHanzi)
                {
                    syllables[last].isErhuaSuffix = true
                }
                appendPending(syllables, slash: slash)
                pendingHanzi.append(contentsOf: chars)
            case .syllables(let list, _):
                // User-lexicon pinyin tokens — already in
                // (base, tone) form. They join the same syllable buffer
                // so sandhi runs across user/dict boundaries naturally
                // (e.g. user word ending in tone-3 followed by dict
                // word starting with tone-3 → 3+3 promotion). They are
                // taken literally for erhua: their `er` tokens stay
                // unflagged, and a `儿` after them has no Hanzi context
                // to fold by.
                appendPending(list, slash: slash)
                pendingHanzi.removeAll(keepingCapacity: true)
            case .punctuation(let s, let source):
                // Sandhi never crosses punctuation; emit accumulated
                // syllables first.
                flushPending()
                if let source { punctuationSources[output.unicodeScalars.count] = source }
                output.append(s)
            case .literal(let s):
                // ASCII letters / digits / unmapped Bopomofo: pass
                // through. KokoroAneVocab will encode what it can and
                // silently drop the rest.
                flushPending()
                output.append(s)
            case .disambiguated(var syllable, let hanzi):
                // A g2pW pick joins the buffer like a dict reading so
                // sandhi, erhua and the vocab encoding treat it the same.
                if syllable.base == "er",
                    MandarinErhua.isSuffix(word: [hanzi], preceding: pendingHanzi)
                {
                    syllable.isErhuaSuffix = true
                }
                appendPending([syllable], slash: slash)
                pendingHanzi.append(hanzi)
            }
        }
        flushPending()

        if output.isEmpty {
            throw KokoroAneError.inputProcessingFailed(
                "Mandarin G2P produced no phonemes for input '\(text)'")
        }
        return (output, punctuationSources)
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
        /// Diacritic-form pinyin syllables for `word`, the Hanzi consumed
        /// from the input. A one-character `word` is a single-char fallback
        /// (eligible for g2pW override); a longer one is a phrase match,
        /// which the dict already context-disambiguated. The text itself
        /// is what lets the erhua pass tell `儿` from other `er` readings.
        case pinyin([String], word: String)
        /// Pre-parsed syllables (user-lexicon source). `word` is the Hanzi
        /// the lexicon hit consumed, on the hit's first segment only.
        case syllables([MandarinPinyinNormalizer.Syllable], word: String)
        /// ASCII punctuation passthrough; `source` is the fullwidth mark
        /// it was normalized from.
        case punctuation(String, source: Character?)
        case literal(String)  // Anything else (ASCII letters, digits,
        // already-phonemised bopomofo, etc.)
        /// Single-char reading chosen by the polyphone disambiguator,
        /// already in `(base, tone)` form. `hanzi` is the source char so
        /// the erhua pass keeps its context.
        case disambiguated(MandarinPinyinNormalizer.Syllable, hanzi: Character)
    }

    /// Segmentation result: the segment list plus a side-channel of
    /// polyphone candidates. Each candidate is a single-char `.pinyin`
    /// segment whose underlying char has > 1 reading in the dict — the
    /// g2pW pass can override `segments[segmentIdx]` by char position.
    struct SegmentResult {
        let segments: [Segment]
        let polyphoneTargets: [PolyphoneTarget]
        /// Parallel to `segments`: the segment starts a word. The
        /// per-character fallbacks of one jieba-HMM word, and the
        /// segments of one lexicon hit, are one word.
        let wordStarts: [Bool]
    }

    struct PolyphoneTarget {
        let segmentIdx: Int
        let charPos: Int
    }

    /// `punctuationSources`, when given, runs parallel to `chars` (see
    /// ``normalizeTracingPunctuation(_:)``).
    func segment(chars: [Character], punctuationSources: [Character?] = []) -> SegmentResult {
        var segments: [Segment] = []
        var wordStarts: [Bool] = []
        var polyphoneTargets: [PolyphoneTarget] = []
        var i = 0
        let upperBound = max(2, dict.maxPhraseCharCount)
        var literalBuffer = ""
        var hanziFallbackRun: [Character] = []
        var hanziFallbackStart = 0

        func emit(_ segment: Segment, startsWord: Bool) {
            segments.append(segment)
            wordStarts.append(startsWord)
        }

        func flushLiteral() {
            if !literalBuffer.isEmpty {
                emit(.literal(literalBuffer), startsWord: false)
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
                    emit(.pinyin(pinyin, word: word), startsWord: true)
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
                        emit(.pinyin([pinyin[0]], word: String(ch)), startsWord: offsetInWord == 0)
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
                    let source = i < punctuationSources.count ? punctuationSources[i] : nil
                    emit(.punctuation(String(ch), source: source), startsWord: false)
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
                flushHanziRun()
                flushLiteral()
                let word = String(chars[i..<(i + hit.length)])
                for (k, hitSegment) in lexiconSegments(hit.tokens, word: word).enumerated() {
                    emit(hitSegment, startsWord: k == 0)
                }
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
                            emit(.pinyin(pinyin, word: candidate), startsWord: true)
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
        return SegmentResult(segments: segments, polyphoneTargets: polyphoneTargets, wordStarts: wordStarts)
    }

    /// A user-lexicon hit as a single segment when possible
    /// (consecutive run of one token kind), or as separate segments
    /// when the hit mixes pinyin + bopomofo tokens. Preserving order
    /// matters because sandhi accumulates across `.syllables` segments
    /// but resets at `.literal` segments (matching the existing
    /// punctuation/dict contract).
    private func lexiconSegments(_ tokens: [MandarinCustomLexicon.Token], word: String) -> [Segment] {
        var segments: [Segment] = []
        var pendingSyls: [MandarinPinyinNormalizer.Syllable] = []
        var pendingBopo = ""
        func flushSyls() {
            if !pendingSyls.isEmpty {
                segments.append(.syllables(pendingSyls, word: segments.isEmpty ? word : ""))
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
        return segments
    }

    /// Parallel to `segments`: whether a `/` goes before the segment.
    /// Words are grouped per `wordStarts`; punctuation and literals end a
    /// run, so no `/` is written next to them (misaki writes none next to
    /// punctuation, and literal bopomofo carries no tones to merge on).
    static func wordSeparators(segments: [Segment], wordStarts: [Bool]) -> [Bool] {
        var separators = [Bool](repeating: false, count: segments.count)
        var words: [MandarinWordBoundaries.Word] = []
        var firstSegments: [Int] = []
        var text: [Character] = []
        var tones: [Int] = []

        func closeWord() {
            guard !text.isEmpty || !tones.isEmpty else { return }
            words.append(MandarinWordBoundaries.Word(text: text, tones: tones))
            text.removeAll()
            tones.removeAll()
        }
        func closeRun() {
            closeWord()
            for (k, separator) in MandarinWordBoundaries.separators(for: words).enumerated() where separator {
                separators[firstSegments[k]] = true
            }
            words.removeAll()
            firstSegments.removeAll()
        }
        func add(_ index: Int, text segmentText: String, tones segmentTones: [Int]) {
            if wordStarts[index] || firstSegments.count == words.count {
                closeWord()
                firstSegments.append(index)
            }
            text.append(contentsOf: segmentText)
            tones.append(contentsOf: segmentTones)
        }

        for (index, segment) in segments.enumerated() {
            switch segment {
            case .pinyin(let list, let word):
                add(index, text: word, tones: list.map { MandarinPinyinNormalizer.normalize($0).tone })
            case .syllables(let list, let word):
                add(index, text: word, tones: list.map(\.tone))
            case .disambiguated(let syllable, let hanzi):
                add(index, text: String(hanzi), tones: [syllable.tone])
            case .punctuation, .literal:
                closeRun()
            }
        }
        closeRun()
        return separators
    }

    // MARK: - Text normalization

    /// misaki's `ZHG2P.map_punctuation`: fullwidth marks → the ASCII marks
    /// the v1.1-zh vocab carries, with the spaces misaki puts around them.
    static let punctuationMap: [Character: String] = [
        "，": ", ", "、": ", ", "。": ". ", "．": ". ", "！": "! ", "？": "? ",
        "；": "; ", "：": ": ", "（": " (", "）": ") ", "／": "/",
        "«": " “", "《": " “", "「": " “", "『": " “", "【": " “",
        "»": "” ", "》": "” ", "」": "” ", "』": "” ", "】": "” ",
    ]

    /// Map fullwidth punctuation per ``punctuationMap`` and collapse
    /// whitespace runs to a single space, trimming the ends. (misaki keeps
    /// both spaces where two mapped marks meet, as in `：「`.) Anything not
    /// handled here falls through to the segmenter.
    static func normalizeText(_ text: String) -> String {
        String(normalizeTracingPunctuation(text).map(\.char))
    }

    /// ``normalizeText(_:)`` per character, with the fullwidth mark each
    /// mapped mark came from.
    static func normalizeTracingPunctuation(_ text: String) -> [(char: Character, source: Character?)] {
        var out: [(char: Character, source: Character?)] = []
        out.reserveCapacity(text.count)
        func appendSpace() {
            if let last = out.last, !last.char.isWhitespace { out.append((" ", nil)) }
        }
        for ch in text {
            if let mapped = punctuationMap[ch] {
                for m in mapped {
                    if m == " " { appendSpace() } else { out.append((m, ch)) }
                }
            } else if ch.isWhitespace {
                appendSpace()
            } else {
                out.append((ch, nil))
            }
        }
        while out.last?.char.isWhitespace == true { out.removeLast() }
        return out
    }
}
#endif
