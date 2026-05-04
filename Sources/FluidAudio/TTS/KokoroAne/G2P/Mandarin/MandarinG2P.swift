import Foundation

/// Hanzi → Bopomofo + tone-digit string for the Kokoro v1.1-zh
/// (`kokoro-82m-coreml/ANE-zh/`) acoustic chain.
///
/// Pipeline (mirrors `misaki/zh_frontend.py`):
///
///   1. **Punctuation normalization** — fullwidth → ASCII for the
///      characters the v1.1-zh vocab actually carries (`，` → `,`,
///      `。` → `.`, …).
///   2. **Segmentation** — forward maximum-matching against
///      `MandarinPinyinDict.phrases`, falling back to single-char
///      lookups in `singles`. Matches `MagpieMandarinTokenizer`'s
///      strategy — full jieba HMM is not ported.
///   3. **Polyphone disambiguation (optional)** — when a g2pW model is
///      wired, every single-char Hanzi whose dict entry has multiple
///      candidate readings (or that the polyphone catalog flags) is
///      handed to the BERT classifier with the full sentence as
///      context. The picked bopomofo overrides the dict's first-listed
///      reading and bypasses the rest of the pipeline (sandhi
///      included) since the classifier output already encodes tone.
///   4. **Diacritic → digit** — each pinyin syllable is normalized to
///      `(base, tone)` via `MandarinPinyinNormalizer`.
///   5. **Tone sandhi** — 3+3 → 2+3, 不 / 一 contextual rules
///      (`MandarinToneSandhi`).
///   6. **Pinyin → Bopomofo** — `MandarinBopomofoMap.encode` produces
///      the final `<initial><final><digit>` string per syllable.
///   7. **Concatenation** — syllables joined with no separator,
///      punctuation interleaved verbatim. The output is fed straight
///      into `KokoroAneVocab.encode`.
///
/// Out of scope (deferred — a future PR can extend this without
/// breaking callers):
///
///   * Erhua merging (`儿` suffix collapse).
///   * POS-conditioned sandhi from `tone_sandhi.py`.
///   * Number / datetime / English-letter normalization
///     (`misaki.zh_normalization`).
public struct MandarinG2P: Sendable {

    private let dict: MandarinPinyinDict
    private let g2pw: MandarinG2pwModel?
    private static let logger = AppLogger(category: "MandarinG2P")

    public init(dict: MandarinPinyinDict, g2pw: MandarinG2pwModel? = nil) {
        self.dict = dict
        self.g2pw = g2pw
    }

    /// Convert text to a Bopomofo + tone-digit string ready for
    /// `KokoroAneVocab.encode`. Empty input is rejected with `throws`
    /// to match the existing English path's behaviour.
    public func phonemize(_ text: String) async throws -> String {
        let normalized = Self.normalizeText(text)
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
            MandarinToneSandhi.apply(&pendingSyllables)
            for syl in pendingSyllables {
                if let bo = MandarinBopomofoMap.encode(syllable: syl.base, tone: syl.tone) {
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
            // CJK Unified Ideographs (U+4E00…U+9FFF) +
            // Extension A (U+3400…U+4DBF). Anything in those ranges is
            // a hanzi the model can't speak directly without G2P.
            let v = scalar.value
            if (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v) {
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

        func flushLiteral() {
            if !literalBuffer.isEmpty {
                segments.append(.literal(literalBuffer))
                literalBuffer.removeAll(keepingCapacity: true)
            }
        }

        while i < chars.count {
            let ch = chars[i]
            // Pure ASCII punctuation passthrough.
            if let scalar = ch.unicodeScalars.first,
                MandarinBopomofoMap.allowedPunctuation.contains(ch) || scalar.value < 0x80
            {
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

            // Forward-max-match against phrases (only worth trying when
            // ≥ 2 hanzi remain).
            var matched = false
            let remaining = chars.count - i
            if remaining > 1 {
                let maxLen = min(upperBound, remaining)
                if maxLen >= 2 {
                    for len in stride(from: maxLen, through: 2, by: -1) {
                        let candidate = String(chars[i..<(i + len)])
                        if let pinyin = dict.phrases[candidate] {
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

            // Single-char lookup.
            let scalar = ch.unicodeScalars.first
            if let scalar, let pinyin = dict.singles[scalar.value], !pinyin.isEmpty {
                flushLiteral()
                if pinyin.count > 1 {
                    polyphoneTargets.append(
                        PolyphoneTarget(segmentIdx: segments.count, charPos: i))
                }
                segments.append(.pinyin([pinyin[0]], hanziCount: 1))
            } else {
                // Unknown char (likely Bopomofo, fullwidth latin, an
                // emoji, etc.). Pass through verbatim — KokoroAneVocab
                // will tokenise what it recognises.
                literalBuffer.append(ch)
            }
            i += 1
        }
        flushLiteral()
        return SegmentResult(segments: segments, polyphoneTargets: polyphoneTargets)
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
