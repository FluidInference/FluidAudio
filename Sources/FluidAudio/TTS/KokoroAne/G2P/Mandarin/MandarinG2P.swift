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
///      lookups in `singles`. Matches `MagpieMandarinTokenizer`'s
///      strategy — full jieba HMM is not ported.
///   4. **Diacritic → digit** — each pinyin syllable is normalized to
///      `(base, tone)` via `MandarinPinyinNormalizer`.
///   5. **Erhua merge** — `MandarinErhua.merge` folds trailing `儿`
///      into the previous syllable so `小孩儿` emits a single
///      r-coloured token (`ㄒㄧㄠ3ㄏㄞㄦ2`).
///   6. **Tone sandhi** — 3+3 → 2+3, 不 / 一 contextual rules
///      (`MandarinToneSandhi`).
///   7. **Pinyin → Bopomofo** — `MandarinBopomofoMap.encode` produces
///      the final `<initial><final><digit>` string per syllable.
///   8. **Concatenation** — syllables joined with no separator,
///      punctuation interleaved verbatim. The output is fed straight
///      into `KokoroAneVocab.encode`.
///
/// Out of scope (deferred — a future PR can extend this without
/// breaking callers):
///
///   * POS-conditioned sandhi from `tone_sandhi.py`.
///   * English-letter normalization (`misaki.zh_normalization`).
public struct MandarinG2P: Sendable {

    private let dict: MandarinPinyinDict
    private static let logger = AppLogger(category: "MandarinG2P")

    public init(dict: MandarinPinyinDict) {
        self.dict = dict
    }

    /// Convert text to a Bopomofo + tone-digit string ready for
    /// `KokoroAneVocab.encode`. Empty input is rejected with `throws`
    /// to match the existing English path's behaviour.
    public func phonemize(_ text: String) throws -> String {
        let verbalized = MandarinNumberNormalizer.normalize(text)
        let normalized = Self.normalizeText(verbalized)
        guard !normalized.isEmpty else {
            throw KokoroAneError.inputProcessingFailed("(empty input)")
        }
        let segments = segment(normalized)
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
            case .pinyin(let list):
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
        case pinyin([String])  // Diacritic-form pinyin syllables.
        case punctuation(String)  // ASCII punctuation passthrough.
        case literal(String)  // Anything else (ASCII letters, digits,
        // already-phonemised bopomofo, etc.)
    }

    func segment(_ text: String) -> [Segment] {
        var segments: [Segment] = []
        let chars = Array(text)
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
                            segments.append(.pinyin(pinyin))
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
                segments.append(.pinyin([pinyin[0]]))
            } else {
                // Unknown char (likely Bopomofo, fullwidth latin, an
                // emoji, etc.). Pass through verbatim — KokoroAneVocab
                // will tokenise what it recognises.
                literalBuffer.append(ch)
            }
            i += 1
        }
        flushLiteral()
        return segments
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
