import Foundation

/// POS-aware Mandarin tone sandhi.
///
/// The baseline `MandarinToneSandhi` ships only the POS-independent
/// rules (3+3, 不+tone-4, 一+tone-X). It deliberately misfires on a
/// handful of contexts that matter for natural speech:
///
///  * Ordinal `一` (`第一`, `一月`, `一号`) — the bare ordinal keeps
///    tone 1 but the baseline promotes it to 2/4 unconditionally.
///  * `不` reduplication (`好不好`, `行不行`, `要不要`) — the `不`
///    sits between two copies of the same word and keeps tone 4
///    rather than promoting to tone 2.
///  * Word-grouped 3+3 — the linguistic norm is to apply 3+3 sandhi
///    *within* a prosodic word; cross-word 3+3 promotes only the
///    word-final tone 3. The baseline's pure-run rule overshoots on
///    sentences like `我也想去` (3 3 3 3 → would emit 2 2 2 3,
///    correct output is 2 2 3 3 — `想去` is its own word).
///
/// This module replaces `MandarinToneSandhi.apply(_:)` for callers
/// that have a jieba POS tagger available. Inputs:
///
///  * `syllables` — the pending syllable buffer (in/out).
///  * `words` — `[Range<Int>]` partitioning `syllables.indices` into
///    prosodic words (e.g. `[0..<2, 2..<3]` = "two-syllable word
///    then a one-syllable word"). Ranges must cover every syllable
///    exactly once and be non-overlapping.
///  * `tags` — POS tag per word (jieba's `paddle`/`ictclas` set:
///    `m` = numeral, `r` = pronoun, `n*` = nouns, `v*` = verbs, …).
///    `tags.count == words.count` is required.
///
/// When the caller cannot supply POS tags (no jieba POS tagger
/// loaded), use `MandarinToneSandhi.apply(_:)` instead — that path
/// keeps the baseline rules and stays backward-compatible.
public enum MandarinToneSandhiPOS {

    /// Apply POS-aware sandhi to `syllables` in place.
    ///
    /// Pass order matches the baseline:
    ///   1. 不 / 一 contextual sandhi with POS carve-outs.
    ///   2. 3+3 sandhi, scoped per-word, with cross-word fallback.
    public static func apply(
        _ syllables: inout [MandarinPinyinNormalizer.Syllable],
        words: [Range<Int>],
        tags: [String]
    ) {
        guard syllables.count >= 2 else { return }
        precondition(
            words.count == tags.count,
            "MandarinToneSandhiPOS: words.count (\(words.count)) != tags.count (\(tags.count))"
        )

        // Pass 1: 不 / 一 contextual sandhi with carve-outs.
        applyBuYiSandhi(&syllables, words: words, tags: tags)

        // Pass 2: word-grouped 3+3.
        applyThirdToneSandhi(&syllables, words: words)
    }

    // MARK: - 不 / 一

    /// Walk the syllable buffer applying 不 and 一 contextual rules,
    /// honouring the POS carve-outs from `tone_sandhi.py`.
    private static func applyBuYiSandhi(
        _ syllables: inout [MandarinPinyinNormalizer.Syllable],
        words: [Range<Int>],
        tags: [String]
    ) {
        // Build a quick `syllableIdx -> (wordIdx, positionWithinWord)`
        // lookup so the rule code can ask "what tag is this 一's
        // word?" in O(1).
        var wordOfSyllable = [Int](repeating: -1, count: syllables.count)
        var positionInWord = [Int](repeating: -1, count: syllables.count)
        for (wIdx, range) in words.enumerated() {
            var pos = 0
            for sIdx in range {
                guard sIdx >= 0, sIdx < syllables.count else { continue }
                wordOfSyllable[sIdx] = wIdx
                positionInWord[sIdx] = pos
                pos += 1
            }
        }

        for i in 0..<(syllables.count - 1) {
            let cur = syllables[i]
            let next = syllables[i + 1]

            if cur.base == "bu" && cur.tone == 4 && next.tone == 4 {
                // Reduplication carve-out: 好不好 / 行不行. Detect when
                // the previous syllable matches the next one's base
                // (case-insensitive on the ASCII pinyin) — that's the
                // [X, 不, X] pattern. In that frame `不` keeps tone 4.
                let isReduplication: Bool = {
                    guard i >= 1 else { return false }
                    let prev = syllables[i - 1]
                    return prev.base == next.base
                }()
                if !isReduplication {
                    syllables[i].tone = 2
                }
            } else if cur.base == "yi" && cur.tone == 1 {
                // Ordinal carve-out: when 一 is its own one-syllable
                // word AND that word is tagged as a numeral (`m`) the
                // surrounding context is an ordinal / unit reading
                // (`第一`, `一月`, `一号`, …). Keep tone 1.
                //
                // `第一` segments as `["第", "一"]` with tags
                // `["m", "m"]` from jieba — both are `m`. The carve-out
                // also covers bare year/month/day numerals where 一
                // stands alone in its own word.
                let wIdx = wordOfSyllable[i]
                let isOrdinal: Bool = {
                    guard wIdx >= 0, wIdx < tags.count else { return false }
                    let tag = tags[wIdx]
                    let wordRange = words[wIdx]
                    let isSoloYi = wordRange.count == 1
                    return isSoloYi && tag == "m"
                }()
                if isOrdinal { continue }

                switch next.tone {
                case 4:
                    syllables[i].tone = 2
                case 1, 2, 3:
                    syllables[i].tone = 4
                default:
                    break
                }
            }
        }
    }

    // MARK: - 3+3

    /// Word-scoped 3+3 sandhi.
    ///
    /// Within each word, apply the standard rule: any consecutive
    /// run of tone-3 syllables shifts every syllable but the last to
    /// tone 2 (`3 3 3 → 2 2 3`).
    ///
    /// Across words, when a word ends in tone 3 and the next word
    /// starts with tone 3, only the word-final syllable of the
    /// preceding word promotes to tone 2 (the prosodic break stops
    /// the chain from cascading further left).
    private static func applyThirdToneSandhi(
        _ syllables: inout [MandarinPinyinNormalizer.Syllable],
        words: [Range<Int>]
    ) {
        // Step 1: in-word 3+3 within each word's range.
        for word in words {
            guard !word.isEmpty else { continue }
            var i = word.lowerBound
            while i < word.upperBound {
                guard syllables[i].tone == 3 else {
                    i += 1
                    continue
                }
                var j = i
                while j < word.upperBound && syllables[j].tone == 3 { j += 1 }
                if j - i >= 2 {
                    for k in i..<(j - 1) {
                        syllables[k].tone = 2
                    }
                }
                i = j
            }
        }

        // Step 2: cross-word 3+3. After in-word sandhi only word
        // boundaries remain as candidate (3, 3) pairs — the
        // word-final syllable kept tone 3, and so might the next
        // word's leading syllable. Promote the word-final to tone 2
        // when the immediately following word starts with tone 3.
        //
        // We do not cascade further: only the boundary syllable
        // shifts, even if both words are entirely tone 3 (the
        // in-word pass already handled the rest).
        for k in 0..<(words.count - 1) {
            let lhs = words[k]
            let rhs = words[k + 1]
            guard !lhs.isEmpty, !rhs.isEmpty else { continue }
            let lastIdx = lhs.upperBound - 1
            let firstIdx = rhs.lowerBound
            if syllables[lastIdx].tone == 3 && syllables[firstIdx].tone == 3 {
                syllables[lastIdx].tone = 2
            }
        }
    }
}
