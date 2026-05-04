import Foundation

/// Minimal Mandarin tone sandhi pass operating on a flat list of
/// `MandarinPinyinNormalizer.Syllable`s.
///
/// We deliberately ship only the high-impact, POS-independent rules from
/// `misaki/tone_sandhi.py` — getting these wrong is *audible* in normal
/// speech, while the other (POS-conditioned) rules need a Chinese POS
/// tagger we don't ship yet.
///
/// Rules implemented:
///
///  * **Third-tone sandhi**: any `3 + 3` pair → `2 + 3`. Repeats roll
///    left-to-right (`3 3 3` → `2 2 3`, then we re-scan from the right
///    so trailing pairs win).
///  * **不 (`bu4`) sandhi**: `bu4 + tone-4` → `bu2 + tone-4`
///    (`不要 bù yào → bú yào`).
///  * **一 (`yi1`) sandhi**: `yi1 + tone-4` → `yi2`,
///    `yi1 + tone-{1,2,3}` → `yi4`. The bare ordinal "一" (e.g. 第一,
///    一月) keeps tone 1 — but distinguishing those needs context we
///    don't have, so we apply the contextual rule unconditionally.
///    Acceptable trade-off: even without the carve-out the result is
///    intelligible.
///
/// All rules operate on the underlying `(base, tone)` tuple — sandhi
/// happens *before* `MandarinBopomofoMap.encode` is called.
public enum MandarinToneSandhi {

    /// Apply sandhi to `syllables` in place. Token boundaries (e.g.
    /// punctuation between two third-tones) prevent the rule from
    /// firing — we only consider adjacent voiced syllables.
    public static func apply(_ syllables: inout [MandarinPinyinNormalizer.Syllable]) {
        guard syllables.count >= 2 else { return }

        // Pass 1: 不 / 一 contextual sandhi. Run before 3+3 because the
        // 不/一 promotion can create a third-tone pair upstream.
        for i in 0..<(syllables.count - 1) {
            let next = syllables[i + 1]
            if syllables[i].base == "bu" && syllables[i].tone == 4 && next.tone == 4 {
                syllables[i].tone = 2
            } else if syllables[i].base == "yi" && syllables[i].tone == 1 {
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

        // Pass 2: 3+3 sandhi over runs. Linguistic norm: in any
        // consecutive run of tone-3 syllables, every syllable except
        // the last shifts to tone 2 (`3 3 3 → 2 2 3`,
        // `3 3 → 2 3`). A POS-aware grouping (jieba words) would
        // produce more nuanced output for very long runs but we don't
        // ship a POS tagger.
        var i = 0
        while i < syllables.count {
            guard syllables[i].tone == 3 else {
                i += 1
                continue
            }
            var j = i
            while j < syllables.count && syllables[j].tone == 3 { j += 1 }
            if j - i >= 2 {
                for k in i..<(j - 1) {
                    syllables[k].tone = 2
                }
            }
            i = j
        }
    }
}
