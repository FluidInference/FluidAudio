import Foundation

/// Merge `儿` (er) suffixes into the preceding syllable so that
/// `这儿`, `小孩儿`, `一会儿` emit a single r-coloured token instead
/// of trailing `er` as a separate audible syllable.
///
/// Mirrors `misaki/zh_frontend.py:_merge_erhua` with one simplification:
/// we don't drop the preceding final's `-n` / `-ng` consonant (e.g.
/// `wan + r` stays `wanr` rather than collapsing to `war`). The Kokoro
/// v1.1-zh acoustic model was trained on misaki-style input where the
/// erhua marker is a standalone `ㄦ` appended to the toned final, so
/// the simpler form preserves intelligibility without per-final tables.
///
/// Boundary rules:
///
///   * Only a syllable flagged `isErhuaSuffix` folds. The flag is set by
///     `MandarinG2P` from the source text via `isSuffix(word:preceding:)`:
///     the character must be `儿` / `兒` (not `二`, `而`, `耳`, `尔`, which
///     share the pinyin `er`), it must end its word or stand alone, and
///     the word must not be one misaki keeps as a full `ér` (`女儿`,
///     `婴儿`, `孤儿`, …).
///   * `儿` at index 0 of the sandhi buffer is *never* merged — there is
///     nothing to fold into.
///   * The preceding syllable must itself not be `er` — back-to-back
///     `er er` is left as two syllables.
///
/// misaki also skips words tagged `a`, `j` or `nr` by jieba's POS tagger;
/// this pipeline has no POS tags, so the word lists carry the common
/// cases instead.
///
/// Operates in place on the same `pendingSyllables` buffer that
/// `MandarinToneSandhi.apply` will see — invoke this *before* sandhi
/// so 3+3 promotion considers the (now shorter) buffer.
public enum MandarinErhua {

    /// Fold flagged trailing `er` syllables into their predecessors.
    /// Mutates `syllables` in place; the merged-into syllable gains
    /// `erhua = true`, the trailing `er` is removed.
    public static func merge(_ syllables: inout [MandarinPinyinNormalizer.Syllable]) {
        guard syllables.count >= 2 else { return }

        // Walk back-to-front so removals don't shift unprocessed indices,
        // and skip past the merged-into slot once a merge fires (chained
        // `er er` patterns shouldn't double-merge).
        var i = syllables.count - 1
        while i >= 1 {
            let cur = syllables[i]
            let prev = syllables[i - 1]
            if cur.base == "er" && cur.isErhuaSuffix && shouldMergeInto(prev: prev) {
                syllables[i - 1].erhua = true
                syllables.remove(at: i)
                // Advance past the now-merged anchor so an immediately
                // preceding `er` (rare but possible across word seams)
                // isn't itself folded into something further back.
                i -= 2
            } else {
                i -= 1
            }
        }
    }

    /// Whether the last character of `word` is an erhua suffix that may
    /// fold into the syllable before it. `preceding` is the Hanzi already
    /// in the sandhi buffer before `word`; it is only consulted when the
    /// segmenter split `儿` off as a word of its own (`这` + `儿` when
    /// `这儿` isn't in the phrase dict), which is the one case where the
    /// syllable to fold into belongs to an earlier word.
    ///
    /// Mirrors misaki's rule: the character is `儿`, it is the last one of
    /// its word, and neither the word nor its last two characters are in
    /// `notErhua` unless the word is in `mustErhua`.
    public static func isSuffix(word: [Character], preceding: [Character]) -> Bool {
        guard let last = word.last, last == "儿" || last == "兒" else { return false }
        // The syllable `儿` folds into, and the text the word lists see.
        let context: [Character]
        if word.count >= 2 {
            context = word
        } else {
            guard !preceding.isEmpty else { return false }
            context = Array(preceding.suffix(2)) + word
        }
        let normalized = String(context).replacingOccurrences(of: "兒", with: "儿")
        if mustErhua.contains(normalized) { return true }
        let tail2 = String(normalized.suffix(2))
        let tail3 = String(normalized.suffix(3))
        return !notErhua.contains(normalized) && !notErhua.contains(tail2) && !notErhua.contains(tail3)
    }

    /// Whitelist for the merge predicate. Conservative on purpose —
    /// any non-empty, non-`er` base is allowed.
    private static func shouldMergeInto(
        prev: MandarinPinyinNormalizer.Syllable
    ) -> Bool {
        !prev.base.isEmpty && prev.base != "er"
    }

    /// Words whose final `儿` is always read as erhua. From misaki's
    /// `ZHFrontend.must_erhua`.
    static let mustErhua: Set<String> = [
        "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿",
    ]

    /// Words whose final `儿` keeps its own `ér` syllable — mostly `儿`
    /// meaning "child" (`女儿`, `婴儿`) or part of a name (`红孩儿`).
    /// From misaki's `ZHFrontend.not_erhua`.
    static let notErhua: Set<String> = [
        "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
        "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
        "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
        "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
        "狗儿", "少儿",
    ]
}
