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
///   * `儿` at index 0 of the sandhi buffer is *never* merged — keeps
///     standalone words (`儿子 érzi`, `儿童 értóng`) intact.
///   * The preceding syllable must itself not be `er` — back-to-back
///     `er er` is left as two syllables.
///   * Any non-`er` base is mergeable. The whitelist is intentionally
///     loose; cases that misaki blocks (e.g. `儿` at the start of a
///     polysyllabic word) are already filtered out by the
///     `dict.phrases` / single-char lookup happening upstream.
///
/// Operates in place on the same `pendingSyllables` buffer that
/// `MandarinToneSandhi.apply` will see — invoke this *before* sandhi
/// so 3+3 promotion considers the (now shorter) buffer.
public enum MandarinErhua {

    /// Fold trailing `er` syllables into their predecessors. Mutates
    /// `syllables` in place; the merged-into syllable gains
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
            if cur.base == "er" && shouldMergeInto(prev: prev) {
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

    /// Whitelist for the merge predicate. Conservative on purpose —
    /// any non-empty, non-`er` base is allowed.
    private static func shouldMergeInto(
        prev: MandarinPinyinNormalizer.Syllable
    ) -> Bool {
        !prev.base.isEmpty && prev.base != "er"
    }
}
