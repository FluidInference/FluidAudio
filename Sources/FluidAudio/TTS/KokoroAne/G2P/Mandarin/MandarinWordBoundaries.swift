#if TTS
import Foundation

/// Word boundaries misaki's `ZHFrontend` marks with `/` in the v1.1-zh
/// phoneme string (`我们去公园吧` → `我3ㄇㄣ5/ㄑㄩ4/ㄍ中1元2/ㄅㄚ5`).
///
/// misaki writes `/` between two adjacent words of jieba's segmentation
/// after `ToneSandhi.pre_merge_for_modify` has glued some of them
/// together, so a merged pair gets no separator. This replays those
/// merge passes, in misaki's order, over the words of one run (adjacent
/// words with no punctuation, space or literal between them):
///
///   1. `不` joins the word after it.
///   2. `X一X` reduplication joins into one word (misaki also requires
///      `X` to be tagged a verb; without a POS tagger `X` must be a single
///      character), then a standalone `一` joins the word after it.
///   3. A word equal to the one before it joins it (`看` `看`).
///   4. Two all-third-tone words join when they total ≤ 3 characters and
///      the first is not an `AA` reduplication.
///   5. Same, when only the boundary syllables are third tone.
///   6. `儿` joins the word before it.
///
/// Tones are the dictionary readings (misaki reads them with pypinyin
/// before any sandhi).
enum MandarinWordBoundaries {

    struct Word: Equatable {
        let text: [Character]
        let tones: [Int]
    }

    /// For each word of a run, whether a `/` goes before it. The first
    /// word never gets one.
    static func separators(for words: [Word]) -> [Bool] {
        guard words.count > 1 else { return [Bool](repeating: false, count: words.count) }
        var units = words.enumerated().map { Unit(text: $1.text, tones: $1.tones, first: $0) }
        units = mergeBu(units)
        units = mergeYi(units)
        units = mergeReduplication(units)
        units = mergeThreeTones(units) { allThirdTone($0.tones) && allThirdTone($1.tones) }
        units = mergeThreeTones(units) { $0.tones.last == 3 && $1.tones.first == 3 }
        units = mergeEr(units)
        let starts = Set(units.map(\.first))
        return words.indices.map { $0 > 0 && starts.contains($0) }
    }

    /// A word after merging, keyed by the index of its first input word.
    private struct Unit {
        var text: [Character]
        var tones: [Int]
        let first: Int

        func appending(_ other: Unit) -> Unit {
            Unit(text: text + other.text, tones: tones + other.tones, first: first)
        }
    }

    private static let bu: [Character] = ["不"]
    private static let yi: [Character] = ["一"]

    private static func allThirdTone(_ tones: [Int]) -> Bool {
        !tones.isEmpty && tones.allSatisfy { $0 == 3 }
    }

    private static func isReduplication(_ text: [Character]) -> Bool {
        text.count == 2 && text[0] == text[1]
    }

    /// `_merge_bu`: a `不` with a word after it is folded into that word.
    private static func mergeBu(_ units: [Unit]) -> [Unit] {
        var out: [Unit] = []
        for (i, unit) in units.enumerated() {
            let merged = i > 0 && units[i - 1].text == bu ? units[i - 1].appending(unit) : unit
            if merged.text != bu || i + 1 == units.count {
                out.append(merged)
            }
        }
        return out
    }

    /// `_merge_yi`: `X一X` first, then a standalone `一` takes the next word.
    private static func mergeYi(_ units: [Unit]) -> [Unit] {
        var reduplicated: [Unit] = []
        var i = 0
        while i < units.count {
            if i > 0, i + 1 < units.count, units[i].text == yi, units[i - 1].text.count == 1,
                units[i - 1].text == units[i + 1].text, let last = reduplicated.popLast()
            {
                reduplicated.append(last.appending(units[i]).appending(units[i + 1]))
                i += 2
            } else {
                reduplicated.append(units[i])
                i += 1
            }
        }
        var out: [Unit] = []
        for unit in reduplicated {
            if let last = out.last, last.text == yi {
                out[out.count - 1] = last.appending(unit)
            } else {
                out.append(unit)
            }
        }
        return out
    }

    /// `_merge_reduplication`: a word equal to the merged word before it.
    private static func mergeReduplication(_ units: [Unit]) -> [Unit] {
        var out: [Unit] = []
        for unit in units {
            if let last = out.last, last.text == unit.text {
                out[out.count - 1] = last.appending(unit)
            } else {
                out.append(unit)
            }
        }
        return out
    }

    /// `_merge_continuous_three_tones` / `_2`: `joins(previous, current)`
    /// decides the tone condition; a word merged into its predecessor
    /// cannot take the next one in the same pass.
    private static func mergeThreeTones(_ units: [Unit], joins: (Unit, Unit) -> Bool) -> [Unit] {
        var out: [Unit] = []
        var mergedLast = [Bool](repeating: false, count: units.count)
        for (i, unit) in units.enumerated() {
            if i > 0, !mergedLast[i - 1], joins(units[i - 1], unit), !isReduplication(units[i - 1].text),
                units[i - 1].text.count + unit.text.count <= 3
            {
                out[out.count - 1] = out[out.count - 1].appending(unit)
                mergedLast[i] = true
            } else {
                out.append(unit)
            }
        }
        return out
    }

    /// `_merge_er`: `儿` (or `兒`) joins the word before it.
    private static func mergeEr(_ units: [Unit]) -> [Unit] {
        var out: [Unit] = []
        for unit in units {
            if !out.isEmpty, unit.text == ["儿"] || unit.text == ["兒"] {
                out[out.count - 1] = out[out.count - 1].appending(unit)
            } else {
                out.append(unit)
            }
        }
        return out
    }
}
#endif
