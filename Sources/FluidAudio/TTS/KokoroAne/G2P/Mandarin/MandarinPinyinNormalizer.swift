import Foundation

/// Convert pypinyin's diacritic-form pinyin (`níhǎo`, `qiū`, `lǜ`) into
/// the `<base><digit>` form (`ni2`, `hao3`, `qiu1`, `lv4`) that
/// `MandarinBopomofoMap` expects.
///
/// Tone is identified by the diacritic on the first vowel that carries
/// one (in standard Mandarin orthography only one vowel per syllable
/// carries a tone mark). After the diacritic is read off, the underlying
/// vowel is restored to its bare ASCII form. The diaeresis on `ü` (and
/// its tone-marked variants) is normalized to the ASCII letter `v` —
/// matching pypinyin `Style.TONE3`'s convention and the keys used in
/// `MandarinBopomofoMap`.
public enum MandarinPinyinNormalizer {

    /// Tone digit (1–5) plus the bare-ASCII syllable, e.g. `níhǎo`'s
    /// first syllable → (`"ni"`, `2`).
    public struct Syllable: Equatable, Sendable {
        public var base: String  // ASCII letters only, with `ü` → `v`.
        public var tone: Int  // 1…4 = explicit, 5 = neutral / unmarked.

        public init(base: String, tone: Int) {
            self.base = base
            self.tone = tone
        }
    }

    /// Map every tone-marked vowel (and the plain `ü`) to its bare-ASCII
    /// counterpart + the implied tone digit. Tone 0 means the character is
    /// untoned and just contributes its base letter; tones 1-4 are
    /// explicit. We treat unmarked syllables as tone 5 (neutral) at the
    /// caller level.
    private static let table: [Character: (base: Character, tone: Int)] = [
        // a-row
        "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
        // e-row
        "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
        // i-row
        "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
        // o-row
        "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
        // u-row
        "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
        // ü-row (collapses to `v` per pypinyin TONE3)
        "ǖ": ("v", 1), "ǘ": ("v", 2), "ǚ": ("v", 3), "ǜ": ("v", 4),
        "ü": ("v", 0),
        // n / m carrying a tone (e.g. 嗯 → "n2", "ḿ")
        "ń": ("n", 2), "ň": ("n", 3), "ǹ": ("n", 4),
        "ḿ": ("m", 2),
    ]

    /// Convert a pypinyin diacritic syllable to (base, tone).
    public static func normalize(_ pinyin: String) -> Syllable {
        var base = ""
        base.reserveCapacity(pinyin.count)
        var tone = 5  // default = neutral / unmarked
        for ch in pinyin {
            if let mapped = table[ch] {
                base.append(mapped.base)
                if mapped.tone != 0 {
                    tone = mapped.tone
                }
            } else {
                base.append(ch)
            }
        }
        return Syllable(base: base, tone: tone)
    }
}
