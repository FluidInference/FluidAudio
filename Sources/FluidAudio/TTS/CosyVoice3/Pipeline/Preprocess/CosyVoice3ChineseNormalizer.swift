import Foundation

/// Minimal Mandarin text normalizer ported from CosyVoice's
/// `cosyvoice/utils/frontend_utils.py` + the Chinese branch of
/// `cosyvoice/cli/frontend.py:text_normalize`.
///
/// **Scope (intentional):** regex-free character-level rules plus digit
/// spellout. The full `wetext.ZhNormalizer` (which rewrites years, phone
/// numbers, decimals, units, chemistry, currency, dates…) is **not** ported.
/// Callers that need production-quality TN should run wetext server-side and
/// pass the result via `synthesize(text:prenormalized: true, ...)`.
///
/// Rules applied (in order):
///   1. strip newlines, leading/trailing whitespace
///   2. `replaceCornerMark` — `²` → `平方`, `³` → `立方`
///   3. ASCII digits → 零一二三四五六七八九 (per-digit fallback; lossy vs wetext
///      but avoids raw Arabic numerals going into the BPE)
///   4. `.` → `。`, ` - ` → `，`
///   5. `replaceBlank` — remove spaces between CJK chars; keep spaces between
///      ASCII tokens. Runs *after* the ASCII→CJK substitutions above so
///      spaces that became CJK-interior are also cleaned up.
///   6. `removeBracket` — drop `（）【】` and backticks, `——` → space
///   7. trailing `，` / `,` / `、` sequences → `。`
public enum CosyVoice3ChineseNormalizer {

    public static func normalize(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: "\n", with: "")
        s = s.trimmingCharacters(in: .whitespaces)
        s = replaceCornerMark(s)
        s = spellOutDigitsZh(s)
        s = s.replacingOccurrences(of: ".", with: "。")
        s = s.replacingOccurrences(of: " - ", with: "，")
        s = replaceBlank(s)
        s = removeBracket(s)
        s = stripTrailingCommaLikes(s)
        return s
    }

    /// True if `text` contains at least one CJK Unified Ideograph
    /// (U+4E00..U+9FFF), matching `contains_chinese` in frontend_utils.py.
    public static func containsChinese(_ text: String) -> Bool {
        for scalar in text.unicodeScalars where (0x4E00...0x9FFF).contains(scalar.value) {
            return true
        }
        return false
    }

    /// True if `text` is empty or consists only of Unicode punctuation /
    /// symbol characters. Mirrors `is_only_punctuation`.
    public static func isOnlyPunctuation(_ text: String) -> Bool {
        if text.isEmpty { return true }
        let allowed: CharacterSet = {
            var s = CharacterSet.punctuationCharacters
            s.formUnion(.symbols)
            s.formUnion(.whitespaces)
            return s
        }()
        for scalar in text.unicodeScalars where !allowed.contains(scalar) {
            return false
        }
        return true
    }

    // MARK: - Individual rules

    /// Drop spaces between non-ASCII chars; keep spaces that sit between two
    /// ASCII tokens (e.g. "hello world" stays, "中 国" → "中国").
    static func replaceBlank(_ text: String) -> String {
        let chars = Array(text)
        var out: [Character] = []
        out.reserveCapacity(chars.count)
        for i in 0..<chars.count {
            let c = chars[i]
            if c == " " {
                let prev = i > 0 ? chars[i - 1] : Character(" ")
                let next = i + 1 < chars.count ? chars[i + 1] : Character(" ")
                let prevOk = prev.isASCII && prev != " "
                let nextOk = next.isASCII && next != " "
                if prevOk && nextOk {
                    out.append(c)
                }
            } else {
                out.append(c)
            }
        }
        return String(out)
    }

    static func replaceCornerMark(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: "²", with: "平方")
        s = s.replacingOccurrences(of: "³", with: "立方")
        return s
    }

    static func removeBracket(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: "（", with: "")
        s = s.replacingOccurrences(of: "）", with: "")
        s = s.replacingOccurrences(of: "【", with: "")
        s = s.replacingOccurrences(of: "】", with: "")
        s = s.replacingOccurrences(of: "`", with: "")
        s = s.replacingOccurrences(of: "——", with: " ")
        return s
    }

    /// Replace each ASCII digit in `text` with its Chinese reading. Lossy
    /// per-digit fallback (e.g. `2024` → `二零二四`); correct for years / IDs
    /// but wrong for decimals or large cardinals. Acceptable as a placeholder
    /// while wetext remains server-side.
    static func spellOutDigitsZh(_ text: String) -> String {
        let map: [Character: String] = [
            "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
        ]
        var out = ""
        out.reserveCapacity(text.count)
        for ch in text {
            if let zh = map[ch] {
                out += zh
            } else {
                out.append(ch)
            }
        }
        return out
    }

    /// Collapse a run of trailing `，` / `,` / `、` into a single `。`.
    /// Equivalent to the Python `re.sub(r'[，,、]+$', '。', text)` rule.
    static func stripTrailingCommaLikes(_ text: String) -> String {
        let commaLikes: Set<Character> = ["，", ",", "、"]
        var chars = Array(text)
        var end = chars.count
        while end > 0, commaLikes.contains(chars[end - 1]) {
            end -= 1
        }
        if end == chars.count {
            return text
        }
        chars = Array(chars[0..<end])
        chars.append("。")
        return String(chars)
    }
}
