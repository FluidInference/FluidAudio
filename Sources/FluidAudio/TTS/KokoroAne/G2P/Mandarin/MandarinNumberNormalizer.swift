import Foundation

/// Mandarin number / date / time / currency verbalization pre-pass.
///
/// Runs *before* `MandarinG2P.normalizeText` so Arabic numerals, dates,
/// times, percentages, fractions, and currency expressions are converted
/// into Hanzi the existing G2P pipeline can speak directly. The
/// transformation is text-in / text-out — pure function, no state.
///
/// Mirrors the most common cases from `misaki/zh/num.py`. Out of scope
/// (deferred — extending later won't break callers):
///
///   * Scientific notation (`1e6`).
///   * English-style ordinals (`1st`, `2nd`).
///   * Unit abbreviations (`kg`, `°C`, `km/h`).
///   * Phone-number / ID grouping (`138-0013-8000`).
///
/// Rule ordering is significant — date / time / currency patterns run
/// before the generic decimal / integer fallthrough so that
/// `2025年5月3日` becomes one cohesive Hanzi span instead of leaving
/// `年` / `月` / `日` glued to digit fragments.
public enum MandarinNumberNormalizer {

    private static let digits: [Character] = [
        "零", "一", "二", "三", "四", "五", "六", "七", "八", "九",
    ]
    private static let groupUnits: [String] = ["", "万", "亿", "兆"]

    /// Convert all numeric expressions in `text` to their Hanzi
    /// verbalization. Non-numeric content passes through unchanged.
    ///
    /// Regexes are compiled per call rather than cached at file scope —
    /// keeps the type Sendable-clean without `@unchecked` annotations.
    /// Compilation cost is microseconds and dominated by the surrounding
    /// G2P pipeline.
    public static func normalize(_ text: String) -> String {
        var s = text
        for (pattern, transform) in pipeline {
            s = apply(pattern: pattern, transform: transform, to: s)
        }
        return s
    }

    // MARK: - Cardinal

    /// Verbalize a non-negative integer up to ~10¹⁶ (`9999_9999_9999_9999`).
    /// Larger inputs degrade gracefully by emitting digit-by-digit.
    static func cardinal(_ n: Int64) -> String {
        if n == 0 { return "零" }
        if n < 0 { return "负" + cardinal(-n) }

        var groups: [Int] = []
        var x = n
        while x > 0 {
            groups.append(Int(x % 10000))
            x /= 10000
        }
        if groups.count > groupUnits.count {
            return digitString(String(n))
        }

        var result = ""
        var emitted = false
        for i in (0..<groups.count).reversed() {
            let g = groups[i]
            if g == 0 { continue }
            if emitted && g < 1000 {
                result += "零"
            }
            result += fourDigitChunk(g, isHighest: !emitted)
            result += groupUnits[i]
            emitted = true
        }
        return result
    }

    /// Render a 4-digit value (0..<10000). `isHighest` collapses the
    /// 1x range to `十` (e.g. `12` → `十二`) when no higher group
    /// precedes it; intra-number tens always render as `一十X`.
    private static func fourDigitChunk(_ n: Int, isHighest: Bool) -> String {
        if n == 0 { return "" }
        var result = ""
        let q = n / 1000
        let h = (n / 100) % 10
        let t = (n / 10) % 10
        let u = n % 10
        var pendingZero = false

        if q > 0 {
            result.append(digits[q])
            result += "千"
        }
        if h > 0 {
            if pendingZero {
                result += "零"
                pendingZero = false
            }
            result.append(digits[h])
            result += "百"
        } else if q > 0 && (t > 0 || u > 0) {
            pendingZero = true
        }
        if t > 0 {
            if pendingZero {
                result += "零"
                pendingZero = false
            }
            if t == 1 && q == 0 && h == 0 && isHighest {
                result += "十"
            } else {
                result.append(digits[t])
                result += "十"
            }
        } else if (q > 0 || h > 0) && u > 0 {
            pendingZero = true
        }
        if u > 0 {
            if pendingZero {
                result += "零"
                pendingZero = false
            }
            result.append(digits[u])
        }
        return result
    }

    /// Spell each digit individually (e.g. `"2025"` → `"二零二五"`).
    /// Used for years and as a fallback for out-of-range integers.
    static func digitString(_ s: String) -> String {
        var out = ""
        for ch in s {
            if let v = ch.wholeNumberValue, (0..<10).contains(v) {
                out.append(digits[v])
            } else if ch == "-" {
                out += "负"
            } else if ch == "." {
                out += "点"
            }
        }
        return out
    }

    /// `"3.14"` → `"三点一四"`. Integer part as cardinal, fractional
    /// part digit-by-digit. Trailing zeros in the fractional part are
    /// stripped to match colloquial Mandarin (`5.50` → `五点五`).
    static func decimal(_ s: String) -> String {
        let parts = s.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        let intPart = String(parts[0])
        if parts.count == 1 {
            return Int64(intPart).map(cardinal) ?? digitString(intPart)
        }
        var fracPart = String(parts[1])
        while fracPart.count > 1 && fracPart.last == "0" {
            fracPart.removeLast()
        }
        let intStr = Int64(intPart).map(cardinal) ?? digitString(intPart)
        if fracPart.isEmpty || fracPart == "0" {
            return intStr
        }
        return intStr + "点" + digitString(fracPart)
    }

    // MARK: - Rule plumbing

    private typealias Transform = ([String]) -> String

    private static func apply(
        pattern: String,
        transform: Transform,
        to text: String
    ) -> String {
        // Patterns are compile-time literals — a failure here is a
        // programmer bug, surface it loudly instead of silently dropping.
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            fatalError("MandarinNumberNormalizer: invalid regex \(pattern)")
        }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        let matches = regex.matches(in: text, options: [], range: range)
        guard !matches.isEmpty else { return text }
        var out = ""
        var cursor = 0
        for m in matches {
            let mr = m.range
            if mr.location > cursor {
                out += ns.substring(
                    with: NSRange(location: cursor, length: mr.location - cursor))
            }
            var groups: [String] = []
            groups.reserveCapacity(m.numberOfRanges)
            for i in 0..<m.numberOfRanges {
                let r = m.range(at: i)
                groups.append(r.location == NSNotFound ? "" : ns.substring(with: r))
            }
            out += transform(groups)
            cursor = mr.location + mr.length
        }
        if cursor < ns.length {
            out += ns.substring(
                with: NSRange(location: cursor, length: ns.length - cursor))
        }
        return out
    }

    private static func intToHanzi(_ s: String) -> String {
        Int64(s).map(cardinal) ?? s
    }

    /// Ordered list of (regex, transform) pairs applied left-to-right.
    /// Order is significant — date / time / currency patterns must run
    /// before generic decimal / integer fallthrough.
    private static var pipeline: [(String, Transform)] {
        [
            // Date: 2025年5月3日 / 2025年5月3号
            (
                #"(\d{4})年(\d{1,2})月(\d{1,2})[日号]"#,
                { g in
                    digitString(g[1]) + "年" + intToHanzi(g[2]) + "月" + intToHanzi(g[3]) + "日"
                }
            ),
            // Date: 2025年5月
            (
                #"(\d{4})年(\d{1,2})月"#,
                { g in
                    digitString(g[1]) + "年" + intToHanzi(g[2]) + "月"
                }
            ),
            // Date: 2025-05-03 / 2025/05/03
            (
                #"(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"#,
                { g in
                    digitString(g[1]) + "年" + intToHanzi(g[2]) + "月" + intToHanzi(g[3]) + "日"
                }
            ),
            // Date: 2025年 (year-only)
            (#"(\d{4})年"#, { g in digitString(g[1]) + "年" }),
            // Time: HH:MM:SS
            (
                #"(\d{1,2}):(\d{2}):(\d{2})"#,
                { g in
                    intToHanzi(g[1]) + "点" + intToHanzi(g[2]) + "分" + intToHanzi(g[3]) + "秒"
                }
            ),
            // Time: HH:MM
            (
                #"(\d{1,2}):(\d{2})"#,
                { g in
                    intToHanzi(g[1]) + "点" + intToHanzi(g[2]) + "分"
                }
            ),
            // Currency: prefix symbol + amount.
            (#"[¥￥](\d+(?:\.\d+)?)"#, { g in decimal(g[1]) + "元" }),
            (#"\$(\d+(?:\.\d+)?)"#, { g in decimal(g[1]) + "美元" }),
            (#"€(\d+(?:\.\d+)?)"#, { g in decimal(g[1]) + "欧元" }),
            (#"£(\d+(?:\.\d+)?)"#, { g in decimal(g[1]) + "英镑" }),
            // Percentage: 99% / 0.5%
            (#"(\d+(?:\.\d+)?)%"#, { g in "百分之" + decimal(g[1]) }),
            // Fraction: a/b — denominator first in Chinese (`二分之一` for `1/2`).
            (#"(\d+)/(\d+)"#, { g in intToHanzi(g[2]) + "分之" + intToHanzi(g[1]) }),
            // Plain decimal (catches what currency / percentage didn't).
            (#"\d+\.\d+"#, { g in decimal(g[0]) }),
            // Plain integer fallthrough.
            (#"\d+"#, { g in intToHanzi(g[0]) }),
        ]
    }
}
