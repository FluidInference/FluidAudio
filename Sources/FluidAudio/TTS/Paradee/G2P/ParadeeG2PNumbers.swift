#if TTS
import Foundation

/// Numbers for Paradee's G2P, put together from the per-language pieces in
/// `paradee_g2p.json` (`numbers`: 0–99, the hundreds, `thousand`, `lakh`,
/// `million` …) the way eSpeak NG reads them. A port of `_hi`, `_fr`, `_es`
/// and `_soften` in upstream `paradee/g2p.py` (package 1.3.1).
enum ParadeeG2PNumbers {

    /// Phonemes for `n` (< 10^10), words separated by one space. `before` is
    /// the sound or letter before the number (`#` after a pause); only Spanish
    /// uses it, to soften a first `b`, `d`, `ɡ`.
    static func read(_ n: Int, language: ParadeeLanguage, table: [String: String], before: String = "#") -> String {
        let parts: [String]
        switch language {
        case .hindi: parts = hindi(n, table)
        case .french: parts = french(n, table)
        case .spanish:
            var soft: [String] = []
            for part in spanish(n, table) {
                let previous = soft.last.flatMap { $0.unicodeScalars.last.map(String.init) } ?? before
                soft.append(soften(part, before: previous))
            }
            parts = soft
        case .americanEnglish, .britishEnglish: parts = []
        }
        return parts.joined(separator: " ")
    }

    static func hindi(_ n: Int, _ t: [String: String]) -> [String] {
        if n < 100 { return [t[String(n)] ?? ""] }
        var n = n
        var out: [String] = []
        for (div, name) in [(10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand")] where n >= div {
            out += hindi(n / div, t) + [t[name] ?? ""]
            n %= div
        }
        if n >= 100 {
            out.append(t[String(n / 100 * 100)] ?? "")
            n %= 100
        }
        if n > 0 { out.append(t[String(n)] ?? "") }
        return out
    }

    /// French stresses only the last word of a number (`top`).
    static func french(_ n: Int, _ t: [String: String], top: Bool = true) -> [String] {
        var n = n
        var out: [String] = []
        for (div, one, many) in [(1_000_000_000, "milliard", "milliards"), (1_000_000, "million", "millions")]
        where n >= div {
            let q = n / div
            out += (q == 1 ? ["œ̃"] : french(q, t, top: false)) + [t[q == 1 ? one : many] ?? ""]
            n %= div
        }
        if n >= 1000 {
            let q = n / 1000
            out += (q == 1 ? [] : french(q, t, top: false)) + [t["thousand"] ?? ""]
            n %= 1000
        }
        if n >= 100 {
            out.append(t[String(n / 100 * 100)] ?? "")
            n %= 100
        }
        if n > 0 || out.isEmpty { out.append(t[String(n)] ?? "") }
        if top, out.count > 1 {
            out = out.dropLast().map(unstress) + [out[out.count - 1]]
        }
        return out
    }

    static func spanish(_ n: Int, _ t: [String: String]) -> [String] {
        var n = n
        var out: [String] = []
        if n >= 1_000_000 {
            let q = n / 1_000_000
            out += q == 1 ? [t["millón"] ?? ""] : spanish(q, t) + [t["millones"] ?? ""]
            n %= 1_000_000
        }
        if n >= 1000 {
            let q = n / 1000
            if q == 1 {
                out.append(t["thousand"] ?? "")
            } else {
                var p = spanish(q, t)
                // veintiuno mil -> veintiún mil
                if q % 10 == 1, q % 100 != 11, let last = p.last, last.unicodeScalars.last == "o" {
                    p[p.count - 1] = String(String.UnicodeScalarView(last.unicodeScalars.dropLast()))
                }
                out += p + [t["thousand"] ?? ""]
            }
            n %= 1000
        }
        if n >= 100 {
            let h = n / 100 * 100
            n %= 100
            out.append((h == 100 && n > 0 ? t["100+"] : t[String(h)]) ?? "")
        }
        if n > 0 || out.isEmpty { out.append(t[String(n)] ?? "") }
        return out
    }

    /// Spanish b, d and g are said β, ð and ɣ after a vowel and most
    /// consonants, also across words, as eSpeak writes them. `before` is the
    /// sound or letter before, `#` after a pause.
    static func soften(_ p: String, before: String) -> String {
        var scalars = Array(p.unicodeScalars)
        var i = 0
        while i < scalars.count, scalars[i] == "ˈ" || scalars[i] == "ˌ" { i += 1 }
        guard i < scalars.count else { return p }
        let c = scalars[i]
        let soft: Unicode.Scalar
        switch c {
        case "b": soft = "β"
        case "d": soft = "ð"
        case "ɡ": soft = "ɣ"
        default: return p
        }
        // Python `before not in "#nmɲŋ"` is a substring test (an empty `before` never softens).
        guard !containsScalars("#nmɲŋ", before), !(c == "d" && before == "l") else { return p }
        scalars[i] = soft
        return String(String.UnicodeScalarView(scalars))
    }

    static func unstress(_ s: String) -> String {
        String(String.UnicodeScalarView(s.unicodeScalars.filter { $0 != "ˈ" && $0 != "ˌ" }))
    }

    /// Code-point substring test, like Python's `needle in haystack` on str.
    private static func containsScalars(_ haystack: String, _ needle: String) -> Bool {
        let h = Array(haystack.unicodeScalars)
        let n = Array(needle.unicodeScalars)
        guard !n.isEmpty else { return true }
        guard n.count <= h.count else { return false }
        for start in 0...(h.count - n.count) where Array(h[start..<(start + n.count)]) == n {
            return true
        }
        return false
    }
}
#endif
