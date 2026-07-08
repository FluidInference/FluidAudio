import Foundation

/// Port of ZipVoice's English text normalization (EmiliaTokenizer path:
/// `map_punctuations` + `zipvoice.tokenizer.normalizer.EnglishTextNormalizer`).
///
/// LuxTTS must match the *upstream* normalizer exactly — the espeak oracle
/// the model was trained against runs behind it — so this is a faithful
/// port (inflect-parity number spelling, hyphens preserved: espeak merges
/// "twenty-six" -> `twˈɛntisˈɪks`), not a reuse of the repo's conservative
/// `EnglishTextNormalizer` (issue #711), whose different output would shift
/// tokens (e.g. "twenty six" with a space).
enum LuxTtsEnglishNormalizer {

    /// `EmiliaTokenizer.preprocess_text` + `EnglishTextNormalizer.normalize`.
    static func normalize(_ text: String) -> String {
        var result = mapPunctuations(text)
        result = expandAbbreviations(result)
        result = normalizeNumbers(result)
        return result
    }

    // MARK: - map_punctuations

    static func mapPunctuations(_ text: String) -> String {
        var t = text
        let pairs: [(String, String)] = [
            ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("；", ";"),
            ("：", ":"), ("、", ","), ("‘", "'"), ("“", "\""), ("”", "\""),
            ("’", "'"), ("⋯", "…"), ("···", "…"), ("・・・", "…"), ("...", "…"),
        ]
        for (from, to) in pairs {
            t = t.replacingOccurrences(of: from, with: to)
        }
        return t
    }

    // MARK: - Abbreviations

    private static let abbreviations: [(NSRegularExpression, String)] = [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"),
        ("st", "saint"), ("co", "company"), ("jr", "junior"),
        ("maj", "major"), ("gen", "general"), ("drs", "doctors"),
        ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"),
        ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"),
        ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
        ("etc", "et cetera"), ("btw", "by the way"),
    ].map { (regex("\\b\($0.0)\\b", caseInsensitive: true), $0.1) }

    private static func expandAbbreviations(_ text: String) -> String {
        var t = text
        for (re, replacement) in abbreviations {
            t = re.stringByReplacingMatches(
                in: t, range: NSRange(location: 0, length: (t as NSString).length),
                withTemplate: replacement)
        }
        return t
    }

    // MARK: - Numbers (upstream regex order)

    private static let commaNumberRe = regex(#"([0-9][0-9\,]+[0-9])"#)
    private static let poundsRe = regex(#"£([0-9\,]*[0-9]+)"#)
    private static let dollarsRe = regex(#"\$([0-9\.\,]*[0-9]+)"#)
    private static let fractionRe = regex(#"([0-9]+)/([0-9]+)"#)
    private static let decimalRe = regex(#"([0-9]+\.[0-9]+)"#)
    private static let percentRe = regex(#"([0-9\.\,]*[0-9]+%)"#)
    private static let ordinalRe = regex(#"[0-9]+(st|nd|rd|th)"#)
    private static let numberRe = regex(#"[0-9]+"#)

    private static func normalizeNumbers(_ text: String) -> String {
        var t = text
        t = replace(commaNumberRe, in: t) { g in g[1].replacingOccurrences(of: ",", with: "") }
        t = replace(poundsRe, in: t) { g in "\(g[1]) pounds" }
        t = replace(dollarsRe, in: t) { g in expandDollars(g[1]) }
        t = replace(fractionRe, in: t) { g in
            expandFraction(numerator: Int(g[1]) ?? 0, denominator: Int(g[2]) ?? 1)
        }
        t = replace(decimalRe, in: t) { g in g[1].replacingOccurrences(of: ".", with: " point ") }
        t = replace(percentRe, in: t) { g in g[1].replacingOccurrences(of: "%", with: " percent") }
        t = replace(ordinalRe, in: t) { g in " \(ordinalWords(g[0])) " }
        t = replace(numberRe, in: t) { g in expandNumber(Int(g[0]) ?? 0) }
        return t
    }

    private static func expandDollars(_ match: String) -> String {
        let parts = match.split(separator: ".", omittingEmptySubsequences: false)
        if parts.count > 2 { return " \(match) dollars " }
        let dollars = parts.count > 0 ? Int(parts[0]) ?? 0 : 0
        let cents = parts.count > 1 ? Int(parts[1]) ?? 0 : 0
        // Digits remain here on purpose (upstream behavior); the trailing
        // cardinal pass expands them.
        if dollars != 0 && cents != 0 {
            let dollarUnit = dollars == 1 ? "dollar" : "dollars"
            let centUnit = cents == 1 ? "cent" : "cents"
            return " \(dollars) \(dollarUnit), \(cents) \(centUnit) "
        }
        if dollars != 0 {
            return " \(dollars) \(dollars == 1 ? "dollar" : "dollars") "
        }
        if cents != 0 {
            return " \(cents) \(cents == 1 ? "cent" : "cents") "
        }
        return " zero dollars "
    }

    private static func expandFraction(numerator: Int, denominator: Int) -> String {
        if numerator == 1 && denominator == 2 { return " one half " }
        if numerator == 1 && denominator == 4 { return " one quarter " }
        if denominator == 2 { return " \(cardinalWords(numerator)) halves " }
        if denominator == 4 { return " \(cardinalWords(numerator)) quarters " }
        return " \(cardinalWords(numerator)) \(ordinalize(cardinalWords(denominator))) "
    }

    /// Upstream `_expand_number`: years 1000..<3000 use pair reading.
    private static func expandNumber(_ num: Int) -> String {
        if num > 1000 && num < 3000 {
            if num == 2000 { return " two thousand " }
            if num > 2000 && num < 2010 {
                return " two thousand \(cardinalWords(num % 100)) "
            }
            if num % 100 == 0 {
                return " \(cardinalWords(num / 100)) hundred "
            }
            // inflect group=2, zero='oh', comma replaced by space
            let high = num / 100
            let low = num % 100
            let lowWords = low < 10 ? "oh \(cardinalWords(low))" : cardinalWords(low)
            return " \(cardinalWords(high)) \(lowWords) "
        }
        return " \(cardinalWords(num)) "
    }

    // MARK: - inflect-parity spelling

    private static let ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    private static let tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety",
    ]
    private static let scales: [(Int, String)] = [
        (1_000_000_000_000, "trillion"), (1_000_000_000, "billion"),
        (1_000_000, "million"), (1_000, "thousand"),
    ]

    /// `inflect.number_to_words(n, andword="")`: hyphenated tens, scale
    /// groups joined with ", " (only between non-zero groups).
    static func cardinalWords(_ n: Int) -> String {
        if n < 0 { return "minus \(cardinalWords(-n))" }
        if n < 20 { return ones[n] }
        if n < 100 {
            let t = tens[n / 10]
            return n % 10 == 0 ? t : "\(t)-\(ones[n % 10])"
        }
        if n < 1000 {
            let head = "\(ones[n / 100]) hundred"
            return n % 100 == 0 ? head : "\(head) \(cardinalWords(n % 100))"
        }
        var remaining = n
        var groups: [String] = []
        for (value, name) in scales {
            if remaining >= value {
                groups.append("\(cardinalWords(remaining / value)) \(name)")
                remaining %= value
            }
        }
        if remaining > 0 { groups.append(cardinalWords(remaining)) }
        return groups.joined(separator: ", ")
    }

    /// `inflect.number_to_words("42nd")` — cardinal words with the final
    /// word ordinalized ("forty-second").
    static func ordinalWords(_ ordinalDigits: String) -> String {
        let digits = String(ordinalDigits.prefix(while: { $0.isNumber }))
        guard let n = Int(digits) else { return ordinalDigits }
        return ordinalize(cardinalWords(n))
    }

    /// Ordinalize the last word of a cardinal phrase (inflect.ordinal).
    static func ordinalize(_ words: String) -> String {
        let irregular: [String: String] = [
            "one": "first", "two": "second", "three": "third",
            "five": "fifth", "eight": "eighth", "nine": "ninth",
            "twelve": "twelfth",
        ]
        // Find the last word (after the last space or hyphen).
        var separator: Character = " "
        var lastRange = words.startIndex..<words.endIndex
        if let idx = words.lastIndex(where: { $0 == " " || $0 == "-" }) {
            separator = words[idx]
            lastRange = words.index(after: idx)..<words.endIndex
        }
        let last = String(words[lastRange])
        let head =
            lastRange.lowerBound == words.startIndex
            ? "" : String(words[..<words.index(before: lastRange.lowerBound)])
        let ordinal: String
        if let irr = irregular[last] {
            ordinal = irr
        } else if last.hasSuffix("y") {
            ordinal = String(last.dropLast()) + "ieth"
        } else {
            ordinal = last + "th"
        }
        return head.isEmpty ? ordinal : "\(head)\(separator)\(ordinal)"
    }

    // MARK: - Regex helpers

    private static func regex(_ pattern: String, caseInsensitive: Bool = false) -> NSRegularExpression {
        // Compile-time constant patterns; a failure is a programmer error.
        try! NSRegularExpression(
            pattern: pattern, options: caseInsensitive ? [.caseInsensitive] : [])
    }

    private static func replace(
        _ re: NSRegularExpression, in text: String,
        transform: ([String]) -> String
    ) -> String {
        let ns = text as NSString
        let matches = re.matches(in: text, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return text }
        let mutable = NSMutableString(string: text)
        for match in matches.reversed() {
            var groups: [String] = []
            groups.reserveCapacity(match.numberOfRanges)
            for index in 0..<match.numberOfRanges {
                let range = match.range(at: index)
                groups.append(range.location == NSNotFound ? "" : ns.substring(with: range))
            }
            mutable.replaceCharacters(in: match.range, with: transform(groups))
        }
        return mutable as String
    }
}
