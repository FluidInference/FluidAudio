import Foundation

/// Conservative pre-tokenization text normalization for English TTS
/// raw-text frontends (issue #711).
///
/// Raw chat-style text often contains standalone numbers, ordinals, and
/// clock times that tokenize poorly (`3.14` splits around `.` and reads
/// closer to `three fourteen` than `three point one four`). This pass
/// rewrites only *strict standalone* numeric forms into spoken words
/// before a frontend tokenizes, reusing ``SayAsInterpreter`` for the
/// actual spelling. It is frontend-agnostic — both ``KokoroAneManager``
/// and (in a follow-up) StyleTTS2 can apply it.
///
/// Raw text carries no caller annotation, so the rules are deliberately
/// stricter than SSML `<say-as>`: anything ambiguous or structured is left
/// untouched. Handled forms:
///   * cardinal integers — `26` → `twenty six`
///   * valid ordinals — `13th` → `thirteenth` (suffix must match the number)
///   * leading-zero digit strings — `007` → `zero zero seven`
///   * decimals — `3.14` → `three point one four`
///   * 12-hour meridiem times — `1:49 PM` → `one forty nine p m`
///   * decade forms — `1770s`/`'90s` → `seventeen seventies`/`nineties` (issue #776)
///   * bare 4-digit years (1000–2099) — `1770` → `seventeen seventy` (issue #776)
///   * roman-numeral list markers — `(ii)`, `ii)`, `ii.` → `(two)`, `two)`, `two.`
///     (issue #972)
///
/// Left unchanged (ambiguous / structured): version strings (`1.2.3`),
/// grouped numbers (`1,234`), embedded digits (`word26`, `26word`), loose
/// colon numbers without a meridiem (`1:49`), invalid times (`1:99 PM`),
/// 24-hour forms (`13:49`), and roman-numeral letters outside an enumerator
/// form (`mix`, `did`, the pronoun `I`).
enum EnglishTextNormalizer {

    /// Rewrite strict standalone numeric forms in `text` to spoken words.
    static func normalize(_ text: String) -> String {
        spellNumericForms(spellRomanEnumerators(text))
    }

    /// TTS-frontend entry point shared by KokoroAne and StyleTTS2.
    ///
    /// Uses the bundled byte-exact NeMo TN via the compiled-FST engine
    /// (``NemoTextNormalizer``) — much richer than the baseline: currency,
    /// measures, dates, ranges, fractions, embedded digits, … The conservative
    /// always-available baseline in ``normalize(_:)`` is used only if the FST
    /// engine leaves the text unchanged (e.g. plain prose, or a build without
    /// the `fst-engine` feature).
    ///
    /// Roman-numeral list markers are rewritten before either pass: NeMo's
    /// roman grammar is uppercase-only, so `(ii)` would otherwise reach G2P
    /// and be read as letters (issue #972).
    static func normalizeForFrontend(_ text: String) -> String {
        let prepared = spellRomanEnumerators(text)
        let fst = NemoTextNormalizer.normalize(prepared, language: .english)
        return fst == prepared ? spellNumericForms(prepared) : fst
    }

    /// Passes run in priority order so a token is consumed by the most
    /// specific rule (a meridiem time before its bare digits, a decimal
    /// before its integer part).
    private static func spellNumericForms(_ text: String) -> String {
        var result = text
        result = apply(Self.meridiemTimeRegex, to: result, transform: Self.spellMeridiemTime)
        result = apply(Self.decadeRegex, to: result, transform: Self.spellDecade)
        result = apply(Self.decimalRegex, to: result, transform: Self.spellDecimal)
        result = apply(Self.ordinalRegex, to: result, transform: Self.spellOrdinal)
        result = apply(Self.leadingZeroRegex, to: result, transform: Self.spellLeadingZero)
        result = apply(Self.yearRegex, to: result, transform: Self.spellYear)
        result = apply(Self.cardinalRegex, to: result, transform: Self.spellCardinal)
        return result
    }

    // MARK: - Roman-numeral list markers (issue #972)

    /// Rewrite roman-numeral enumerators — `(ii)`, `ii)`, `ii.` — to spoken
    /// cardinals, leaving the surrounding punctuation in place (`(two)`).
    ///
    /// Keyed on the enumerator *form*, never on the letters alone: `mix`,
    /// `did`, `civil` and the pronoun `I` are all spelled from roman letters.
    /// The parenthesized form may appear anywhere (not glued to a letter, so
    /// `f(x)` stays); the half-paren and dot forms only in enumerator position
    /// — line start, or after `; : , .` + space — which is what keeps `did I)`,
    /// `I use vi.` and `i.e.` intact.
    ///
    /// A numeral that is not self-evidently a list marker (see
    /// ``isSelfEvidentEnumerator(_:)``) additionally needs a second enumerator
    /// somewhere in the text: a list has members, while medical `(IV)`, a
    /// checkbox `(x)`, a kiss sign-off `xx.`, the citation `v. Madison` and the
    /// initials `I. M. Pei` stand alone. Conversely an uppercase outline
    /// `I. … II. … III.` converts as a whole.
    private static func spellRomanEnumerators(_ text: String) -> String {
        let enumerators = romanEnumeratorMatches(in: text)
        guard !enumerators.isEmpty else { return text }
        let isList = enumerators.count >= 2

        func spell(_ numeral: String) -> String? {
            guard isList || isSelfEvidentEnumerator(numeral), let value = romanValue(numeral) else { return nil }
            return cardinalWords(String(value))
        }

        var result = text
        result = apply(Self.romanParenthesizedRegex, to: result) { groups in
            spell(groups[1]).map { "(\($0))" }
        }
        result = apply(Self.romanEnumeratorRegex, to: result) { groups in
            spell(groups[2]).map { "\(groups[1])\($0)\(groups[3])" }
        }
        return result
    }

    /// Every well-formed roman numeral in an enumerator form, in text order.
    private static func romanEnumeratorMatches(in text: String) -> [String] {
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        let parenthesized = romanParenthesizedRegex.matches(in: text, range: range).map {
            ns.substring(with: $0.range(at: 1))
        }
        let bare = romanEnumeratorRegex.matches(in: text, range: range).map { ns.substring(with: $0.range(at: 2)) }
        return (parenthesized + bare).filter { romanValue($0) != nil }
    }

    /// Lowercase `i`, or a lowercase multi-letter numeral that is not all `x`:
    /// nothing else reads as a word or an abbreviation in enumerator position.
    /// Uppercase (`IV` intravenous, `I` pronoun/initial), a lone `v`/`x`
    /// (verb marker, variable, checkbox) and `xx`/`xxx` (sign-off, rating) are
    /// only list markers in company.
    private static func isSelfEvidentEnumerator(_ numeral: String) -> Bool {
        guard numeral == numeral.lowercased() else { return false }
        if numeral == "i" { return true }
        return numeral.count >= 2 && numeral.contains { $0 != "x" }
    }

    /// `[ivx]`-only numerals (1–39) in a single case; the strict form is
    /// checked in ``romanValue(_:)``. Longer/richer numerals (`L C D M`) are
    /// deliberately out: list markers never get there, and admitting `l`/`c`/
    /// `d`/`m` would add words like `mix`, `cd`, `mm`, `xl` to the trap set.
    private static let romanNumeral = #"([ivx]{1,7}|[IVX]{1,7})"#

    /// `(ii)` — both parentheses present, not glued to a letter (`f(x)`).
    private static let romanParenthesizedRegex = regex(
        #"(?<!\p{L})\("# + romanNumeral + #"\)(?![\p{L}0-9])"#)

    /// `ii)` / `ii.` in enumerator position. Group 1 is the lead context
    /// (line start incl. indentation, or clause punctuation + space) and
    /// group 3 the closer, both re-emitted verbatim. The dot form requires
    /// following whitespace so `i.e.` and a text-final `ii.` stay put.
    private static let romanEnumeratorRegex = regex(
        #"(^[ \t]*|[;:,.]\s)"# + romanNumeral + #"(\)(?![\p{L}0-9])|\.(?=\s))"#,
        options: [.anchorsMatchLines])

    /// Strict subtractive form up to 39: tens `X{0,3}`, units `IX|IV|V?I{0,3}`.
    private static let romanFormRegex = regex(#"^(X{0,3})(IX|IV|V?I{0,3})$"#)

    private static let romanUnits: [String: Int] = [
        "": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
    ]

    /// Value of a strict-form numeral (1–39) in either case, else `nil`.
    private static func romanValue(_ numeral: String) -> Int? {
        let upper = numeral.uppercased() as NSString
        guard
            let match = romanFormRegex.firstMatch(
                in: upper as String, range: NSRange(location: 0, length: upper.length)),
            let units = romanUnits[upper.substring(with: match.range(at: 2))]
        else { return nil }
        let value = 10 * match.range(at: 1).length + units
        return value > 0 ? value : nil
    }

    // MARK: - Boundaries
    //
    // A standalone number must not be glued to a letter, another digit, or
    // a `. , :` separator that would make it part of a word, version
    // string, grouped number, or clock value. `leadBoundary` guards the
    // left edge; `trailBoundary` guards the right edge while still allowing
    // a trailing sentence period (`26.` / `3.14.`) — a `.`/`,`/`:` only
    // disqualifies when it is itself followed by a digit.
    private static let leadBoundary = #"(?<![A-Za-z0-9.,:])"#
    private static let trailBoundary = #"(?![A-Za-z0-9])(?![.,:][0-9])"#

    // MARK: - Compiled patterns

    /// `1:49 PM`, `1:49 p.m.` — hour 1-12, minute 00-59, explicit meridiem.
    /// The `m` half is an explicit alternation (`.m`/`.m.` for the dotted
    /// form, bare `m` otherwise) so a sentence period after `PM` is left as
    /// punctuation instead of being swallowed (`1:49 PM.`).
    private static let meridiemTimeRegex = regex(
        leadBoundary + #"(1[0-2]|[1-9]):([0-5][0-9])\s*([AaPp])(?:\.[Mm]\.?|[Mm])"# + #"(?![A-Za-z])"#)

    /// `1770s`, `'90s`, `1900s` — a decade written as a 2- or 4-digit number
    /// with a trailing `s` (issue #776). An optional leading apostrophe is
    /// absorbed (`'90s`). The 4-digit alternative is listed first so the full
    /// number wins over its leading two digits (`1770s` → `1770`, not `17`).
    private static let decadeRegex = regex(
        leadBoundary + #"'?([0-9]{4}|[0-9]{2})s"# + #"(?![A-Za-z0-9])"#)

    /// `3.14` — integer and fractional parts, not part of a version string.
    private static let decimalRegex = regex(
        leadBoundary + #"([0-9]+)\.([0-9]+)"# + trailBoundary)

    /// `13th`, `21st` — digits immediately followed by an ordinal suffix.
    private static let ordinalRegex = regex(
        leadBoundary + #"([0-9]+)(st|nd|rd|th)"# + #"(?![A-Za-z])"#)

    /// `007` — leading zero forces a digit-by-digit reading.
    private static let leadingZeroRegex = regex(
        leadBoundary + #"(0[0-9]+)"# + trailBoundary)

    /// `1770` — a standalone 4-digit integer read year-style (issue #776).
    /// The range is narrowed to 1000–2099 in ``spellYear`` so only plausible
    /// years are rewritten; everything else falls through to ``cardinalRegex``.
    private static let yearRegex = regex(
        leadBoundary + #"([0-9]{4})"# + trailBoundary)

    /// `26` — a plain standalone integer.
    private static let cardinalRegex = regex(
        leadBoundary + #"([0-9]+)"# + trailBoundary)

    // MARK: - Per-match spelling (return nil to leave the match unchanged)

    private static func spellMeridiemTime(_ groups: [String]) -> String? {
        let clock = "\(groups[1]):\(groups[2])"
        let spoken = spaced(SayAsInterpreter.interpret(content: clock, interpretAs: "time", format: nil))
        guard !containsDigit(spoken) else { return nil }
        let meridiem = groups[3].lowercased() == "p" ? "p m" : "a m"
        return "\(spoken) \(meridiem)"
    }

    private static func spellDecade(_ groups: [String]) -> String? {
        // Only conventional decades (`1770s`, `90s`) — the number must end in 0.
        // An all-zero base (`'00s`, `0000s`) is skipped: it has no century to
        // anchor it (`'00s` is ambiguous between the 1900s and 2000s) and would
        // cardinal-read as a misleading "zeros", so leave it for the frontend.
        let digits = groups[1]
        guard digits.last == "0", let value = Int(digits), value != 0 else { return nil }
        // 4-digit decades read year-style (`1770` → `seventeen seventy`),
        // 2-digit decades as a bare cardinal (`90` → `ninety`); the trailing
        // `s` then pluralizes the last word (`seventy` → `seventies`).
        let interpretAs = digits.count == 4 ? "year" : "cardinal"
        let base = spaced(SayAsInterpreter.interpret(content: digits, interpretAs: interpretAs, format: nil))
        guard !base.isEmpty, !containsDigit(base) else { return nil }
        return pluralizeLastWord(base)
    }

    private static func spellYear(_ groups: [String]) -> String? {
        // Only rewrite plausible years; other 4-digit integers fall through
        // to the cardinal rule.
        guard let year = Int(groups[1]), (1000...2099).contains(year) else { return nil }
        let spoken = spaced(SayAsInterpreter.interpret(content: groups[1], interpretAs: "year", format: nil))
        return containsDigit(spoken) ? nil : spoken
    }

    private static func spellDecimal(_ groups: [String]) -> String? {
        guard let integerPart = cardinalWords(groups[1]) else { return nil }
        let fractionalPart = SayAsInterpreter.interpret(
            content: groups[2], interpretAs: "digits", format: nil)
        guard !containsDigit(fractionalPart) else { return nil }
        return "\(integerPart) point \(fractionalPart)"
    }

    private static func spellOrdinal(_ groups: [String]) -> String? {
        // Only rewrite grammatically valid ordinals (`13th`, not `13st`).
        guard let number = Int(groups[1]), expectedOrdinalSuffix(for: number) == groups[2].lowercased()
        else { return nil }
        let spoken = spaced(SayAsInterpreter.interpret(content: groups[1], interpretAs: "ordinal", format: nil))
        return containsDigit(spoken) ? nil : spoken
    }

    private static func spellLeadingZero(_ groups: [String]) -> String? {
        let spoken = SayAsInterpreter.interpret(content: groups[1], interpretAs: "digits", format: nil)
        return containsDigit(spoken) ? nil : spoken
    }

    private static func spellCardinal(_ groups: [String]) -> String? {
        cardinalWords(groups[1])
    }

    // MARK: - Helpers

    /// Spell a non-negative integer string with spaces between words
    /// (`twenty-six` → `twenty six`); `nil` if it overflows `Int` and the
    /// interpreter hands the digits back unchanged.
    private static func cardinalWords(_ digits: String) -> String? {
        let spoken = spaced(SayAsInterpreter.interpret(content: digits, interpretAs: "cardinal", format: nil))
        return containsDigit(spoken) ? nil : spoken
    }

    /// The grammatically correct ordinal suffix for `number`.
    private static func expectedOrdinalSuffix(for number: Int) -> String {
        if (11...13).contains(number % 100) { return "th" }
        switch number % 10 {
        case 1: return "st"
        case 2: return "nd"
        case 3: return "rd"
        default: return "th"
        }
    }

    private static func spaced(_ text: String) -> String {
        text.replacingOccurrences(of: "-", with: " ")
    }

    /// Pluralize the final word of a spelled decade base: a `-y` ending
    /// becomes `-ies` (`seventy` → `seventies`), otherwise `s` is appended
    /// (`hundred` → `hundreds`, `thousand` → `thousands`, `ten` → `tens`).
    private static func pluralizeLastWord(_ text: String) -> String {
        var words = text.split(separator: " ").map(String.init)
        guard let last = words.last else { return text }
        let plural = last.hasSuffix("y") ? String(last.dropLast()) + "ies" : last + "s"
        words[words.count - 1] = plural
        return words.joined(separator: " ")
    }

    private static func containsDigit(_ text: String) -> Bool {
        text.contains { $0.isNumber }
    }

    private static func regex(
        _ pattern: String, options: NSRegularExpression.Options = []
    ) -> NSRegularExpression {
        // Patterns are compile-time constants; a failure is a programmer error
        // (mirrors `SayAsInterpreter`'s regex initialization).
        try! NSRegularExpression(pattern: pattern, options: options)
    }

    /// Apply `regex` to `text`, replacing each match with `transform`'s
    /// result. Matches are spliced in reverse so earlier ranges stay valid.
    /// `transform` receives the full match plus capture groups (index 0 is
    /// the whole match); returning `nil` leaves that match untouched.
    private static func apply(
        _ regex: NSRegularExpression,
        to text: String,
        transform: ([String]) -> String?
    ) -> String {
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return text }

        let mutable = NSMutableString(string: text)
        for match in matches.reversed() {
            var groups: [String] = []
            groups.reserveCapacity(match.numberOfRanges)
            for index in 0..<match.numberOfRanges {
                let range = match.range(at: index)
                groups.append(range.location == NSNotFound ? "" : ns.substring(with: range))
            }
            if let replacement = transform(groups) {
                mutable.replaceCharacters(in: match.range, with: replacement)
            }
        }
        return mutable as String
    }
}
