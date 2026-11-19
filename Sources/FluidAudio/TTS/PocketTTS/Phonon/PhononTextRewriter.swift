#if TTS
import Foundation

/// Word-level rewrites run after the Phonon character pass, one space-separated word at a time.
/// The first rule that claims a word wins, and rules are never chained. Ported from
/// `ptts/src/preprocess/rewrite.rs` for English with the default rule set, in the upstream
/// order: numbers, currency, dashed digits, emails, URLs, abbreviations, elongations.
enum PhononTextRewriter {

    typealias Scalars = [Unicode.Scalar]

    /// The sentence punctuation a rule carries through at the end of a word.
    private static let punctuation: Set<Unicode.Scalar> = ["!", "?", ".", ":", ",", ";", "\u{2026}"]

    // MARK: - Text and words

    /// Rewrite every word of `text`, letting the multi-word rules (scaled amounts, digit runs,
    /// "ending in" codes, ranks) claim several words at once. A word no rule claims has its `@`
    /// and `+` spelled out.
    static func rewriteText(_ text: Scalars) -> String {
        let words = PhononScalar.split(text, on: " ")
        var out: [String] = []
        out.reserveCapacity(words.count)
        var index = 0
        while index < words.count {
            let rest = words[index...]
            let span =
                scaled(rest)
                ?? digitRun(rest)
                ?? endingIn(previous: out.last, rest)
                ?? ranks(rest)
            if let (rewritten, used) = span {
                out.append(rewritten)
                index += used
                continue
            }
            out.append(rewriteWord(words[index]) ?? spellSymbols(words[index]))
            index += 1
        }
        return out.joined(separator: " ")
    }

    /// Rewrite `word` under the default rules, or `nil` when no rule claims it.
    static func rewriteWord(_ word: Scalars) -> String? {
        numbers(word)
            ?? currency(word)
            ?? dashedDigits(word)
            ?? emails(word)
            ?? urls(word)
            ?? abbreviations(word)
            ?? elongations(word)
    }

    /// Spell out the `@` and `+` of a word no rule claimed, and give the pieces they leave their
    /// own turn at the rules. A slash between words ("and/or") is read as a pause.
    private static func spellSymbols(_ word: Scalars) -> String {
        if let pieces = splitSlash(word) {
            return pieces.map { rewriteWord($0) ?? PhononScalar.string($0) }.joined(separator: " ")
        }
        guard word.contains("@") || word.contains("+") else {
            return PhononScalar.string(word)
        }
        var res = PhononStringAppender()
        for c in word {
            switch c {
            case "@": res.pushSpoken("at")
            case "+": res.pushSpoken("plus")
            default: res.push(c)
            }
        }
        let pieces = PhononScalar.split(res.intoScalars(), on: " ")
        return pieces.map { rewriteWord($0) ?? PhononScalar.string($0) }.joined(separator: " ")
    }

    /// Units written with a slash between letters ("mg/kg", "kJ/mol"), which keep it.
    private static let slashUnits: Set<String> = [
        "km", "kg", "mg", "ml", "mL", "dL", "mol", "kJ", "kW", "kWh", "Hz", "min", "hr", "hrs", "sec",
        "ft", "lb", "lbs", "oz", "mi", "mph", "kph", "yd", "cm", "mm", "yr", "wk", "mo", "kcal", "rpm",
    ]

    /// `word` cut at its slashes into space-separated words, when every part is a word of two
    /// letters or more and not a unit; `nil` otherwise ("c/o", "km/h", dates, fractions).
    private static func splitSlash(_ word: Scalars) -> [Scalars]? {
        var bodyEnd = word.count
        while bodyEnd > 0, PhononScalar.isASCIIPunctuation(word[bodyEnd - 1]), word[bodyEnd - 1] != "/" {
            bodyEnd -= 1
        }
        let parts = PhononScalar.split(word[..<bodyEnd], on: "/")
        let wordlike: (Scalars) -> Bool = { part in
            part.count >= 2 && part.allSatisfy(PhononScalar.isAlphabetic)
                && !(PhononScalar.ascii(part).map { slashUnits.contains($0) } ?? false)
        }
        guard parts.count > 1, parts.allSatisfy(wordlike) else { return nil }
        // The parts hold no spaces, so the joined text splits back into them; the trailing
        // punctuation stays on the last one.
        var pieces = parts
        pieces[pieces.count - 1].append(contentsOf: word[bodyEnd...])
        return pieces
    }

    // MARK: - Shared helpers

    /// The trailing punctuation a rule carries through, or `nil` when `rest` is anything else.
    static func suffix<C: Collection>(_ rest: C) -> Scalars? where C.Element == Unicode.Scalar {
        rest.allSatisfy { punctuation.contains($0) } ? Array(rest) : nil
    }

    /// `word` without its trailing punctuation, and that punctuation.
    static func splitSuffix<C: BidirectionalCollection>(_ word: C) -> (body: Scalars, suffix: Scalars)
    where C.Element == Unicode.Scalar {
        let scalars = Array(word)
        var end = scalars.count
        while end > 0, punctuation.contains(scalars[end - 1]) {
            end -= 1
        }
        return (Array(scalars[..<end]), Array(scalars[end...]))
    }

    /// Whether `s` is a non-empty run of ASCII digits.
    static func allDigits<C: Collection>(_ s: C) -> Bool where C.Element == Unicode.Scalar {
        !s.isEmpty && s.allSatisfy(PhononScalar.isASCIIDigit)
    }

    /// A two-letter part of an address spelled out ("fr" becomes "F-R", "www" "W-W-W");
    /// anything else is read as a word.
    static func spellShort<C: Collection>(_ word: C) -> String where C.Element == Unicode.Scalar {
        if PhononScalar.equals(word, "www") || PhononScalar.equals(word, "WWW") {
            return "W-W-W"
        }
        let scalars = Array(word)
        if scalars.count == 2, scalars[0].isASCII, scalars[1].isASCII {
            return PhononScalar.string([
                PhononScalar.asciiUppercased(scalars[0]), "-", PhononScalar.asciiUppercased(scalars[1]),
            ])
        }
        return PhononScalar.string(scalars)
    }

    // MARK: - Dashed digits

    /// Digit sequences with at least two `-` or `_` separators: dashes become spaces,
    /// underscores are spoken, groups of up to 3 digits stay numbers and longer ones are spelled
    /// digit by digit ("555-123-4567" becomes "555 123 4 5 6 7"). Dates are left alone.
    static func dashedDigits(_ word: Scalars) -> String? {
        let (body, suffix) = splitSuffix(word)
        let isSeparator: (Unicode.Scalar) -> Bool = { $0 == "-" || $0 == "_" }
        guard body.allSatisfy({ PhononScalar.isASCIIDigit($0) || isSeparator($0) }),
            body.filter(isSeparator).count >= 2
        else { return nil }

        // Groups, each with the separator after it; nil when a separator is leading, trailing
        // or doubled.
        var groups: [(digits: Scalars, separator: Unicode.Scalar?)] = []
        var start = 0
        for (i, c) in body.enumerated() where isSeparator(c) {
            if i == start { return nil }
            groups.append((Array(body[start..<i]), c))
            start = i + 1
        }
        if start == body.count { return nil }
        groups.append((Array(body[start...]), nil))

        if !body.contains("_") && looksLikeDate(groups.map { $0.digits }) {
            return nil
        }
        var words: [String] = []
        for (digits, separator) in groups {
            if digits.count >= 4 {
                words.append(contentsOf: digits.map { String($0) })
            } else {
                words.append(PhononScalar.string(digits))
            }
            if separator == "_" {
                words.append("underscore")
            }
        }
        return words.joined(separator: " ") + PhononScalar.string(suffix)
    }

    /// Whether three groups read as a date: `YYYY-MM-DD`, or `DD-MM-YYYY` / `MM-DD-YYYY` with a
    /// 2- or 4-digit year.
    private static func looksLikeDate(_ groups: [Scalars]) -> Bool {
        func dayMonth(_ x: Scalars, _ y: Scalars) -> Bool {
            guard x.count <= 2, y.count <= 2,
                let x = UInt32(PhononScalar.string(x)), let y = UInt32(PhononScalar.string(y))
            else { return false }
            return ((1...31).contains(x) && (1...12).contains(y)) || ((1...12).contains(x) && (1...31).contains(y))
        }
        guard groups.count == 3 else { return false }
        let (a, b, c) = (groups[0], groups[1], groups[2])
        return (a.count == 4 && dayMonth(b, c)) || ((c.count == 4 || c.count == 2) && dayMonth(a, b))
    }

    // MARK: - Emails

    /// An email address with its separators spoken and a two-letter top-level domain spelled
    /// out: "a.b@gra-dium.fr" becomes "a dot b at gra dash dium dot F-R".
    static func emails(_ word: Scalars) -> String? {
        guard let at = word.firstIndex(of: "@") else { return nil }
        let user = word[..<at]
        guard !user.isEmpty,
            user.allSatisfy({ PhononScalar.isASCIIAlphanumeric($0) || "-_.+".unicodeScalars.contains($0) })
        else { return nil }

        func label(_ s: ArraySlice<Unicode.Scalar>) -> (ArraySlice<Unicode.Scalar>, ArraySlice<Unicode.Scalar>)? {
            let n = s.prefix { PhononScalar.isASCIIAlphanumeric($0) || $0 == "-" }.count
            guard n > 0 else { return nil }
            return (s.prefix(n), s.dropFirst(n))
        }

        guard let (first, afterFirst) = label(word[(at + 1)...]) else { return nil }
        var labels = [first]
        var rest = afterFirst
        while rest.first == ".", let (next, after) = label(rest.dropFirst()) {
            labels.append(next)
            rest = after
        }
        guard let suffix = suffix(rest), labels.count >= 2, let tld = labels.last else { return nil }

        var words: [String] = []
        speak(user, into: &words) { c in
            switch c {
            case "-": return "dash"
            case ".": return "dot"
            case "+": return "plus"
            case "_": return "underscore"
            default: return nil
            }
        }
        words.append("at")
        for (i, part) in labels.dropLast().enumerated() {
            if i > 0 { words.append("dot") }
            speak(part, into: &words) { $0 == "-" ? "dash" : nil }
        }
        words.append("dot")
        return words.joined(separator: " ") + " " + spellShort(tld) + PhononScalar.string(suffix)
    }

    /// Append the words of `part` to `words`, each separator for which `spoken` has a word
    /// replaced by that word.
    private static func speak(
        _ part: ArraySlice<Unicode.Scalar>, into words: inout [String], spoken: (Unicode.Scalar) -> String?
    ) {
        var start = part.startIndex
        for i in part.indices {
            if let word = spoken(part[i]) {
                if start < i { words.append(PhononScalar.string(part[start..<i])) }
                words.append(word)
                start = i + 1
            }
        }
        if start < part.endIndex { words.append(PhononScalar.string(part[start...])) }
    }

    // MARK: - URLs

    /// A web address with an optional `http(s)://` and path, its last label a top-level domain,
    /// read with its separators spoken: `https://www.kyutai.fr` becomes "H-T-T-P-S colon slash
    /// slash W-W-W dot kyutai dot F-R".
    static func urls(_ word: Scalars) -> String? {
        var words: [String] = []
        var rest = word[...]
        if rest.starts(with: "https://".unicodeScalars) {
            words.append("H-T-T-P-S")
            rest = rest.dropFirst(8)
        } else if rest.starts(with: "http://".unicodeScalars) {
            words.append("H-T-T-P")
            rest = rest.dropFirst(7)
        }
        if !words.isEmpty {
            words.append(contentsOf: ["colon", "slash", "slash"])
        }

        func run(
            _ s: ArraySlice<Unicode.Scalar>, _ keep: (Unicode.Scalar) -> Bool
        )
            -> (ArraySlice<Unicode.Scalar>, ArraySlice<Unicode.Scalar>)?
        {
            let n = s.prefix(while: keep).count
            guard n > 0 else { return nil }
            return (s.prefix(n), s.dropFirst(n))
        }
        let labelChar: (Unicode.Scalar) -> Bool = { PhononScalar.isAlphanumeric($0) || $0 == "-" }

        guard let (first, afterFirst) = run(rest, labelChar) else { return nil }
        var labels = [first]
        rest = afterFirst
        while rest.first == ".", let (next, after) = run(rest.dropFirst(), labelChar) {
            labels.append(next)
            rest = after
        }
        var path: ArraySlice<Unicode.Scalar>?
        if let (found, after) = run(rest, { labelChar($0) || $0 == "/" }) {
            path = found
            rest = after
        }
        guard suffix(rest) != nil, let tld = labels.last, labels.count >= 2, PhononTopLevelDomains.contains(tld)
        else { return nil }
        let capitalized = tld.contains(where: PhononScalar.isUppercase) && tld.contains(where: PhononScalar.isLowercase)
        if capitalized && words.isEmpty
            && !labels[0].map(PhononScalar.asciiLowercased).elementsEqual("www".unicodeScalars)
        {
            return nil
        }

        for (i, label) in labels.enumerated() {
            if i > 0 { words.append("dot") }
            speakAddressPart(label, into: &words)
        }
        if let path {
            speakAddressPart(path, into: &words)
        }
        return words.joined(separator: " ") + PhononScalar.string(rest)
    }

    /// Append the words of a label or a path: dashes and slashes spoken, each ASCII digit a word
    /// of its own, and two-letter words spelled out.
    private static func speakAddressPart(_ part: ArraySlice<Unicode.Scalar>, into words: inout [String]) {
        var start = part.startIndex
        for i in part.indices {
            let c = part[i]
            let word: String
            switch c {
            case "-": word = "dash"
            case "/": word = "slash"
            case "0"..."9": word = String(c)
            default: continue
            }
            if start < i { words.append(spellShort(part[start..<i])) }
            words.append(word)
            start = i + 1
        }
        if start < part.endIndex { words.append(spellShort(part[start...])) }
    }

    // MARK: - Abbreviations

    private static let abbreviationTable: [String: String] = [
        "Lt": "Lieutenant", "Lt.": "Lieutenant", "Col.": "Colonel", "Sgt.": "Sergeant", "Sgt": "Sergeant",
        "Capt.": "Captain", "Cpl.": "Corporal", "Gen.": "General", "Adm.": "Admiral", "Sen.": "Senator",
        "Sens.": "Senators", "Rep.": "Representative", "Reps.": "Representatives", "Gov.": "Governor",
        "Pres.": "President", "Det.": "Detective", "Insp.": "Inspector", "Supt.": "Superintendent",
        "Ft.": "Fort", "Mt.": "Mount", "Jan.": "January", "Feb.": "February", "Mar.": "March",
        "Apr.": "April", "Jun.": "June", "Jul.": "July", "Aug.": "August", "Sep.": "September",
        "Sept.": "September", "Oct.": "October", "Nov.": "November", "Dec.": "December", "Mon.": "Monday",
        "Tue.": "Tuesday", "Tues.": "Tuesday", "Wed.": "Wednesday", "Thu.": "Thursday", "Thur.": "Thursday",
        "Thurs.": "Thursday", "Fri.": "Friday", "Sat.": "Saturday", "Sun.": "Sunday", "vol.": "volume",
        "Vol.": "Volume", "ed.": "edition", "approx.": "approximately", "Approx.": "Approximately",
        "dept.": "department", "Dept.": "Department", "vs.": "versus", "w/": "with", "W/": "With",
        "w/o": "without", "W/o": "Without",
    ]

    static func abbreviations(_ word: Scalars) -> String? {
        let (body, suffix) = splitSuffix(word)
        var abbreviation = body
        var rest = suffix[...]
        if suffix.first == "." {
            abbreviation.append(".")
            rest = suffix.dropFirst()
        }
        guard let key = PhononScalar.ascii(abbreviation), let expansion = abbreviationTable[key] else {
            return nil
        }
        return expansion + PhononScalar.string(rest)
    }

    /// "Lt Col" as one rank.
    private static func ranks(_ words: ArraySlice<Scalars>) -> (String, Int)? {
        guard words.count >= 2 else { return nil }
        let first = words[words.startIndex]
        let (rank, suffix) = splitSuffix(words[words.startIndex + 1])
        guard PhononScalar.equals(first, "Lt"), PhononScalar.equals(rank, "Col") else { return nil }
        return ("Lieutenant Colonel" + PhononScalar.string(suffix), 2)
    }

    // MARK: - Elongations

    /// Stretched words shortened: "sooo" becomes "so", "Ahhh!" "Ah!".
    static func elongations(_ word: Scalars) -> String? {
        let (chars, suffix) = splitSuffix(word)
        let stretched: (Unicode.Scalar) -> Bool = {
            "aeiouyhw".unicodeScalars.contains(PhononScalar.asciiLowercased($0))
        }
        guard chars.allSatisfy(PhononScalar.isASCIIAlphabetic), !chars.allSatisfy(PhononScalar.isASCIIUppercase)
        else { return nil }
        var hasRun = false
        if chars.count >= 3 {
            for i in 0...(chars.count - 3)
            where chars[i] == chars[i + 1] && chars[i + 1] == chars[i + 2] && stretched(chars[i]) {
                hasRun = true
                break
            }
        }
        guard hasRun else { return nil }
        var out: Scalars = []
        for (i, c) in chars.enumerated() {
            let run = i >= 1 && chars[i - 1] == c && stretched(c)
            let window = Array(chars[max(i - 2, 0)..<min(i + 3, chars.count)])
            var long = false
            if window.count >= 3 {
                for j in 0...(window.count - 3) where window[j] == c && window[j + 1] == c && window[j + 2] == c {
                    long = true
                    break
                }
            }
            if !(run && long) {
                out.append(c)
            }
        }
        return PhononScalar.string(out + suffix)
    }
}
#endif
