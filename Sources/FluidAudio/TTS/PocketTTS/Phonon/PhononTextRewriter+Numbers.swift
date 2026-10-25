#if TTS
import Foundation

/// Numbers and amounts of money, ported from `ptts/src/preprocess/rewrite/numbers.rs` and
/// `currency.rs` (English only).
extension PhononTextRewriter {

    // MARK: - Numbers

    /// Large numbers with their scale words: "1,234.56" becomes "1 thousand 234 point 56".
    /// Values under a thousand and years stay as digits, and a leading zero marks a code, read
    /// digit by digit.
    static func numbers(_ word: Scalars) -> String? {
        let (body, end) = splitSuffix(word)
        if body.count >= 2 && body.first == "0" && allDigits(body) {
            return spell(body) + PhononScalar.string(end)
        }
        if let range = range(body) {
            return range + PhononScalar.string(end)
        }
        if let (words, suffix) = numberWords(word[...]) {
            return words + PhononScalar.string(suffix)
        }
        // "7,000-ton", "20-year-old": a number joined to a word by a dash.
        guard let dash = word.firstIndex(of: "-") else { return nil }
        let head = word[..<dash]
        let tail = word[dash...]
        guard let afterDash = tail.dropFirst().first, PhononScalar.isAlphabetic(afterDash),
            let (words, suffix) = numberWords(head), suffix.isEmpty
        else { return nil }
        return words + PhononScalar.string(tail)
    }

    /// "20-25" read "20 to 25"; year ranges too ("1445-1450", "2024-25").
    private static func range(_ body: Scalars) -> String? {
        guard let dash = body.firstIndex(of: "-") else { return nil }
        let from = Array(body[..<dash])
        let to = Array(body[(dash + 1)...])
        func small(_ s: Scalars) -> Bool {
            (2...3).contains(s.count) && allDigits(s) && (s.count == 1 || s.first != "0")
        }
        func year(_ s: Scalars) -> Bool {
            s.count == 4 && allDigits(s) && (s.first == "1" || s.starts(with: "20".unicodeScalars))
        }
        let ok: Bool
        if small(from) && small(to) {
            guard let a = UInt32(PhononScalar.string(from)), let b = UInt32(PhononScalar.string(to)) else {
                return nil
            }
            ok = a < b
        } else if from.count == 4 && to.count == 4 {
            ok = year(from) && year(to) && lexicographicallyLess(from[...], to[...])
        } else if from.count == 4 && to.count == 2 {
            ok = year(from) && allDigits(to) && lexicographicallyLess(from[2...], to[...])
        } else {
            ok = false
        }
        guard ok else { return nil }
        return PhononScalar.string(from) + " to " + PhononScalar.string(to)
    }

    private static func lexicographicallyLess(_ a: ArraySlice<Unicode.Scalar>, _ b: ArraySlice<Unicode.Scalar>) -> Bool
    {
        a.lexicographicallyPrecedes(b) { $0.value < $1.value }
    }

    private static let digitNames = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    /// Each digit spelled out, dashes dropped, anything else kept as it is.
    static func spell<C: Collection>(_ s: C) -> String where C.Element == Unicode.Scalar {
        s.filter { $0 != "-" }
            .map { c in PhononScalar.isASCIIDigit(c) ? digitNames[Int(c.value - 0x30)] : String(c) }
            .joined(separator: " ")
    }

    /// Runs of digit groups read digit by digit: "+44 20 7946 0958" or "07700 900123".
    static func digitRun(_ words: ArraySlice<Scalars>) -> (String, Int)? {
        func group(_ s: ArraySlice<Unicode.Scalar>) -> Bool {
            (1...8).contains(s.count)
                && s.allSatisfy { PhononScalar.isASCIIDigit($0) || PhononScalar.isASCIIUppercase($0) }
                && s.contains(where: PhononScalar.isASCIIDigit)
        }
        guard let firstWord = words.first else { return nil }
        let plus = firstWord.first == "+"
        var bodies: [ArraySlice<Unicode.Scalar>] = []
        var end: Scalars = []
        for word in words {
            let (body, suffix) = splitSuffix(word)
            let trimmed = bodies.isEmpty && plus ? body.dropFirst() : body[...]
            if !group(trimmed) { break }
            bodies.append(trimmed)
            if !suffix.isEmpty {
                end = suffix
                break
            }
        }
        let years = bodies.allSatisfy { b in
            b.count == 4 && (b.starts(with: "19".unicodeScalars) || b.starts(with: "20".unicodeScalars))
        }
        let long = bodies.count >= 3 && !years
        if !long && !(bodies.count == 2 && (plus || bodies.contains { $0.first == "0" })) {
            return nil
        }
        var spelled = bodies.map { spell($0) }.joined(separator: ", ")
        if plus { spelled = "plus " + spelled }
        return (spelled + PhononScalar.string(end), bodies.count)
    }

    /// "ending in -4098": a card or account code after "ending in", read digit by digit.
    static func endingIn(previous: String?, _ words: ArraySlice<Scalars>) -> (String, Int)? {
        guard let previous else { return nil }
        let lowered = previous.lowercased()
        guard ["ending", "ends", "ended"].contains(where: { PhononScalar.equals(lowered.unicodeScalars, $0) }),
            let first = words.first, PhononScalar.equals(first, "in"), words.count >= 2
        else { return nil }
        let (body, end) = splitSuffix(words[words.startIndex + 1])
        guard body.first == "-" else { return nil }
        let code = body.dropFirst()
        let ok: (Unicode.Scalar) -> Bool = {
            PhononScalar.isASCIIDigit($0) || PhononScalar.isASCIIUppercase($0) || $0 == "-"
        }
        guard code.contains(where: PhononScalar.isASCIIDigit), code.allSatisfy(ok) else { return nil }
        return ("in " + spell(code) + PhononScalar.string(end), 2)
    }

    /// What `numbers` reads `word` as, and the trailing punctuation it carries through, kept
    /// apart so that an amount of money can put its currency in between. English groups
    /// thousands with commas and marks decimals with a dot.
    static func numberWords(_ word: ArraySlice<Unicode.Scalar>) -> (String, Scalars)? {
        let scalars = Array(word)
        if scalars.count >= 2 && scalars[0] == "0" && PhononScalar.isASCIIDigit(scalars[1]) {
            return nil
        }
        let negative = scalars.first == "-"
        let rest = negative ? Array(scalars.dropFirst()) : scalars
        let intEnd = rest.firstIndex { !PhononScalar.isASCIIDigit($0) && $0 != "," } ?? rest.count
        var intPart = rest[..<intEnd]
        while intPart.last == "," { intPart = intPart.dropLast() }
        guard !intPart.isEmpty else { return nil }
        var tail = rest[intPart.endIndex...]
        var decimal: ArraySlice<Unicode.Scalar>?
        if tail.first == "." {
            let after = tail.dropFirst()
            let digitCount = after.prefix(while: PhononScalar.isASCIIDigit).count
            if digitCount > 0 {
                decimal = after.prefix(digitCount)
                tail = after.dropFirst(digitCount)
            }
        }
        guard let suffix = suffix(tail) else { return nil }
        let digits = PhononScalar.string(intPart.filter(PhononScalar.isASCIIDigit))
        guard let value = Int64(digits), value < 1_000_000_000_000 else { return nil }

        var words: String
        if value < 1000 || (1900..<2100).contains(value) {
            words = String(value)
        } else {
            var parts: [String] = []
            for (divisor, name) in [(Int64(1_000_000_000), "billion"), (1_000_000, "million"), (1000, "thousand")] {
                let group = (value / divisor) % 1000
                if group != 0 {
                    parts.append("\(group) \(name)")
                }
            }
            if value % 1000 != 0 {
                parts.append(String(value % 1000))
            }
            words = parts.joined(separator: " ")
        }
        if let decimal {
            words = "\(words) point \(PhononScalar.string(decimal))"
        }
        if negative {
            words = "minus \(words)"
        }
        return (words, suffix)
    }

    // MARK: - Currency

    private enum Currency {
        case dollar, euro, pound, yen, cent

        init?(symbol: Unicode.Scalar) {
            switch symbol {
            case "$": self = .dollar
            case "\u{20AC}": self = .euro
            case "\u{00A3}": self = .pound
            case "\u{00A5}": self = .yen
            case "\u{00A2}": self = .cent
            default: return nil
            }
        }

        /// The currency's English name, singular and plural.
        var name: (one: String, many: String) {
            switch self {
            case .dollar: return ("dollar", "dollars")
            case .euro: return ("euro", "euros")
            case .pound: return ("pound", "pounds")
            case .yen: return ("yen", "yen")
            case .cent: return ("cent", "cents")
            }
        }
    }

    /// Whether `c` is a currency symbol the currency rule reads.
    static func isCurrencySymbol(_ c: Unicode.Scalar) -> Bool {
        Currency(symbol: c) != nil
    }

    /// A number with a currency symbol before or after it, read as `numbers` reads the number,
    /// followed by the currency's name: "$4.59" becomes "4 dollars and 59 cents".
    static func currency(_ word: Scalars) -> String? {
        let (body, suffix) = splitSuffix(word)
        // "-$500" reads as "$-500" does.
        if body.first == "-", body.count >= 2, Currency(symbol: body[1]) != nil {
            return currency([body[1], "-"] + body[2...] + suffix)
        }
        guard let (currency, amount) = splitSymbol(body[...]) else { return nil }
        if let (number, scale) = splitScale(amount) {
            return withScale(number, scale: scale, currency: currency).map { $0 + PhononScalar.string(suffix) }
        }
        if currency != .cent, let spoken = withCents(amount, currency: currency) {
            return spoken + PhononScalar.string(suffix)
        }
        guard let (spoken, _) = numberWords(amount) else { return nil }
        // Always the plural, even for 1 ("$1" reads "1 dollars"), a quirk kept upstream; the
        // cent sign is newer, so "1¢" reads "1 cent".
        let name = currency == .cent && PhononScalar.equals(amount, "1") ? currency.name.one : currency.name.many
        return "\(spoken) \(name)" + PhononScalar.string(suffix)
    }

    private static let scaleSuffixes: [(String, String)] = [
        ("thousand", "thousand"), ("million", "million"), ("billion", "billion"), ("trillion", "trillion"),
        ("bn", "billion"), ("BN", "billion"), ("mn", "million"), ("tn", "trillion"), ("k", "thousand"),
        ("K", "thousand"), ("m", "million"), ("M", "million"), ("b", "billion"), ("B", "billion"),
    ]

    /// "3.2m" as "3.2" and "million".
    private static func splitScale(_ amount: ArraySlice<Unicode.Scalar>) -> (ArraySlice<Unicode.Scalar>, String)? {
        for (suffix, scale) in scaleSuffixes {
            let suffixScalars = Array(suffix.unicodeScalars)
            guard amount.count >= suffixScalars.count, amount.suffix(suffixScalars.count).elementsEqual(suffixScalars)
            else { continue }
            let number = amount.dropLast(suffixScalars.count)
            if let last = number.last, PhononScalar.isASCIIDigit(last) {
                return (number, scale)
            }
        }
        return nil
    }

    private static func withScale(_ amount: ArraySlice<Unicode.Scalar>, scale: String, currency: Currency) -> String? {
        guard let (words, rest) = numberWords(amount), rest.isEmpty else { return nil }
        return "\(words) \(scale) \(currency.name.many)"
    }

    /// "$2 million": an amount followed by a scale word, as two words.
    static func scaled(_ words: ArraySlice<Scalars>) -> (String, Int)? {
        guard words.count >= 2 else { return nil }
        let word = words[words.startIndex]
        let (scale, suffix) = splitSuffix(words[words.startIndex + 1])
        guard let (currency, amount) = splitSymbol(word[...]),
            let scaleWord = PhononScalar.ascii(scale),
            ["thousand", "million", "billion", "trillion"].contains(scaleWord),
            let spoken = withScale(amount, scale: scaleWord, currency: currency)
        else { return nil }
        return (spoken + PhononScalar.string(suffix), 2)
    }

    /// The currency of a leading or trailing symbol, and the amount it marks.
    private static func splitSymbol(_ word: ArraySlice<Unicode.Scalar>) -> (Currency, ArraySlice<Unicode.Scalar>)? {
        if let first = word.first, let currency = Currency(symbol: first) {
            return (currency, word.dropFirst())
        }
        if let last = word.last, let currency = Currency(symbol: last) {
            return (currency, word.dropLast())
        }
        return nil
    }

    /// `amount` read as whole units and cents, when it has exactly two decimals.
    private static func withCents(_ amount: ArraySlice<Unicode.Scalar>, currency: Currency) -> String? {
        guard let dot = amount.lastIndex(of: ".") else { return nil }
        let whole = amount[..<dot]
        let cents = amount[(dot + 1)...]
        let grouped = PhononScalar.split(whole, on: ",").enumerated().allSatisfy { i, group in
            allDigits(group) && (i == 0 ? group.count <= 3 : group.count == 3)
        }
        guard allDigits(whole) || grouped, cents.count == 2, allDigits(cents),
            let (wholeWords, _) = numberWords(whole)
        else { return nil }
        let units = "\(wholeWords) \(wholeWords == "1" ? currency.name.one : currency.name.many)"
        if PhononScalar.equals(cents, "00") {
            return units
        }
        // "05" is read "5"; English says cents even for pounds, a quirk kept upstream.
        let spokenCents = PhononScalar.string(cents.first == "0" ? cents.dropFirst() : cents)
        if wholeWords == "0" {
            return "\(spokenCents) cents"
        }
        return "\(units) and \(spokenCents) cents"
    }
}
#endif
