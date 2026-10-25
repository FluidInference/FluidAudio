#if TTS
import Foundation

/// English text normalization applied by Gradium's Phonon TTS before tokenizing.
///
/// A line-for-line port of `normalize_text(input, Lang::En, Rules::DEFAULT)` from the `ptts`
/// crate (`ptts/src/preprocess.rs` and `ptts/src/preprocess/rewrite/`). Phonon checkpoints are
/// trained on ordinary written prose, so characters the model never saw (typographic quotes,
/// bullets, arrows, emoji) are removed or folded to ASCII, and symbols that are read aloud
/// (`@`, `+`, `=`, `°`, `×`, …) are spelled out. Each whitespace-separated word then goes to the
/// default rewrite rules: numbers, currency, dashed digits, emails, URLs, abbreviations and
/// elongations. Phone numbers, times and dates are opt-in upstream and are not ported.
///
/// The output must match the Rust implementation byte for byte; the golden test
/// `PhononTextNormalizerTests` runs every case of a Rust-generated fixture. All processing is
/// done on Unicode scalars (Rust `char`s), never on Swift `Character`s, so grapheme clustering
/// and canonical equivalence cannot change a result.
public enum PhononTextNormalizer {

    /// Normalize `text` the way Phonon's English frontend does before tokenizing.
    public static func normalize(_ text: String) -> String {
        let input = Array(text.unicodeScalars)
        var res = PhononStringAppender()
        var prev: Unicode.Scalar?
        var index = 0

        func peek(_ offset: Int = 0) -> Unicode.Scalar? {
            index + offset < input.count ? input[index + offset] : nil
        }

        while index < input.count {
            let c = input[index]
            index += 1
            defer { prev = c }

            if PhononCharTables.isDoubleQuote(c) {
                res.push("\"")
            } else if PhononCharTables.isSingleQuote(c) {
                res.push("'")
            } else if c == "\u{2010}" || c == "\u{2011}" || c == "\u{2012}" || c == "\u{2015}" {
                res.push("-")
            } else if c == "\u{2605}" || c == "\u{2606}" {
                // Star ratings: "★★★★☆" is "4 out of 5 stars", "★★★½" is "3.5 stars".
                var full = c == "\u{2605}" ? 1 : 0
                var empty = c == "\u{2606}" ? 1 : 0
                while let next = peek(), next == "\u{2605}" || next == "\u{2606}" {
                    full += next == "\u{2605}" ? 1 : 0
                    empty += next == "\u{2606}" ? 1 : 0
                    index += 1
                }
                let half = peek() == "\u{00BD}"
                if half { index += 1 }
                let count = half ? "\(full).5" : "\(full)"
                if empty > 0 {
                    res.pushSpoken("\(count) out of \(full + empty + (half ? 1 : 0)) stars")
                } else {
                    res.pushSpoken("\(count) stars")
                }
            } else if (c == "\u{00B0}" || c == "\u{00BA}") && res.endsWithN() {
                // "n°5" is the number sign, not "n degrees".
                if res.buffer.last == "." { res.buffer.removeLast() }
                let letter = res.buffer.popLast()
                res.pushSpoken(letter == "N" ? "Number" : "number")
            } else if c == "\u{00B0}" {
                res.pushSpoken(res.endsWithOne() ? "degree" : "degrees")
                let scale: String?
                switch peek() {
                case "C": scale = "Celsius"
                case "F": scale = "Fahrenheit"
                default: scale = nil
                }
                if let scale, !(peek(1).map(PhononScalar.isAlphanumeric) ?? false) {
                    index += 1
                    res.pushString(scale)
                }
            } else if let folded = PhononCharTables.unaccented(c) {
                res.push(folded)
            } else if let lowered = PhononCharTables.subscriptBase(c) {
                res.push(lowered)
            } else if let raised = PhononCharTables.superscript(c) {
                var digits = String(raised)
                while let next = peek(), let more = PhononCharTables.superscript(next) {
                    digits.unicodeScalars.append(more)
                    index += 1
                }
                if let prev, PhononScalar.isAlphanumeric(prev) || prev == ")" {
                    res.pushSpoken(power(digits))
                } else {
                    res.pushString(digits)
                }
            } else if let name = PhononCharTables.greek(c) {
                res.pushSpoken(name)
            } else if let name = PhononCharTables.symbol(c) {
                res.pushSpoken(name)
            } else if c == "\u{00B1}" {
                res.pushSpoken("plus or minus")
            } else if c == "\u{03C0}" {
                res.pushSpoken("pi")
            } else if c == "\u{2764}" || c == "\u{2665}" {
                if peek() == "\u{FE0F}" { index += 1 }
                res.pushSpoken("love")
            } else if c == "~", let next = peek(), PhononScalar.isASCIIDigit(next) {
                res.pushSpoken("about")
            } else if c == "*", prev.map(PhononScalar.isWhitespace) ?? true,
                let next = peek(), PhononScalar.isASCIIDigit(next)
            {
                res.pushSpoken("star")
            } else if PhononCharTables.isSpaceLike(c) {
                res.pushWhitespace()
            } else if PhononTextRewriter.isCurrencySymbol(c), res.endsWithSpacedDigit(),
                !(peek().map(PhononScalar.isASCIIDigit) ?? false)
            {
                // "250 €": a symbol after the number and a space joins the number.
                res.popWhitespace()
                res.push(c)
            } else if c == "\u{2026}" {
                res.push(".")
            } else if c == "=" {
                res.pushSpoken("equals")
            } else if c == ":", !(prev.map(PhononScalar.isWhitespace) ?? true),
                !(peek().map(PhononScalar.isWhitespace) ?? true)
            {
                // A `:` between two non-space characters, as in `10:30`, is kept.
                res.push(":")
            } else if c == ";" || c == ":" || c == "(" || c == ")" {
                res.popWhitespace()
                res.push(",")
                res.pushWhitespace()
            } else if PhononCharTables.isEmoji(c) || PhononScalar.isControl(c) || PhononScalar.isWhitespace(c) {
                res.pushWhitespace()
            } else {
                res.push(c)
            }
        }

        let text = res.intoScalars()
        let rewritten = PhononTextRewriter.rewriteText(text)
        return stripClosingQuote(rewritten)
    }

    /// Prepare one chunk of (already normalized) text for the model: collapse whitespace,
    /// capitalize the first letter and end on a full stop when the text ends on a letter or
    /// digit. Mirrors `prepare_text_prompt` in `ptts/src/tts_model.rs`.
    ///
    /// - Returns: The prompt text and the number of frames to keep generating after the model
    ///   signals end of speech: 3 for texts of at most four words, 1 otherwise.
    public static func prepareTextPrompt(_ text: String) -> (text: String, framesAfterEos: Int) {
        let scalars = PhononScalar.trimWhitespace(Array(text.unicodeScalars))
        if scalars.isEmpty {
            return ("", 3)
        }
        let unbroken = scalars.map { $0 == "\n" || $0 == "\r" ? " " : $0 }
        let words = PhononScalar.splitWhitespace(unbroken)
        let framesAfterEos = words.count <= 4 ? 3 : 1
        var joined = words.map(PhononScalar.string).joined(separator: " ")
        var joinedScalars = Array(joined.unicodeScalars)
        if let first = joinedScalars.first {
            joined = String(first).uppercased() + PhononScalar.string(joinedScalars.dropFirst())
            joinedScalars = Array(joined.unicodeScalars)
        }
        if let last = joinedScalars.last, PhononScalar.isAlphanumeric(last) {
            joined += "."
        }
        return (joined, framesAfterEos)
    }

    // MARK: - Helpers

    private static func power(_ digits: String) -> String {
        switch digits {
        case "2": return "squared"
        case "3": return "cubed"
        default: return "to the power of \(digits)"
        }
    }

    /// Drop a closing double quote at the end, and a closing single quote after sentence
    /// punctuation: the model reads a dangling quote as a sound.
    private static func stripClosingQuote(_ text: String) -> String {
        var scalars = Array(text.unicodeScalars)
        while scalars.last == "\"" { scalars.removeLast() }
        if scalars.last == "'" {
            let rest = scalars.dropLast()
            if let end = rest.last, ".!?,;".unicodeScalars.contains(end) {
                return PhononScalar.string(rest)
            }
        }
        return PhononScalar.string(scalars)
    }
}

/// Character sink that keeps the output free of the punctuation pile-ups the substitutions
/// would otherwise produce: a `.` or `,` swallows any whitespace and punctuation immediately
/// before it, and runs of whitespace collapse to a single space. Mirrors `StringAppender`.
struct PhononStringAppender {
    var buffer: [Unicode.Scalar] = []

    mutating func push(_ c: Unicode.Scalar) {
        if c == "." || c == "," {
            while let last = buffer.last,
                PhononScalar.isWhitespace(last) || (PhononScalar.isASCIIPunctuation(last) && !keeps(last))
            {
                buffer.removeLast()
            }
        }
        buffer.append(c)
    }

    mutating func pushString(_ s: String) {
        for c in s.unicodeScalars {
            push(c)
        }
    }

    /// Whether `last` stays before a `.` or `,`: quotes, the symbols the rules read, and the `$`
    /// of an amount such as "5$.".
    private func keeps(_ last: Unicode.Scalar) -> Bool {
        switch last {
        case "\"", "'", "@", "+", "%", "#", "&":
            return true
        case "$":
            return buffer.count >= 2 && PhononScalar.isASCIIDigit(buffer[buffer.count - 2])
        default:
            return false
        }
    }

    /// Whether the text ends with a number that is exactly "1" (for "1 degree").
    func endsWithOne() -> Bool {
        let end = (buffer.lastIndex { !PhononScalar.isWhitespace($0) }).map { $0 + 1 } ?? 0
        guard end > 0, buffer[end - 1] == "1" else { return false }
        if end >= 2 {
            let before = buffer[end - 2]
            if PhononScalar.isASCIIDigit(before) || before == "." || before == "," {
                return false
            }
        }
        return true
    }

    /// Whether the text ends with an "n" or "N" that is a word of its own, with or without a
    /// period, as in "n°5".
    func endsWithN() -> Bool {
        let b = buffer.last == "." ? Array(buffer.dropLast()) : buffer
        guard let last = b.last, last == "n" || last == "N" else { return false }
        return !(b.count >= 2 && PhononScalar.isAlphanumeric(b[b.count - 2]))
    }

    /// Whether the text ends with a digit and then whitespace, as before the `€` of "250 €".
    func endsWithSpacedDigit() -> Bool {
        guard lastIsWhitespace else { return false }
        return buffer.last { !PhononScalar.isWhitespace($0) }.map(PhononScalar.isASCIIDigit) ?? false
    }

    /// `word` as a word of its own, for a symbol read aloud.
    mutating func pushSpoken(_ word: String) {
        pushWhitespace()
        pushString(word)
        pushWhitespace()
    }

    mutating func intoScalars() -> [Unicode.Scalar] {
        popWhitespace()
        return buffer
    }

    var lastIsWhitespace: Bool {
        buffer.last.map(PhononScalar.isWhitespace) ?? false
    }

    mutating func pushWhitespace() {
        if !lastIsWhitespace && !buffer.isEmpty {
            push(" ")
        }
    }

    mutating func popWhitespace() {
        while lastIsWhitespace {
            buffer.removeLast()
        }
    }
}

/// Rust `char` predicates on Unicode scalars, so that the port classifies characters exactly as
/// `ptts` does.
enum PhononScalar {
    /// Rust `char::is_whitespace`: the Unicode `White_Space` property.
    static func isWhitespace(_ c: Unicode.Scalar) -> Bool { c.properties.isWhitespace }

    /// Rust `char::is_alphabetic`: the Unicode `Alphabetic` property.
    static func isAlphabetic(_ c: Unicode.Scalar) -> Bool { c.properties.isAlphabetic }

    /// Rust `char::is_alphanumeric`: `Alphabetic`, or a general category of `Nd`, `Nl` or `No`.
    static func isAlphanumeric(_ c: Unicode.Scalar) -> Bool {
        if c.isASCII {
            return isASCIIAlphanumeric(c)
        }
        if c.properties.isAlphabetic { return true }
        switch c.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }

    static func isUppercase(_ c: Unicode.Scalar) -> Bool { c.properties.isUppercase }
    static func isLowercase(_ c: Unicode.Scalar) -> Bool { c.properties.isLowercase }
    static func isControl(_ c: Unicode.Scalar) -> Bool { c.properties.generalCategory == .control }

    static func isASCIIDigit(_ c: Unicode.Scalar) -> Bool { c.value >= 0x30 && c.value <= 0x39 }
    static func isASCIIUppercase(_ c: Unicode.Scalar) -> Bool { c.value >= 0x41 && c.value <= 0x5A }
    static func isASCIILowercase(_ c: Unicode.Scalar) -> Bool { c.value >= 0x61 && c.value <= 0x7A }
    static func isASCIIAlphabetic(_ c: Unicode.Scalar) -> Bool { isASCIIUppercase(c) || isASCIILowercase(c) }
    static func isASCIIAlphanumeric(_ c: Unicode.Scalar) -> Bool { isASCIIAlphabetic(c) || isASCIIDigit(c) }

    /// Rust `char::is_ascii_punctuation`.
    static func isASCIIPunctuation(_ c: Unicode.Scalar) -> Bool {
        switch c.value {
        case 0x21...0x2F, 0x3A...0x40, 0x5B...0x60, 0x7B...0x7E: return true
        default: return false
        }
    }

    static func asciiUppercased(_ c: Unicode.Scalar) -> Unicode.Scalar {
        isASCIILowercase(c) ? Unicode.Scalar(UInt8(c.value - 32)) : c
    }

    static func asciiLowercased(_ c: Unicode.Scalar) -> Unicode.Scalar {
        isASCIIUppercase(c) ? Unicode.Scalar(UInt8(c.value + 32)) : c
    }

    static func string<S: Sequence>(_ scalars: S) -> String where S.Element == Unicode.Scalar {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }

    /// Rust `str::trim`.
    static func trimWhitespace(_ s: [Unicode.Scalar]) -> [Unicode.Scalar] {
        guard let start = s.firstIndex(where: { !isWhitespace($0) }),
            let end = s.lastIndex(where: { !isWhitespace($0) })
        else { return [] }
        return Array(s[start...end])
    }

    static func trimWhitespace(_ s: String) -> String {
        string(trimWhitespace(Array(s.unicodeScalars)))
    }

    /// Rust `str::split_whitespace`.
    static func splitWhitespace(_ s: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var words: [[Unicode.Scalar]] = []
        var current: [Unicode.Scalar] = []
        for c in s {
            if isWhitespace(c) {
                if !current.isEmpty {
                    words.append(current)
                    current = []
                }
            } else {
                current.append(c)
            }
        }
        if !current.isEmpty { words.append(current) }
        return words
    }

    /// Rust `str::split(separator)`: every piece, empty ones included.
    static func split<C: Collection>(_ s: C, on separator: Unicode.Scalar) -> [[Unicode.Scalar]]
    where C.Element == Unicode.Scalar {
        var pieces: [[Unicode.Scalar]] = [[]]
        for c in s {
            if c == separator {
                pieces.append([])
            } else {
                pieces[pieces.count - 1].append(c)
            }
        }
        return pieces
    }

    /// `s` as a `String` when every scalar is ASCII, for exact lookups in ASCII tables (a Swift
    /// `String` compares by canonical equivalence, under which e.g. the Kelvin sign equals "K").
    static func ascii<C: Collection>(_ s: C) -> String? where C.Element == Unicode.Scalar {
        guard s.allSatisfy({ $0.isASCII }) else { return nil }
        return string(s)
    }

    /// Whether the scalars of `s` are exactly those of `literal`.
    static func equals<C: Collection>(_ s: C, _ literal: String) -> Bool where C.Element == Unicode.Scalar {
        s.elementsEqual(literal.unicodeScalars)
    }
}
#endif
