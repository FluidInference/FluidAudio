import Foundation

/// Spanish / French number reading for builds without the NeMo engine
/// (`NemoTextProcessing` trait off), so digits are spoken instead of dropped
/// by the G2P tokenizers. Covers cardinals up to 10¹², decimals and
/// percentages, and follows NeMo's readings so output does not change with
/// the trait (`21` → `veintiún`, `1001` → `mille et un`). Everything else,
/// including dates, currency and ordinals, needs the full engine.
enum RomanceNumberNormalizer {

    enum Language { case spanish, french }

    static func normalize(_ text: String, language: Language) -> String {
        // Thousands groups: `.` in both languages, spaces (incl. NBSP and
        // narrow NBSP) in French. Decimal comma, or a point not followed by
        // exactly three digits.
        let group = language == .french ? "[.\\u00A0\\u202F ]" : "[.]"
        // French sets a (narrow) space before %.
        let percent = language == .french ? "(?:[\\u00A0\\u202F ]?%)?" : "%?"
        let pattern = "\\d{1,3}(?:\(group)\\d{3})+(?:,\\d+)?\(percent)|\\d+(?:[,.]\\d+)?\(percent)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        var result = ""
        var cursor = text.startIndex
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range, in: text) else { continue }
            result += text[cursor..<range.lowerBound]
            result += read(String(text[range]), language: language, groupSeparator: group)
            cursor = range.upperBound
        }
        result += text[cursor...]
        return result
    }

    private static func read(_ token: String, language: Language, groupSeparator: String) -> String {
        var body = token
        let percent = body.hasSuffix("%")
        if percent {
            body.removeLast()
            while let last = body.last, last.isWhitespace { body.removeLast() }
        }

        var integerPart = body
        var fraction: String?
        if let comma = body.firstIndex(of: ",") {
            integerPart = String(body[..<comma])
            fraction = String(body[body.index(after: comma)...])
        } else if body.range(of: "^\\d{1,3}(" + groupSeparator + "\\d{3})+$", options: .regularExpression) == nil,
            let point = body.firstIndex(of: ".")
        {
            integerPart = String(body[..<point])
            fraction = String(body[body.index(after: point)...])
        }
        let digits = integerPart.filter(\.isNumber)
        guard let value = Int64(digits), value < 1_000_000_000_000 else {
            return token.map { String($0) }.joined(separator: " ")
        }

        var words = cardinal(value, language)
        if let fraction, !fraction.isEmpty {
            let separator = language == .spanish ? "coma" : "virgule"
            let leadingZeros = fraction.prefix { $0 == "0" }.count
            let zero = language == .spanish ? "cero" : "zéro"
            var fractionWords = Array(repeating: zero, count: leadingZeros)
            if let rest = Int64(fraction.dropFirst(leadingZeros)), leadingZeros < fraction.count {
                fractionWords.append(cardinal(rest, language))
            }
            words += " \(separator) " + fractionWords.joined(separator: " ")
        }
        if percent {
            words += language == .spanish ? " por ciento" : " pour cent"
        }
        return words
    }

    static func cardinal(_ n: Int64, _ language: Language) -> String {
        language == .spanish ? spanish(n) : french(n)
    }

    // MARK: - Spanish

    private static let spanishUnits = [
        "cero", "un", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
        "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiún",
        "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
        "veintinueve",
    ]
    private static let spanishTens = [
        "", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa",
    ]
    private static let spanishHundreds = [
        "", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos", "setecientos",
        "ochocientos", "novecientos",
    ]

    private static func spanish(_ n: Int64) -> String {
        if n >= 1_000_000 {
            let millions = n / 1_000_000
            let rest = n % 1_000_000
            let head = millions == 1 ? "un millón" : spanish(millions) + " millones"
            return rest == 0 ? head : head + " " + spanish(rest)
        }
        if n >= 1000 {
            let thousands = n / 1000
            let rest = n % 1000
            let head = thousands == 1 ? "mil" : spanishBelowThousand(Int(thousands)) + " mil"
            return rest == 0 ? head : head + " " + spanishBelowThousand(Int(rest))
        }
        return spanishBelowThousand(Int(n))
    }

    private static func spanishBelowThousand(_ n: Int) -> String {
        if n == 100 { return "cien" }
        let hundreds = n / 100
        let rest = n % 100
        var parts: [String] = []
        if hundreds > 0 { parts.append(spanishHundreds[hundreds]) }
        if rest > 0 || n == 0 {
            if rest < 30 {
                parts.append(spanishUnits[rest])
            } else {
                parts.append(
                    rest % 10 == 0 ? spanishTens[rest / 10] : "\(spanishTens[rest / 10]) y \(spanishUnits[rest % 10])")
            }
        }
        return parts.joined(separator: " ")
    }

    // MARK: - French

    private static let frenchUnits = [
        "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze",
        "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf",
    ]
    private static let frenchTens = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante"]

    private static func french(_ n: Int64) -> String {
        if n >= 1_000_000_000 {
            let billions = n / 1_000_000_000
            let rest = n % 1_000_000_000
            let head = billions == 1 ? "un milliard" : french(billions) + " milliards"
            return rest == 0 ? head : head + " " + french(rest)
        }
        if n >= 1_000_000 {
            let millions = n / 1_000_000
            let rest = n % 1_000_000
            let head = millions == 1 ? "un million" : french(millions) + " millions"
            return rest == 0 ? head : head + " " + french(rest)
        }
        if n >= 1000 {
            let thousands = n / 1000
            let rest = n % 1000
            let head = thousands == 1 ? "mille" : frenchBelowThousand(Int(thousands)) + " mille"
            if rest == 0 { return head }
            return head + (thousands == 1 && rest == 1 ? " et un" : " " + frenchBelowThousand(Int(rest)))
        }
        return frenchBelowThousand(Int(n))
    }

    private static func frenchBelowThousand(_ n: Int) -> String {
        let hundreds = n / 100
        let rest = n % 100
        var parts: [String] = []
        if hundreds == 1 {
            parts.append("cent")
        } else if hundreds > 1 {
            parts.append(frenchUnits[hundreds] + (rest == 0 ? " cents" : " cent"))
        }
        if rest > 0 || n == 0 { parts.append(frenchBelowHundred(rest)) }
        return parts.joined(separator: " ")
    }

    private static func frenchBelowHundred(_ n: Int) -> String {
        if n < 20 { return frenchUnits[n] }
        let tens = n / 10
        let unit = n % 10
        switch tens {
        case 7:  // soixante-dix … soixante-dix-neuf
            return unit == 1 ? "soixante et onze" : "soixante-" + frenchUnits[10 + unit]
        case 8:
            return unit == 0 ? "quatre-vingts" : "quatre-vingt-" + frenchUnits[unit]
        case 9:
            return "quatre-vingt-" + frenchUnits[10 + unit]
        default:
            if unit == 0 { return frenchTens[tens] }
            return unit == 1 ? "\(frenchTens[tens]) et un" : "\(frenchTens[tens])-\(frenchUnits[unit])"
        }
    }
}
