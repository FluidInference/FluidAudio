import Foundation

enum ChunkPreprocessor {
    private static let currencies: [Character: (bill: String, cent: String)] = [
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent"),
    ]

    private static let currencyRegex = try! NSRegularExpression(
        pattern: #"[\$£€]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[\$£€]\d+\.\d\d?\b"#
    )

    private static let timeRegex = try! NSRegularExpression(
        pattern: #"\b(?:[1-9]|1[0-2]):[0-5]\d\b"#
    )

    private static let decimalRegex = try! NSRegularExpression(
        pattern: #"\b\d*\.\d+\b"#
    )

    private static let rangeRegex = try! NSRegularExpression(
        pattern: #"([\$£€]?\d+)-([\$£€]?\d+)"#
    )

    private static let commaInNumberRegex = try! NSRegularExpression(
        pattern: #"(^|[^\d])(\d+(?:,\d+)*)([^\d]|$)"#
    )

    private static let linkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]+)\]\(([^\)]*)\)"#
    )

    static func process(_ text: String) -> String {
        var processed = text
        processed = removeCommas(from: processed)
        processed = Self.rangeRegex.stringByReplacingMatches(
            in: processed,
            range: NSRange(processed.startIndex..., in: processed),
            withTemplate: "$1 to $2"
        )
        processed = flipMoney(processed)
        processed = splitTimes(processed)
        processed = spellDecimals(processed)
        processed = applyAliasReplacements(processed)
        return processed
    }

    private static func removeCommas(from text: String) -> String {
        let replaced = commaInNumberRegex.stringByReplacingMatches(
            in: text,
            range: NSRange(text.startIndex..., in: text),
            withTemplate: "$1$2$3"
        )
        return replaced.replacingOccurrences(of: ",", with: "")
    }

    private static func flipMoney(_ text: String) -> String {
        var result = text
        let matches = currencyRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            guard let range = Range(match.range, in: text) else { continue }
            let token = String(text[range])
            guard let symbol = token.first,
                let currency = currencies[symbol]
            else { continue }

            let value = String(token.dropFirst())
            let components = value.components(separatedBy: ".")
            let dollars = components[0]
            let cents = components.count > 1 ? components[1] : "0"

            let replacement: String
            if let centValue = Int(cents), centValue == 0 {
                if let dollarValue = Int(dollars), dollarValue == 1 {
                    replacement = "\(dollars) \(currency.bill)"
                } else {
                    replacement = "\(dollars) \(currency.bill)s"
                }
            } else {
                let dollarPart: String
                if let dollarValue = Int(dollars), dollarValue == 1 {
                    dollarPart = "\(dollars) \(currency.bill)"
                } else {
                    dollarPart = "\(dollars) \(currency.bill)s"
                }
                replacement = "\(dollarPart) and \(cents) \(currency.cent)s"
            }

            result = result.replacingCharacters(in: range, with: replacement)
        }

        return result
    }

    private static func splitTimes(_ text: String) -> String {
        var result = text
        let matches = timeRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            guard let range = Range(match.range, in: text) else { continue }
            let substring = String(text[range])
            let parts = substring.components(separatedBy: ":")
            guard parts.count == 2,
                let hour = Int(parts[0]),
                let minute = Int(parts[1])
            else { continue }

            let replacement: String
            if minute == 0 {
                replacement = "\(hour) o'clock"
            } else if minute < 10 {
                replacement = "\(hour) oh \(minute)"
            } else {
                replacement = "\(hour) \(minute)"
            }

            result = result.replacingCharacters(in: range, with: replacement)
        }

        return result
    }

    private static func spellDecimals(_ text: String) -> String {
        var result = text
        let decimalMatches = decimalRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        let linkMatches = linkRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        var excludedRanges: [NSRange] = []
        for match in linkMatches where match.numberOfRanges >= 3 {
            excludedRanges.append(match.range(at: 2))
        }

        for match in decimalMatches.reversed() {
            let range = match.range
            guard excludedRanges.allSatisfy({ NSIntersectionRange($0, range).length == 0 }) else { continue }
            guard let swiftRange = Range(range, in: text) else { continue }
            let substring = String(text[swiftRange])
            let pieces = substring.components(separatedBy: ".")
            guard pieces.count == 2 else { continue }
            let integerPart = pieces[0]
            let decimalPart = pieces[1]
            let spelledDigits = decimalPart.map { String($0) }.joined(separator: " ")
            let replacement = "\(integerPart) point \(spelledDigits)"
            result = result.replacingCharacters(in: swiftRange, with: replacement)
        }

        return result
    }

    private static func applyAliasReplacements(_ text: String) -> String {
        var result = text
        let matches = linkRegex.matches(in: text, range: NSRange(text.startIndex..., in: text))

        for match in matches.reversed() {
            guard match.numberOfRanges >= 3,
                let fullRange = Range(match.range, in: text),
                let originalRange = Range(match.range(at: 1), in: text),
                let replacementRange = Range(match.range(at: 2), in: text)
            else { continue }

            let original = String(text[originalRange])
            let replacement = String(text[replacementRange])

            let trimmedReplacement = replacement.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedReplacement.isEmpty else { continue }

            let isPhonemeDirective = trimmedReplacement.hasPrefix("/") && trimmedReplacement.hasSuffix("/")
            let isNumericDirective = Float(trimmedReplacement) != nil

            if isPhonemeDirective || isNumericDirective {
                // For phoneme overrides or stress controls, keep the original surface text.
                result.replaceSubrange(fullRange, with: original)
            } else {
                result.replaceSubrange(fullRange, with: replacement)
            }
        }

        return result
    }
}
