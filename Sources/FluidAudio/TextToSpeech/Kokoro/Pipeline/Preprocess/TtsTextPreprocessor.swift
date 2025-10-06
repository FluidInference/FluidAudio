import Foundation

/// Text preprocessing for TTS following mlx-audio's comprehensive approach
/// Handles numbers, currencies, times, units, and other text normalization
/// All preprocessing happens before tokenization to prevent splitting issues
enum TtsTextPreprocessor {

    /// Main preprocessing entry point that normalizes text for better TTS synthesis
    /// Following mlx-audio's order: commas → ranges → currencies → times → decimals → units → abbreviations
    static func preprocess(_ text: String) -> String {
        var processed = text

        // 1. Remove commas from numbers (1,000 → 1000)
        processed = removeCommasFromNumbers(processed)

        // 2. Handle ranges (5-10 → 5 to 10)
        processed = processRanges(processed)

        // 3. Process currencies ($12.50 → 12 dollars and 50 cents)
        processed = processCurrencies(processed)

        // 4. Process times (12:30 → 12 30)
        processed = processTimes(processed)

        // 5. Handle decimal numbers (12.3 → 12 point 3)
        processed = processDecimalNumbers(processed)

        // 6. Handle unit abbreviations (g → grams)
        processed = processUnitAbbreviations(processed)

        // 7. Handle common abbreviations and symbols
        processed = processCommonAbbreviations(processed)

        return processed
    }

    // MARK: - Number Processing

    /// Remove commas from numbers (1,000 → 1000)
    private static func removeCommasFromNumbers(_ text: String) -> String {
        let commaInNumberPattern = try! NSRegularExpression(
            pattern: "(^|[^\\d])(\\d+(?:,\\d+)*)([^\\d]|$)",
            options: []
        )

        return commaInNumberPattern.stringByReplacingMatches(
            in: text,
            options: [],
            range: NSRange(location: 0, length: text.count),
            withTemplate: "$1$2$3"
        ).replacingOccurrences(of: ",", with: "")
    }

    /// Handle ranges (5-10 → 5 to 10)
    private static func processRanges(_ text: String) -> String {
        let rangePattern = try! NSRegularExpression(
            pattern: "([\\$£€]?\\d+)-([\\$£€]?\\d+)",
            options: []
        )

        return rangePattern.stringByReplacingMatches(
            in: text,
            options: [],
            range: NSRange(location: 0, length: text.count),
            withTemplate: "$1 to $2"
        )
    }

    /// Process decimal numbers (12.3 → twelve point three) - fully spelled out approach
    private static func processDecimalNumbers(_ text: String) -> String {
        let decimalPattern = try! NSRegularExpression(
            pattern: "\\b\\d*\\.\\d+(?=\\s|[a-zA-Z]|$)",
            options: []
        )

        let matches = decimalPattern.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))

        var result = text

        // Process matches in reverse order to maintain string indices
        for match in matches.reversed() {
            guard let fullRange = Range(match.range, in: text) else { continue }

            let matchText = String(text[fullRange])
            let components = matchText.components(separatedBy: ".")
            guard components.count == 2 else { continue }

            let integerPart = components[0]
            let decimalPart = components[1]

            // Convert integer part to words
            let integerWords: String
            if let integerValue = Int(integerPart) {
                let formatter = NumberFormatter()
                formatter.numberStyle = .spellOut
                integerWords = formatter.string(from: NSNumber(value: integerValue)) ?? integerPart
            } else {
                integerWords = integerPart
            }

            // Convert each decimal digit to individual words
            let digitWords = [
                "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
            ]
            let decimalWords = decimalPart.compactMap { digitWords[String($0)] }.joined(separator: " ")

            let replacement = "\(integerWords) point \(decimalWords)"

            // Check if the decimal number is immediately followed by a letter (like "g")
            // and add a space if needed
            let endIndex = fullRange.upperBound
            let needsSpace = endIndex < result.endIndex && result[endIndex].isLetter
            let finalReplacement = needsSpace ? replacement + " " : replacement

            result.replaceSubrange(fullRange, with: finalReplacement)
        }

        return result
    }

    // MARK: - Currency Processing

    /// Process currencies ($12.50 → 12 dollars and 50 cents)
    private static func processCurrencies(_ text: String) -> String {
        let currencyPattern = try! NSRegularExpression(
            pattern:
                "[\\$£€]\\d+(?:\\.\\d+)?(?:\\ hundred|\\ thousand|\\ (?:[bm]|tr)illion)*\\b|[\\$£€]\\d+\\.\\d\\d?\\b",
            options: []
        )

        var result = text
        let matches = currencyPattern.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))

        for match in matches.reversed() {
            guard let fullRange = Range(match.range, in: text) else { continue }

            let matchText = String(text[fullRange])
            guard let currencySymbol = matchText.first,
                let currency = currencies[currencySymbol]
            else { continue }

            let value = String(matchText.dropFirst())
            let components = value.components(separatedBy: ".")
            let dollars = components[0]
            let cents = components.count > 1 ? components[1] : "0"

            let replacement: String
            if Int(cents) == 0 {
                replacement = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
            } else {
                let dollarPart = Int(dollars) == 1 ? "\(dollars) \(currency.bill)" : "\(dollars) \(currency.bill)s"
                replacement = "\(dollarPart) and \(cents) \(currency.cent)s"
            }

            result.replaceSubrange(fullRange, with: replacement)
        }

        return result
    }

    // MARK: - Time Processing

    /// Process times (12:30 → 12 30, 12:00 → 12 o'clock)
    private static func processTimes(_ text: String) -> String {
        let timePattern = try! NSRegularExpression(
            pattern: "\\b(?:[1-9]|1[0-2]):[0-5]\\d\\b",
            options: []
        )

        var result = text
        let matches = timePattern.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))

        for match in matches.reversed() {
            guard let fullRange = Range(match.range, in: text) else { continue }

            let matchText = String(text[fullRange])
            let components = matchText.components(separatedBy: ":")
            guard components.count == 2,
                let hour = Int(components[0]),
                let minute = Int(components[1])
            else { continue }

            let replacement: String
            if minute == 0 {
                replacement = "\(hour) o'clock"
            } else if minute < 10 {
                replacement = "\(hour) oh \(minute)"
            } else {
                replacement = "\(hour) \(minute)"
            }

            result.replaceSubrange(fullRange, with: replacement)
        }

        return result
    }

    // MARK: - Unit Abbreviations

    private static func processUnitAbbreviations(_ text: String) -> String {
        var processed = text

        // Process weight units
        processed = processUnits(processed, units: weightUnits)

        // Process volume units
        processed = processUnits(processed, units: volumeUnits)

        // Process length units
        processed = processUnits(processed, units: lengthUnits)

        // Process temperature units
        processed = processUnits(processed, units: temperatureUnits)

        // Process time units
        processed = processUnits(processed, units: timeUnits)

        return processed
    }

    private static func processUnits(_ text: String, units: [String: String]) -> String {
        var processed = text

        for (abbreviation, expansion) in units {
            // First pattern: Match numeric values with units (e.g., "5g", "12.3g")
            let numericPattern = "\\b(\\d+(?:\\.\\d+)?)\\s*\(NSRegularExpression.escapedPattern(for: abbreviation))\\b"
            let numericRegex = try! NSRegularExpression(pattern: numericPattern, options: [])

            let numericMatches = numericRegex.matches(
                in: processed, options: [], range: NSRange(location: 0, length: processed.count))

            // Process numeric matches in reverse order to maintain string indices
            for match in numericMatches.reversed() {
                guard let numberRange = Range(match.range(at: 1), in: processed),
                    let fullRange = Range(match.range, in: processed)
                else {
                    continue
                }

                let numberStr = String(processed[numberRange])
                let number = Double(numberStr) ?? 1.0

                // Use plural form for numbers != 1
                let unit = (number == 1.0) ? expansion : pluralize(expansion)
                let replacement = "\(numberStr) \(unit)"

                processed.replaceSubrange(fullRange, with: replacement)
            }

            // Second pattern: Match spelled-out numbers with units (e.g., "twelve point three g")
            // Only match valid spelled-out numbers, not arbitrary words
            let spelledNumberWords = [
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                "nineteen",
                "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                "hundred", "thousand", "million", "billion", "trillion", "point",
            ]

            // Create pattern that only matches actual spelled-out numbers
            let numberPattern = spelledNumberWords.map { NSRegularExpression.escapedPattern(for: $0) }.joined(
                separator: "|")
            let spelledPattern =
                "\\b((?:\(numberPattern))(?:\\s+(?:\(numberPattern)))*)\\s+\(NSRegularExpression.escapedPattern(for: abbreviation))\\b"
            let spelledRegex = try! NSRegularExpression(pattern: spelledPattern, options: [.caseInsensitive])

            let spelledMatches = spelledRegex.matches(
                in: processed, options: [], range: NSRange(location: 0, length: processed.count))

            // Process spelled matches in reverse order to maintain string indices
            for match in spelledMatches.reversed() {
                guard let numberRange = Range(match.range(at: 1), in: processed),
                    let fullRange = Range(match.range, in: processed)
                else {
                    continue
                }

                let numberStr = String(processed[numberRange])

                // For spelled-out numbers, always use plural form unless it's exactly "one"
                let unit = (numberStr == "one") ? expansion : pluralize(expansion)
                let replacement = "\(numberStr) \(unit)"

                processed.replaceSubrange(fullRange, with: replacement)
            }
        }

        return processed
    }

    private static func pluralize(_ word: String) -> String {
        // Simple pluralization rules
        if word.hasSuffix("s") || word.hasSuffix("x") || word.hasSuffix("ch") || word.hasSuffix("sh") {
            return word + "es"
        } else if word.hasSuffix("y") && word.count > 1 {
            let beforeY = word.dropLast()
            if !["a", "e", "i", "o", "u"].contains(String(beforeY.last ?? " ")) {
                return String(beforeY) + "ies"
            }
        } else if word.hasSuffix("f") {
            return String(word.dropLast()) + "ves"
        } else if word.hasSuffix("fe") {
            return String(word.dropLast(2)) + "ves"
        }

        return word + "s"
    }

    // MARK: - Common Abbreviations

    private static func processCommonAbbreviations(_ text: String) -> String {
        var processed = text

        for (abbreviation, expansion) in commonAbbreviations {
            let pattern = "\\b" + NSRegularExpression.escapedPattern(for: abbreviation) + "\\b"
            let regex = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])

            processed = regex.stringByReplacingMatches(
                in: processed,
                options: [],
                range: NSRange(location: 0, length: processed.count),
                withTemplate: expansion
            )
        }

        return processed
    }

    // MARK: - Constants

    private static let currencies: [Character: (bill: String, cent: String)] = [
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent"),
    ]

    private static let weightUnits: [String: String] = [
        "g": "gram",
        "kg": "kilogram",
        "mg": "milligram",
        "lb": "pound",
        "lbs": "pounds",
        "oz": "ounce",
        "t": "ton",
        "ton": "ton",
    ]

    private static let volumeUnits: [String: String] = [
        "ml": "milliliter",
        "l": "liter",
        "cl": "centiliter",
        "dl": "deciliter",
        "fl oz": "fluid ounce",
        "cup": "cup",
        "pt": "pint",
        "qt": "quart",
        "gal": "gallon",
        "tsp": "teaspoon",
        "tbsp": "tablespoon",
    ]

    private static let lengthUnits: [String: String] = [
        "mm": "millimeter",
        "cm": "centimeter",
        "m": "meter",
        "km": "kilometer",
        "in": "inch",
        "ft": "foot",
        "yd": "yard",
        "mi": "mile",
    ]

    private static let temperatureUnits: [String: String] = [
        "°C": "degrees Celsius",
        "°F": "degrees Fahrenheit",
        "°K": "degrees Kelvin",
        "C": "degrees Celsius",
        "F": "degrees Fahrenheit",
    ]

    private static let timeUnits: [String: String] = [
        "s": "second",
        "sec": "second",
        "min": "minute",
        "hr": "hour",
        "h": "hour",
    ]

    private static let commonAbbreviations: [String: String] = [
        // Common text abbreviations that affect speech
        "vs": "versus",
        "vs.": "versus",
        "etc": "etcetera",
        "etc.": "etcetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Dr.": "Doctor",
        "Prof.": "Professor",
        "St.": "Saint",
    ]
}
