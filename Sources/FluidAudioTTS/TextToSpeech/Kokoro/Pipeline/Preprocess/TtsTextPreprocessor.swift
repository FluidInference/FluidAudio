import FluidAudio
import Foundation

public struct TtsPreprocessingResult {
    public let text: String
    public let phoneticOverrides: [TtsPhoneticOverride]

    public init(text: String, phoneticOverrides: [TtsPhoneticOverride]) {
        self.text = text
        self.phoneticOverrides = phoneticOverrides
    }
}

public struct TtsPhoneticOverride: Sendable {
    public let wordIndex: Int
    public let tokens: [String]
    public let scalarTokens: [String]
    public let raw: String
    public let word: String

    public init(wordIndex: Int, tokens: [String], scalarTokens: [String], raw: String, word: String) {
        self.wordIndex = wordIndex
        self.tokens = tokens
        self.scalarTokens = scalarTokens
        self.raw = raw
        self.word = word
    }
}

/// Text preprocessing for TTS following mlx-audio's comprehensive approach
/// Handles numbers, currencies, times, units, and other text normalization
/// All preprocessing happens before tokenization to prevent splitting issues
enum TtsTextPreprocessor {

    /// Main preprocessing entry point that normalizes text for better TTS synthesis
    /// Following mlx-audio's order: commas → ranges → currencies → times → decimals → units → abbreviations
    static func preprocess(_ text: String) -> String {
        preprocessDetailed(text).text
    }

    static func preprocessDetailed(_ text: String) -> TtsPreprocessingResult {
        var processed = text

        // 0. Process SSML tags first (before all other preprocessing)
        let ssmlResult = SSMLProcessor.process(processed)
        processed = ssmlResult.text
        let ssmlOverrides = ssmlResult.phoneticOverrides

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

        // 8. Handle alias replacement [LOL](laugh out loud)
        processed = processAliasReplacement(processed)

        // 9. Handle phonetic replacement [Kokoro](/kˈOkəɹO/)
        let phoneticResult = processPhoneticReplacement(processed)
        processed = phoneticResult.text

        // Merge SSML overrides with markdown-style overrides
        // SSML overrides come first (processed first), then markdown-style
        let mergedOverrides = mergePhoneticOverrides(
            ssmlOverrides: ssmlOverrides,
            markdownOverrides: phoneticResult.overrides
        )

        return TtsPreprocessingResult(text: processed, phoneticOverrides: mergedOverrides)
    }

    /// Merge phonetic overrides from SSML and markdown sources
    /// Both are already sorted by word index; merge them maintaining sort order
    private static func mergePhoneticOverrides(
        ssmlOverrides: [TtsPhoneticOverride],
        markdownOverrides: [TtsPhoneticOverride]
    ) -> [TtsPhoneticOverride] {
        // If either is empty, return the other
        if ssmlOverrides.isEmpty { return markdownOverrides }
        if markdownOverrides.isEmpty { return ssmlOverrides }

        // Merge maintaining sort order by word index
        var merged: [TtsPhoneticOverride] = []
        var ssmlIndex = 0
        var markdownIndex = 0

        while ssmlIndex < ssmlOverrides.count && markdownIndex < markdownOverrides.count {
            let ssml = ssmlOverrides[ssmlIndex]
            let markdown = markdownOverrides[markdownIndex]

            if ssml.wordIndex <= markdown.wordIndex {
                merged.append(ssml)
                ssmlIndex += 1
            } else {
                merged.append(markdown)
                markdownIndex += 1
            }
        }

        // Append remaining
        merged.append(contentsOf: ssmlOverrides[ssmlIndex...])
        merged.append(contentsOf: markdownOverrides[markdownIndex...])

        return merged
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
                integerWords =
                    FluidAudioTTS.spellOutFormatter.string(from: NSNumber(value: integerValue)) ?? integerPart
            } else {
                integerWords = integerPart
            }

            // Convert each decimal digit to individual words
            let decimalWords = decimalPart.compactMap { FluidAudioTTS.digitToWord($0) }.joined(separator: " ")

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

    // MARK: - Alias Replacement

    private static func processAliasReplacement(_ text: String) -> String {
        guard text.contains("[") && text.contains("](") else {
            return text
        }

        let searchRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = aliasRegex.matches(in: text, options: [], range: searchRange)

        var result = text
        for match in matches.reversed() {
            guard
                let fullRange = Range(match.range, in: result),
                let replacementRange = Range(match.range(at: 1), in: result)
            else {
                continue
            }

            let replacement = result[replacementRange].trimmingCharacters(in: .whitespacesAndNewlines)
            if isPhoneticReplacement(replacement) {
                continue
            }
            result.replaceSubrange(fullRange, with: replacement)
        }

        return result
    }

    private static func processPhoneticReplacement(_ text: String) -> (text: String, overrides: [TtsPhoneticOverride]) {
        guard text.contains("[") else {
            return (text, [])
        }

        var overrides: [TtsPhoneticOverride] = []
        overrides.reserveCapacity(4)

        var accumulator = WordAccumulator(
            expectedScalarCount: text.unicodeScalars.count,
            apostrophes: FluidAudioTTS.phoneticApostropheCharacters
        )

        let end = text.endIndex
        var index = text.startIndex

        while index < end {
            let character = text[index]
            if character == "[" {
                if let parsed = parsePhoneticSpan(in: text, startingAt: index) {
                    let (word, raw, tokens, scalarTokens, nextIndex) = parsed
                    let trimmed = word.trimmingCharacters(in: .whitespacesAndNewlines)
                    let normalizedWord = normalizeAliasWord(trimmed)

                    if !normalizedWord.isEmpty {
                        accumulator.finalizeCurrentWord()
                        let overrideIndex = accumulator.completedWords
                        accumulator.append(text: normalizedWord)
                        overrides.append(
                            TtsPhoneticOverride(
                                wordIndex: overrideIndex,
                                tokens: tokens,
                                scalarTokens: scalarTokens,
                                raw: raw,
                                word: normalizedWord
                            )
                        )
                        index = nextIndex
                        continue
                    }
                }

                accumulator.append(character: character)
                index = text.index(after: index)
                continue
            }

            accumulator.append(character: character)
            index = text.index(after: index)
        }

        accumulator.finalizeCurrentWord()
        return (accumulator.rendered(), overrides)
    }

    private static func parsePhoneticSpan(
        in text: String,
        startingAt start: String.Index
    ) -> (word: String, raw: String, tokens: [String], scalarTokens: [String], nextIndex: String.Index)? {
        let end = text.endIndex
        var cursor = text.index(after: start)
        guard cursor < end else { return nil }

        var closingBracket: String.Index?
        while cursor < end {
            let character = text[cursor]
            if character == "]" {
                closingBracket = cursor
                break
            }
            if character == "[" {
                return nil
            }
            cursor = text.index(after: cursor)
        }

        guard let bracket = closingBracket else { return nil }
        let wordRange = text.index(after: start)..<bracket
        let afterBracket = text.index(after: bracket)
        guard afterBracket < end, text[afterBracket] == "(" else { return nil }

        var search = text.index(after: afterBracket)
        var closingParen: String.Index?
        while search < end {
            if text[search] == ")" {
                closingParen = search
                break
            }
            search = text.index(after: search)
        }

        guard let paren = closingParen else { return nil }
        let phonemeRange = text.index(after: afterBracket)..<paren
        let outer = text[phonemeRange]
        let trimmedOuter = outer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedOuter.first == "/", trimmedOuter.last == "/" else { return nil }

        let inner = trimmedOuter.dropFirst().dropLast()
        let raw = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return nil }

        let (tokens, scalarTokens) = tokenizePhonemeString(raw)
        guard !tokens.isEmpty || !scalarTokens.isEmpty else { return nil }

        let nextIndex = text.index(after: paren)
        let word = String(text[wordRange])
        return (word, raw, tokens, scalarTokens, nextIndex)
    }

    private static func tokenizePhonemeString(_ value: String) -> (tokens: [String], scalarTokens: [String]) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return ([], []) }

        let scalarTokens = trimmed.unicodeScalars.map { String($0) }
        let components = trimmed.split(whereSeparator: { $0.isWhitespace })
        if components.count > 1 {
            let tokens = components.map { String($0) }
            return (tokens, scalarTokens)
        }

        return ([trimmed], scalarTokens)
    }

    private static func normalizeAliasWord(_ value: String) -> String {
        value.precomposedStringWithCanonicalMapping
    }

    private static func isPhoneticReplacement(_ value: String) -> Bool {
        guard value.count >= 2 else { return false }
        return value.first == "/" && value.last == "/"
    }

    private struct WordAccumulator {
        private var scalars: [UnicodeScalar]
        private(set) var completedWords: Int = 0
        private var inWord = false
        private let apostrophes: Set<Character>

        init(expectedScalarCount: Int, apostrophes: Set<Character>) {
            self.scalars = []
            self.scalars.reserveCapacity(max(expectedScalarCount, 0))
            self.apostrophes = apostrophes
        }

        mutating func append(character: Character) {
            updateWordState(for: character)
            scalars.append(contentsOf: character.unicodeScalars)
        }

        mutating func append<T: StringProtocol>(text: T) {
            for character in text {
                append(character: character)
            }
        }

        mutating func finalizeCurrentWord() {
            if inWord {
                completedWords += 1
                inWord = false
            }
        }

        func rendered() -> String {
            String(String.UnicodeScalarView(scalars))
        }

        private mutating func updateWordState(for character: Character) {
            if Self.isWordLike(character, apostrophes: apostrophes) {
                if !inWord {
                    inWord = true
                }
                return
            }

            if inWord && Self.isCombiningMark(character) {
                return
            }

            finalizeCurrentWord()
        }

        private static func isWordLike(_ character: Character, apostrophes: Set<Character>) -> Bool {
            FluidAudioTTS.isWordCharacter(character, apostrophes: apostrophes)
        }

        private static func isCombiningMark(_ character: Character) -> Bool {
            character.unicodeScalars.allSatisfy { scalar in
                switch scalar.properties.generalCategory {
                case .nonspacingMark, .spacingMark, .enclosingMark:
                    return true
                default:
                    return false
                }
            }
        }
    }

    // MARK: - Constants

    private static let aliasWordMaxLength = 256
    private static let aliasReplacementMaxLength = 512

    private static let aliasRegex: NSRegularExpression = {
        let pattern =
            "\\[[^\\[\\]]{1,\(aliasWordMaxLength)}\\]\\(\\s*([^\\)]{1,\(aliasReplacementMaxLength)}?)\\s*\\)"
        return try! NSRegularExpression(pattern: pattern, options: [])
    }()

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
