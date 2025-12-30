import Foundation

/// Interprets say-as content based on interpret-as type
/// Converts specialized content (numbers, dates, etc.) to speakable text
public enum SayAsInterpreter {

    // MARK: - Cached Static Resources

    /// Ordinal words for numbers 1-19
    private static let ordinalWords: [Int: String] = [
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
        16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
    ]

    /// Month names (index 0 is empty for 1-based indexing)
    private static let monthNames = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    /// Regex for extracting numeric components from dates
    private static let digitPattern = try! NSRegularExpression(pattern: #"\d+"#, options: [])

    /// Regex for duration minutes
    private static let minutePattern = try! NSRegularExpression(pattern: #"(\d+)'"#, options: [])

    /// Regex for duration seconds
    private static let secondPattern = try! NSRegularExpression(pattern: #"(\d+)\""#, options: [])

    // MARK: - Public API

    /// Interpret content based on the interpret-as type
    /// - Parameters:
    ///   - content: The text content to interpret
    ///   - interpretAs: The interpretation type (e.g., "cardinal", "date")
    ///   - format: Optional format specifier (e.g., date format)
    /// - Returns: The interpreted speakable text
    public static func interpret(content: String, interpretAs: String, format: String?) -> String {
        let key = interpretAs.lowercased().trimmingCharacters(in: .whitespaces)
        let trimmedContent = content.trimmingCharacters(in: .whitespaces)

        switch key {
        case "characters", "spell-out":
            return interpretCharacters(trimmedContent)
        case "cardinal", "number":
            return interpretCardinal(trimmedContent)
        case "ordinal":
            return interpretOrdinal(trimmedContent)
        case "digits":
            return interpretDigits(trimmedContent)
        case "date":
            return interpretDate(trimmedContent, format: format)
        case "time":
            return interpretTime(trimmedContent)
        case "telephone":
            return interpretTelephone(trimmedContent)
        case "fraction":
            return interpretFraction(trimmedContent)
        default:
            // Unknown interpret-as: return content unchanged
            return content
        }
    }

    // MARK: - Interpreters

    /// Spell out each character: "ABC" -> "A B C"
    private static func interpretCharacters(_ content: String) -> String {
        content.map { String($0) }.joined(separator: " ")
    }

    /// Cardinal number: "123" -> "one hundred twenty three"
    private static func interpretCardinal(_ content: String) -> String {
        guard let number = Int(content.filter { $0.isNumber || $0 == "-" }) else {
            return content
        }
        return FluidAudioTTS.spellOutFormatter.string(from: NSNumber(value: number)) ?? content
    }

    /// Ordinal number: "1" -> "first", "23" -> "twenty third"
    private static func interpretOrdinal(_ content: String) -> String {
        guard let number = Int(content.filter { $0.isNumber }) else {
            return content
        }
        return ordinalWord(for: number)
    }

    /// Spell each digit: "123" -> "one two three"
    private static func interpretDigits(_ content: String) -> String {
        content.compactMap { char -> String? in
            FluidAudioTTS.digitToWord(char)
        }.joined(separator: " ")
    }

    /// Date interpretation with format support
    /// Formats: mdy, dmy, ymd, md, dm, ym, my, y, m, d
    private static func interpretDate(_ content: String, format: String?) -> String {
        let components = extractDateComponents(content)
        guard !components.isEmpty else { return content }

        let formatKey = format?.lowercased() ?? "mdy"
        let result = formatDate(components: components, format: formatKey)
        // Return original content if formatting failed (e.g., invalid month)
        return result.isEmpty ? content : result
    }

    /// Time/duration: "1'21\"" -> "one minute twenty one seconds"
    /// Also handles "2:30" -> "two thirty"
    private static func interpretTime(_ content: String) -> String {
        // Check for duration format: 1'21"
        if content.contains("'") || content.contains("\"") {
            return interpretDuration(content)
        }

        // Check for clock time: 2:30
        if content.contains(":") {
            return interpretClockTime(content)
        }

        return content
    }

    /// Telephone: "555-1234" -> "five five five one two three four"
    private static func interpretTelephone(_ content: String) -> String {
        let digits = content.filter { $0.isNumber }
        return interpretDigits(digits)
    }

    /// Fraction: "2/9" -> "two ninths", "1/2" -> "one half"
    private static func interpretFraction(_ content: String) -> String {
        // Handle mixed fractions like "3+1/2" or "3 1/2"
        let normalized = content.replacingOccurrences(of: "+", with: " ")
        let parts = normalized.split(separator: " ")

        if parts.count == 2,
            let wholeNumber = Int(parts[0]),
            let fractionPart = parseFractionPart(String(parts[1]))
        {
            let wholePart = interpretCardinal(String(wholeNumber))
            return "\(wholePart) and \(fractionPart)"
        }

        // Simple fraction
        if let result = parseFractionPart(content) {
            return result
        }

        return content
    }

    // MARK: - Helper Methods

    private static func ordinalWord(for number: Int) -> String {
        // Check cached ordinals for 1-19
        if let ordinal = ordinalWords[number] {
            return ordinal
        }

        // For numbers 20+, spell out the cardinal and add appropriate suffix
        guard let spelled = FluidAudioTTS.spellOutFormatter.string(from: NSNumber(value: number)) else {
            return "\(number)th"
        }

        return addOrdinalSuffix(to: spelled, number: number)
    }

    private static func addOrdinalSuffix(to spelled: String, number: Int) -> String {
        let lastDigit = number % 10
        let lastTwoDigits = number % 100

        // Handle teen numbers (11th, 12th, 13th, etc.)
        if lastTwoDigits >= 11 && lastTwoDigits <= 13 {
            if spelled.hasSuffix("one") {
                return String(spelled.dropLast(3)) + "eleventh"
            } else if spelled.hasSuffix("two") {
                return String(spelled.dropLast(3)) + "twelfth"
            } else if spelled.hasSuffix("three") {
                return String(spelled.dropLast(5)) + "thirteenth"
            }
        }

        // Handle regular endings
        switch lastDigit {
        case 1:
            if spelled.hasSuffix("one") {
                return String(spelled.dropLast(3)) + "first"
            }
        case 2:
            if spelled.hasSuffix("two") {
                return String(spelled.dropLast(3)) + "second"
            }
        case 3:
            if spelled.hasSuffix("three") {
                return String(spelled.dropLast(5)) + "third"
            }
        case 5:
            if spelled.hasSuffix("five") {
                return String(spelled.dropLast(4)) + "fifth"
            }
        case 8:
            if spelled.hasSuffix("eight") {
                return String(spelled.dropLast(5)) + "eighth"
            }
        case 9:
            if spelled.hasSuffix("nine") {
                return String(spelled.dropLast(4)) + "ninth"
            }
        case 0:
            if spelled.hasSuffix("y") {
                return String(spelled.dropLast(1)) + "ieth"
            }
        default:
            break
        }

        // Default: add "th"
        if spelled.hasSuffix("e") {
            return spelled + "th"
        }
        return spelled + "th"
    }

    private static func extractDateComponents(_ content: String) -> [Int] {
        let nsContent = content as NSString
        let matches = digitPattern.matches(
            in: content, options: [], range: NSRange(location: 0, length: nsContent.length))

        return matches.compactMap { match -> Int? in
            let str = nsContent.substring(with: match.range)
            return Int(str)
        }
    }

    private static func formatDate(components: [Int], format: String) -> String {
        var result: [String] = []

        switch format {
        case "mdy":
            if components.count >= 3 {
                guard components[0] >= 1 && components[0] <= 12 else { return "" }
                result.append(monthNames[components[0]])
                result.append(ordinalWord(for: components[1]))
                result.append(interpretYear(components[2]))
            }
        case "dmy":
            if components.count >= 3 {
                guard components[1] >= 1 && components[1] <= 12 else { return "" }
                result.append(ordinalWord(for: components[0]))
                result.append(monthNames[components[1]])
                result.append(interpretYear(components[2]))
            }
        case "ymd":
            if components.count >= 3 {
                guard components[1] >= 1 && components[1] <= 12 else { return "" }
                result.append(interpretYear(components[0]))
                result.append(monthNames[components[1]])
                result.append(ordinalWord(for: components[2]))
            }
        case "md":
            if components.count >= 2 {
                guard components[0] >= 1 && components[0] <= 12 else { return "" }
                result.append(monthNames[components[0]])
                result.append(ordinalWord(for: components[1]))
            }
        case "dm":
            if components.count >= 2 {
                guard components[1] >= 1 && components[1] <= 12 else { return "" }
                result.append(ordinalWord(for: components[0]))
                result.append(monthNames[components[1]])
            }
        case "y":
            if !components.isEmpty {
                result.append(interpretYear(components[0]))
            }
        case "m":
            if !components.isEmpty && components[0] >= 1 && components[0] <= 12 {
                result.append(monthNames[components[0]])
            }
        case "d":
            if !components.isEmpty {
                result.append(ordinalWord(for: components[0]))
            }
        default:
            // Default to mdy
            return formatDate(components: components, format: "mdy")
        }

        return result.joined(separator: " ")
    }

    private static func interpretYear(_ year: Int) -> String {
        // Handle 4-digit years specially
        if year >= 1000 && year <= 9999 {
            let century = year / 100
            let remainder = year % 100

            if remainder == 0 {
                // 2000 -> "two thousand", 1900 -> "nineteen hundred"
                if year == 2000 {
                    return "two thousand"
                }
                return interpretCardinal(String(century)) + " hundred"
            } else if year >= 2000 && year <= 2009 {
                // 2001-2009 -> "two thousand one" etc.
                return "two thousand " + interpretCardinal(String(remainder))
            } else if remainder >= 1 && remainder <= 9 {
                // 1905 -> "nineteen oh five", 2101 -> "twenty one oh one"
                let centuryPart = interpretCardinal(String(century))
                return "\(centuryPart) oh \(interpretCardinal(String(remainder)))"
            } else {
                // 1985 -> "nineteen eighty five", 2024 -> "twenty twenty four"
                let centuryPart = interpretCardinal(String(century))
                let remainderPart = interpretCardinal(String(remainder))
                return "\(centuryPart) \(remainderPart)"
            }
        }

        return interpretCardinal(String(year))
    }

    private static func interpretDuration(_ content: String) -> String {
        // Parse format like 1'21" (1 minute 21 seconds)
        var minutes = 0
        var seconds = 0
        let nsContent = content as NSString
        let range = NSRange(location: 0, length: nsContent.length)

        if let match = minutePattern.firstMatch(in: content, options: [], range: range) {
            minutes = Int(nsContent.substring(with: match.range(at: 1))) ?? 0
        }

        if let match = secondPattern.firstMatch(in: content, options: [], range: range) {
            seconds = Int(nsContent.substring(with: match.range(at: 1))) ?? 0
        }

        var parts: [String] = []
        if minutes > 0 {
            let minuteWord = minutes == 1 ? "minute" : "minutes"
            parts.append("\(interpretCardinal(String(minutes))) \(minuteWord)")
        }
        if seconds > 0 {
            let secondWord = seconds == 1 ? "second" : "seconds"
            parts.append("\(interpretCardinal(String(seconds))) \(secondWord)")
        }

        return parts.isEmpty ? content : parts.joined(separator: " ")
    }

    private static func interpretClockTime(_ content: String) -> String {
        let components = content.split(separator: ":").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard components.count >= 2 else { return content }

        let hours = components[0]
        let minutes = components[1]

        if minutes == 0 {
            return "\(interpretCardinal(String(hours))) o'clock"
        }

        // Single-digit minutes: "3:05" -> "three oh five"
        if minutes >= 1 && minutes <= 9 {
            return "\(interpretCardinal(String(hours))) oh \(interpretCardinal(String(minutes)))"
        }

        return "\(interpretCardinal(String(hours))) \(interpretCardinal(String(minutes)))"
    }

    private static func parseFractionPart(_ content: String) -> String? {
        let parts = content.split(separator: "/")
        guard parts.count == 2,
            let numerator = Int(parts[0].trimmingCharacters(in: .whitespaces)),
            let denominator = Int(parts[1].trimmingCharacters(in: .whitespaces)),
            denominator > 0
        else {
            return nil
        }

        return spellFraction(numerator: numerator, denominator: denominator)
    }

    private static func spellFraction(numerator: Int, denominator: Int) -> String {
        // Special cases
        if denominator == 2 {
            let halfWord = numerator == 1 ? "half" : "halves"
            if numerator == 1 {
                return "one \(halfWord)"
            }
            return "\(interpretCardinal(String(numerator))) \(halfWord)"
        }

        if denominator == 4 {
            let quarterWord = numerator == 1 ? "quarter" : "quarters"
            if numerator == 1 {
                return "one \(quarterWord)"
            }
            return "\(interpretCardinal(String(numerator))) \(quarterWord)"
        }

        // General case: use ordinal for denominator
        let numeratorWord = interpretCardinal(String(numerator))
        let denominatorOrdinal = ordinalWord(for: denominator)

        // Pluralize if needed
        let denominatorWord = numerator == 1 ? denominatorOrdinal : denominatorOrdinal + "s"

        return "\(numeratorWord) \(denominatorWord)"
    }
}
