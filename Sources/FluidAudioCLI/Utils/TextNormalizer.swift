import Foundation
import RegexBuilder

/// HuggingFace-compatible text normalizer for ASR evaluation
/// Matches the normalization used in the Open ASR Leaderboard
struct TextNormalizer {

    private static let additionalDiacritics: [Character: String] = [
        "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
        "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
    ]

    private static let britishToAmerican: [String: String] = {
        guard let url = Bundle.module.url(forResource: "english", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let dictionary = try? JSONSerialization.jsonObject(with: data) as? [String: String]
        else {
            return [:]
        }
        return dictionary
    }()

    /// Basic text normalizer that matches the reference Python implementation
    static func basicNormalize(_ text: String, removeDiacritics: Bool = false) -> String {
        var normalized = text.lowercased()

        // Remove words between brackets
        let bracketsPattern = try! NSRegularExpression(pattern: "[<\\[].*?[>\\]]", options: [])
        normalized = bracketsPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )

        // Remove words between parentheses
        let parenthesesPattern = try! NSRegularExpression(pattern: "\\([^)]+?\\)", options: [])
        normalized = parenthesesPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )

        // Apply NFKD normalization and handle diacritics/symbols
        normalized = normalized.precomposedStringWithCompatibilityMapping

        if removeDiacritics {
            normalized = removeSymbolsAndDiacritics(normalized)
        } else {
            normalized = removeSymbols(normalized)
        }

        // Replace successive whitespace with single space
        let whitespacePattern = try! NSRegularExpression(pattern: "\\s+", options: [])
        normalized = whitespacePattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func removeSymbolsAndDiacritics(_ text: String) -> String {
        return text.compactMap { char in
            if let replacement = additionalDiacritics[char] {
                return replacement
            }

            let category = char.unicodeScalars.first?.properties.generalCategory

            // Remove combining marks (diacritics)
            if category == .nonspacingMark {
                return ""
            }

            // Replace symbols, punctuation, separators with space
            if let cat = category,
                ["symbol", "punctuation", "separator"].contains(String(describing: cat).prefix(1).lowercased())
            {
                return " "
            }

            return String(char)
        }.joined()
    }

    private static func removeSymbols(_ text: String) -> String {
        return text.compactMap { char in
            let category = char.unicodeScalars.first?.properties.generalCategory

            // Replace symbols, punctuation, separators with space, but keep diacritics
            if let cat = category,
                ["symbol", "punctuation", "separator"].contains(String(describing: cat).prefix(1).lowercased())
            {
                return " "
            }

            return String(char)
        }.joined()
    }

    /// Normalize text using HuggingFace ASR leaderboard standards
    /// This matches the normalization used in the official leaderboard evaluation
    static func normalize(_ text: String) -> String {
        var normalized = text

        normalized = normalized.lowercased()

        // British to American normalization
        for (british, american) in britishToAmerican {
            let pattern = "\\b" + NSRegularExpression.escapedPattern(for: british) + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: american,
                options: .regularExpression
            )
        }

        let bracketsPattern = try! NSRegularExpression(pattern: "[<\\[].*?[>\\]]", options: [])
        normalized = bracketsPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )

        let parenthesesPattern = try! NSRegularExpression(pattern: "\\([^)]+?\\)", options: [])
        normalized = parenthesesPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )

        // Remove filler words and interjections
        let fillerPattern = try! NSRegularExpression(pattern: "\\b(hmm|mm|mhm|mmm|uh|um)\\b", options: [])
        normalized = fillerPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )

        // Standardize spaces before apostrophes
        normalized = normalized.replacingOccurrences(of: " '", with: "'")
        
        // Handle "and a half" → "point five" 
        normalized = normalized.replacingOccurrences(of: " and a half", with: " point five")
        
        // Add spaces at number/letter boundaries
        let numberLetterPattern1 = try! NSRegularExpression(pattern: "([a-z])([0-9])", options: [])
        normalized = numberLetterPattern1.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: "$1 $2"
        )
        
        let numberLetterPattern2 = try! NSRegularExpression(pattern: "([0-9])([a-z])", options: [])
        normalized = numberLetterPattern2.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: "$1 $2"
        )
        
        // Remove spaces before suffixes
        let suffixPattern = try! NSRegularExpression(pattern: "([0-9])\\s+(st|nd|rd|th|s)\\b", options: [])
        normalized = suffixPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: "$1$2"
        )

        normalized = normalized.map { char in
            if let replacement = additionalDiacritics[char] {
                return replacement
            }
            return String(char)
        }.joined()

        normalized = normalized.replacingOccurrences(of: "$", with: " dollar ")
        normalized = normalized.replacingOccurrences(of: "&", with: " and ")
        normalized = normalized.replacingOccurrences(of: "%", with: " percent ")

        let punctuationPattern = try! NSRegularExpression(pattern: "[^\\w\\s']", options: [])
        normalized = punctuationPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        let contractions = [
            // Basic contractions
            "can't": "can not",
            "won't": "will not",
            "ain't": "aint",
            "let's": "let us",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'t": " not",
            "'s": " is",

            // Informal contractions
            "y'all": "you all",
            "wanna": "want to",
            "gonna": "going to",
            "gotta": "got to",
            "i'ma": "i am going to",
            "imma": "i am going to",
            "woulda": "would have",
            "coulda": "could have",
            "shoulda": "should have",
            "ma'am": "madam",

            // Perfect tenses
            "'d been": " had been",
            "'s been": " has been",
            "'d gone": " had gone",
            "'s gone": " has gone",
            "'d done": " had done",
            "'s got": " has got",

            // Specific contractions
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "how's": "how is",
            "i'm": "i am",
            "you're": "you are",
            "we're": "we are",
            "they're": "they are",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i've": "i have",
            "you'll": "you will",
            "we'll": "we will",
            "they'll": "they will",
            "i'll": "i will",
            "you'd": "you would",
            "we'd": "we would",
            "they'd": "they would",
            "i'd": "i would",
            "she's": "she is",
            "he's": "he is",
            "she'll": "she will",
            "he'll": "he will",
            "she'd": "she would",
            "he'd": "he would",
        ]

        for (contraction, expansion) in contractions {
            normalized = normalized.replacingOccurrences(of: contraction, with: expansion)
        }

        let abbreviations = [
            // Titles and names
            "mr": "mister",
            "mrs": "missus",
            "ms": "miss",
            "dr": "doctor",
            "prof": "professor",
            "st": "saint",
            "jr": "junior",
            "sr": "senior",
            "esq": "esquire",

            // Government and military titles
            "capt": "captain",
            "gov": "governor",
            "ald": "alderman",
            "gen": "general",
            "sen": "senator",
            "rep": "representative",
            "pres": "president",
            "rev": "reverend",
            "hon": "honorable",
            "asst": "assistant",
            "assoc": "associate",
            "lt": "lieutenant",
            "col": "colonel",

            // Business and other
            "vs": "versus",
            "inc": "incorporated",
            "ltd": "limited",
            "co": "company",

            // Time and date abbreviations
            "am": "a m",
            "pm": "p m",
            "ad": "ad",
            "bc": "bc",
        ]

        for (abbrev, expansion) in abbreviations {
            let pattern = "\\b" + abbrev + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: expansion,
                options: .regularExpression
            )
        }

        let numberWords = [
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
            "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
            "million": "1000000", "billion": "1000000000",
            "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
            "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
            "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
            "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
            "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
            "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th",
            "fortieth": "40th", "fiftieth": "50th", "sixtieth": "60th",
            "seventieth": "70th", "eightieth": "80th", "ninetieth": "90th",
            "hundredth": "100th", "thousandth": "1000th",
        ]

        for (word, digit) in numberWords {
            let pattern = "\\b" + word + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: digit,
                options: .regularExpression
            )
        }

        // Remove commas between digits
        let commaPattern = try! NSRegularExpression(pattern: "(\\d),(\\d)", options: [])
        normalized = commaPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: "$1$2"
        )

        // Remove periods not followed by numbers
        let periodPattern = try! NSRegularExpression(pattern: "\\.([^0-9]|$)", options: [])
        normalized = periodPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " $1"
        )

        // Normalize "a d" back to "ad" (handles A.D. -> a d -> ad)
        normalized = normalized.replacingOccurrences(of: "a d", with: "ad")

        // Handle time formats: "11 35 pm" -> "11 35 p m"
        let timePattern = try! NSRegularExpression(pattern: "\\b(\\d{1,2})\\s+(\\d{2})\\s+(am|pm)\\b", options: [])
        normalized = timePattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: "$1 $2 $3"
        )

        normalized = normalized.replacingOccurrences(of: "€", with: " euro ")
        normalized = normalized.replacingOccurrences(of: "£", with: " pound ")
        normalized = normalized.replacingOccurrences(of: "¥", with: " yen ")
        normalized = normalized.replacingOccurrences(of: "©", with: " copyright ")
        normalized = normalized.replacingOccurrences(of: "®", with: " registered ")
        normalized = normalized.replacingOccurrences(of: "™", with: " trademark ")

        let finalCleanPattern = try! NSRegularExpression(pattern: "[^\\w\\s]", options: [])
        normalized = finalCleanPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        let whitespacePattern = try! NSRegularExpression(pattern: "\\s+", options: [])
        normalized = whitespacePattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        normalized = normalized.trimmingCharacters(in: .whitespacesAndNewlines)

        return normalized
    }

}
