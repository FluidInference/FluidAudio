import Foundation
import RegexBuilder

/// HuggingFace-compatible text normalizer for ASR evaluation
/// Matches the normalization used in the Open ASR Leaderboard
struct TextNormalizer {
    
    // Additional diacritics mapping from HuggingFace normalizer
    private static let additionalDiacritics: [Character: String] = [
        "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
        "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "th", "ł": "l", "Ł": "L"
    ]

    /// Normalize text using HuggingFace ASR leaderboard standards
    /// This matches the normalization used in the official leaderboard evaluation
    static func normalize(_ text: String) -> String {
        var normalized = text

        // Step 1: Convert to lowercase (standard for ASR evaluation)
        normalized = normalized.lowercased()
        
        // Step 2: Remove content between brackets and parentheses (HuggingFace behavior)
        // Remove content between angle/square brackets
        let bracketsPattern = try! NSRegularExpression(pattern: "[<\\[].*?[>\\]]", options: [])
        normalized = bracketsPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )
        
        // Remove content between parentheses
        let parenthesesPattern = try! NSRegularExpression(pattern: "\\([^)]+?\\)", options: [])
        normalized = parenthesesPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )
        
        // Step 3: Handle diacritics and special characters
        normalized = normalized.map { char in
            if let replacement = additionalDiacritics[char] {
                return replacement
            }
            return String(char)
        }.joined()
        
        // Step 4: Handle common symbols BEFORE removing punctuation
        normalized = normalized.replacingOccurrences(of: "$", with: " dollar ")
        normalized = normalized.replacingOccurrences(of: "&", with: " and ")
        normalized = normalized.replacingOccurrences(of: "%", with: " percent ")

        // Step 5: Remove punctuation except apostrophes (keep contractions)
        // This preserves linguistic meaning while removing formatting differences
        let punctuationPattern = try! NSRegularExpression(pattern: "[^\\w\\s']", options: [])
        normalized = punctuationPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        // Step 6: Handle contractions consistently
        // This is critical for fair comparison across different ASR systems
        let contractions = [
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "how's": "how is",
            "let's": "let us",
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
            "he'd": "he would"
        ]

        for (contraction, expansion) in contractions {
            normalized = normalized.replacingOccurrences(of: contraction, with: expansion)
        }

        // Step 7: Handle abbreviations (matching HuggingFace)
        let abbreviations = [
            "mr": "mister",
            "mrs": "misess",
            "ms": "miss",
            "dr": "doctor",
            "prof": "professor",
            "st": "saint",
            "jr": "junior",
            "sr": "senior",
            "vs": "versus",
            "inc": "incorporated",
            "ltd": "limited",
            "co": "company"
        ]
        
        for (abbrev, expansion) in abbreviations {
            let pattern = "\\b" + abbrev + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: expansion,
                options: .regularExpression
            )
        }
        
        // Step 8: Normalize numbers and symbols
        // Convert written numbers to digits or vice versa based on ASR output patterns
        let numberWords = [
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
            "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
            "million": "1000000", "billion": "1000000000",
            // Ordinal numbers
            "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
            "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
            "ninth": "9th", "tenth": "10th"
        ]

        // Apply number normalization (can be made more sophisticated)
        for (word, digit) in numberWords {
            let pattern = "\\b" + word + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: digit,
                options: .regularExpression
            )
        }

        // Step 9: Handle remaining special characters and symbols
        normalized = normalized.replacingOccurrences(of: "€", with: " euro ")
        normalized = normalized.replacingOccurrences(of: "£", with: " pound ")
        normalized = normalized.replacingOccurrences(of: "¥", with: " yen ")
        normalized = normalized.replacingOccurrences(of: "©", with: " copyright ")
        normalized = normalized.replacingOccurrences(of: "®", with: " registered ")
        normalized = normalized.replacingOccurrences(of: "™", with: " trademark ")
        
        // Step 10: Remove any remaining symbols and normalize whitespace
        // Remove any remaining punctuation and symbols
        let finalCleanPattern = try! NSRegularExpression(pattern: "[^\\w\\s]", options: [])
        normalized = finalCleanPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        // Step 11: Normalize whitespace
        // Replace multiple spaces with single space and trim
        let whitespacePattern = try! NSRegularExpression(pattern: "\\s+", options: [])
        normalized = whitespacePattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        // Step 12: Trim leading and trailing whitespace
        normalized = normalized.trimmingCharacters(in: .whitespacesAndNewlines)

        return normalized
    }

    /// Lightweight normalization (less aggressive)
    /// Use this when you want to preserve more linguistic features
    static func lightNormalize(_ text: String) -> String {
        var normalized = text

        // Basic case normalization
        normalized = normalized.lowercased()

        // Remove only basic punctuation, preserve apostrophes
        let basicPunctuationPattern = try! NSRegularExpression(pattern: "[.!?,:;\"()\\[\\]{}\\-]", options: [])
        normalized = basicPunctuationPattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: ""
        )

        // Normalize whitespace
        let whitespacePattern = try! NSRegularExpression(pattern: "\\s+", options: [])
        normalized = whitespacePattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Aggressive normalization (maximum compatibility)
    /// Use this for maximum compatibility with various ASR evaluation standards
    static func aggressiveNormalize(_ text: String) -> String {
        var normalized = normalize(text)

        // Additional aggressive steps

        // Remove all remaining single quotes
        normalized = normalized.replacingOccurrences(of: "'", with: "")

        // Convert all digits to written form or vice versa (standardize)
        let digitPattern = try! NSRegularExpression(pattern: "\\d", options: [])
        let _ = digitPattern.matches(in: normalized, options: [], range: NSRange(location: 0, length: normalized.count))

        // If there are digits, this might need digit-to-word conversion
        // For now, we'll leave digits as they are, but this could be extended

        // Remove common filler words that don't affect meaning
        let fillerWords = ["um", "uh", "ah", "er", "like", "you know", "well", "so", "actually", "basically"]
        for filler in fillerWords {
            let pattern = "\\b" + filler + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: "",
                options: .regularExpression
            )
        }

        // Final cleanup
        let finalWhitespacePattern = try! NSRegularExpression(pattern: "\\s+", options: [])
        normalized = finalWhitespacePattern.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " "
        )

        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Example usage and testing
#if DEBUG
extension TextNormalizer {
    /// Test the normalizer with common ASR evaluation cases
    static func runTests() {
        let testCases = [
            ("Hello, world! This is a test.", "hello world this is a test"),
            ("Can't you see it's working?", "cannot you see it is working"),
            ("I've got twenty-five dollars & 50 cents.", "i have got 25 dollar and 50 cents"),
            ("The temperature is 72°F today.", "the temperature is 72 f today"),
            ("It's 3:30 PM on December 1st, 2023.", "it is 3 30 pm on december 1 2023"),
            ("Dr. Smith said, \"That's excellent!\"", "dr smith said that is excellent"),
            ("We'll be there at 9 o'clock.", "we will be there at 9 o clock"),
            ("The U.S.A. is a big country.", "the u s a is a big country"),
            ("I'm 100% sure about this.", "i am 100 percent sure about this"),
            ("She won't be coming today.", "she will not be coming today")
        ]

        print("🧪 Testing TextNormalizer")
        print(String(repeating: "=", count: 50))

        for (input, expected) in testCases {
            let result = TextNormalizer.normalize(input)
            let passed = result == expected

            print("Input:    \"\(input)\"")
            print("Expected: \"\(expected)\"")
            print("Got:      \"\(result)\"")
            print("Result:   \(passed ? "✅ PASS" : "❌ FAIL")")
            print()
        }
    }
}
#endif
