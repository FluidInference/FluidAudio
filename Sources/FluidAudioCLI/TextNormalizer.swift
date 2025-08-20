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

    /// Normalize text using HuggingFace ASR leaderboard standards
    /// This matches the normalization used in the official leaderboard evaluation
    static func normalize(_ text: String) -> String {
        var normalized = text

        normalized = normalized.lowercased()

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
            "he'd": "he would",
        ]

        for (contraction, expansion) in contractions {
            normalized = normalized.replacingOccurrences(of: contraction, with: expansion)
        }

        // Normalize British to American spelling variations
        let spellingVariations = [
            "counselled": "counseled",
            "counselling": "counseling",
            "travelled": "traveled",
            "travelling": "traveling",
            "modelled": "modeled",
            "modelling": "modeling",
            "labelled": "labeled",
            "labelling": "labeling",
            "honour": "honor",
            "honours": "honors",
            "honoured": "honored",
            "honouring": "honoring",
            "colour": "color",
            "colours": "colors",
            "coloured": "colored",
            "colouring": "coloring",
            "favour": "favor",
            "favours": "favors",
            "favoured": "favored",
            "favouring": "favoring",
            "behaviour": "behavior",
            "behaviours": "behaviors",
            "neighbour": "neighbor",
            "neighbours": "neighbors",
            "labour": "labor",
            "labours": "labors",
            "laboured": "labored",
            "labouring": "laboring",
            "centre": "center",
            "centres": "centers",
            "centred": "centered",
            "theatre": "theater",
            "theatres": "theaters",
            "metre": "meter",
            "metres": "meters",
            "litre": "liter",
            "litres": "liters",
            "defence": "defense",
            "defences": "defenses",
            "offence": "offense",
            "offences": "offenses",
            "licence": "license",
            "licences": "licenses",
            "practise": "practice",
            "practising": "practicing",
            "practised": "practiced",
            "analyse": "analyze",
            "analyses": "analyzes",
            "analysed": "analyzed",
            "analysing": "analyzing",
            "paralyse": "paralyze",
            "paralysed": "paralyzed",
            "recognise": "recognize",
            "recognised": "recognized",
            "recognising": "recognizing",
            "organise": "organize",
            "organised": "organized",
            "organising": "organizing",
            "realise": "realize",
            "realised": "realized",
            "realising": "realizing",
            "apologise": "apologize",
            "apologised": "apologized",
            "apologising": "apologizing",
            "emphasise": "emphasize",
            "emphasised": "emphasized",
            "emphasising": "emphasizing",
            "specialise": "specialize",
            "specialised": "specialized",
            "specialising": "specializing",
            "summarise": "summarize",
            "summarised": "summarized",
            "summarising": "summarizing",
            "authorise": "authorize",
            "authorised": "authorized",
            "authorising": "authorizing",
            "maximise": "maximize",
            "maximised": "maximized",
            "maximising": "maximizing",
            "minimise": "minimize",
            "minimised": "minimized",
            "minimising": "minimizing",
            "optimise": "optimize",
            "optimised": "optimized",
            "optimising": "optimizing",
            "standardise": "standardize",
            "standardised": "standardized",
            "standardising": "standardizing",
            "categorise": "categorize",
            "categorised": "categorized",
            "categorising": "categorizing",
            "prioritise": "prioritize",
            "prioritised": "prioritized",
            "prioritising": "prioritizing",
            "memorise": "memorize",
            "memorised": "memorized",
            "memorising": "memorizing",
            "criticise": "criticize",
            "criticised": "criticized",
            "criticising": "criticizing",
            "publicise": "publicize",
            "publicised": "publicized",
            "publicising": "publicizing",
            "advertise": "advertize",
            "advertised": "advertized",
            "advertising": "advertizing",
            "grey": "gray",
            "greys": "grays",
            "greyed": "grayed",
            "greying": "graying",
        ]

        for (british, american) in spellingVariations {
            let pattern = "\\b" + british + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: american,
                options: .regularExpression
            )
        }

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
            "co": "company",
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
            "ninth": "9th", "tenth": "10th",
        ]

        for (word, digit) in numberWords {
            let pattern = "\\b" + word + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: digit,
                options: .regularExpression
            )
        }

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
