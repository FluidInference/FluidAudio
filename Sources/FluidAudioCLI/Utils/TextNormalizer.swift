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

    /// Basic text normalizer that matches the reference Python implementation.
    ///
    /// When `spellOutLocale` is non-nil, an inverse text normalization (ITN)
    /// pass runs FIRST: every digit-run in the input is replaced by its
    /// spelled-out form for that locale (`NumberFormatter` `.spellOut`). This
    /// matches what NeMo / NVIDIA's leaderboard pipeline does for multilingual
    /// FLEURS scoring — non-English ASR models emit numbers as words, but
    /// FLEURS references keep digits, which manufactures large substitution
    /// gaps otherwise. Hyphens and soft hyphens produced by NumberFormatter
    /// (e.g. fr "soixante-seize", it/de soft-hyphen U+00AD between morphemes)
    /// are stripped here / by the downstream punct strip so the output
    /// tokenizes the same way as the model's words.
    static func basicNormalize(
        _ text: String,
        removeDiacritics: Bool = false,
        spellOutLocale: Locale? = nil
    ) -> String {
        var normalized = text
        if let locale = spellOutLocale {
            normalized = spellOutDigits(normalized, locale: locale)
        }
        normalized = normalized.lowercased()

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

        // Apply Unicode normalization. Matches Whisper BasicTextNormalizer:
        // - remove_diacritics=False: NFKC (precomposed), then replace M/S/P
        //   category chars with space (diacritics survive as precomposed
        //   letters in Ll/Lu).
        // - remove_diacritics=True:  NFKD (decomposed), then drop Mn chars
        //   and replace remaining M/S/P with space (combining marks now
        //   appear as separate Mn code points that get stripped).
        if removeDiacritics {
            normalized = normalized.decomposedStringWithCompatibilityMapping
            normalized = removeSymbolsAndDiacritics(normalized)
        } else {
            normalized = normalized.precomposedStringWithCompatibilityMapping
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

    /// Replace every digit-run in `text` with its spelled-out form for
    /// `locale`. Handles two surface forms:
    ///   1. Bare digit runs: `1976` → `mille neuf cent soixante-seize` (fr).
    ///   2. Thousands-separated groups: `30 000` / `1\u{00A0}000\u{00A0}000`
    ///      → `trente mille` / `un million` — required because FLEURS keeps
    ///      the locale's thousands separator and our basic punct strip would
    ///      otherwise treat the parts as independent numbers.
    /// Soft hyphens (`U+00AD`) inserted by NumberFormatter inside compounds
    /// (e.g. de "ein­tausend­neun­hundert…", it "otto­cento­due") are
    /// stripped. ASCII hyphens (fr "soixante-seize") are left for the
    /// downstream punct strip in `basicNormalize` to convert into spaces.
    ///
    /// The thousands-separator character class covers the Unicode space
    /// variants that FLEURS references actually contain across locales:
    ///   - U+0020 SPACE
    ///   - U+00A0 NO-BREAK SPACE
    ///   - U+2007 FIGURE SPACE
    ///   - U+2009 THIN SPACE                 (common in fr_fr "800 000")
    ///   - U+202F NARROW NO-BREAK SPACE      (typographic French / SI groups)
    static func spellOutDigits(_ text: String, locale: Locale) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .spellOut
        formatter.locale = locale

        // Prefer the thousands-grouped alternative (longest match first) so
        // "30 000" / "800\u{2009}000" parses as a single number, falling back
        // to a bare digit run for non-grouped occurrences.
        let pattern = #"\d+(?:[\u00A0\u2007\u2009\u202F ]\d{3})+|\d+"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text
        }

        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        if matches.isEmpty { return text }

        var result = ""
        var lastEnd = 0
        for m in matches {
            let r = m.range
            result += nsText.substring(with: NSRange(location: lastEnd, length: r.location - lastEnd))
            let matched = nsText.substring(with: r)
            let digitsOnly =
                matched
                .replacingOccurrences(of: " ", with: "")
                .replacingOccurrences(of: "\u{00A0}", with: "")
                .replacingOccurrences(of: "\u{2007}", with: "")
                .replacingOccurrences(of: "\u{2009}", with: "")
                .replacingOccurrences(of: "\u{202F}", with: "")
            if let n = Int(digitsOnly),
                let spelled = formatter.string(from: NSNumber(value: n))
            {
                result += spelled.replacingOccurrences(of: "\u{00AD}", with: "")
            } else {
                result += matched
            }
            lastEnd = r.location + r.length
        }
        result += nsText.substring(from: lastEnd)
        return result
    }

    /// Matches Whisper `remove_symbols_and_diacritics`: drop combining marks
    /// (Mn) entirely, fold the `ADDITIONAL_DIACRITICS` map (œ→oe, ß→ss, …),
    /// replace any remaining M/S/P category char with a single space.
    private static func removeSymbolsAndDiacritics(_ text: String) -> String {
        return text.compactMap { char in
            if let replacement = additionalDiacritics[char] {
                return replacement
            }
            guard let category = char.unicodeScalars.first?.properties.generalCategory else {
                return String(char)
            }
            // Drop non-spacing combining marks entirely (the diacritic itself).
            if category == .nonspacingMark {
                return ""
            }
            if Self.isSymbolPunctOrSpacingMark(category) {
                return " "
            }
            return String(char)
        }.joined()
    }

    /// Matches Whisper `remove_symbols` (the default `BasicTextNormalizer`
    /// path with `remove_diacritics=False`): NFKC has already been applied
    /// upstream; here we just replace every char whose Unicode category
    /// starts with M, S, or P (combining mark / symbol / punctuation) with
    /// a single space. Diacritics that survived NFKC composition are kept
    /// as part of the precomposed letter (Ll / Lu).
    private static func removeSymbols(_ text: String) -> String {
        return text.compactMap { char in
            guard let category = char.unicodeScalars.first?.properties.generalCategory else {
                return String(char)
            }
            if Self.isSymbolPunctOrMark(category) {
                return " "
            }
            return String(char)
        }.joined()
    }

    /// Mirror of Python's `unicodedata.category(c)[0] in "MSP"` — covers all
    /// Mark / Symbol / Punctuation subcategories of the Unicode standard.
    private static func isSymbolPunctOrMark(_ category: Unicode.GeneralCategory) -> Bool {
        switch category {
        case .nonspacingMark, .spacingMark, .enclosingMark,
            .currencySymbol, .mathSymbol, .modifierSymbol, .otherSymbol,
            .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
            .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }

    /// Same as `isSymbolPunctOrMark` but excludes `.nonspacingMark` because
    /// the diacritic-stripping path drops Mn entirely (returns "" not " ").
    private static func isSymbolPunctOrSpacingMark(_ category: Unicode.GeneralCategory) -> Bool {
        switch category {
        case .spacingMark, .enclosingMark,
            .currencySymbol, .mathSymbol, .modifierSymbol, .otherSymbol,
            .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
            .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }

    /// Normalize text using HuggingFace ASR leaderboard standards
    /// This matches the normalization used in the official leaderboard evaluation
    static func normalize(_ text: String) -> String {
        var normalized = text

        normalized = normalized.lowercased()

        // British to American normalization
        if britishToAmerican.isEmpty {
            print("WARNING: english.json failed to load or is empty!")
        }
        for (british, american) in britishToAmerican {
            let pattern = "\\b" + NSRegularExpression.escapedPattern(for: british) + "\\b"
            normalized = normalized.replacingOccurrences(
                of: pattern,
                with: american,
                options: .regularExpression
            )
        }

        // Abbreviations (Moved to top)
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

        // Remove stuttering patterns like "th-", "o-", "y-" (1-2 letter prefix
        // followed by a dash). Require trailing whitespace so we do not strip
        // hyphenated words ("Frazer-Nash", "well-known") or trailing-dash
        // tokens before alphanumerics (e.g. "Bose- Okay" with a space after
        // the dash is still a stutter; "Bose-Okay" without one would be a
        // compound and is left alone).
        let stutterPattern = try! NSRegularExpression(pattern: "\\b[a-z]{1,2}-\\s+", options: [])
        normalized = stutterPattern.stringByReplacingMatches(
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

        // Abbreviations moved to top

        let numberWords = [
            // English numbers
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
            "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
            "billion": "1000000000",
            "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
            "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
            "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
            "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
            "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
            "nineteenth": "19th", "twentieth": "20th", "thirtieth": "30th",
            "fortieth": "40th", "fiftieth": "50th", "sixtieth": "60th",
            "seventieth": "70th", "eightieth": "80th", "ninetieth": "90th",
            "hundredth": "100th", "thousandth": "1000th",

            // Italian numbers
            "uno": "1", "due": "2", "tre": "3", "quattro": "4", "cinque": "5",
            "sei": "6", "sette": "7", "otto": "8", "nove": "9", "dieci": "10",
            "undici": "11", "dodici": "12", "tredici": "13", "quattordici": "14",
            "quindici": "15", "sedici": "16", "diciassette": "17", "diciotto": "18",
            "diciannove": "19", "venti": "20", "trenta": "30", "quaranta": "40",
            "cinquanta": "50", "sessanta": "60", "settanta": "70", "ottanta": "80",
            "novanta": "90", "cento": "100", "mila": "1000", "milione": "1000000",
            "milioni": "1000000", "miliardo": "1000000000", "miliardi": "1000000000",

            // Italian ordinals
            "primo": "1st", "secondo": "2nd", "terzo": "3rd", "quarto": "4th",
            "quinto": "5th", "sesto": "6th", "settimo": "7th", "ottavo": "8th",
            "nono": "9th", "decimo": "10th", "undicesimo": "11th", "dodicesimo": "12th",
            "tredicesimo": "13th", "quattordicesimo": "14th", "quindicesimo": "15th",
            "ventesimo": "20th", "trentesimo": "30th", "centesimo": "100th",

            // French numbers
            "zéro": "0", "un": "1", "deux": "2", "trois": "3", "quatre": "4",
            "cinq": "5", "six": "6", "sept": "7", "huit": "8", "neuf": "9",
            "dix": "10", "onze": "11", "douze": "12", "treize": "13", "quatorze": "14",
            "quinze": "15", "seize": "16", "dix-sept": "17", "dix-huit": "18",
            "dix-neuf": "19", "vingt": "20", "trente": "30", "quarante": "40",
            "cinquante": "50", "soixante": "60", "soixante-dix": "70", "quatre-vingts": "80",
            "quatre-vingt-dix": "90", "cent": "100", "mille": "1000", "million": "1000000",
            "millions": "1000000", "milliard": "1000000000", "milliards": "1000000000",

            // French ordinals
            "premier": "1st", "première": "1st", "deuxième": "2nd", "troisième": "3rd",
            "quatrième": "4th", "cinquième": "5th", "sixième": "6th", "septième": "7th",
            "huitième": "8th", "neuvième": "9th", "dixième": "10th", "onzième": "11th",
            "douzième": "12th", "treizième": "13th", "quatorzième": "14th", "quinzième": "15th",
            "seizième": "16th", "vingtième": "20th", "trentième": "30th", "centième": "100th",
        ]

        // Advanced Number Conversion (e.g. "one hundred" -> "100")
        normalized = convertNumbers(normalized)

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

        // Merge consecutive digits logic removed due to regression

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

        // Remove symbols not preceded/followed by numbers
        let symbolCleanup1 = try! NSRegularExpression(pattern: "[.$¢€£]([^0-9])", options: [])
        normalized = symbolCleanup1.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: " $1"
        )

        let symbolCleanup2 = try! NSRegularExpression(pattern: "([^0-9])%", options: [])
        normalized = symbolCleanup2.stringByReplacingMatches(
            in: normalized,
            options: [],
            range: NSRange(location: 0, length: normalized.count),
            withTemplate: "$1 "
        )

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

    // MARK: - Advanced Number Parsing

    private static let numberValues: [String: Int] = [
        "zero": 0, "oh": 0,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    ]

    private static let multipliers: [String: Int] = [
        "hundred": 100,
        "thousand": 1000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
    ]

    private static func convertNumbers(_ text: String) -> String {
        let words = text.components(separatedBy: .whitespaces)
        var result: [String] = []
        var currentNumberWords: [String] = []

        for word in words {
            if isNumberWord(word) {
                currentNumberWords.append(word)
            } else {
                if !currentNumberWords.isEmpty {
                    result.append(parseNumberSequence(currentNumberWords))
                    currentNumberWords = []
                }
                result.append(word)
            }
        }

        if !currentNumberWords.isEmpty {
            result.append(parseNumberSequence(currentNumberWords))
        }

        return result.joined(separator: " ")
    }

    private static func isNumberWord(_ word: String) -> Bool {
        return numberValues.keys.contains(word) || multipliers.keys.contains(word)
    }

    private static func parseNumberSequence(_ words: [String]) -> String {
        var results: [String] = []
        var currentSum = 0
        var lastScale = 0

        for word in words {
            let val = numberValues[word] ?? multipliers[word] ?? 0
            let isMultiplier = multipliers.keys.contains(word)

            if isMultiplier {
                if currentSum == 0 {
                    currentSum = 1
                }
                currentSum *= val
                lastScale = val
            } else {
                if currentSum == 0 {
                    currentSum = val
                    lastScale = 1
                } else {
                    // Merge conditions:
                    // 1. Previous was a multiplier larger than current (e.g. "hundred" ... "five")
                    // 2. Previous was a ten (20-90) and current is unit (1-9)

                    var canMerge = false
                    if lastScale >= 100 && val < lastScale {
                        canMerge = true
                    } else if lastScale == 1 && (currentSum % 100 >= 20 && currentSum % 10 == 0) && val < 10 {
                        canMerge = true
                    }

                    if canMerge {
                        currentSum += val
                        lastScale = 1
                    } else {
                        results.append(String(currentSum))
                        currentSum = val
                        lastScale = 1
                    }
                }
            }
        }

        if currentSum > 0 {
            results.append(String(currentSum))
        }

        return results.joined(separator: " ")
    }

}
