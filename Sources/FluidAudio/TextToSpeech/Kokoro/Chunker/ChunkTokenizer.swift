import Foundation

enum ChunkTokenizer {
    struct ClauseUnit {
        let words: [String]
        let pause: Int
    }

    private struct Token {
        enum Kind {
            case word
            case punctuation
            case lineBreak
        }

        let text: String
        let kind: Kind
    }

    private enum Boundary {
        case none
        case clause
        case sentence
    }

    private enum Constants {
        static let sentenceTerminators: Set<String> = [
            ".", "!", "?", "?!", "!?", "...", "…",
            "。", "！", "？", "？！", "！？", "！！", "？？", "｡", "．",
        ]
        static let sentenceTerminatorCharacters: Set<Character> = [
            ".", "!", "?", "…", "。", "！", "？", "｡", "．",
        ]
        static let clauseDelimiters: Set<String> = [
            ",", ";", ":", "—", "–", "-", "、", "，", "；", "：", "､",
        ]
        static let clauseDelimiterCharacters: Set<Character> = [
            ",", ";", ":", "—", "–", "-", "、", "，", "；", "：", "､",
        ]
        static let parentheticalOpeners: Set<String> = ["(", "[", "{"]
        static let parentheticalClosers: Set<String> = [")", "]", "}"]
        static let quotePunctuation: Set<String> = ["\"", "'", "“", "”", "‘", "’", "«", "»", "‹", "›", "„", "‟"]
        static let abbreviationTokens: Set<String> = [
            "mr", "mrs", "ms", "dr", "prof", "ph", "sr", "jr", "st", "no", "vs", "etc", "fig", "al",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
            "col", "gen", "lt", "hon", "rev", "messrs", "dept", "est", "inc", "co", "corp",
        ]
        static let apostrophes: Set<Character> = ["'", "\u{2019}", "\u{2018}", "\u{201B}", "\u{2032}"]
        static let hyphenCharacters: Set<Character> = ["-", "\u{2010}", "\u{2011}"]
    }

    static func paragraphs(from text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let normalized =
            text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        guard let regex = try? NSRegularExpression(pattern: "(?:\\n\\s*){2,}") else {
            return [normalized]
        }

        var result: [String] = []
        var start = normalized.startIndex
        let searchRange = NSRange(normalized.startIndex..., in: normalized)
        let matches = regex.matches(in: normalized, range: searchRange)

        for match in matches {
            guard let range = Range(match.range, in: normalized) else { continue }
            let paragraph = normalized[start..<range.lowerBound]
            if !paragraph.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                result.append(String(paragraph))
            }
            start = range.upperBound
        }

        if start < normalized.endIndex {
            let tail = normalized[start..<normalized.endIndex]
            if !tail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                result.append(String(tail))
            }
        } else if result.isEmpty {
            result.append(normalized)
        }

        return result
    }

    static func clauseUnits(
        for paragraph: String,
        pauseSentence: Int,
        pauseClause: Int,
        pauseLineBreak: Int
    ) -> [ClauseUnit] {
        let tokens = tokenizeParagraph(paragraph)
        var units: [ClauseUnit] = []
        var currentWords: [String] = []
        var lastWord: String?

        func flushCurrent(pause: Int) {
            guard !currentWords.isEmpty else { return }
            units.append(ClauseUnit(words: currentWords, pause: pause))
            currentWords.removeAll(keepingCapacity: true)
        }

        for (index, token) in tokens.enumerated() {
            switch token.kind {
            case .word:
                currentWords.append(token.text)
                lastWord = token.text

            case .lineBreak:
                flushCurrent(pause: pauseLineBreak)
                lastWord = nil

            case .punctuation:
                let nextWord = nextWord(in: tokens, from: index + 1)
                let boundary = boundary(for: token.text, previousWord: lastWord, nextWord: nextWord)

                switch boundary {
                case .sentence:
                    flushCurrent(pause: pauseSentence)
                    lastWord = nil
                case .clause:
                    flushCurrent(pause: pauseClause)
                    lastWord = nil
                case .none:
                    if Constants.parentheticalOpeners.contains(token.text) {
                        flushCurrent(pause: pauseClause)
                        lastWord = nil
                    }
                }
            }
        }

        flushCurrent(pause: 0)
        return units
    }

    private static func tokenizeParagraph(_ paragraph: String) -> [Token] {
        guard !paragraph.isEmpty else { return [] }
        let normalized =
            paragraph
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let characters = Array(normalized)

        var tokens: [Token] = []
        var buffer = ""
        var index = 0

        func flushWord() {
            guard !buffer.isEmpty else { return }
            tokens.append(Token(text: buffer, kind: .word))
            buffer.removeAll(keepingCapacity: true)
        }

        while index < characters.count {
            let current = characters[index]
            let previous = index > 0 ? characters[index - 1] : nil
            let next = (index + 1) < characters.count ? characters[index + 1] : nil

            if current == "\n" {
                flushWord()
                tokens.append(Token(text: "\n", kind: .lineBreak))
                index += 1
                continue
            }

            if current.isWhitespace {
                flushWord()
                index += 1
                continue
            }

            if isWordCharacter(current, previous: previous, next: next) {
                buffer.append(normalizedWordCharacter(current))
                index += 1
                continue
            }

            flushWord()
            let (punct, nextIndex) = readPunctuation(from: characters, start: index)
            if !punct.isEmpty {
                tokens.append(Token(text: punct, kind: .punctuation))
            }
            index = nextIndex
        }

        flushWord()
        return tokens
    }

    private static func nextWord(in tokens: [Token], from startIndex: Int) -> String? {
        guard startIndex < tokens.count else { return nil }
        for index in startIndex..<tokens.count {
            let token = tokens[index]
            switch token.kind {
            case .word:
                return token.text
            case .lineBreak:
                return nil
            case .punctuation:
                continue
            }
        }
        return nil
    }

    private static func boundary(
        for punctuation: String,
        previousWord: String?,
        nextWord: String?
    ) -> Boundary {
        guard !punctuation.isEmpty else { return .none }

        if Constants.quotePunctuation.contains(punctuation) {
            return .none
        }

        if Constants.parentheticalClosers.contains(punctuation) {
            return .clause
        }

        if punctuation.allSatisfy({ $0 == "." }) {
            if let previousWord, isLikelyAbbreviation(previousWord, nextWord: nextWord) {
                return .none
            }
            return .sentence
        }

        if Constants.sentenceTerminators.contains(punctuation) {
            return .sentence
        }

        if punctuation.allSatisfy({ Constants.sentenceTerminatorCharacters.contains($0) }) {
            return .sentence
        }

        if punctuation.allSatisfy({ Constants.clauseDelimiterCharacters.contains($0) }) {
            return .clause
        }

        if Constants.clauseDelimiters.contains(punctuation) {
            return .clause
        }

        if Constants.parentheticalOpeners.contains(punctuation) {
            return .clause
        }

        return .none
    }

    private static func isLikelyAbbreviation(_ word: String, nextWord: String?) -> Bool {
        let trimmed = word.trimmingCharacters(in: .punctuationCharacters)
        guard !trimmed.isEmpty else { return false }

        let lower = trimmed.lowercased()
        if Constants.abbreviationTokens.contains(lower) {
            return true
        }

        if lower.hasSuffix("s"), Constants.abbreviationTokens.contains(String(lower.dropLast())) {
            return true
        }

        if trimmed.count == 1 {
            if trimmed == "I" || trimmed == "i" {
                return false
            }
            if trimmed.uppercased() == trimmed {
                if let nextWord {
                    return nextWord.uppercased() == nextWord
                }
                return true
            }
        }

        if trimmed.count <= 2 && trimmed.uppercased() == trimmed {
            if let nextWord {
                return nextWord.uppercased() == nextWord
            }
        }

        if trimmed.count <= 3,
            trimmed.uppercased() == trimmed,
            let first = nextWord?.first,
            first.isUppercase
        {
            return true
        }

        return false
    }

    private static func isWordCharacter(
        _ character: Character,
        previous: Character?,
        next: Character?
    ) -> Bool {
        if character.isLetter || character.isNumber {
            return true
        }

        if Constants.apostrophes.contains(character) {
            return (previous?.isLetterOrNumber ?? false) && (next?.isLetterOrNumber ?? false)
        }

        if Constants.hyphenCharacters.contains(character) {
            return (previous?.isLetterOrNumber ?? false) && (next?.isLetterOrNumber ?? false)
        }

        if character == "," {
            return (previous?.isNumber ?? false) && (next?.isNumber ?? false)
        }

        if character == "." {
            return (previous?.isNumber ?? false) && (next?.isNumber ?? false)
        }

        return false
    }

    private static func normalizedWordCharacter(_ character: Character) -> Character {
        if Constants.apostrophes.contains(character) {
            return "'"
        }
        if Constants.hyphenCharacters.contains(character) {
            return "-"
        }
        return character
    }

    private static func readPunctuation(from characters: [Character], start: Int) -> (String, Int) {
        var index = start
        guard index < characters.count else { return ("", start) }

        let first = characters[index]
        var token = String(first)
        index += 1

        func canGroup(_ candidate: Character) -> Bool {
            if first == "." {
                return candidate == "."
            }
            if first == "!" || first == "?" {
                return candidate == "!" || candidate == "?"
            }
            if first == "-" {
                return candidate == "-"
            }
            return false
        }

        while index < characters.count {
            let candidate = characters[index]
            if canGroup(candidate) {
                token.append(candidate)
                index += 1
            } else {
                break
            }
        }

        return (token, index)
    }
}

extension Character {
    fileprivate var isLetterOrNumber: Bool { isLetter || isNumber }
}
