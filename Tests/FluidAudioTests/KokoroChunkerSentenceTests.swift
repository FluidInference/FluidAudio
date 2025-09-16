import Foundation
import XCTest

@testable import FluidAudio

final class KokoroChunkerSentenceTests: XCTestCase {
    func testEnglishBasicSentences() {
        let text = "Hello world. This is a test. How are you?"
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Should produce at least one chunk")
        XCTAssertEqual(chunks.count, 1, "Short text should merge into one chunk")
        XCTAssertEqual(chunks[0].words, expectedWords, "Chunk should retain word order")

        XCTAssertFalse(chunks[0].words.isEmpty, "Chunk should contain words")
    }

    func testEnglishWithAbbreviations() {
        let text =
            "Dr. Smith went to the U.S.A. yesterday. He met Prof. Johnson at 3:30 p.m. They discussed the project."
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Should produce at least one chunk")
        XCTAssertLessThanOrEqual(chunks.count, 2, "Should merge into 1-2 chunks for optimal TTS")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords, "Token sequence should be preserved")

        XCTAssertTrue(flattened.containsSubsequence(["Dr", "Smith"]))
        XCTAssertTrue(flattened.containsSubsequence(["U", "S", "A"]))
        XCTAssertTrue(flattened.containsSubsequence(["Prof", "Johnson"]))
        XCTAssertTrue(flattened.containsSubsequence(["p", "m"]))

        for chunk in chunks {
            let chunkText = chunk.words.joined(separator: " ")
            XCTAssertLessThanOrEqual(chunkText.count, 300, "Latin chunks should be ≤300 chars")
            XCTAssertGreaterThan(chunkText.trimmingCharacters(in: .whitespacesAndNewlines).count, 0)
        }
    }

    func testEnglishLongText() {
        let text =
            "This is a very long sentence that should test the chunking mechanism. "
            + "It contains multiple clauses and should be split appropriately for TTS processing. "
            + "The tokenizer should optimize chunk sizes for better speech synthesis. "
            + "Each chunk should be within the optimal length range for natural speech flow."
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Long text should produce chunks")
        XCTAssertGreaterThanOrEqual(chunks.count, 1, "Should produce at least one chunk")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords, "All words should be preserved across chunks")

        for (index, chunk) in chunks.enumerated() {
            let chunkString = chunk.words.joined(separator: " ")
            XCTAssertLessThanOrEqual(chunkString.count, 300, "Chunk \(index) should be ≤300 chars")
            XCTAssertGreaterThan(
                chunkString.trimmingCharacters(in: .whitespacesAndNewlines).count, 0, "Chunk should not be empty")
        }
    }

    func testEnglishQuestionAndExclamation() {
        let text = "What's your name? I'm excited! This is amazing."
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Should produce at least one chunk")
        XCTAssertEqual(chunks.count, 1, "Short text with strong punctuation should merge into one chunk")
        XCTAssertEqual(chunks[0].words, expectedWords)

        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("?"))
        XCTAssertTrue(normalized.contains("!"))
    }

    // MARK: - Helpers

    private func chunkText(_ text: String) -> ([TextChunk], [String]) {
        let normalized = ChunkerTestSupport.normalize(text)
        let expectedWords = ChunkerTestSupport.tokenizeWords(in: normalized)
        let (lexicon, allowedTokens) = ChunkerTestSupport.lexicon(for: expectedWords)

        let chunks = KokoroChunker.chunk(
            text: text,
            wordToPhonemes: lexicon,
            targetTokens: 512,
            hasLanguageToken: false,
            allowedPhonemeTokens: allowedTokens
        )

        return (chunks, expectedWords)
    }
}

private enum ChunkerTestSupport {
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

    static func normalize(_ text: String) -> String {
        var processed = text
        processed = removeCommas(from: processed)
        processed = rangeRegex.stringByReplacingMatches(
            in: processed,
            range: NSRange(processed.startIndex..., in: processed),
            withTemplate: "$1 to $2"
        )
        processed = flipMoney(processed)
        processed = splitTimes(processed)
        processed = spellDecimals(processed)
        return processed
    }

    static func tokenizeWords(in text: String) -> [String] {
        let normalized =
            text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let characters = Array(normalized)

        var words: [String] = []
        var buffer = ""
        var index = 0

        func flush() {
            guard !buffer.isEmpty else { return }
            words.append(buffer)
            buffer.removeAll(keepingCapacity: true)
        }

        while index < characters.count {
            let current = characters[index]
            let previous = index > 0 ? characters[index - 1] : nil
            let next = (index + 1) < characters.count ? characters[index + 1] : nil

            if current == "\n" || current.isWhitespace {
                flush()
                index += 1
                continue
            }

            if isWordCharacter(current, previous: previous, next: next) {
                buffer.append(normalizedWordCharacter(current))
                index += 1
                continue
            }

            flush()
            index += 1
        }

        flush()
        return words
    }

    static func lexicon(for words: [String]) -> ([String: [String]], Set<String>) {
        var dictionary: [String: [String]] = [:]
        var allowedTokens: Set<String> = [
            " ", ".", ",", "!", "?", "\"", "'", "…", ":", ";", "-", "—", "–",
        ]

        for word in words {
            let key = normalizeWordKey(word)
            guard !key.isEmpty else { continue }
            if dictionary[key] == nil {
                let token = "tok_\(dictionary.count)"
                dictionary[key] = [token]
                allowedTokens.insert(token)
            }
        }

        return (dictionary, allowedTokens)
    }

    private static func normalizeWordKey(_ word: String) -> String {
        let lowered = word.lowercased()
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{201B}", with: "'")
        let allowedSet = CharacterSet.letters
            .union(.decimalDigits)
            .union(CharacterSet(charactersIn: "'"))
        return String(lowered.unicodeScalars.filter { allowedSet.contains($0) })
    }

    private static func isWordCharacter(
        _ character: Character,
        previous: Character?,
        next: Character?
    ) -> Bool {
        if character.isLetter || character.isNumber {
            return true
        }

        if ["'", "\u{2019}", "\u{2018}", "\u{201B}", "\u{2032}"].contains(character) {
            let prevAlnum = previous.map { $0.isLetter || $0.isNumber } ?? false
            let nextAlnum = next.map { $0.isLetter || $0.isNumber } ?? false
            return prevAlnum && nextAlnum
        }

        if ["-", "\u{2010}", "\u{2011}"].contains(character) {
            let prevAlnum = previous.map { $0.isLetter || $0.isNumber } ?? false
            let nextAlnum = next.map { $0.isLetter || $0.isNumber } ?? false
            return prevAlnum && nextAlnum
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
        if ["'", "\u{2019}", "\u{2018}", "\u{201B}", "\u{2032}"].contains(character) {
            return "'"
        }
        if ["-", "\u{2010}", "\u{2011}"].contains(character) {
            return "-"
        }
        return character
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

    private static let linkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]+)\]\(([^\)]*)\)"#
    )
}

extension Array where Element == String {
    fileprivate func containsSubsequence(_ sequence: [String]) -> Bool {
        guard !sequence.isEmpty, sequence.count <= count else { return false }
        for start in 0...(count - sequence.count) {
            if Array(self[start..<(start + sequence.count)]) == sequence {
                return true
            }
        }
        return false
    }
}
