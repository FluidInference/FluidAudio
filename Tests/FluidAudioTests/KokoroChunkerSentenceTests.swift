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

    func testHyphenatedWordsPreserved() {
        let text = "The self-made inventor amazed everyone."
        let (chunks, expectedWords) = chunkText(text)

        let flattened = chunks.flatMap { $0.words }
        XCTAssertTrue(flattened.contains("self-made"), "Chunks should retain hyphenated words")
        XCTAssertTrue(expectedWords.contains("self-made"), "Expected word list should include hyphenated forms")
    }

    func testHyphenatedDictionaryLookup() {
        let wordToPhonemes: [String: [String]] = [
            "self-made": ["S1"],
            "inventor": ["S2"],
        ]
        let allowedTokens: Set<String> = ["S1", "S2", " "]

        let chunks = KokoroChunker.chunk(
            text: "self-made inventor",
            wordToPhonemes: wordToPhonemes,
            targetTokens: 128,
            hasLanguageToken: false,
            languageCode: "en-us",
            voiceIdentifier: nil,
            allowedPhonemeTokens: allowedTokens
        )

        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(
            chunks[0].phonemes, ["S1", " ", "S2"], "Hyphenated word should map through dictionary without fallback")
    }

    func testFrenchBasicSentences() {
        let text = "Bonjour le monde. Comment allez-vous? J'espère que vous allez bien."
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "French text should produce chunks")
        XCTAssertEqual(chunks.count, 1, "Short French text should merge into one chunk")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords, "Word sequence should be preserved")
        XCTAssertTrue(flattened.contains("Bonjour"))
        XCTAssertTrue(flattened.contains("allez-vous"))
        XCTAssertTrue(flattened.contains("J'espère"))

        let chunkTextValue = flattened.joined(separator: " ")
        XCTAssertLessThanOrEqual(chunkTextValue.count, 300)
    }

    func testFrenchWithAccents() {
        let text = "C'est très intéressant. Les élèves étudient français. Où êtes-vous né?"
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "French text with accents should produce chunks")
        XCTAssertLessThanOrEqual(chunks.count, 2, "Should merge into a small number of chunks")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords)
        XCTAssertTrue(flattened.contains("très"))
        XCTAssertTrue(flattened.contains("élèves"))
        XCTAssertTrue(flattened.contains("Où"))
    }

    func testSpanishBasicSentences() {
        let text = "Hola mundo. ¿Cómo estás? ¡Esto es increíble!"
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Spanish text should produce chunks")
        XCTAssertEqual(chunks.count, 1, "Short Spanish text should merge into one chunk")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords)
        XCTAssertTrue(flattened.contains("Hola"))
        XCTAssertTrue(flattened.contains("Cómo"))
        XCTAssertTrue(flattened.contains("increíble"))
    }

    func testSpanishInvertedPunctuation() {
        let text = "¿Hablas español? ¡Qué maravilloso! Me gusta mucho este idioma."
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Spanish text should produce chunks")
        XCTAssertEqual(chunks.count, 1, "Short Spanish text should merge into one chunk")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords)
        XCTAssertTrue(flattened.contains("Hablas"))
        XCTAssertTrue(flattened.contains("español"))
        XCTAssertTrue(flattened.contains("idioma"))
    }

    func testItalianBasicSentences() {
        let text = "Ciao mondo. Come stai? Spero che tu stia bene."
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Italian text should produce chunks")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords)
        XCTAssertTrue(flattened.contains("Ciao"))
        XCTAssertTrue(flattened.contains("Come"))
        XCTAssertTrue(flattened.contains("bene"))

        for (index, chunk) in chunks.enumerated() {
            let chunkString = chunk.words.joined(separator: " ")
            XCTAssertLessThanOrEqual(chunkString.count, 300, "Chunk \(index) should follow Latin limits")
        }
    }

    func testItalianWithApostrophes() {
        let text = "L'Italia è bella. Non c'è problema. Quest'anno andrò in vacanza."
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Italian text with apostrophes should produce chunks")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords)
        XCTAssertTrue(flattened.contains("L'Italia"))
        XCTAssertTrue(flattened.contains("c'è"))
        XCTAssertTrue(flattened.contains("Quest'anno"))
    }

    func testJapanesePunctuation() {
        let text = "こんにちは世界。元気ですか？これはテストです！"
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Japanese text should produce chunks")
        XCTAssertEqual(chunks.count, 1, "Short Japanese text should stay in one chunk")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords)
        XCTAssertTrue(flattened.contains("こんにちは世界"))
        XCTAssertTrue(flattened.contains("元気ですか"))
        XCTAssertTrue(flattened.contains("これはテストです"))
    }

    func testMandarinPunctuation() {
        let text = "你好世界。你今天好吗？这是一个测试！"
        let (chunks, expectedWords) = chunkText(text)

        XCTAssertFalse(chunks.isEmpty, "Mandarin text should produce chunks")
        XCTAssertEqual(chunks.count, 1, "Short Mandarin text should stay in one chunk")

        let flattened = chunks.flatMap { $0.words }
        XCTAssertEqual(flattened, expectedWords)
        XCTAssertTrue(flattened.contains("你好世界"))
        XCTAssertTrue(flattened.contains("你今天好吗"))
        XCTAssertTrue(flattened.contains("这是一个测试"))
    }

    func testCurrencyNormalizationMatchesTokenizer() {
        let text = "I paid $5.50 for lunch."
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("5 dollars and 50 cents"))
    }

    func testTimeNormalizationMatchesTokenizer() {
        let text = "The meeting is at 5:30."
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("5 30") || normalized.contains("5 oh 30"))
    }

    func testDecimalNormalizationMatchesTokenizer() {
        let text = "Pi is approximately 3.14."
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("3 point 1 4"))
    }

    func testDecimalInStressDirectiveNotExpanded() {
        let text = "[important](1.5) topic"
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("important"))
        XCTAssertFalse(normalized.contains("point"), "Stress directive decimals should remain untouched")
    }

    func testAliasReplacementUsesReplacementText() {
        let text = "[Dr.](Doctor) Smith"
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("Doctor Smith"))
        XCTAssertFalse(normalized.contains("Dr."))
    }

    func testAcronymAliasSpelling() {
        let text = "[NASA](N A S A) launched a rocket."
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("N A S A launched"))
    }

    func testDirectPhonemeDirectiveKeepsSurfaceForm() {
        let text = "[hello](/həˈloʊ/) world"
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("hello world"))
    }

    func testReplacementTextWithTimeNotAutoProcessed() {
        let text = "[5:30](half past five) reminder"
        let normalized = ChunkerTestSupport.normalize(text)
        XCTAssertTrue(normalized.contains("half past five"))
        XCTAssertFalse(normalized.contains("5 30"))
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
            languageCode: "en-us",
            voiceIdentifier: KokoroVoiceCatalog.defaultVoiceId,
            allowedPhonemeTokens: allowedTokens
        )

        return (chunks, expectedWords)
    }
}

private enum ChunkerTestSupport {
    static func normalize(_ text: String) -> String {
        ChunkPreprocessor.process(text)
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
            "。", "！", "？", "、", "，", "；", "：", "¿", "¡", "｡", "．",
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
            .union(CharacterSet(charactersIn: "'-"))
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
