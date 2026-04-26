import XCTest
@testable import FluidAudio

final class TokenLanguageFilterTests: XCTestCase {

    // MARK: - Script Property Tests

    func testLatinScriptLanguages() {
        let latinLanguages: [Language] = [
            .english, .spanish, .french, .german, .italian, .portuguese, .romanian,
            .polish, .czech, .slovak, .slovenian, .croatian, .bosnian,
        ]

        for language in latinLanguages {
            XCTAssertEqual(
                language.script, .latin,
                "\(language.rawValue) should use Latin script")
        }
    }

    func testCyrillicScriptLanguages() {
        let cyrillicLanguages: [Language] = [
            .russian, .ukrainian, .belarusian, .bulgarian, .serbian,
        ]

        for language in cyrillicLanguages {
            XCTAssertEqual(
                language.script, .cyrillic,
                "\(language.rawValue) should use Cyrillic script")
        }
    }

    // MARK: - Basic Script Matching Tests

    func testMatchesLatinText() {
        XCTAssertTrue(TokenLanguageFilter.matches("hello", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("world", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("Hello World!", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("123 abc", script: .latin))
    }

    func testMatchesCyrillicText() {
        XCTAssertTrue(TokenLanguageFilter.matches("привет", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("мир", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("Привет мир!", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("123 абв", script: .cyrillic))
    }

    func testDoesNotMatchMixedScripts() {
        XCTAssertFalse(TokenLanguageFilter.matches("hello мир", script: .latin))
        XCTAssertFalse(TokenLanguageFilter.matches("hello мир", script: .cyrillic))
        XCTAssertFalse(TokenLanguageFilter.matches("привет world", script: .latin))
        XCTAssertFalse(TokenLanguageFilter.matches("привет world", script: .cyrillic))
    }

    // MARK: - SentencePiece Boundary Marker Tests

    func testStripsSentencePieceBoundaryMarker() {
        // U+2581 (LOWER ONE EIGHTH BLOCK) is SentencePiece word boundary marker
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}hello", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}world", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}привет", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}мир", script: .cyrillic))
    }

    func testMultipleBoundaryMarkers() {
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}\u{2581}hello", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}\u{2581}привет", script: .cyrillic))
    }

    func testBoundaryMarkerOnly() {
        // A token that is *only* the SentencePiece boundary marker carries no
        // script signal, so we treat it as script-neutral (matches both).
        // filterTopK's argmax is then free to rank it alongside real candidates
        // rather than skipping it outright.
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}\u{2581}", script: .latin))
    }

    // MARK: - Polish Language Tests (Issue #512)

    func testPolishLatinCharacters() {
        // Polish uses Latin Extended-A for special characters
        XCTAssertTrue(TokenLanguageFilter.matches("ą", script: .latin))  // U+0105
        XCTAssertTrue(TokenLanguageFilter.matches("ć", script: .latin))  // U+0107
        XCTAssertTrue(TokenLanguageFilter.matches("ę", script: .latin))  // U+0119
        XCTAssertTrue(TokenLanguageFilter.matches("ł", script: .latin))  // U+0142
        XCTAssertTrue(TokenLanguageFilter.matches("ń", script: .latin))  // U+0144
        XCTAssertTrue(TokenLanguageFilter.matches("ó", script: .latin))  // U+00F3
        XCTAssertTrue(TokenLanguageFilter.matches("ś", script: .latin))  // U+015B
        XCTAssertTrue(TokenLanguageFilter.matches("ź", script: .latin))  // U+017A
        XCTAssertTrue(TokenLanguageFilter.matches("ż", script: .latin))  // U+017C
    }

    func testPolishWords() {
        XCTAssertTrue(TokenLanguageFilter.matches("cześć", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("świat", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("Polska", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("zażółć", script: .latin))
    }

    func testPolishWordsWithBoundaryMarker() {
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}cześć", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("\u{2581}świat", script: .latin))
    }

    func testRejectsPolishTextAsCyrillic() {
        XCTAssertFalse(TokenLanguageFilter.matches("cześć", script: .cyrillic))
        XCTAssertFalse(TokenLanguageFilter.matches("świat", script: .cyrillic))
    }

    // MARK: - Other Latin-Script Slavic Language Tests

    func testCzechDiacritics() {
        // Czech: á č ď é ě í ň ó ř š ť ú ů ý ž — all Latin Extended-A
        for token in ["Příliš", "žluťoučký", "kůň", "ďábelské", "ódy"] {
            XCTAssertTrue(
                TokenLanguageFilter.matches(token, script: .latin),
                "Czech token '\(token)' should match Latin")
        }
    }

    func testSlovakDiacritics() {
        // Slovak adds ĺ ľ ŕ ô — all Latin Extended-A
        for token in ["Kŕdeľ", "šťastných", "ďatľov", "mĺkveho", "ústí", "koňa"] {
            XCTAssertTrue(
                TokenLanguageFilter.matches(token, script: .latin),
                "Slovak token '\(token)' should match Latin")
        }
    }

    func testSlovenianAndCroatianDiacritics() {
        // Slovenian: č š ž
        for token in ["Češnja", "sočna"] {
            XCTAssertTrue(TokenLanguageFilter.matches(token, script: .latin))
        }
        // Croatian: č ć đ š ž
        for token in ["Džemper", "čokolada", "svježe"] {
            XCTAssertTrue(TokenLanguageFilter.matches(token, script: .latin))
        }
    }

    func testRomanianDiacritics() {
        // Romanian: ă â î ș ț — ș (U+0219) and ț (U+021B) are in Latin Extended-B
        for token in ["Înșelător", "mâine", "scrisoare", "ți-a"] {
            XCTAssertTrue(
                TokenLanguageFilter.matches(token, script: .latin),
                "Romanian token '\(token)' should match Latin")
        }
    }

    func testSlavicLatinLanguagesRejectedAsCyrillic() {
        // These should never be classified as Cyrillic
        XCTAssertFalse(TokenLanguageFilter.matches("Příliš", script: .cyrillic))
        XCTAssertFalse(TokenLanguageFilter.matches("žluťoučký", script: .cyrillic))
        XCTAssertFalse(TokenLanguageFilter.matches("Kŕdeľ", script: .cyrillic))
        XCTAssertFalse(TokenLanguageFilter.matches("Džemper", script: .cyrillic))
        XCTAssertFalse(TokenLanguageFilter.matches("Înșelător", script: .cyrillic))
    }

    // MARK: - Punctuation and Special Characters

    func testPunctuationWithLatin() {
        XCTAssertTrue(TokenLanguageFilter.matches("hello!", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("world?", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("test.", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("hello, world!", script: .latin))
    }

    func testPunctuationWithCyrillic() {
        XCTAssertTrue(TokenLanguageFilter.matches("привет!", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("мир?", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("тест.", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("привет, мир!", script: .cyrillic))
    }

    func testSpacesAndWhitespace() {
        XCTAssertTrue(TokenLanguageFilter.matches("hello world", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("  hello  ", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("привет мир", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("  привет  ", script: .cyrillic))
    }

    // MARK: - Edge Cases

    func testEmptyString() {
        // Mirrors testBoundaryMarkerOnly: no content means no script signal,
        // so we return true (script-neutral) for both Latin and Cyrillic.
        XCTAssertTrue(TokenLanguageFilter.matches("", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("", script: .cyrillic))
    }

    func testWhitespaceOnly() {
        XCTAssertTrue(TokenLanguageFilter.matches(" ", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("   ", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches(" ", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("   ", script: .cyrillic))
    }

    func testNumbers() {
        XCTAssertTrue(TokenLanguageFilter.matches("123", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("123", script: .cyrillic))
        XCTAssertTrue(TokenLanguageFilter.matches("456 789", script: .latin))
        XCTAssertTrue(TokenLanguageFilter.matches("456 789", script: .cyrillic))
    }

    // MARK: - Filter Top-K Tests

    func testFilterTopKReturnsHighestLogitMatch() {
        let topKIds = [1, 2, 3, 4]
        let topKLogits: [Float] = [0.9, 0.7, 0.5, 0.3]
        let vocabulary = [
            1: "привет",  // Cyrillic
            2: "hello",  // Latin
            3: "мир",  // Cyrillic
            4: "world",  // Latin
        ]

        // Highest-logit right-language (Latin) candidate is ID=2 ("hello") at logit 0.7
        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)
        if let probability = result?.probability {
            // Softmax over [0.9, 0.7, 0.5, 0.3]; probability of index 1 ≈ 0.2695
            XCTAssertGreaterThan(probability, 0.0)
            XCTAssertLessThan(probability, 1.0)
            XCTAssertEqual(probability, 0.2695, accuracy: 0.01)
        }
    }

    func testFilterTopKArgmaxOverUnsortedLogits() {
        // Regression test: do not assume CoreML returned top-K in sorted order.
        // Here the first Latin match (index 1, logit 0.4) is NOT the highest-logit
        // Latin candidate (index 3, logit 0.8). filterTopK must pick index 3.
        let topKIds = [1, 2, 3, 4]
        let topKLogits: [Float] = [0.9, 0.4, 0.1, 0.8]
        let vocabulary = [
            1: "привет",  // Cyrillic
            2: "hi",  // Latin (first match, but NOT highest logit)
            3: "мир",  // Cyrillic
            4: "world",  // Latin (highest-logit right-language candidate)
        ]

        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 4, "Must pick argmax-logit match, not first match")
    }

    func testFilterTopKWithSentencePieceBoundaryMarker() {
        let topKIds = [1, 2, 3]
        let topKLogits: [Float] = [0.9, 0.7, 0.5]
        let vocabulary = [
            1: "\u{2581}привет",  // Cyrillic with boundary marker
            2: "\u{2581}hello",  // Latin with boundary marker
            3: "\u{2581}мир",  // Cyrillic with boundary marker
        ]

        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)
        if let probability = result?.probability {
            // Softmax over [0.9, 0.7, 0.5]; probability of index 1 ≈ 0.329
            XCTAssertGreaterThan(probability, 0.0)
            XCTAssertLessThan(probability, 1.0)
            XCTAssertEqual(probability, 0.329, accuracy: 0.01)
        }
    }

    func testFilterTopKReturnsNilWhenNoMatch() {
        let topKIds = [1, 2, 3]
        let topKLogits: [Float] = [0.9, 0.7, 0.5]
        let vocabulary = [
            1: "привет",
            2: "мир",
            3: "тест",
        ]

        // All tokens are Cyrillic, should return nil for Latin
        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNil(result)
    }

    func testFilterTopKSkipsMissingVocabularyEntries() {
        let topKIds = [1, 2, 3, 4]
        let topKLogits: [Float] = [0.9, 0.7, 0.5, 0.3]
        let vocabulary = [
            1: "привет",
            // 2 is missing
            3: "мир",
            4: "world",  // Latin
        ]

        // Should skip missing ID=2 and return ID=4 ("world")
        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 4)
        if let probability = result?.probability {
            // Softmax over [0.9, 0.7, 0.5, 0.3]; probability of index 3 ≈ 0.1806
            XCTAssertGreaterThan(probability, 0.0)
            XCTAssertLessThan(probability, 1.0)
            XCTAssertEqual(probability, 0.1806, accuracy: 0.01)
        }
    }

    func testFilterTopKPicksNegativeInfinityLogit() {
        // Edge case: if the only right-language candidate has logit == -infinity,
        // the earlier `logit > bestLogit` check (bestLogit seeded at -infinity)
        // incorrectly returned nil. The "first match wins unconditionally"
        // branch ensures we pick it.
        let topKIds = [1, 2]
        let topKLogits: [Float] = [0.9, -.infinity]
        let vocabulary = [
            1: "привет",  // Cyrillic, not our script
            2: "hello",  // Latin, but -inf logit
        ]

        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)
    }

    func testFilterTopKEmptyArrays() {
        let result = TokenLanguageFilter.filterTopK(
            topKIds: [],
            topKLogits: [],
            vocabulary: [:],
            preferredScript: .latin
        )

        XCTAssertNil(result)
    }

    func testFilterTopKHandlesLengthMismatch() {
        // Safety guard: if the two top-K arrays disagree on length, the function
        // should only iterate over the common prefix without crashing.
        let topKIds = [1, 2, 3]
        let topKLogits: [Float] = [0.9, 0.7]  // shorter than topKIds
        let vocabulary = [
            1: "привет",  // Cyrillic
            2: "hello",  // Latin
            3: "world",  // Latin (but out of logits range)
        ]

        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        // Should find ID=2 ("hello") within the common prefix of length 2.
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)
    }

    func testFilterTopKProbabilityInValidRange() {
        // The returned probability must always be a valid probability in [0, 1].
        let topKIds = [1, 2, 3, 4, 5]
        let topKLogits: [Float] = [10.0, 5.0, 2.0, 1.0, 0.0]
        let vocabulary = [
            1: "привет",
            2: "hello",
            3: "world",
            4: "test",
            5: "foo",
        ]

        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        if let probability = result?.probability {
            XCTAssertGreaterThanOrEqual(probability, 0.0)
            XCTAssertLessThanOrEqual(probability, 1.0)
        }
    }

    func testFilterTopKPolishScenario() {
        // Real-world scenario from issue #512
        let topKIds = [1, 2, 3]
        let topKLogits: [Float] = [0.9, 0.6, 0.4]
        let vocabulary = [
            1: "\u{2581}при",  // Cyrillic (top-1, wrong script)
            2: "\u{2581}prz",  // Polish/Latin (top-2, correct script)
            3: "\u{2581}прі",  // Cyrillic
        ]

        let result = TokenLanguageFilter.filterTopK(
            topKIds: topKIds,
            topKLogits: topKLogits,
            vocabulary: vocabulary,
            preferredScript: .latin
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tokenId, 2)  // Should select Polish token
        if let probability = result?.probability {
            // Softmax over [0.9, 0.6, 0.4]; probability of index 1 ≈ 0.3156
            XCTAssertGreaterThan(probability, 0.0)
            XCTAssertLessThan(probability, 1.0)
            XCTAssertEqual(probability, 0.3156, accuracy: 0.01)
        }
    }

    // MARK: - Language Enum Tests

    func testAllLanguagesHaveScript() {
        // Ensure all languages have a defined script
        for language in Language.allCases {
            let script = language.script
            XCTAssertTrue(
                script == .latin || script == .cyrillic,
                "\(language.rawValue) must have a valid script")
        }
    }

    func testLanguageRawValues() {
        XCTAssertEqual(Language.english.rawValue, "en")
        XCTAssertEqual(Language.polish.rawValue, "pl")
        XCTAssertEqual(Language.russian.rawValue, "ru")
        XCTAssertEqual(Language.ukrainian.rawValue, "uk")
        // Newly-supported Latin-script Slavic + Romanian
        XCTAssertEqual(Language.czech.rawValue, "cs")
        XCTAssertEqual(Language.slovak.rawValue, "sk")
        XCTAssertEqual(Language.slovenian.rawValue, "sl")
        XCTAssertEqual(Language.croatian.rawValue, "hr")
        XCTAssertEqual(Language.bosnian.rawValue, "bs")
        XCTAssertEqual(Language.romanian.rawValue, "ro")
    }

    // MARK: - Unicode Range Tests

    func testLatinExtendedARange() {
        // Test characters in Latin Extended-A (U+0100 to U+017F)
        XCTAssertTrue(TokenLanguageFilter.matches("Ā", script: .latin))  // U+0100
        XCTAssertTrue(TokenLanguageFilter.matches("ž", script: .latin))  // U+017E
        XCTAssertTrue(TokenLanguageFilter.matches("ſ", script: .latin))  // U+017F
    }

    func testLatinExtendedBRange() {
        // Test characters in Latin Extended-B (U+0180 to U+024F)
        XCTAssertTrue(TokenLanguageFilter.matches("ƀ", script: .latin))  // U+0180
        XCTAssertTrue(TokenLanguageFilter.matches("ș", script: .latin))  // U+0219 (Romanian)
        XCTAssertTrue(TokenLanguageFilter.matches("ț", script: .latin))  // U+021B (Romanian)
        XCTAssertTrue(TokenLanguageFilter.matches("ɏ", script: .latin))  // U+024F
    }

    func testLatinExtendedAdditionalRange() {
        // Test characters in Latin Extended Additional (U+1E00 to U+1EFF)
        XCTAssertTrue(TokenLanguageFilter.matches("Ḁ", script: .latin))  // U+1E00
        XCTAssertTrue(TokenLanguageFilter.matches("ế", script: .latin))  // U+1EBF (Vietnamese)
    }

    func testCombiningDiacriticsRange() {
        // Combining Diacritical Marks (U+0300–U+036F) accepted as Latin so
        // decomposed (NFD) sequences like "e" + U+0301 don't get rejected.
        XCTAssertTrue(TokenLanguageFilter.matches("e\u{0301}", script: .latin))  // "e" + combining acute
        XCTAssertTrue(TokenLanguageFilter.matches("a\u{0300}", script: .latin))  // "a" + combining grave
        XCTAssertTrue(TokenLanguageFilter.matches("n\u{0303}", script: .latin))  // "n" + combining tilde (Spanish ñ NFD)
        // And Cyrillic must still reject bare ASCII letter + combining mark.
        XCTAssertFalse(TokenLanguageFilter.matches("e\u{0301}", script: .cyrillic))
    }

    func testCyrillicRange() {
        // Test characters in Cyrillic (U+0400 to U+04FF)
        XCTAssertTrue(TokenLanguageFilter.matches("Ѐ", script: .cyrillic))  // U+0400
        XCTAssertTrue(TokenLanguageFilter.matches("ӿ", script: .cyrillic))  // U+04FF
        XCTAssertTrue(TokenLanguageFilter.matches("Ӏ", script: .cyrillic))  // U+04C0
    }
}
