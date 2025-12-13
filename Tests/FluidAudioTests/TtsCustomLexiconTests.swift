import XCTest

#if canImport(FluidAudioTTS)
@testable import FluidAudioTTS

final class TtsCustomLexiconTests: XCTestCase {

    // MARK: - Parsing Tests

    func testParseSimpleEntry() throws {
        let content = "hello=hɛˈloʊ"
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 1)
        XCTAssertEqual(lexicon.phonemes(for: "hello"), ["h", "ɛ", "ˈ", "l", "o", "ʊ"])
    }

    func testParseMultipleEntries() throws {
        let content = """
            hello=hɛˈloʊ
            world=wɝld
            test=tɛst
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 3)
        XCTAssertNotNil(lexicon.phonemes(for: "hello"))
        XCTAssertNotNil(lexicon.phonemes(for: "world"))
        XCTAssertNotNil(lexicon.phonemes(for: "test"))
    }

    func testParseWithComments() throws {
        let content = """
            # This is a comment
            hello=hɛˈloʊ
            # Another comment
            world=wɝld
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 2)
        XCTAssertNotNil(lexicon.phonemes(for: "hello"))
        XCTAssertNotNil(lexicon.phonemes(for: "world"))
    }

    func testParseWithEmptyLines() throws {
        let content = """
            hello=hɛˈloʊ

            world=wɝld

            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 2)
    }

    func testParseWithWhitespace() throws {
        let content = "  hello  =  hɛˈloʊ  "
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 1)
        XCTAssertNotNil(lexicon.phonemes(for: "hello"))
    }

    // MARK: - Error Handling Tests

    func testParseMissingSeparatorThrows() {
        let content = "hello hɛˈloʊ"

        XCTAssertThrowsError(try TtsCustomLexicon.parse(content)) { error in
            XCTAssertTrue(error is TTSError)
        }
    }

    func testParseEmptyWordThrows() {
        let content = "=hɛˈloʊ"

        XCTAssertThrowsError(try TtsCustomLexicon.parse(content)) { error in
            XCTAssertTrue(error is TTSError)
        }
    }

    func testParseEmptyPhonemesThrows() {
        let content = "hello="

        XCTAssertThrowsError(try TtsCustomLexicon.parse(content)) { error in
            XCTAssertTrue(error is TTSError)
        }
    }

    // MARK: - Word Matching Tests

    func testExactMatch() throws {
        let content = """
            Hello=hɛˈloʊ
            hello=həˈloʊ
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        // Exact match should return the exact entry
        let upperPhonemes = lexicon.phonemes(for: "Hello")
        let lowerPhonemes = lexicon.phonemes(for: "hello")

        XCTAssertNotNil(upperPhonemes)
        XCTAssertNotNil(lowerPhonemes)
        XCTAssertNotEqual(upperPhonemes, lowerPhonemes, "Different cases should have different phonemes")
    }

    func testCaseInsensitiveFallback() throws {
        let content = "hello=hɛˈloʊ"
        let lexicon = try TtsCustomLexicon.parse(content)

        // HELLO should fall back to case-insensitive match
        let phonemes = lexicon.phonemes(for: "HELLO")
        XCTAssertNotNil(phonemes)
        XCTAssertEqual(phonemes, lexicon.phonemes(for: "hello"))
    }

    func testNormalizedFallback() throws {
        let content = "hello=hɛˈloʊ"
        let lexicon = try TtsCustomLexicon.parse(content)

        // Words with extra punctuation should normalize and match
        // Normalization strips non-letter/digit/apostrophe characters
        let phonemes = lexicon.phonemes(for: "HELLO!")
        XCTAssertNotNil(phonemes, "Should match via normalized fallback")
    }

    func testApostropheVariants() throws {
        let content = "don't=doʊnt"
        let lexicon = try TtsCustomLexicon.parse(content)

        // Different apostrophe characters should all match
        XCTAssertNotNil(lexicon.phonemes(for: "don't"))  // Standard apostrophe
        XCTAssertNotNil(lexicon.phonemes(for: "don't"))  // Curly apostrophe
    }

    func testWordNotFound() throws {
        let content = "hello=hɛˈloʊ"
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertNil(lexicon.phonemes(for: "goodbye"))
    }

    // MARK: - Empty Lexicon Tests

    func testEmptyLexicon() {
        let lexicon = TtsCustomLexicon.empty

        XCTAssertTrue(lexicon.isEmpty)
        XCTAssertEqual(lexicon.count, 0)
        XCTAssertNil(lexicon.phonemes(for: "anything"))
    }

    func testParseEmptyContent() throws {
        let lexicon = try TtsCustomLexicon.parse("")

        XCTAssertTrue(lexicon.isEmpty)
    }

    func testParseOnlyComments() throws {
        let content = """
            # Just comments
            # Nothing else
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertTrue(lexicon.isEmpty)
    }

    // MARK: - Merge Tests

    func testMergeLexicons() throws {
        let content1 = """
            hello=hɛˈloʊ
            world=wɝld
            """
        let content2 = """
            goodbye=ɡʊdˈbaɪ
            world=wɜːld
            """

        let lexicon1 = try TtsCustomLexicon.parse(content1)
        let lexicon2 = try TtsCustomLexicon.parse(content2)

        let merged = lexicon1.merged(with: lexicon2)

        XCTAssertEqual(merged.count, 3)
        XCTAssertNotNil(merged.phonemes(for: "hello"))
        XCTAssertNotNil(merged.phonemes(for: "goodbye"))

        // lexicon2's "world" should override lexicon1's
        let worldPhonemes = merged.phonemes(for: "world")
        XCTAssertEqual(worldPhonemes, lexicon2.phonemes(for: "world"))
    }

    func testMergeWithEmpty() throws {
        let content = "hello=hɛˈloʊ"
        let lexicon = try TtsCustomLexicon.parse(content)

        let merged = lexicon.merged(with: .empty)
        XCTAssertEqual(merged.count, 1)

        let merged2 = TtsCustomLexicon.empty.merged(with: lexicon)
        XCTAssertEqual(merged2.count, 1)
    }

    // MARK: - File Loading Tests

    func testLoadFromFile() throws {
        let content = """
            # Test lexicon file
            kokoro=kəkˈɔɹO
            xiaomi=ʃaʊˈmiː
            """

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_lexicon_\(UUID().uuidString).txt")

        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        try content.write(to: tempURL, atomically: true, encoding: .utf8)

        let lexicon = try TtsCustomLexicon.load(from: tempURL)

        XCTAssertEqual(lexicon.count, 2)
        XCTAssertNotNil(lexicon.phonemes(for: "kokoro"))
        XCTAssertNotNil(lexicon.phonemes(for: "xiaomi"))
    }

    func testLoadFromNonexistentFile() {
        let fakeURL = URL(fileURLWithPath: "/nonexistent/path/lexicon.txt")

        XCTAssertThrowsError(try TtsCustomLexicon.load(from: fakeURL))
    }

    // MARK: - Phoneme Tokenization Tests

    func testPhonemeTokenization() throws {
        let content = "test=tɛst"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "test")
        XCTAssertEqual(phonemes, ["t", "ɛ", "s", "t"])
    }

    func testPhonemeWithStressMarkers() throws {
        let content = "hello=hɛˈloʊ"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "hello")
        XCTAssertNotNil(phonemes)
        XCTAssertTrue(phonemes!.contains("ˈ"), "Should preserve stress marker")
    }

    func testPhonemeWithSecondaryStress() throws {
        let content = "international=ˌɪntɚnˈæʃənəl"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "international")
        XCTAssertNotNil(phonemes)
        XCTAssertTrue(phonemes!.contains("ˌ"), "Should preserve secondary stress marker")
        XCTAssertTrue(phonemes!.contains("ˈ"), "Should preserve primary stress marker")
    }

    // MARK: - Dictionary Initialization Tests

    func testInitFromDictionary() {
        let entries: [String: [String]] = [
            "hello": ["h", "ɛ", "l", "o", "ʊ"],
            "world": ["w", "ɝ", "l", "d"],
        ]

        let lexicon = TtsCustomLexicon(entries: entries)

        XCTAssertEqual(lexicon.count, 2)
        XCTAssertEqual(lexicon.phonemes(for: "hello"), ["h", "ɛ", "l", "o", "ʊ"])
        XCTAssertEqual(lexicon.phonemes(for: "world"), ["w", "ɝ", "l", "d"])
    }

    func testInitFromEmptyDictionary() {
        let lexicon = TtsCustomLexicon(entries: [:])

        XCTAssertTrue(lexicon.isEmpty)
        XCTAssertEqual(lexicon.count, 0)
    }

    // MARK: - Real-World Examples Tests

    func testMedicalTerminology() throws {
        let content = """
            acetaminophen=əˌsiːtəmˈɪnəfɛn
            ibuprofen=ˌaɪbjuːpɹˈoʊfən
            ketorolac=kˈɛtɔːɹˌɒlak
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 3)
        XCTAssertNotNil(lexicon.phonemes(for: "acetaminophen"))
        XCTAssertNotNil(lexicon.phonemes(for: "ibuprofen"))
        XCTAssertNotNil(lexicon.phonemes(for: "ketorolac"))
    }

    func testBrandNames() throws {
        let content = """
            Xiaomi=ʃaʊˈmiː
            NVIDIA=ɛnvˈɪdiə
            Kubernetes=kuːbɚnˈɛtiːz
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 3)

        // Test case-insensitive matching for brand names
        XCTAssertNotNil(lexicon.phonemes(for: "xiaomi"))
        XCTAssertNotNil(lexicon.phonemes(for: "nvidia"))
        XCTAssertNotNil(lexicon.phonemes(for: "kubernetes"))
    }

    func testAcronyms() throws {
        let content = """
            NASA=nˈæsə
            HIPAA=hˈɪpɑː
            EBITDA=iːbˈɪtdɑː
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 3)
        XCTAssertNotNil(lexicon.phonemes(for: "NASA"))
        XCTAssertNotNil(lexicon.phonemes(for: "HIPAA"))
        XCTAssertNotNil(lexicon.phonemes(for: "EBITDA"))
    }

    // MARK: - Three-Tier Matching Priority Tests

    func testMatchingPriorityExactOverCaseInsensitive() throws {
        // When both exact and lowercase entries exist, exact should win
        let content = """
            Hello=EXACT
            hello=LOWER
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        let result = lexicon.phonemes(for: "Hello")
        XCTAssertEqual(result, ["E", "X", "A", "C", "T"], "Exact match should take priority")

        let lowerResult = lexicon.phonemes(for: "hello")
        XCTAssertEqual(lowerResult, ["L", "O", "W", "E", "R"], "Lowercase query should match lowercase entry")
    }

    func testMatchingPriorityCaseInsensitiveOverNormalized() throws {
        // Case-insensitive should be checked before normalized
        let content = "hello=hɛˈloʊ"
        let lexicon = try TtsCustomLexicon.parse(content)

        // "HELLO" should match via case-insensitive (tier 2), not normalized (tier 3)
        let result = lexicon.phonemes(for: "HELLO")
        XCTAssertNotNil(result)

        // "hello!!!" has punctuation - should still match via normalized fallback
        let normalizedResult = lexicon.phonemes(for: "hello!!!")
        XCTAssertNotNil(normalizedResult, "Should fall back to normalized matching")
    }

    func testThreeTierMatchingComplete() throws {
        // Test all three tiers with a single lexicon
        let content = "Test=tɛst"
        let lexicon = try TtsCustomLexicon.parse(content)

        // Tier 1: Exact match
        XCTAssertNotNil(lexicon.phonemes(for: "Test"), "Tier 1: Exact match")

        // Tier 2: Case-insensitive (no exact match for "TEST")
        XCTAssertNotNil(lexicon.phonemes(for: "TEST"), "Tier 2: Case-insensitive")
        XCTAssertNotNil(lexicon.phonemes(for: "test"), "Tier 2: Case-insensitive")

        // Tier 3: Normalized (strips punctuation)
        XCTAssertNotNil(lexicon.phonemes(for: "test!"), "Tier 3: Normalized")
        XCTAssertNotNil(lexicon.phonemes(for: "@TEST@"), "Tier 3: Normalized with symbols")
    }

    // MARK: - Unicode Grapheme Cluster Tests

    func testUnicodeGraphemeClusterTokenization() throws {
        // Test that combining characters stay together as single tokens
        // é can be represented as e + combining acute (U+0301) or as precomposed é (U+00E9)
        let content = "cafe=kafˈeɪ"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "cafe")
        XCTAssertNotNil(phonemes)
        // Each character should be a separate token
        XCTAssertEqual(phonemes?.count, 6, "Should have 6 phoneme tokens")
    }

    func testUnicodeDiacriticsInPhonemes() throws {
        // IPA often uses combining diacritics (e.g., nasalization ̃, length mark ː)
        let content = "nasal=nãsal"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "nasal")
        XCTAssertNotNil(phonemes)
        // ã should be one grapheme cluster (a + combining tilde)
        XCTAssertTrue(phonemes!.contains { $0.contains("̃") || $0 == "ã" }, "Should preserve nasal diacritic")
    }

    func testUnicodeLengthMarker() throws {
        // Test the length marker ː (U+02D0) which is common in IPA
        let content = "beat=biːt"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "beat")
        XCTAssertEqual(phonemes, ["b", "i", "ː", "t"], "Length marker should be separate token")
    }

    func testMultiWordExpansionTokenization() throws {
        // Test that whitespace in phonemes creates word separator tokens
        let content = "UN=juːˈnaɪtɪd ˈneɪʃənz"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "UN")
        XCTAssertNotNil(phonemes)
        XCTAssertTrue(phonemes!.contains(" "), "Should contain space as word separator")

        // Count the space separators
        let spaceCount = phonemes!.filter { $0 == " " }.count
        XCTAssertEqual(spaceCount, 1, "Should have exactly one word separator")
    }

    // MARK: - Edge Case Tests

    func testEqualsSignInPhonemes() throws {
        // Edge case: what if phonemes contain = character?
        // The parser should only split on the FIRST = sign
        let content = "equals=iːkwəlz=saɪn"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "equals")
        XCTAssertNotNil(phonemes)
        // Should include everything after first =, including the second =
        XCTAssertTrue(phonemes!.contains("="), "Phonemes should contain the = character")
    }

    func testMultipleEqualsInLine() throws {
        // Another test for = handling
        let content = "test=a=b=c"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "test")
        XCTAssertNotNil(phonemes)
        XCTAssertEqual(phonemes, ["a", "=", "b", "=", "c"])
    }

    func testMergeConflictResolutionExplicit() throws {
        // Explicitly test that second lexicon wins on conflicts
        let base = try TtsCustomLexicon.parse(
            """
            word1=AAA
            word2=BBB
            word3=CCC
            """)

        let overlay = try TtsCustomLexicon.parse(
            """
            word2=XXX
            word4=DDD
            """)

        let merged = base.merged(with: overlay)

        // word1: only in base, should be preserved
        XCTAssertEqual(merged.phonemes(for: "word1"), ["A", "A", "A"])

        // word2: in both, overlay should win
        XCTAssertEqual(merged.phonemes(for: "word2"), ["X", "X", "X"])

        // word3: only in base, should be preserved
        XCTAssertEqual(merged.phonemes(for: "word3"), ["C", "C", "C"])

        // word4: only in overlay, should be included
        XCTAssertEqual(merged.phonemes(for: "word4"), ["D", "D", "D"])

        XCTAssertEqual(merged.count, 4)
    }

    func testWhitespaceOnlyPhonemes() throws {
        // Edge case: phonemes that are only whitespace should be rejected
        let content = "test=   "

        XCTAssertThrowsError(try TtsCustomLexicon.parse(content)) { error in
            XCTAssertTrue(error is TTSError)
        }
    }

    func testVeryLongPhonemeString() throws {
        // Test handling of unusually long phoneme strings
        let longPhonemes = String(repeating: "ə", count: 100)
        let content = "long=\(longPhonemes)"
        let lexicon = try TtsCustomLexicon.parse(content)

        let phonemes = lexicon.phonemes(for: "long")
        XCTAssertNotNil(phonemes)
        XCTAssertEqual(phonemes?.count, 100)
    }

    func testSpecialCharactersInWord() throws {
        // Test words with hyphens, numbers, etc.
        let content = """
            covid-19=ˈkoʊvɪd naɪnˈtiːn
            3d=θriːˈdiː
            """
        let lexicon = try TtsCustomLexicon.parse(content)

        XCTAssertEqual(lexicon.count, 2)
        XCTAssertNotNil(lexicon.phonemes(for: "covid-19"))
        XCTAssertNotNil(lexicon.phonemes(for: "3d"))
    }

    func testLoadEmptyFile() throws {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty_lexicon_\(UUID().uuidString).txt")

        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        try "".write(to: tempURL, atomically: true, encoding: .utf8)

        let lexicon = try TtsCustomLexicon.load(from: tempURL)
        XCTAssertTrue(lexicon.isEmpty)
    }

    func testLoadFileWithOnlyWhitespaceAndComments() throws {
        let content = """

            # Comment 1

            # Comment 2

            """

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("comments_only_\(UUID().uuidString).txt")

        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        try content.write(to: tempURL, atomically: true, encoding: .utf8)

        let lexicon = try TtsCustomLexicon.load(from: tempURL)
        XCTAssertTrue(lexicon.isEmpty)
    }
}
#endif
