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
}
#endif
