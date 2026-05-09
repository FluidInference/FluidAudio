import XCTest

@testable import FluidAudio

final class StyleTTS2GoldLexiconTests: XCTestCase {

    // MARK: - MisakiNormalizer

    func testMisakiNormalizerRewritesUppercaseDiphthongs() {
        // Each Misaki shorthand expands to its espeak-style two-char form,
        // and stress markers / lower-case phonemes pass through unchanged.
        let input = "fɹˈAd"  // 'fraid: Misaki A → eɪ
        XCTAssertEqual(MisakiNormalizer.normalize(input), "fɹˈeɪd")
    }

    func testMisakiNormalizerHandlesAllDiphthongLetters() {
        XCTAssertEqual(MisakiNormalizer.normalize("A"), "eɪ")
        XCTAssertEqual(MisakiNormalizer.normalize("I"), "aɪ")
        XCTAssertEqual(MisakiNormalizer.normalize("O"), "oʊ")
        XCTAssertEqual(MisakiNormalizer.normalize("W"), "aʊ")
        XCTAssertEqual(MisakiNormalizer.normalize("Y"), "ɔɪ")
        XCTAssertEqual(MisakiNormalizer.normalize("Q"), "oʊ")
        XCTAssertEqual(MisakiNormalizer.normalize("R"), "ɹ")
    }

    func testMisakiNormalizerCollapsesSuperscriptSchwa() {
        // "ˈʌbᵊl" (Misaki '-able') → "ˈʌbəl"
        XCTAssertEqual(MisakiNormalizer.normalize("ˈʌbᵊl"), "ˈʌbəl")
    }

    func testMisakiNormalizerLeavesEspeakCharsAlone() {
        let espeak = "ˈmɛɹəkə"
        XCTAssertEqual(MisakiNormalizer.normalize(espeak), espeak)
    }

    // MARK: - normalizeKey

    func testNormalizeKeyLowercasesAndCanonicalizesApostrophes() {
        XCTAssertEqual(StyleTTS2GoldLexicon.normalizeKey("Don\u{2019}t"), "don't")
        XCTAssertEqual(StyleTTS2GoldLexicon.normalizeKey(" HELLO  "), "hello")
        XCTAssertEqual(StyleTTS2GoldLexicon.normalizeKey(""), "")
    }

    // MARK: - parse

    func testParseHandlesStringAndDictEntries() throws {
        // String entries pass through; DICT entries pull "DEFAULT" first
        // and fall back to any non-null POS value if DEFAULT is null.
        let json = """
            {
              "Hello": "həlˈO",
              "AI": "ˈAˌI",
              "ACT": {"DEFAULT": "ˈækt", "NOUN": null},
              "OnlyVerb": {"DEFAULT": null, "VERB": "vˈɜɹb"},
              "EmptyEntry": {"DEFAULT": null, "NOUN": null}
            }
            """
        let url = try writeTempJSON(json)
        defer { try? FileManager.default.removeItem(at: url) }

        let (lower, cased) = try StyleTTS2GoldLexicon.parse(url: url)

        // Lowercased map covers every survivable entry.
        XCTAssertEqual(lower["hello"], "həlˈO")
        XCTAssertEqual(lower["ai"], "ˈAˌI")
        XCTAssertEqual(lower["act"], "ˈækt")
        XCTAssertEqual(lower["onlyverb"], "vˈɜɹb")
        XCTAssertNil(lower["emptyentry"], "all-null DICT entries should be dropped")

        // Case-sensitive map only retains keys whose original casing
        // differs from the normalized one.
        XCTAssertEqual(cased["Hello"], "həlˈO")
        XCTAssertEqual(cased["AI"], "ˈAˌI")
        XCTAssertEqual(cased["ACT"], "ˈækt")
        XCTAssertEqual(cased["OnlyVerb"], "vˈɜɹb")
        XCTAssertNil(cased["EmptyEntry"])
    }

    // MARK: - phonemes(for:)

    func testPhonemesForLooksUpThenNormalizesMisakiSymbols() async throws {
        let json = """
            {
              "AI": "ˈAˌI",
              "again": "əɡˈɛn"
            }
            """
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(json.utf8).write(to: dir.appendingPathComponent("us_gold.json"))

        let lexicon = StyleTTS2GoldLexicon()
        try await lexicon.load(directory: dir)

        // Case-sensitive hit + Misaki normalization (A → eɪ, I → aɪ).
        let aiIPA = await lexicon.phonemes(for: "AI")
        XCTAssertEqual(aiIPA, "ˈeɪˌaɪ")

        // Lower-case fallthrough.
        let againIPA = await lexicon.phonemes(for: "Again")
        XCTAssertEqual(againIPA, "əɡˈɛn")

        // Unknown words return nil so the caller can fall through to G2P.
        let missing = await lexicon.phonemes(for: "thisisnotaword")
        XCTAssertNil(missing)
    }

    func testLoadThrowsWhenGoldFileMissing() async {
        let dir = try? makeTempDir()
        let lexicon = StyleTTS2GoldLexicon()
        do {
            try await lexicon.load(directory: dir!)
            XCTFail("expected fileMissing error")
        } catch StyleTTS2GoldLexicon.LoadError.fileMissing {
            // expected
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        if let dir { try? FileManager.default.removeItem(at: dir) }
    }

    // MARK: - Helpers

    private func writeTempJSON(_ content: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("styletts2-gold-test-\(UUID().uuidString).json")
        try Data(content.utf8).write(to: url)
        return url
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("styletts2-gold-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
