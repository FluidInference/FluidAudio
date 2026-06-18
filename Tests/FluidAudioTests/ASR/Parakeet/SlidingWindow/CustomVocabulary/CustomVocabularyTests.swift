import XCTest

@testable import FluidAudio

final class CustomVocabularyTests: XCTestCase {

    // MARK: - CustomVocabularyTerm Creation

    func testTermDefaultInit() {
        let term = CustomVocabularyTerm(text: "NVIDIA")
        XCTAssertEqual(term.text, "NVIDIA")
        XCTAssertNil(term.weight)
        XCTAssertNil(term.aliases)
        XCTAssertNil(term.tokenIds)
        XCTAssertNil(term.ctcTokenIds)
        XCTAssertNil(term.minSimilarity)
    }

    func testTermFullInit() {
        let term = CustomVocabularyTerm(
            text: "Bose",
            weight: 5.0,
            aliases: ["boz", "boss"],
            tokenIds: [100, 200],
            ctcTokenIds: [50, 60],
            minSimilarity: 0.45
        )
        XCTAssertEqual(term.text, "Bose")
        XCTAssertEqual(term.weight, 5.0)
        XCTAssertEqual(term.aliases, ["boz", "boss"])
        XCTAssertEqual(term.tokenIds, [100, 200])
        XCTAssertEqual(term.ctcTokenIds, [50, 60])
        XCTAssertEqual(term.minSimilarity ?? -1, 0.45, accuracy: 0.001)
    }

    // MARK: - Per-Term minSimilarity (#647)

    func testTermMinSimilarityClampedToRange() {
        XCTAssertEqual(
            CustomVocabularyTerm(text: "Caivex", minSimilarity: -0.5).minSimilarity ?? -1, 0.0, accuracy: 0.001)
        XCTAssertEqual(
            CustomVocabularyTerm(text: "Andre", minSimilarity: 1.7).minSimilarity ?? -1, 1.0, accuracy: 0.001)
        XCTAssertEqual(CustomVocabularyTerm(text: "Mid", minSimilarity: 0.6).minSimilarity ?? -1, 0.6, accuracy: 0.001)
    }

    func testTextLowercased() {
        XCTAssertEqual(CustomVocabularyTerm(text: "NVIDIA").textLowercased, "nvidia")
        XCTAssertEqual(CustomVocabularyTerm(text: "McDonald's").textLowercased, "mcdonald's")
    }

    func testTextLowercasedEmptyString() {
        XCTAssertEqual(CustomVocabularyTerm(text: "").textLowercased, "")
    }

    // MARK: - Codable Conformance

    func testTermEncodeDecode() throws {
        let original = CustomVocabularyTerm(
            text: "TensorRT",
            weight: 3.0,
            aliases: ["tensor-rt"],
            tokenIds: [10, 20],
            ctcTokenIds: [30, 40],
            minSimilarity: 0.52
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(CustomVocabularyTerm.self, from: data)

        XCTAssertEqual(decoded.text, original.text)
        XCTAssertEqual(decoded.weight, original.weight)
        XCTAssertEqual(decoded.aliases, original.aliases)
        XCTAssertEqual(decoded.tokenIds, original.tokenIds)
        XCTAssertEqual(decoded.ctcTokenIds, original.ctcTokenIds)
        XCTAssertEqual(decoded.minSimilarity ?? -1, 0.52, accuracy: 0.001)
        XCTAssertEqual(decoded.textLowercased, "tensorrt")
    }

    func testTermDecodeWithMissingOptionals() throws {
        let json = """
            {"text": "Nequi"}
            """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(CustomVocabularyTerm.self, from: json)
        XCTAssertEqual(decoded.text, "Nequi")
        XCTAssertNil(decoded.weight)
        XCTAssertNil(decoded.aliases)
        XCTAssertNil(decoded.tokenIds)
        XCTAssertNil(decoded.ctcTokenIds)
        XCTAssertNil(decoded.minSimilarity)
    }

    func testTermDecodeMinSimilarityFromJSON() throws {
        let json = """
            {"text": "Caivex", "minSimilarity": 0.42}
            """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(CustomVocabularyTerm.self, from: json)
        XCTAssertEqual(decoded.minSimilarity ?? -1, 0.42, accuracy: 0.001)
    }

    func testTermDecodeSetsLowercased() throws {
        let json = """
            {"text": "PyTorch"}
            """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(CustomVocabularyTerm.self, from: json)
        XCTAssertEqual(decoded.textLowercased, "pytorch")
    }

    // MARK: - CustomVocabularyContext Creation

    func testContextDefaultInit() {
        let terms = [
            CustomVocabularyTerm(text: "NVIDIA"),
            CustomVocabularyTerm(text: "AMD"),
        ]
        let context = CustomVocabularyContext(terms: terms)
        XCTAssertEqual(context.terms.count, 2)
        XCTAssertEqual(context.alpha, ContextBiasingConstants.defaultAlpha, accuracy: 0.01)
        XCTAssertEqual(context.minCtcScore, ContextBiasingConstants.defaultMinVocabCtcScore, accuracy: 0.01)
        XCTAssertEqual(context.minSimilarity, ContextBiasingConstants.defaultMinSimilarity, accuracy: 0.01)
        XCTAssertEqual(
            context.minCombinedConfidence, ContextBiasingConstants.defaultMinCombinedConfidence, accuracy: 0.01)
    }

    func testContextCustomInit() {
        let context = CustomVocabularyContext(
            terms: [],
            alpha: 0.8,
            minCtcScore: -10.0,
            minSimilarity: 0.7,
            minCombinedConfidence: 0.75,
            minTermLength: 5
        )
        XCTAssertEqual(context.alpha, 0.8, accuracy: 0.01)
        XCTAssertEqual(context.minCtcScore, -10.0, accuracy: 0.01)
        XCTAssertEqual(context.minSimilarity, 0.7, accuracy: 0.01)
        XCTAssertEqual(context.minCombinedConfidence, 0.75, accuracy: 0.01)
        XCTAssertEqual(context.minTermLength, 5)
    }

    func testContextMinTermLengthDefault() {
        let context = CustomVocabularyContext(terms: [])
        XCTAssertEqual(context.minTermLength, 3)
    }

    // MARK: - load(from:) Per-Term minSimilarity

    func testLoadPreservesPerTermMinSimilarity() throws {
        let json = """
            {
              "terms": [
                {"text": "Caivex", "minSimilarity": 0.40, "ctcTokenIds": [1, 2, 3]},
                {"text": "Andre", "minSimilarity": 0.80, "ctcTokenIds": [4, 5]},
                {"text": "Mobius", "ctcTokenIds": [6, 7]}
              ]
            }
            """
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vocab-647-\(UUID().uuidString).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let context = try CustomVocabularyContext.load(from: url)
        XCTAssertEqual(context.terms.count, 3)

        let byText = Dictionary(uniqueKeysWithValues: context.terms.map { ($0.text, $0) })
        XCTAssertEqual(byText["Caivex"]?.minSimilarity ?? -1, 0.40, accuracy: 0.001)
        XCTAssertEqual(byText["Andre"]?.minSimilarity ?? -1, 0.80, accuracy: 0.001)
        // Term without an override keeps nil → falls back to vocabulary-level threshold.
        let mobius = try XCTUnwrap(byText["Mobius"])
        XCTAssertNil(mobius.minSimilarity)
    }

    // MARK: - loadVocabularyFile Format Detection (JSON vs simple text)

    func testLoadVocabularyFileDetectsJSON() throws {
        let json = """
            {
              "minSimilarity": 0.6,
              "terms": [
                {"text": "Caivex", "minSimilarity": 0.40},
                {"text": "Andre"}
              ]
            }
            """
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vocab-647-json-\(UUID().uuidString).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let context = try CustomVocabularyContext.loadVocabularyFile(at: url)
        XCTAssertEqual(context.terms.count, 2)
        // Vocabulary-level threshold from JSON is honored.
        XCTAssertEqual(context.minSimilarity, 0.6, accuracy: 0.001)
        let byText = Dictionary(uniqueKeysWithValues: context.terms.map { ($0.text, $0) })
        XCTAssertEqual(byText["Caivex"]?.minSimilarity ?? -1, 0.40, accuracy: 0.001)
        XCTAssertNil(try XCTUnwrap(byText["Andre"]).minSimilarity)
    }

    func testLoadVocabularyFileDetectsJSONWithLeadingWhitespace() throws {
        let json = "\n\n   {\"terms\": [{\"text\": \"Caivex\", \"minSimilarity\": 0.4}]}"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vocab-647-ws-\(UUID().uuidString).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let context = try CustomVocabularyContext.loadVocabularyFile(at: url)
        XCTAssertEqual(context.terms.count, 1)
        XCTAssertEqual(context.terms.first?.minSimilarity ?? -1, 0.4, accuracy: 0.001)
    }

    func testLoadVocabularyFileDetectsSimpleText() throws {
        let text = """
            # comment line
            NVIDIA
            Bose: boz, boss
            """
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vocab-647-txt-\(UUID().uuidString).txt")
        try text.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        let context = try CustomVocabularyContext.loadVocabularyFile(at: url)
        XCTAssertEqual(context.terms.count, 2)
        let byText = Dictionary(uniqueKeysWithValues: context.terms.map { ($0.text, $0) })
        XCTAssertNotNil(byText["NVIDIA"])
        XCTAssertEqual(byText["Bose"]?.aliases, ["boz", "boss"])
        // Simple text format has no per-term threshold field.
        XCTAssertNil(try XCTUnwrap(byText["NVIDIA"]).minSimilarity)
    }

    // MARK: - Edge Cases

    func testTermWithEmptyAliases() {
        let term = CustomVocabularyTerm(text: "X", aliases: [])
        XCTAssertNotNil(term.aliases)
        XCTAssertTrue(term.aliases?.isEmpty == true)
    }
}
