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
    }

    func testTermFullInit() {
        let term = CustomVocabularyTerm(
            text: "Bose",
            weight: 5.0,
            aliases: ["boz", "boss"],
            tokenIds: [100, 200],
            ctcTokenIds: [50, 60]
        )
        XCTAssertEqual(term.text, "Bose")
        XCTAssertEqual(term.weight, 5.0)
        XCTAssertEqual(term.aliases, ["boz", "boss"])
        XCTAssertEqual(term.tokenIds, [100, 200])
        XCTAssertEqual(term.ctcTokenIds, [50, 60])
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
            ctcTokenIds: [30, 40]
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(CustomVocabularyTerm.self, from: data)

        XCTAssertEqual(decoded.text, original.text)
        XCTAssertEqual(decoded.weight, original.weight)
        XCTAssertEqual(decoded.aliases, original.aliases)
        XCTAssertEqual(decoded.tokenIds, original.tokenIds)
        XCTAssertEqual(decoded.ctcTokenIds, original.ctcTokenIds)
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

    // MARK: - Edge Cases

    func testTermWithEmptyAliases() {
        let term = CustomVocabularyTerm(text: "X", aliases: [])
        XCTAssertNotNil(term.aliases)
        XCTAssertTrue(term.aliases?.isEmpty == true)
    }
}
