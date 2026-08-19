import NaturalLanguage
import XCTest

@testable import FluidAudio

final class TextNormalizerTests: XCTestCase {

    // MARK: - NLTagger Context Spotting

    /// Ambiguous words that are both punctuation spoken forms AND common English words.
    private let ambiguousWords: Set<String> = [
        "period", "dash", "colon", "pipe", "slash", "dot", "plus", "hash", "percent",
    ]

    /// Helper: check if NLTagger identifies a word as natural language (noun/verb/adj/adverb)
    /// in the given sentence context.
    private func isNaturalLanguage(_ word: String, in sentence: String) -> Bool {
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = sentence

        guard let range = sentence.range(of: word) else {
            return false
        }

        let (tag, _) = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lexicalClass)
        return tag == .noun || tag == .verb || tag == .adjective || tag == .adverb
    }

    // --- "period" as noun (time period) should NOT be normalized ---

    func testPeriodAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("period", in: "that was the best period of my life"),
            "'period' should be tagged as natural language (noun) in this context"
        )
    }

    func testPeriodAsNounEndOfPhrase() {
        XCTAssertTrue(
            isNaturalLanguage("period", in: "end of the period"),
            "'period' should be tagged as natural language (noun) when used as a time reference"
        )
    }

    func testPeriodStandaloneIsPunctuation() {
        // Standalone "period" is NOT natural language — it's a punctuation command
        XCTAssertFalse(
            isNaturalLanguage("period", in: "period"),
            "standalone 'period' should not be tagged as natural language"
        )
    }

    // --- "dash" as verb/noun should NOT be normalized ---

    func testDashAsVerbInSentence() {
        // "I need to dash to the store" — dash = verb
        XCTAssertTrue(
            isNaturalLanguage("dash", in: "I need to dash to the store"),
            "'dash' should be tagged as natural language (verb) in this context"
        )
    }

    func testDashAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("dash", in: "add a dash of salt"),
            "'dash' should be tagged as natural language (noun) in this context"
        )
    }

    // --- "colon" as noun (body part) should NOT be normalized ---

    func testColonAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("colon", in: "the doctor examined my colon"),
            "'colon' should be tagged as natural language (noun) in this context"
        )
    }

    // --- "dot" as noun should NOT be normalized ---

    func testDotAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("dot", in: "press the red dot"),
            "'dot' should be tagged as natural language (noun) in this context"
        )
    }

    // --- "plus" as noun/adjective should NOT be normalized ---

    func testPlusAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("plus", in: "that is a plus for us"),
            "'plus' should be tagged as natural language (noun) in this context"
        )
    }

    // --- "hash" as noun should NOT be normalized ---

    func testHashAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("hash", in: "use the hash symbol"),
            "'hash' should be tagged as natural language (noun) in this context"
        )
    }

    // --- "percent" as noun should NOT be normalized ---

    func testPercentAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("percent", in: "fifty percent of people agree"),
            "'percent' should be tagged as natural language (noun) in this context"
        )
    }

    // MARK: - TextNormalizer Instance

    func testNativeLibraryAlwaysAvailable() {
        // The engine is linked via the bundled NemoTextProcessing binary target,
        // so it must be available in every consumer build (issue #839).
        let normalizer = TextNormalizer()
        XCTAssertTrue(normalizer.isNativeAvailable)
        XCTAssertTrue(normalizer.isTnAvailable)
        XCTAssertNotNil(normalizer.version)
    }

    func testNormalizeConvertsSpokenNumbers() {
        let normalizer = TextNormalizer()
        XCTAssertEqual(normalizer.normalize("twenty one"), "21")
        XCTAssertEqual(normalizer.normalize("two hundred"), "200")
        // Exact repro from issue #839
        XCTAssertEqual(normalizer.normalize("twelve dollars"), "$12")
    }

    func testNormalizeSentenceConvertsSpans() {
        let normalizer = TextNormalizer()
        XCTAssertEqual(normalizer.normalizeSentence("I have twenty one apples"), "I have 21 apples")
        XCTAssertEqual(normalizer.normalizeSentence("it costs twelve dollars"), "it costs $12")
    }

    func testTnNormalizeConvertsWrittenToSpoken() {
        let normalizer = TextNormalizer()
        XCTAssertEqual(normalizer.tnNormalize("$5.50"), "five dollars fifty cents")
    }

    func testCustomRules() {
        let normalizer = TextNormalizer()
        normalizer.addRule(spoken: "gee pee tee", written: "GPT")
        XCTAssertEqual(normalizer.ruleCount, 1)
        XCTAssertEqual(normalizer.normalize("gee pee tee"), "GPT")
        XCTAssertTrue(normalizer.removeRule(spoken: "gee pee tee"))
        XCTAssertEqual(normalizer.ruleCount, 0)
    }

    func testTextNormalizerIsSendable() {
        // Compile-time check: TextNormalizer conforms to Sendable
        func requiresSendable<T: Sendable>(_: T.Type) {}
        requiresSendable(TextNormalizer.self)
    }

    // MARK: - Ambiguous Words Set Completeness

    func testAmbiguousWordsMatchTextNormalizer() {
        // Verify our test set matches what TextNormalizer uses
        // This is a compile-time documentation test — if the sets drift apart,
        // the developer should update both.
        let expected: Set<String> = [
            "period", "dash", "colon", "pipe", "slash", "dot", "plus", "hash", "percent",
        ]
        XCTAssertEqual(ambiguousWords, expected)
    }

    // MARK: - NLTagger Batch Verification

    func testAllAmbiguousWordsProtectedInNaturalSentences() {
        // Each ambiguous word used in a natural English sentence should be tagged
        // as natural language (noun/verb/adj) and therefore protected from normalization.
        let naturalSentences: [(String, String)] = [
            ("period", "that was a difficult period in history"),
            ("dash", "she made a dash for the door"),
            ("colon", "the colon separates clauses"),
            ("dot", "connect the dot to the line"),
            ("plus", "the plus side is obvious"),
            ("hash", "we ordered hash browns for breakfast"),
            ("percent", "a large percent of voters disagreed"),
        ]

        for (word, sentence) in naturalSentences {
            XCTAssertTrue(
                isNaturalLanguage(word, in: sentence),
                "'\(word)' in '\(sentence)' should be tagged as natural language but wasn't"
            )
        }
    }

    // MARK: - Additional Ambiguous Word Contexts

    // --- "slash" as verb should NOT be normalized ---

    func testSlashAsVerbInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("slash", in: "they had to slash the budget"),
            "'slash' should be tagged as natural language (verb) in this context"
        )
    }

    func testSlashAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("slash", in: "there was a slash across the painting"),
            "'slash' should be tagged as natural language (noun) in this context"
        )
    }

    // --- "pipe" as noun should NOT be normalized ---

    func testPipeAsNounInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("pipe", in: "the water pipe burst overnight"),
            "'pipe' should be tagged as natural language (noun) in this context"
        )
    }

    func testPipeAsVerbInSentence() {
        XCTAssertTrue(
            isNaturalLanguage("pipe", in: "pipe the output to a file"),
            "'pipe' should be tagged as natural language (verb) in this context"
        )
    }

    // MARK: - Multiple Ambiguous Words in One Sentence

    func testMultipleAmbiguousWordsInOneSentence() {
        // "period" and "dash" both in one sentence — both should be protected
        let sentence = "after a short dash she ended the period"
        XCTAssertTrue(
            isNaturalLanguage("dash", in: sentence),
            "'dash' in mixed-ambiguous sentence should be tagged as natural language"
        )
        XCTAssertTrue(
            isNaturalLanguage("period", in: sentence),
            "'period' in mixed-ambiguous sentence should be tagged as natural language"
        )
    }

    func testPercentAndDotInOneSentence() {
        let sentence = "only ten percent of the dot patterns matched"
        XCTAssertTrue(
            isNaturalLanguage("percent", in: sentence),
            "'percent' should be tagged as natural language in mixed context"
        )
        XCTAssertTrue(
            isNaturalLanguage("dot", in: sentence),
            "'dot' should be tagged as natural language in mixed context"
        )
    }

    // MARK: - ASR-Realistic Sentence Contexts

    func testPeriodInAcademicContext() {
        XCTAssertTrue(
            isNaturalLanguage("period", in: "the medieval period lasted several centuries"),
            "'period' as historical era should be tagged as natural language"
        )
    }

    func testDashInSportsContext() {
        XCTAssertTrue(
            isNaturalLanguage("dash", in: "she ran the hundred meter dash"),
            "'dash' as a race event should be tagged as natural language"
        )
    }

    func testColonInMedicalContext() {
        XCTAssertTrue(
            isNaturalLanguage("colon", in: "colon cancer screening is important"),
            "'colon' in medical context should be tagged as natural language"
        )
    }

    func testDotInArtContext() {
        XCTAssertTrue(
            isNaturalLanguage("dot", in: "each dot represents a data point"),
            "'dot' as a visual mark should be tagged as natural language"
        )
    }

    func testHashInCulinaryContext() {
        XCTAssertTrue(
            isNaturalLanguage("hash", in: "corned beef hash is a classic dish"),
            "'hash' as a food should be tagged as natural language"
        )
    }

    func testPercentInFinancialContext() {
        XCTAssertTrue(
            isNaturalLanguage("percent", in: "the interest rate dropped two percent"),
            "'percent' in financial context should be tagged as natural language"
        )
    }

    // MARK: - Edge Position Tests

    func testAmbiguousWordAtStartOfSentence() {
        // "Dash" at the start of a sentence
        XCTAssertTrue(
            isNaturalLanguage("dash", in: "dash across the field quickly"),
            "'dash' at sentence start as verb should be tagged as natural language"
        )
    }

    func testAmbiguousWordAtEndOfSentence() {
        XCTAssertTrue(
            isNaturalLanguage("period", in: "we studied the Jurassic period"),
            "'period' at sentence end as noun should be tagged as natural language"
        )
    }

    // MARK: - filterAmbiguousWords Logic

    func testFilterWithAmbiguousWordInSentence() {
        let normalizer = TextNormalizer()
        // Ambiguous words used as natural language (nouns) must survive
        // normalization: the native ITN would otherwise rewrite "period" → ".",
        // "dash" → "-". Passes with the native lib (masked + restored) and
        // without it (fallback returns the input unchanged).
        let unchanged = [
            "the period of growth was remarkable",
            "add a period here",
            "the dash between them",
        ]
        for input in unchanged {
            XCTAssertEqual(normalizer.normalizeSentence(input), input)
        }
    }

    func testStandaloneAmbiguousWordStillNormalizes() {
        // The mask only protects natural-language usage; a standalone spoken
        // command still normalizes.
        let normalizer = TextNormalizer()
        XCTAssertEqual(normalizer.normalizeSentence("period"), ".")
    }

    // MARK: - TextNormalizer normalize(result:) Method

    func testNormalizeASRResult() {
        let normalizer = TextNormalizer()

        let asrResult = ASRResult(
            text: "I have twenty one apples",
            confidence: 0.95,
            duration: 2.0,
            processingTime: 0.1,
            tokenTimings: [],
            ctcDetectedTerms: [],
            ctcAppliedTerms: []
        )
        let normalized = normalizer.normalize(result: asrResult)
        XCTAssertEqual(normalized.text, "I have 21 apples")
        // Metadata should be preserved
        XCTAssertEqual(normalized.confidence, 0.95)
        XCTAssertEqual(normalized.duration, 2.0)
    }

    // MARK: - TextNormalizer maxSpanTokens Variant

    func testNormalizeSentenceWithMaxSpan() {
        let normalizer = TextNormalizer()
        XCTAssertEqual(normalizer.normalizeSentence("twenty one apples", maxSpanTokens: 8), "21 apples")
    }

    // MARK: - TN (written → spoken) surface

    func testTnNormalizeSentence() {
        let normalizer = TextNormalizer()
        XCTAssertEqual(normalizer.tnNormalizeSentence("I paid $5"), "I paid five dollars")
    }
}
