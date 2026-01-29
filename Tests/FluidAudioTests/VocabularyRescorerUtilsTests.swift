import XCTest

@testable import FluidAudio

final class VocabularyRescorerUtilsTests: XCTestCase {

    // MARK: - stringSimilarity

    func testIdenticalStrings() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("nvidia", "nvidia"), 1.0, accuracy: 0.01)
    }

    func testCompletelyDifferent() {
        // "abc" vs "xyz" -> distance 3, maxLen 3 -> sim = 0.0
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("abc", "xyz"), 0.0, accuracy: 0.01)
    }

    func testCaseInsensitive() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("NVIDIA", "nvidia"), 1.0, accuracy: 0.01)
    }

    func testOneCharDifference() {
        // "bose" vs "boz" -> distance 1 (e vs z) + length diff -> distance 2, maxLen 4
        // Actually: "bose" (4) vs "boz" (3) -> distance 2, maxLen 4 -> 1 - 2/4 = 0.5
        let sim = VocabularyRescorer.stringSimilarity("bose", "boz")
        XCTAssertEqual(sim, 0.5, accuracy: 0.01)
    }

    func testBothEmpty() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("", ""), 1.0, accuracy: 0.01)
    }

    func testOneEmpty() {
        XCTAssertEqual(VocabularyRescorer.stringSimilarity("abc", ""), 0.0, accuracy: 0.01)
    }

    func testKnownPair() {
        // "nvida" vs "nvidia" -> distance 1, maxLen 6 -> 1 - 1/6 ≈ 0.833
        let sim = VocabularyRescorer.stringSimilarity("nvida", "nvidia")
        XCTAssertEqual(sim, 1.0 - 1.0 / 6.0, accuracy: 0.01)
    }

    // MARK: - lengthPenalizedSimilarity

    func testEqualLengthNoPenalty() {
        // Same length -> lengthRatio = 1.0 -> sqrt(1.0) = 1.0 -> no penalty
        let lps = VocabularyRescorer.lengthPenalizedSimilarity("abcde", "abcde")
        let base = VocabularyRescorer.stringSimilarity("abcde", "abcde")
        XCTAssertEqual(lps, base, accuracy: 0.01)
    }

    func testShorterCompoundPenalized() {
        // "ab" (2) vs "abcdef" (6) -> lengthRatio = 2/6 ≈ 0.33
        // penalty = sqrt(0.33) ≈ 0.577
        let lps = VocabularyRescorer.lengthPenalizedSimilarity("ab", "abcdef")
        let base = VocabularyRescorer.stringSimilarity("ab", "abcdef")
        XCTAssertLessThan(lps, base)
    }

    func testSameLengthSimilarWords() {
        // "newres" (6) vs "newrez" (6) -> equal length, sqrt(1.0) = 1.0
        let lps = VocabularyRescorer.lengthPenalizedSimilarity("newres", "newrez")
        let base = VocabularyRescorer.stringSimilarity("newres", "newrez")
        XCTAssertEqual(lps, base, accuracy: 0.01)
    }

    // MARK: - normalizeForSimilarity

    func testNormalizeBasic() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("Hello World!"), "hello world")
    }

    func testNormalizePreservesApostrophe() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("It's"), "it's")
    }

    func testNormalizePreservesHyphen() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("Ramirez-Santos"), "ramirez-santos")
    }

    func testNormalizeMultipleSpaces() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("  hello   world  "), "hello world")
    }

    func testNormalizeEmptyString() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity(""), "")
    }

    func testNormalizeNumbers() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("Test123"), "test123")
    }

    func testNormalizeTabsNewlines() {
        XCTAssertEqual(VocabularyRescorer.normalizeForSimilarity("hello\tworld\nfoo"), "hello world foo")
    }

    // MARK: - Config Adaptive Thresholds

    func testAdaptiveMinVocabScoreAtReference() {
        let config = VocabularyRescorer.Config.default
        // referenceTokenCount = 3, no adjustment
        XCTAssertEqual(config.adaptiveMinVocabScore(tokenCount: 3), config.minVocabScore, accuracy: 0.01)
    }

    func testAdaptiveMinVocabScoreLongerPhrase() {
        let config = VocabularyRescorer.Config.default
        // 5 tokens: extraTokens = 5 - 3 = 2, threshold = -12.0 - 2*1.0 = -14.0
        XCTAssertEqual(config.adaptiveMinVocabScore(tokenCount: 5), -14.0, accuracy: 0.01)
    }

    func testAdaptiveMinVocabScoreShortPhrase() {
        let config = VocabularyRescorer.Config.default
        // 1 token: extraTokens = max(0, 1-3) = 0, no adjustment
        XCTAssertEqual(config.adaptiveMinVocabScore(tokenCount: 1), config.minVocabScore, accuracy: 0.01)
    }

    func testAdaptiveMinVocabScoreDisabled() {
        let config = VocabularyRescorer.Config(useAdaptiveThresholds: false)
        XCTAssertEqual(config.adaptiveMinVocabScore(tokenCount: 10), config.minVocabScore, accuracy: 0.01)
    }

    func testAdaptiveCbwAtReference() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 3), 3.0, accuracy: 0.01)
    }

    func testAdaptiveCbwLongerPhrase() {
        let config = VocabularyRescorer.Config.default
        // 6 tokens: ratio = 6/3 = 2.0, scaleFactor = 1.0 + log2(2.0)*0.3 = 1.3
        // result = 3.0 * 1.3 = 3.9
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 6), 3.9, accuracy: 0.01)
    }

    func testAdaptiveCbwBelowReference() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 2), 3.0, accuracy: 0.01)
    }

    func testAdaptiveCbwDisabled() {
        let config = VocabularyRescorer.Config(useAdaptiveThresholds: false)
        XCTAssertEqual(config.adaptiveCbw(baseCbw: 3.0, tokenCount: 10), 3.0, accuracy: 0.01)
    }

    func testAdaptiveMinScoreAdvantageAtReference() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.adaptiveMinScoreAdvantage(tokenCount: 3), config.minScoreAdvantage, accuracy: 0.01)
    }

    func testAdaptiveMinScoreAdvantageLongerPhrase() {
        let config = VocabularyRescorer.Config.default
        // 6 tokens: ratio = 6/3 = 2.0, result = 2.0 / sqrt(2.0) ≈ 1.414
        let expected = config.minScoreAdvantage / sqrt(2.0)
        XCTAssertEqual(config.adaptiveMinScoreAdvantage(tokenCount: 6), expected, accuracy: 0.01)
    }

    func testAdaptiveMinScoreAdvantageBelowReference() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.adaptiveMinScoreAdvantage(tokenCount: 1), config.minScoreAdvantage, accuracy: 0.01)
    }

    func testAdaptiveMinScoreAdvantageDisabled() {
        let config = VocabularyRescorer.Config(useAdaptiveThresholds: false)
        XCTAssertEqual(config.adaptiveMinScoreAdvantage(tokenCount: 10), config.minScoreAdvantage, accuracy: 0.01)
    }

    // MARK: - Config Defaults

    func testConfigDefaultValues() {
        let config = VocabularyRescorer.Config.default
        XCTAssertEqual(config.minScoreAdvantage, ContextBiasingConstants.defaultMinScoreAdvantage, accuracy: 0.01)
        XCTAssertEqual(config.minVocabScore, ContextBiasingConstants.defaultMinVocabScore, accuracy: 0.01)
        XCTAssertEqual(
            config.maxOriginalScoreForReplacement,
            ContextBiasingConstants.defaultMaxOriginalScoreForReplacement,
            accuracy: 0.01
        )
        XCTAssertEqual(config.vocabBoostWeight, ContextBiasingConstants.defaultVocabBoostWeight, accuracy: 0.01)
    }
}
