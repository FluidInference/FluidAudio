import XCTest

@testable import FluidAudio

final class ContextBiasingConstantsTests: XCTestCase {

    // MARK: - Token ID Constants

    func testWildcardTokenId() {
        XCTAssertEqual(ContextBiasingConstants.wildcardTokenId, -1)
    }

    func testDefaultBlankId() {
        XCTAssertEqual(ContextBiasingConstants.defaultBlankId, 1024)
    }

    // MARK: - Similarity Threshold Hierarchy

    func testSimilarityThresholdHierarchy() {
        // Thresholds should form a strict ordering from lenient to strict
        let floor = ContextBiasingConstants.minSimilarityFloor
        let defaultMin = ContextBiasingConstants.defaultMinSimilarity
        let singleWord = ContextBiasingConstants.singleWordSpanSimilarity
        let alias = ContextBiasingConstants.highConfidenceAliasSimilarity
        let lengthRatio = ContextBiasingConstants.lengthRatioThreshold
        let multiWord = ContextBiasingConstants.multiWordSpanSimilarity
        let shortWord = ContextBiasingConstants.shortWordSimilarity
        let stopword = ContextBiasingConstants.stopwordSpanSimilarity

        XCTAssertLessThan(floor, defaultMin)
        XCTAssertLessThan(defaultMin, singleWord)
        XCTAssertLessThan(singleWord, alias)
        XCTAssertLessThan(alias, lengthRatio)
        XCTAssertLessThanOrEqual(lengthRatio, multiWord)
        XCTAssertEqual(multiWord, shortWord)
        XCTAssertLessThanOrEqual(shortWord, stopword)
    }

    func testAllSimilarityThresholdsInRange() {
        let thresholds: [Float] = [
            ContextBiasingConstants.minSimilarityFloor,
            ContextBiasingConstants.defaultMinSimilarity,
            ContextBiasingConstants.singleWordSpanSimilarity,
            ContextBiasingConstants.highConfidenceAliasSimilarity,
            ContextBiasingConstants.lengthRatioThreshold,
            ContextBiasingConstants.multiWordSpanSimilarity,
            ContextBiasingConstants.shortWordSimilarity,
            ContextBiasingConstants.stopwordSpanSimilarity,
        ]
        for threshold in thresholds {
            XCTAssertGreaterThan(threshold, 0.0)
            XCTAssertLessThanOrEqual(threshold, 1.0)
        }
    }

    // MARK: - Context Biasing Weights

    func testCbwPositive() {
        XCTAssertGreaterThan(ContextBiasingConstants.defaultCbw, 0)
    }

    func testDefaultAlphaInRange() {
        XCTAssertGreaterThanOrEqual(ContextBiasingConstants.defaultAlpha, 0.0)
        XCTAssertLessThanOrEqual(ContextBiasingConstants.defaultAlpha, 1.0)
    }

    // MARK: - rescorerConfig(forVocabSize:)

    func testSmallVocabConfig() {
        let config = ContextBiasingConstants.rescorerConfig(forVocabSize: 5)
        XCTAssertEqual(config.minScoreAdvantage, 1.0, accuracy: 0.01)
        XCTAssertEqual(config.minVocabScore, -15.0, accuracy: 0.01)
        XCTAssertEqual(config.vocabBoostWeight, 3.0, accuracy: 0.01)
        XCTAssertEqual(config.minSimilarity, 0.50, accuracy: 0.01)
        XCTAssertEqual(config.cbw, 3.0, accuracy: 0.01)
    }

    func testLargeVocabConfig() {
        let config = ContextBiasingConstants.rescorerConfig(forVocabSize: 15)
        XCTAssertEqual(config.minScoreAdvantage, 1.5, accuracy: 0.01)
        XCTAssertEqual(config.minVocabScore, -14.0, accuracy: 0.01)
        XCTAssertEqual(config.vocabBoostWeight, 2.5, accuracy: 0.01)
        XCTAssertEqual(config.minSimilarity, 0.60, accuracy: 0.01)
        XCTAssertEqual(config.cbw, 2.5, accuracy: 0.01)
    }

    func testBoundaryVocabConfig() {
        // Exactly 10 = threshold, NOT large (>10 is large)
        let config = ContextBiasingConstants.rescorerConfig(forVocabSize: 10)
        XCTAssertEqual(config.minScoreAdvantage, 1.0, accuracy: 0.01)
    }

    func testLargeVocabStricterThresholds() {
        let small = ContextBiasingConstants.rescorerConfig(forVocabSize: 5)
        let large = ContextBiasingConstants.rescorerConfig(forVocabSize: 15)
        XCTAssertGreaterThan(large.minSimilarity, small.minSimilarity)
        XCTAssertGreaterThan(large.minScoreAdvantage, small.minScoreAdvantage)
    }
}
