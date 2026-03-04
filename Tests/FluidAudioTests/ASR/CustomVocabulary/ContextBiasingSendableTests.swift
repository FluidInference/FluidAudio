import XCTest

@testable import FluidAudio

final class ContextBiasingSendableTests: XCTestCase {

    private func requiresSendable<T: Sendable>(_: T.Type) {}

    // MARK: - Core Types

    func testCustomVocabularyTermIsSendable() {
        requiresSendable(CustomVocabularyTerm.self)
    }

    func testCustomVocabularyContextIsSendable() {
        requiresSendable(CustomVocabularyContext.self)
    }

    // MARK: - Constants

    func testVocabSizeConfigIsSendable() {
        requiresSendable(ContextBiasingConstants.VocabSizeConfig.self)
    }

    // MARK: - Keyword Spotter

    func testCtcKeywordSpotterIsSendable() {
        requiresSendable(CtcKeywordSpotter.self)
    }

    func testKeywordDetectionIsSendable() {
        requiresSendable(CtcKeywordSpotter.KeywordDetection.self)
    }

    func testSpotKeywordsResultIsSendable() {
        requiresSendable(CtcKeywordSpotter.SpotKeywordsResult.self)
    }

    // MARK: - BK-Tree

    func testBKTreeIsSendable() {
        requiresSendable(BKTree.self)
    }

    // MARK: - Rescorer

    func testVocabularyRescorerIsSendable() {
        requiresSendable(VocabularyRescorer.self)
    }

    func testRescorerConfigIsSendable() {
        requiresSendable(VocabularyRescorer.Config.self)
    }
}
