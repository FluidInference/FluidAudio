import FluidAudio
import Foundation
import Testing

@testable import FluidAudioCLI

struct BatchVocabularyThresholdTests {
    @Test func batchRescoringHonorsTheVocabularyFloor() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(#"{"terms":[{"text":"Rentvine"}],"minSimilarity":0.8}"#.utf8).write(to: url)
        let vocabulary = try CustomVocabularyContext.load(from: url)
        var args = TranscribeCommand.ParsedArgs()

        let defaults = args.batchRescoreParameters(for: vocabulary)
        #expect(defaults.minSimilarity == 0.8)
        #expect(defaults.marginSeconds == ContextBiasingConstants.defaultMarginSeconds)

        args.vocabMinSimilarity = 0.65
        #expect(args.batchRescoreParameters(for: vocabulary).minSimilarity == 0.8)

        args.vocabMinSimilarity = 0.9
        #expect(args.batchRescoreParameters(for: vocabulary).minSimilarity == 0.9)
    }
}
