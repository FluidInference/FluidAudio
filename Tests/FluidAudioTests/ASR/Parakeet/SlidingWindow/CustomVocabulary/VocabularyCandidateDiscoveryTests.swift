import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

@MainActor
final class VocabularyCandidateDiscoveryTests: XCTestCase {
    func testPreparedFormsPreserveAliasOrderingAndGuardSetWithoutModels() throws {
        let terms = [
            CustomVocabularyTerm(text: "ESLint", aliases: ["E S lint", "es lint"]),
            CustomVocabularyTerm(text: "eslint", aliases: ["E S LINT", "easylint"]),
            CustomVocabularyTerm(text: "Claude Code", aliases: ["cloud code", "", "!!!"]),
            CustomVocabularyTerm(text: "AI", aliases: nil),
        ]
        let forms = VocabularyRescorer.prepareNormalizedForms(for: CustomVocabularyContext(terms: terms))
        for term in terms {
            let aliases =
                terms.filter { $0.textLowercased == term.textLowercased }
                .flatMap { $0.aliases ?? [] } + (term.aliases ?? [])
            let expected = VocabularyRescorer.normalizedForms(canonicalTerm: term.text, aliases: aliases)
            XCTAssertEqual(try XCTUnwrap(forms[VocabularyRescorer.TermFormKey(term)]), expected)
        }
        let originalGuardSet = Set(
            terms.flatMap { [$0.text] + ($0.aliases ?? []) }
                .map(VocabularyRescorer.normalizeForSimilarity).filter { !$0.isEmpty })
        XCTAssertEqual(Set(forms.values.flatMap { $0.map(\.normalized) }), originalGuardSet)
        XCTAssertTrue(VocabularyRescorer.prepareNormalizedForms(for: CustomVocabularyContext(terms: [])).isEmpty)
    }

    func testDiscoveryMatchesExactScoringCandidatesWithInstalledTokenizerAndAcousticEvidence() async throws {
        let directory = CtcModels.defaultCacheDirectory(for: .ctc110m)
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path),
            let path = ProcessInfo.processInfo.environment["FLUID_AUDIO_VOCABULARY_AUDIO_PATH"]
        else {
            throw XCTSkip("Install CTC 110M and set FLUID_AUDIO_VOCABULARY_AUDIO_PATH to real 16 kHz mono speech")
        }
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        XCTAssertEqual(file.processingFormat.channelCount, 1)
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let pointer = try XCTUnwrap(buffer.floatChannelData?[0])
        let audio = Array(UnsafeBufferPointer(start: pointer, count: Int(buffer.frameLength)))
        let models = try await CtcModels.loadDirect(from: directory)
        let spotter = CtcKeywordSpotter(models: models)
        let probabilities = try await spotter.computeLogProbs(for: audio)
        XCTAssertFalse(probabilities.logProbs.isEmpty)
        let tokenizer = try await CtcTokenizer.load(from: directory)
        // Duplicate canonical spellings retain alias order and provenance.
        let aliasTerms = [
            CustomVocabularyTerm(text: "ESLint", aliases: ["E S lint", "es lint"]),
            CustomVocabularyTerm(text: "eslint", aliases: ["E S LINT", "easylint"]),
            CustomVocabularyTerm(text: "Claude Code", aliases: ["cloud code"]),
        ]
        let aliasContext = CustomVocabularyContext(terms: aliasTerms)
        let prepared = try await VocabularyRescorer.create(
            spotter: spotter, vocabulary: aliasContext,
            config: .init(spotterRescueEnabled: false), ctcModelDirectory: directory)
        for term in aliasTerms + [CustomVocabularyTerm(text: "ESLint", aliases: ["ee ess lint"])] {
            let allAliases =
                aliasTerms.filter { $0.textLowercased == term.textLowercased }
                .flatMap { $0.aliases ?? [] } + (term.aliases ?? [])
            let expected = VocabularyRescorer.normalizedForms(canonicalTerm: term.text, aliases: allAliases)
            XCTAssertEqual(prepared.buildNormalizedForms(for: term), expected)
        }
        let expectedSet = Set(
            aliasTerms.flatMap { [$0.text] + ($0.aliases ?? []) }
                .map(VocabularyRescorer.normalizeForSimilarity).filter { !$0.isEmpty })
        XCTAssertEqual(prepared.vocabularyNormalizedSet, expectedSet)

        let cases: [(String, String, [String], Float?)] = [
            ("quiltor", "Quilter", [], nil),
            ("Quilter", "Quilter", [], nil),
            ("completely unrelated ordinary speech", "Quilter", [], nil),
            ("cloud code", "Claude Code", ["cloud code"], nil),
            ("cloudcode", "Claude Code", ["cloud code"], nil),
            ("Chandra shaker Solanki", "Chandra Shekhar Solanki", [], nil),
            ("E S lint", "ESLint", ["E S lint"], nil),
            ("the", "Azure", [], 0.1),
            ("AI", "AI", [], nil),
            ("quiltor", "Quilter", [], 0.99),
            ("quiltor", "Quilter", [], 0.45),
            ("QUILTOR,", "Quilter", ["quiltor"], nil),
        ]
        var positive = 0
        var negative = 0
        for (transcript, term, aliases, threshold) in cases {
            let context = CustomVocabularyContext(terms: [
                CustomVocabularyTerm(
                    text: term, aliases: aliases, ctcTokenIds: tokenizer.encode(term), minSimilarity: threshold)
            ])
            let rescorer = try await VocabularyRescorer.create(
                spotter: spotter, vocabulary: context,
                config: .init(spotterRescueEnabled: false), ctcModelDirectory: directory)
            let words = transcript.split(separator: " ")
            let duration = Double(audio.count) / 16_000
            let timings = words.enumerated().map { index, word in
                TokenTiming(
                    token: "▁" + word, tokenId: index + 1,
                    startTime: duration * Double(index) / Double(words.count),
                    endTime: duration * Double(index + 1) / Double(words.count), confidence: 1)
            }
            let discovery = rescorer.hasCTCRescoringCandidates(transcript: transcript, tokenTimings: timings)
            let evidence = rescorer.ctcTokenEvaluateCandidates(
                transcript: transcript, tokenTimings: timings,
                logProbs: probabilities.logProbs, frameDuration: probabilities.frameDuration)
            XCTAssertEqual(discovery, !evidence.candidates.isEmpty, "\(transcript) -> \(term)")
            let session = try await VocabularyBoostingSession(
                vocabulary: context, ctcModels: models, config: .init(spotterRescueEnabled: false))
            let threshold = ContextBiasingConstants.rescorerConfig(forVocabSize: context.terms.count).minSimilarity
            XCTAssertEqual(
                session.hasCTCRescoringCandidates(text: transcript, tokenTimings: timings),
                rescorer.hasCTCRescoringCandidates(
                    transcript: transcript, tokenTimings: timings,
                    minSimilarity: max(threshold, context.minSimilarity)))
            if discovery { positive += 1 } else { negative += 1 }
            XCTAssertFalse(rescorer.hasCTCRescoringCandidates(transcript: transcript, tokenTimings: []))
            if threshold == 0.99 { XCTAssertFalse(discovery) }
        }
        XCTAssertGreaterThan(positive, 0)
        XCTAssertGreaterThan(negative, 0)
        let rescue = try await VocabularyRescorer.create(
            spotter: spotter,
            vocabulary: CustomVocabularyContext(terms: [CustomVocabularyTerm(text: "Quilter")]),
            ctcModelDirectory: directory)
        XCTAssertFalse(rescue.hasCTCRescoringCandidates(transcript: "", tokenTimings: []))
        let unrelatedTimings = [TokenTiming(token: "▁unrelated", tokenId: 1, startTime: 0, endTime: 1, confidence: 1)]
        XCTAssertTrue(
            rescue.hasCTCRescoringCandidates(transcript: "unrelated", tokenTimings: unrelatedTimings),
            "Acoustic rescue can require evidence without a text candidate")
        let empty = try await VocabularyRescorer.create(
            spotter: spotter, vocabulary: CustomVocabularyContext(terms: []), ctcModelDirectory: directory)
        XCTAssertFalse(empty.hasCTCRescoringCandidates(transcript: "unrelated", tokenTimings: unrelatedTimings))
    }
}
