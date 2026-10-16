import Foundation
import XCTest

@testable import FluidAudio

@MainActor
final class VocabularySpanPreservationTests: XCTestCase {
    private struct Fixture: Decodable {
        let id: String
        let text: String
        let tokenTimings: [TokenTiming]
    }

    func testRecognizedPrefixesSurviveLongerCorrectionsWithoutConsumingOtherWords() {
        let words = ["John", "Collar", "met", "mister", "Quilter’s", "and", "Birkett", "Foster’s"]
        let timings = VocabularyRescorer.buildWordTimings(
            from: words.enumerated().map { index, word in
                TokenTiming(
                    token: "▁" + word, tokenId: index, startTime: Double(index),
                    endTime: Double(index + 1), confidence: 1)
            })
        let spans = VocabularyRescorer.recognizedVocabularySpans(
            in: timings, canonicalTerms: ["John", "Quilter", "Birkett Foster"].map { CustomVocabularyTerm(text: $0) })
        func candidate(_ phrase: String, term: String, range: Range<Int>) -> VocabularyRescorer.CTCMatchCandidate {
            .init(
                origin: .termCentricMultiWord, originalPhrase: phrase, vocabTerm: term,
                matchedAlias: nil, vocabTokens: [], similarity: 0.8, spanLength: range.count,
                spanIndices: Array(range), tokenRange: range,
                spanStartTime: Double(range.lowerBound), spanEndTime: Double(range.upperBound))
        }
        XCTAssertTrue(
            VocabularyRescorer.preservesRecognizedSpans(
                candidate("John Collar", term: "John Collier", range: 0..<2),
                recognizedSpans: spans, wordTimings: timings))
        XCTAssertFalse(
            VocabularyRescorer.preservesRecognizedSpans(
                candidate("mister Quilter’s", term: "Quilter", range: 3..<5),
                recognizedSpans: spans, wordTimings: timings))
        XCTAssertFalse(
            VocabularyRescorer.preservesRecognizedSpans(
                candidate("Birkett Foster’s", term: "Foster", range: 6..<8),
                recognizedSpans: spans, wordTimings: timings))
    }

    // Real CTC inference over recorded speech, with fixed TDT output so decoder
    // changes cannot conceal a rescorer regression. Never downloads models.
    func testRecognizedNamesKeepNeighboringWordsAndPossessives() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        guard CtcModels.modelsExist(at: directory) else { throw XCTSkip("Install CTC 110M for vocabulary regressions") }
        let models = try await CtcModels.loadDirect(from: directory)
        let spotter = CtcKeywordSpotter(models: models)
        let tokenizer = try await CtcTokenizer.load(from: directory)
        let root = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
            .appendingPathComponent("Vocabulary")
        let fixtures = try JSONDecoder().decode(
            [Fixture].self, from: Data(contentsOf: root.appendingPathComponent("tdt-v3.json")))
        for fixture in fixtures where fixture.id != "solanki" {
            let audio = try AudioConverter().resampleAudioFile(root.appendingPathComponent(fixture.id + ".wav"))
            let probabilities = try await spotter.computeLogProbs(for: audio)
            let vocabulary = CustomVocabularyContext(terms: [
                CustomVocabularyTerm(text: "Quilter", ctcTokenIds: tokenizer.encode("Quilter"))
            ])
            for config in [VocabularyRescorer.Config(spotterRescueEnabled: false), taperedConfig] {
                let rescorer = try await VocabularyRescorer.create(
                    spotter: spotter, vocabulary: vocabulary, config: config, ctcModelDirectory: directory)
                let result = rescorer.ctcTokenRescore(
                    transcript: fixture.text, tokenTimings: fixture.tokenTimings,
                    logProbs: probabilities.logProbs, frameDuration: probabilities.frameDuration,
                    cbw: 4.5, marginSeconds: 0.5)
                XCTAssertEqual(result.text, fixture.text, fixture.id)
                XCTAssertFalse(result.wasModified, fixture.id)
            }
        }
    }

    func testLegitimateNameCorrectionsRemainAvailable() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        guard CtcModels.modelsExist(at: directory) else { throw XCTSkip("Install CTC 110M for vocabulary regressions") }
        let models = try await CtcModels.loadDirect(from: directory)
        let spotter = CtcKeywordSpotter(models: models)
        let tokenizer = try await CtcTokenizer.load(from: directory)
        let root = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
            .appendingPathComponent("Vocabulary")
        let fixtures = try JSONDecoder().decode(
            [Fixture].self, from: Data(contentsOf: root.appendingPathComponent("tdt-v3.json")))
        let fixture = try XCTUnwrap(fixtures.first { $0.id == "solanki" })
        let audio = try AudioConverter().resampleAudioFile(root.appendingPathComponent("solanki.wav"))
        let probabilities = try await spotter.computeLogProbs(for: audio)
        let cases: [([String], String)] = [
            (["Solanki"], fixture.text.replacingOccurrences(of: "Solansky", with: "Solanki")),
            // The listed prefix is already correct and must survive inside
            // the longer, legitimate full-name correction.
            (
                ["Superintendent", "Superintendent Chandra Shekhar Solanki"],
                fixture.text.replacingOccurrences(of: "Shanra Shankhar Solansky", with: "Chandra Shekhar Solanki")
            ),
        ]
        for (terms, expected) in cases {
            let vocabulary = CustomVocabularyContext(
                terms: terms.map {
                    CustomVocabularyTerm(text: $0, ctcTokenIds: tokenizer.encode($0))
                })
            let rescorer = try await VocabularyRescorer.create(
                spotter: spotter, vocabulary: vocabulary, config: taperedConfig, ctcModelDirectory: directory)
            let result = rescorer.ctcTokenRescore(
                transcript: fixture.text, tokenTimings: fixture.tokenTimings,
                logProbs: probabilities.logProbs, frameDuration: probabilities.frameDuration,
                cbw: 4.5, marginSeconds: 0.5)
            XCTAssertTrue(result.wasModified)
            XCTAssertEqual(result.text, expected)
        }
    }

    func testMisspelledAndSplitNamesAreCorrectedWithoutBorrowingTheirEvidenceTwice() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        guard CtcModels.modelsExist(at: directory) else { throw XCTSkip("Install CTC 110M for vocabulary regressions") }
        let models = try await CtcModels.loadDirect(from: directory)
        let spotter = CtcKeywordSpotter(models: models)
        let tokenizer = try await CtcTokenizer.load(from: directory)
        let root = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
            .appendingPathComponent("Vocabulary")
        let fixtures = try JSONDecoder().decode(
            [Fixture].self, from: Data(contentsOf: root.appendingPathComponent("tdt-v3.json")))
        for fixture in fixtures where fixture.id != "solanki" {
            let audio = try AudioConverter().resampleAudioFile(root.appendingPathComponent(fixture.id + ".wav"))
            let probabilities = try await spotter.computeLogProbs(for: audio)
            // Perturb the spelling/segmentation of the recorded TDT tokens only;
            // the human recording and its timestamps remain unchanged.
            for variant in ["misspelled", "alias", "split", "curly"] {
                let text = fixture.text.replacingOccurrences(
                    of: "Quilter", with: variant == "split" ? "Quil ter" : "Quiltor"
                )
                .replacingOccurrences(of: "'", with: variant == "curly" ? "’" : "'")
                let nameTokenIndex = try XCTUnwrap(fixture.tokenTimings.firstIndex { $0.token == "ter" })
                let timings = fixture.tokenTimings.enumerated().map { index, timing in
                    let token =
                        index == nameTokenIndex
                        ? (variant == "split" ? " ter" : "tor")
                        : timing.token.replacingOccurrences(of: "'", with: variant == "curly" ? "’" : "'")
                    return TokenTiming(
                        token: token, tokenId: timing.tokenId, startTime: timing.startTime,
                        endTime: timing.endTime, confidence: timing.confidence)
                }
                let vocabulary = CustomVocabularyContext(terms: [
                    CustomVocabularyTerm(
                        text: "Quilter", aliases: variant == "alias" ? ["Quiltor"] : nil,
                        ctcTokenIds: tokenizer.encode("Quilter"))
                ])
                let rescorer = try await VocabularyRescorer.create(
                    spotter: spotter, vocabulary: vocabulary, config: taperedConfig, ctcModelDirectory: directory)
                let output = rescorer.ctcTokenRescore(
                    transcript: text, tokenTimings: timings,
                    logProbs: probabilities.logProbs, frameDuration: probabilities.frameDuration,
                    cbw: 4.5, marginSeconds: 0.5)
                XCTAssertTrue(output.wasModified, "\(fixture.id)/\(variant)")
                XCTAssertEqual(
                    output.text, fixture.text.replacingOccurrences(of: "'", with: variant == "curly" ? "’" : "'"),
                    "\(fixture.id)/\(variant)")
            }
        }
    }

    func testRepeatedNamesWithNormalAndCoarseTimingsRetainTheLegitimateCorrection() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        guard CtcModels.modelsExist(at: directory) else { throw XCTSkip("Install CTC 110M for vocabulary regressions") }
        let models = try await CtcModels.loadDirect(from: directory)
        let spotter = CtcKeywordSpotter(models: models)
        let tokenizer = try await CtcTokenizer.load(from: directory)
        let root = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
            .appendingPathComponent("Vocabulary")
        let first = try AudioConverter().resampleAudioFile(root.appendingPathComponent("quilter.wav"))
        let second = try AudioConverter().resampleAudioFile(root.appendingPathComponent("quilter-possessive.wav"))
        // Derived human speech: retain the complete name spans from the two
        // recordings using the stored TDT boundaries, then join the PCM samples.
        // This is an edited repetition control, not a natural continuous utterance.
        let firstName = Array(first[Int(0.56 * 16_000)..<Int(1.28 * 16_000)])
        let secondName = Array(second[Int(1.12 * 16_000)..<Int(1.92 * 16_000)])
        let audio = firstName + secondName
        let duration = Double(audio.count) / 16_000
        let boundary = Double(firstName.count) / 16_000
        let probabilities = try await spotter.computeLogProbs(for: audio)
        let vocabulary = CustomVocabularyContext(terms: [
            CustomVocabularyTerm(text: "Quilter", ctcTokenIds: tokenizer.encode("Quilter"))
        ])
        let rescorer = try await VocabularyRescorer.create(
            spotter: spotter, vocabulary: vocabulary, config: taperedConfig, ctcModelDirectory: directory)
        for coarse in [false, true] {
            let timings = [
                TokenTiming(
                    token: "▁Quilter", tokenId: 1, startTime: 0, endTime: coarse ? duration : boundary, confidence: 1),
                TokenTiming(
                    token: "▁Quiltor's", tokenId: 2, startTime: coarse ? 0 : boundary, endTime: duration, confidence: 1),
            ]
            let result = rescorer.ctcTokenRescore(
                transcript: "Quilter Quiltor's", tokenTimings: timings,
                logProbs: probabilities.logProbs, frameDuration: probabilities.frameDuration, cbw: 4.5,
                marginSeconds: 0.5)
            XCTAssertEqual(result.text, "Quilter Quilter's", "coarse=\(coarse)")
            XCTAssertTrue(result.wasModified, "coarse=\(coarse)")
        }
    }

    private var taperedConfig: VocabularyRescorer.Config {
        .init(shortTermCbwTaperPivot: 5, shortTermCbwTaperExponent: 2, spotterRescueEnabled: false)
    }
}
