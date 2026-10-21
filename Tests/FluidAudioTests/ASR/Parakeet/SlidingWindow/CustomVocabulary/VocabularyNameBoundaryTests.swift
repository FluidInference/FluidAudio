import Foundation
import XCTest

@testable import FluidAudio

@MainActor
final class VocabularyNameBoundaryTests: XCTestCase {
    private struct Fixture: Decodable {
        let text: String
        let tokenTimings: [TokenTiming]
    }

    // The recorded v2 name occupies two timing words because of its hyphen.
    // The old three-word matcher includes the following verb in the correction.
    func testFullNameCorrectionPreservesEverySurroundingWordAcrossNameBoundaries() async throws {
        let directory = CtcModels.defaultCacheDirectory()
        guard CtcModels.modelsExist(at: directory) else {
            throw XCTSkip("Install CTC 110M for name boundary regressions")
        }
        let models = try await CtcModels.loadDirect(from: directory)
        let spotter = CtcKeywordSpotter(models: models)
        let tokenizer = try await CtcTokenizer.load(from: directory)
        let root = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
            .appendingPathComponent("Vocabulary")
        let fixture = try JSONDecoder().decode(
            Fixture.self, from: Data(contentsOf: root.appendingPathComponent("tdt-v2-solanki.json")))
        let audio = try AudioConverter().resampleAudioFile(root.appendingPathComponent("solanki.wav"))
        let probabilities = try await spotter.computeLogProbs(for: audio)
        let canonical = "Chandra Shekhar Solanki"
        let originalName = "Chandra Shankar-Solansky"
        let expected = fixture.text.replacingOccurrences(of: originalName, with: canonical)
        let hyphen = try XCTUnwrap(fixture.tokenTimings.firstIndex { $0.token == "-" })
        let nameStart = try XCTUnwrap(fixture.tokenTimings.firstIndex { $0.token == " Ch" })
        let middleName = try XCTUnwrap(fixture.tokenTimings[(nameStart + 1)..<hyphen].firstIndex { $0.token == " S" })
        let surname = hyphen + 1
        let split = try XCTUnwrap(fixture.tokenTimings[middleName..<hyphen].firstIndex { $0.token == "ar" })
        for variant in ["recorded", "joined", "ordinary", "split", "alias", "joined-tail"] {
            let variantName: String
            switch variant {
            case "joined", "joined-tail": variantName = "Chandra-Shankar-Solansky"
            case "ordinary": variantName = "Chandra Shankar Solansky"
            case "split": variantName = "Chandra Shank ar Solansky"
            default: variantName = originalName
            }
            // Controlled spelling/segmentation changes only. The human PCM and
            // all surviving timing intervals stay as recorded, not new ASR samples.
            let timings = fixture.tokenTimings.enumerated().compactMap { index, timing -> TokenTiming? in
                if ["ordinary", "split"].contains(variant), index == hyphen { return nil }
                var token = timing.token
                if ["joined", "joined-tail"].contains(variant), index == middleName { token = "-S" }
                if variant == "joined-tail", timing.token == " said" { token = "-said" }
                if ["ordinary", "split"].contains(variant), index == surname { token = " " + token }
                if variant == "split", index == split { token = " " + token }
                return TokenTiming(
                    token: token, tokenId: timing.tokenId, startTime: timing.startTime,
                    endTime: timing.endTime, confidence: timing.confidence)
            }
            let vocabulary = CustomVocabularyContext(terms: [
                CustomVocabularyTerm(
                    text: canonical, aliases: variant == "alias" ? [originalName] : nil,
                    ctcTokenIds: tokenizer.encode(canonical)),
                CustomVocabularyTerm(text: "Solanki", ctcTokenIds: tokenizer.encode("Solanki")),
                CustomVocabularyTerm(text: "Quilter", ctcTokenIds: tokenizer.encode("Quilter")),
            ])
            let rescorer = try await VocabularyRescorer.create(
                spotter: spotter, vocabulary: vocabulary,
                config: .init(shortTermCbwTaperPivot: 5, shortTermCbwTaperExponent: 2, spotterRescueEnabled: false),
                ctcModelDirectory: directory)
            var transcript = fixture.text.replacingOccurrences(of: originalName, with: variantName)
            if variant == "joined-tail" {
                transcript = transcript.replacingOccurrences(of: variantName + " said", with: variantName + "-said")
            }
            let output = rescorer.ctcTokenRescore(
                transcript: transcript,
                tokenTimings: timings, logProbs: probabilities.logProbs, frameDuration: probabilities.frameDuration,
                cbw: 4.5, marginSeconds: 0.5)
            if variant == "joined-tail" {
                // A name plus an ordinary word is not one complete name span.
                // Leave this unsupported partial-word correction intact.
                XCTAssertFalse(output.wasModified, variant)
                XCTAssertEqual(output.text, transcript, variant)
            } else {
                XCTAssertTrue(
                    rescorer.hasCTCRescoringCandidates(transcript: transcript, tokenTimings: timings), variant)
                XCTAssertTrue(output.wasModified, variant)
                XCTAssertEqual(output.text, expected, variant)
            }
        }
    }
}
