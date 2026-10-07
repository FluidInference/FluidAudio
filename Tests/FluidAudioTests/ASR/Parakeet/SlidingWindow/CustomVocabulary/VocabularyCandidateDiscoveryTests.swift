import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

@MainActor
final class VocabularyCandidateDiscoveryTests: XCTestCase {
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
            if discovery { positive += 1 } else { negative += 1 }
            XCTAssertFalse(rescorer.hasCTCRescoringCandidates(transcript: transcript, tokenTimings: []))
            if threshold == 0.99 { XCTAssertFalse(discovery) }
        }
        XCTAssertGreaterThan(positive, 0)
        XCTAssertGreaterThan(negative, 0)
        let rescue = try await VocabularyRescorer.create(
            spotter: spotter,
            vocabulary: CustomVocabularyContext(terms: []), ctcModelDirectory: directory)
        XCTAssertTrue(
            rescue.hasCTCRescoringCandidates(transcript: "", tokenTimings: []),
            "Acoustic rescue cannot be skipped by a text-only probe")
    }
}
