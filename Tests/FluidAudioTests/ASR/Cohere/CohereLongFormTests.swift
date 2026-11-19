import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 14, iOS 17, *)
final class CohereLongFormTests: XCTestCase {

    // MARK: - Config

    func testChunkOverlapMatchesUpstream() {
        // Matches `overlap_chunk_second: 5` in upstream cohere-pytorch config.
        XCTAssertEqual(CohereAsrConfig.chunkOverlapSeconds, 5.0)
    }

    func testChunkHopIsMaxMinusOverlap() {
        XCTAssertEqual(
            CohereAsrConfig.chunkHopSeconds,
            CohereAsrConfig.maxAudioSeconds - CohereAsrConfig.chunkOverlapSeconds)
        XCTAssertEqual(CohereAsrConfig.chunkHopSeconds, 30.0)
    }

    // MARK: - mergeTokenStreams

    func testMergePrefixEmpty() {
        let merged = CoherePipeline.mergeTokenStreams(prefix: [], suffix: [1, 2, 3])
        XCTAssertEqual(merged, [1, 2, 3])
    }

    func testMergeSuffixEmpty() {
        let merged = CoherePipeline.mergeTokenStreams(prefix: [1, 2, 3], suffix: [])
        XCTAssertEqual(merged, [1, 2, 3])
    }

    func testMergeNoCommonRunFallsBackToConcat() {
        let merged = CoherePipeline.mergeTokenStreams(
            prefix: [10, 11, 12, 13],
            suffix: [20, 21, 22, 23])
        XCTAssertEqual(merged, [10, 11, 12, 13, 20, 21, 22, 23])
    }

    func testMergeShortMatchBelowThresholdFallsBackToConcat() {
        // A common run of 3 tokens is below the default minMatch=4 threshold.
        let merged = CoherePipeline.mergeTokenStreams(
            prefix: [1, 2, 3, 7, 8, 9],
            suffix: [7, 8, 9, 100, 200])
        XCTAssertEqual(merged, [1, 2, 3, 7, 8, 9, 7, 8, 9, 100, 200])
    }

    func testMergeOverlapAtBoundary() {
        // Prefix tail and suffix head share a 5-token run. Merge keeps one
        // copy of the run and continues with the suffix after it.
        let prefix = [1, 2, 3, 4, 50, 51, 52, 53, 54]
        let suffix = [50, 51, 52, 53, 54, 60, 61, 62]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 4, 50, 51, 52, 53, 54, 60, 61, 62])
    }

    func testMergeOverlapOffsetWithinWindow() {
        // The overlap doesn't have to align with the absolute boundary — the
        // merge uses an LCS over a bounded window, so a shifted match still
        // works.
        let prefix = [1, 2, 3, 90, 91, 92, 93, 94, 95]
        let suffix = [91, 92, 93, 94, 95, 200, 201]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 90, 91, 92, 93, 94, 95, 200, 201])
    }

    func testMergePrefersLongestRun() {
        // Two candidate matches: a 4-token run earlier and a 5-token run later.
        // The longer run wins.
        let prefix = [1, 2, 3, 4, 7, 8, 9, 10, 11]
        let suffix = [1, 2, 3, 4, 7, 8, 9, 10, 11, 99]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        // The 9-token tail of prefix matches the 9-token head of suffix → the
        // suffix tail '99' is the only thing kept after the prefix.
        XCTAssertEqual(merged, [1, 2, 3, 4, 7, 8, 9, 10, 11, 99])
    }

    func testMergeWindowBoundsLcsCost() {
        // 4-token threshold; the run lives entirely inside the merge window
        // (default 32 tokens). Verifies behavior when prefix is much longer
        // than the window.
        let leadIn = Array(0..<200)
        let prefix = leadIn + [500, 501, 502, 503, 504]
        let suffix = [500, 501, 502, 503, 504, 700, 701]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, leadIn + [500, 501, 502, 503, 504, 700, 701])
    }

    // MARK: - Seam cut after the matched run (#1013)

    func testMergeDropsPrefixTailAfterMatchedRun() {
        // Both windows transcribe the overlap audio. The prefix's tokens after
        // the matched run (70, 71) and the suffix's copy of that same audio
        // (70, 71) must not both survive — the seam is cut at the run's end
        // and the suffix carries on from there.
        let prefix = [1, 2, 3, 50, 51, 52, 53, 70, 71]
        let suffix = [50, 51, 52, 53, 70, 71, 80, 81]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 50, 51, 52, 53, 70, 71, 80, 81])
    }

    func testMergeDropsPrefixTailWhenSuffixDecodesItDifferently() {
        // The prefix ends with a different reading of the overlap tail (90,
        // 91) than the suffix (70, 71). Keeping the prefix verbatim would emit
        // the tail twice; the cut after the run keeps only the suffix's copy.
        let prefix = [1, 2, 3, 50, 51, 52, 53, 90, 91]
        let suffix = [50, 51, 52, 53, 70, 71, 80, 81]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 50, 51, 52, 53, 70, 71, 80, 81])
    }

    func testMergeRepeatedPhraseCutsAtLaterPrefixOccurrence() {
        // The same 4-token phrase occurs twice inside the prefix window. The
        // seam must be cut at the later occurrence so the text between the
        // two occurrences (10, 11, 12) is not lost.
        let prefix = [1, 2, 50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53]
        let suffix = [50, 51, 52, 53, 80, 81]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53, 80, 81])
    }

    func testMergeRepeatedPhraseKeepsEarlierSuffixOccurrence() {
        // The phrase repeats inside the suffix window. Cutting at the later
        // suffix occurrence would drop (10, 11, 12); the earlier one wins.
        let prefix = [1, 2, 3, 50, 51, 52, 53]
        let suffix = [50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53, 80]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53, 80])
    }

    func testMergeMatchInsideLongDenseWindows() {
        // Dense 35 s windows now decode ~180–250 tokens; the ~5 s overlap is
        // ~30 tokens, so the run can sit well inside a 48-token window.
        let body = Array(1000..<1180)
        let overlap = Array(2000..<2030)
        let prefix = body + overlap
        let suffix = overlap + Array(3000..<3150)
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, body + overlap + Array(3000..<3150))
    }

    func testMergeIgnoresChancePhraseFarFromTheSeam() {
        // A common 4-token phrase sits at the very start of the prefix window
        // and at the very end of the suffix window. Cutting there would drop
        // ~90 tokens; the cut distance exceeds the window, so the match is
        // treated as a chance repeat and the streams are concatenated.
        let phrase = [7, 8, 9, 10]
        let prefix = Array(0..<100) + phrase + Array(200..<244)
        let suffix = Array(300..<344) + phrase + Array(400..<460)
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, prefix + suffix)
    }

    // MARK: - Pause-aligned chunks (languages written without spaces, #1140)

    private let fixtures = [
        "01-validation-request-21.4s.wav",
        "02-release-readiness-19.8s.wav",
        "03-diff-explanation-16.9s.wav",
    ]

    func testPauseChunksFitTheEncoderWindow() {
        XCTAssertLessThanOrEqual(CohereAsrConfig.pauseChunkSeconds, CohereAsrConfig.maxAudioSeconds)
        XCTAssertLessThan(CohereAsrConfig.pauseSearchSeconds, CohereAsrConfig.pauseChunkSeconds)
        // Upstream `min_energy_window_samples`.
        XCTAssertEqual(CohereAsrConfig.pauseEnergyWindowSamples, 1_600)
    }

    func testOnlyJapaneseAndChineseWriteWithoutSpaces() {
        for language in CohereAsrConfig.Language.allCases {
            XCTAssertEqual(
                language.writesWithoutSpaces, language == .japanese || language == .chinese,
                language.rawValue)
        }
    }

    func testPauseChunksTileRealSpeechAndCutAtTheQuietestWindow() throws {
        // Three recordings back to back: ~58 s of speech, so the splitter has
        // to find pauses inside speech rather than at file boundaries.
        let audio = try fixtures.flatMap { try loadSamples(fixtureURL($0)) }
        let sr = Float(CohereAsrConfig.sampleRate)
        let maxChunk = Int(CohereAsrConfig.pauseChunkSeconds * sr)
        let search = Int(CohereAsrConfig.pauseSearchSeconds * sr)
        let window = CohereAsrConfig.pauseEnergyWindowSamples

        let chunks = CoherePipeline.pauseAlignedChunks(
            audio, maxChunkSamples: maxChunk, searchSamples: search, windowSamples: window)

        XCTAssertGreaterThanOrEqual(chunks.count, 3)
        XCTAssertEqual(chunks.first?.lowerBound, 0)
        XCTAssertEqual(chunks.last?.upperBound, audio.count)
        for (previous, next) in zip(chunks, chunks.dropFirst()) {
            XCTAssertEqual(previous.upperBound, next.lowerBound, "chunks must be contiguous")
        }
        for chunk in chunks {
            XCTAssertFalse(chunk.isEmpty)
            XCTAssertLessThanOrEqual(chunk.count, maxChunk)
        }
        for chunk in chunks.dropLast() {
            let limit = chunk.lowerBound + maxChunk
            let searchStart = limit - search
            XCTAssertGreaterThanOrEqual(chunk.upperBound, searchStart)
            XCTAssertLessThan(chunk.upperBound, limit)
            // The cut is the start of the lowest-energy window in the search.
            var lowest = Float.infinity
            var quietest = searchStart
            var offset = searchStart
            while offset + window < limit {
                let energy = audio[offset..<(offset + window)].reduce(0) { $0 + $1 * $1 }
                if energy < lowest {
                    lowest = energy
                    quietest = offset
                }
                offset += window
            }
            XCTAssertEqual(chunk.upperBound, quietest)
        }
    }

    func testAudioThatFitsOnePauseChunkStaysWhole() throws {
        let audio = try loadSamples(fixtureURL(fixtures[2]))  // 16.9 s
        let chunks = CoherePipeline.pauseAlignedChunks(
            audio, maxChunkSamples: 20 * 16_000, searchSamples: 10 * 16_000, windowSamples: 1_600)
        XCTAssertEqual(chunks, [0..<audio.count])
    }

    func testPauseSearchShorterThanOneWindowCutsAtItsMidpoint() throws {
        let audio = try loadSamples(fixtureURL(fixtures[0]))
        XCTAssertEqual(
            CoherePipeline.quietestWindowStart(audio, from: 16_000, to: 17_000, windowSamples: 1_600),
            16_500)
    }

    private func fixtureURL(_ name: String) throws -> URL {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else {
            throw XCTSkip("fixture \(name) missing from test bundle")
        }
        return url
    }

    /// Decode a fixture to 16 kHz mono Float32 samples.
    private func loadSamples(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        XCTAssertEqual(format.sampleRate, 16_000, "fixtures are 16 kHz")
        XCTAssertEqual(format.channelCount, 1, "fixtures are mono")
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))
        else {
            throw XCTSkip("could not allocate a buffer for \(url.lastPathComponent)")
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else {
            throw XCTSkip("fixture \(url.lastPathComponent) is not float PCM")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }
}
