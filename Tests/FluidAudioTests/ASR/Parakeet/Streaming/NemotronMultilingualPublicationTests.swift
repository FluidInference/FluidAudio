import Foundation
import os
import XCTest

@testable import FluidAudio

final class NemotronMultilingualPublicationTests: XCTestCase {
    private typealias Manager = StreamingNemotronMultilingualAsrManager
    private let langTag = 1
    private let hello = 2
    private let period = 3
    private let world = 4
    private let later = 5
    private let boundary = 6

    func testRescueInsertionBeforeAccumulatedPunctuationPublishesOnlyExtensions() throws {
        let tokenizer = try makeTokenizer()
        let ids = [langTag, hello, period, later]
        let timings = [timing(hello, 5), timing(period, 16), timing(later, 30)]
        let before = Manager.partialPublicationTokenIds(
            liveIds: ids, liveTimings: timings, langTagTokenIds: [langTag],
            openSpan: (14, 15, 3, false), nextSpanStartFrame: 31)
        let merged = Manager.mergeRescuedTokens(
            liveIds: ids, liveTimings: timings,
            rescuedIds: [world], rescuedTimings: [timing(world, 13)],
            langTagTokenIds: [langTag], spanStartSec: 0.88)
        let after = Manager.partialPublicationTokenIds(
            liveIds: merged.ids, liveTimings: merged.timings, langTagTokenIds: [langTag],
            openSpan: nil, nextSpanStartFrame: 35)
        let publications = [tokenizer.decode(ids: before).text, tokenizer.decode(ids: after).text]

        XCTAssertEqual(publications, ["Hallo", "Hallo Welt. Später"])
        assertPrefixExtensions(publications)
        XCTAssertEqual(merged.ids, [langTag, hello, world, period, later])
        XCTAssertEqual(merged.timings.map(\.tokenId), [hello, world, period, later])
        XCTAssertEqual(ids, [langTag, hello, period, later])
    }

    func testUnresolvedSpanAcrossChunksKeepsItsOriginalInsertionFrontier() throws {
        let tokenizer = try makeTokenizer()
        let firstIds = [langTag, hello, period]
        let firstTimings = [timing(hello, 5), timing(period, 17)]
        let first = Manager.partialPublicationTokenIds(
            liveIds: firstIds, liveTimings: firstTimings, langTagTokenIds: [langTag],
            openSpan: (14, 17, 3, false), nextSpanStartFrame: 18)
        let secondIds = firstIds + [boundary]
        let secondTimings = firstTimings + [timing(boundary, 27)]
        let second = Manager.partialPublicationTokenIds(
            liveIds: secondIds, liveTimings: secondTimings, langTagTokenIds: [langTag],
            openSpan: (14, 27, 3, false), nextSpanStartFrame: 28)
        let merged = Manager.mergeRescuedTokens(
            liveIds: secondIds, liveTimings: secondTimings,
            rescuedIds: [world], rescuedTimings: [timing(world, 14)],
            langTagTokenIds: [langTag], spanStartSec: 0.88)
        let resolved = Manager.partialPublicationTokenIds(
            liveIds: merged.ids, liveTimings: merged.timings, langTagTokenIds: [langTag],
            openSpan: nil, nextSpanStartFrame: 30)
        let publications = [first, second, resolved].map { tokenizer.decode(ids: $0).text }

        XCTAssertEqual(publications, ["Hallo", "Hallo", "Hallo Welt."])
        assertPrefixExtensions(publications)
        XCTAssertEqual(merged.ids, [langTag, hello, world, period, boundary])
    }

    func testSilentPreRollProtectsAgainstARescueOpeningInTheNextChunk() throws {
        let tokenizer = try makeTokenizer()
        let ids = [langTag, hello, period]
        let timings = [timing(hello, 5), timing(period, 19)]
        // The last three silent frames of this chunk become a future span's
        // pre-roll. That rescue can emit before the punctuation at frame 19.
        let silentTail = Manager.partialPublicationTokenIds(
            liveIds: ids, liveTimings: timings, langTagTokenIds: [langTag],
            openSpan: nil, nextSpanStartFrame: 17)
        let open = Manager.partialPublicationTokenIds(
            liveIds: ids, liveTimings: timings, langTagTokenIds: [langTag],
            openSpan: (20, 22, 3, false), nextSpanStartFrame: 23)
        let merged = Manager.mergeRescuedTokens(
            liveIds: ids, liveTimings: timings,
            rescuedIds: [world], rescuedTimings: [timing(world, 18)],
            langTagTokenIds: [langTag], spanStartSec: 1.36)
        let resolved = Manager.partialPublicationTokenIds(
            liveIds: merged.ids, liveTimings: merged.timings, langTagTokenIds: [langTag],
            openSpan: nil, nextSpanStartFrame: 30)
        let publications = [silentTail, open, resolved].map { tokenizer.decode(ids: $0).text }

        XCTAssertEqual(publications, ["Hallo", "Hallo", "Hallo Welt."])
        assertPrefixExtensions(publications)
    }

    func testFrontierUsesStrictTimestampOrderingAndSkipsUntimedLanguageTags() {
        let ids = [langTag, hello, langTag, period, boundary]
        let timings = [timing(hello, 5), timing(period, 11), timing(boundary, 12)]
        let published = Manager.partialPublicationTokenIds(
            liveIds: ids, liveTimings: timings, langTagTokenIds: [langTag],
            openSpan: (14, 15, 3, false), nextSpanStartFrame: 16)

        // mergeRescuedTokens inserts BEFORE the first strictly later timing;
        // a token exactly at the earliest rescue key cannot be displaced.
        XCTAssertEqual(published, [langTag, hello, langTag, period])
        XCTAssertEqual(ids.count, 5)
        XCTAssertEqual(timings.count, 3)
    }

    func testLexicalEmissionOrOverflowReleasesAnOpenSpanWithoutWaitingForSilence() {
        let ids = [langTag, hello, world, period]
        let timings = [timing(hello, 5), timing(world, 14), timing(period, 17)]
        let lexical = Manager.partialPublicationTokenIds(
            liveIds: ids, liveTimings: timings, langTagTokenIds: [langTag],
            openSpan: (14, 17, 3, false), nextSpanStartFrame: 18)
        let overflowed = Manager.partialPublicationTokenIds(
            liveIds: [langTag, hello, period], liveTimings: [timing(hello, 5), timing(period, 17)],
            langTagTokenIds: [langTag], openSpan: (14, 200, 3, true), nextSpanStartFrame: 201)

        XCTAssertEqual(lexical, ids)
        XCTAssertEqual(overflowed, [langTag, hello, period])
    }

    func testAbandonedSpanReleasesWithheldTextWithoutChangingFullPollingOutput() async throws {
        let tokenizer = try makeTokenizer()
        let manager = Manager()
        let ids = [hello, period]
        let timings = [timing(hello, 5), timing(period, 17)]
        let updates = OSAllocatedUnfairLock<[String]>(initialState: [])
        await manager.setPartialCallback { text in updates.withLock { $0.append(text) } }
        await manager.setPublicationState(tokenizer: tokenizer, ids: ids, timings: timings, openSpanStart: 14)
        await manager.publishCurrentState()
        let fullBefore = await manager.getPartialTranscript()
        let timingsBefore = await manager.getTokenTimings()

        // One speech window is below the rescue minimum: finish's existing
        // hook abandons it without any model call, then must flush the suffix.
        try await manager.finalizeRescueSpanIfNeeded()
        let fullAfter = await manager.getPartialTranscript()
        let timingsAfter = await manager.getTokenTimings()

        XCTAssertEqual(updates.withLock { $0 }, ["Hallo", "Hallo."])
        assertPrefixExtensions(updates.withLock { $0 })
        XCTAssertEqual(fullBefore, tokenizer.decode(ids: ids).text)
        XCTAssertEqual(fullAfter, fullBefore)
        XCTAssertEqual(timingsBefore.map(\.tokenId), ids)
        XCTAssertEqual(timingsAfter.map(\.startTime), timingsBefore.map(\.startTime))
    }

    func testFinishHookPublishesTheFullRescuedAccumulatorEvenWithoutAnOpenSpan() async throws {
        let tokenizer = try makeTokenizer()
        let manager = Manager()
        let merged = Manager.mergeRescuedTokens(
            liveIds: [hello, period], liveTimings: [timing(hello, 5), timing(period, 17)],
            rescuedIds: [world], rescuedTimings: [timing(world, 14)],
            langTagTokenIds: [], spanStartSec: 0.88)
        let updates = OSAllocatedUnfairLock<[String]>(initialState: [])
        await manager.setPartialCallback { text in updates.withLock { $0.append(text) } }
        await manager.setPublicationState(tokenizer: tokenizer, ids: merged.ids, timings: merged.timings)
        let full = await manager.getPartialTranscript()
        try await manager.finalizeRescueSpanIfNeeded()
        let idsAfter = await manager.accumulatedTokenIds

        XCTAssertEqual(full, "Hallo Welt.")
        XCTAssertEqual(updates.withLock { $0 }, [full])
        XCTAssertEqual(idsAfter, merged.ids)
        // finish() decodes these same full IDs immediately after this hook.
        XCTAssertEqual(tokenizer.decode(ids: idsAfter).text, full)
    }

    func testPlainTokenAppendsAreScalarPrefixMonotonicAfterWhitespaceNormalization() throws {
        let tokenizer = try makeTokenizer()
        var ids: [Int] = []
        var publications: [String] = []
        for id in [langTag, boundary, hello, boundary, boundary, world, period, boundary, 7, 8] {
            ids.append(id)
            publications.append(tokenizer.decode(ids: ids).text)
        }
        assertPrefixExtensions(publications)
        XCTAssertEqual(publications.last, "Hallo Welt. कि")
    }

    func testCallbackInstalledDuringRescueCannotPublishTrialTokens() async throws {
        let manager = Manager()
        let tokenizer = try makeTokenizer()
        let originalUpdates = OSAllocatedUnfairLock<[String]>(initialState: [])
        let replacementUpdates = OSAllocatedUnfairLock<[String]>(initialState: [])
        await manager.setPartialCallback { text in originalUpdates.withLock { $0.append(text) } }
        await manager.setPublicationState(
            tokenizer: tokenizer, ids: [hello, world], timings: [timing(hello, 5), timing(world, 14)])
        await manager.setRescueDecodeActive(true)

        // The actor may accept a new callback at any decode await. Neither
        // ordinary nor final publication may expose the trial accumulator.
        await manager.setPartialCallback { text in replacementUpdates.withLock { $0.append(text) } }
        await manager.publishCurrentState()
        await manager.publishCurrentState(isFinal: true)
        XCTAssertEqual(originalUpdates.withLock { $0 }, [])
        XCTAssertEqual(replacementUpdates.withLock { $0 }, [])

        await manager.setPublicationState(
            tokenizer: tokenizer, ids: [hello, period], timings: [timing(hello, 5), timing(period, 17)])
        await manager.setRescueDecodeActive(false)
        await manager.publishCurrentState()
        await manager.publishCurrentState()
        XCTAssertEqual(originalUpdates.withLock { $0 }, [])
        XCTAssertEqual(replacementUpdates.withLock { $0 }, ["Hallo.", "Hallo."])
    }

    func testNestedSuppressionKeepsTheCurrentCallbackUntilTheChunkSettles() async throws {
        let manager = Manager()
        let tokenizer = try makeTokenizer()
        let originalUpdates = OSAllocatedUnfairLock<[String]>(initialState: [])
        let replacementUpdates = OSAllocatedUnfairLock<[String]>(initialState: [])
        await manager.setPartialCallback { text in originalUpdates.withLock { $0.append(text) } }
        await manager.setPublicationState(
            tokenizer: tokenizer, ids: [hello, period], timings: [timing(hello, 5), timing(period, 17)])

        // Model-free state fixtures use the same counter and publication
        // method as the tracked chunk and its nested rescue trial decode.
        await manager.adjustPublicationSuppressionDepth(by: 1)
        await manager.setPartialCallback { text in replacementUpdates.withLock { $0.append(text) } }
        await manager.publishCurrentState()
        await manager.adjustPublicationSuppressionDepth(by: 1)
        await manager.publishCurrentState(isFinal: true)
        await manager.adjustPublicationSuppressionDepth(by: -1)
        await manager.publishCurrentState()
        XCTAssertEqual(originalUpdates.withLock { $0 }, [])
        XCTAssertEqual(replacementUpdates.withLock { $0 }, [])

        await manager.adjustPublicationSuppressionDepth(by: -1)
        await manager.publishCurrentState()
        await manager.publishCurrentState(isFinal: true)
        let depth = await manager.partialPublicationSuppressionDepth
        XCTAssertEqual(depth, 0)
        XCTAssertEqual(originalUpdates.withLock { $0 }, [])
        XCTAssertEqual(replacementUpdates.withLock { $0 }, ["Hallo.", "Hallo."])
    }

    private func timing(_ id: Int, _ frame: Int) -> TokenTiming {
        let pieces = [hello: "▁Hallo", period: ".", world: "▁Welt", later: "▁Später", boundary: "▁"]
        let time = Double(frame) * ASRConstants.secondsPerEncoderFrame
        return TokenTiming(
            token: pieces[id] ?? "", tokenId: id, startTime: time,
            endTime: time + ASRConstants.secondsPerEncoderFrame, confidence: 1)
    }

    private func makeTokenizer() throws -> NemotronMultilingualTokenizer {
        // Same flat vocabulary fixture pattern as NemotronMultilingualTests;
        // these tests exercise real tokenizer code, never models or audio.
        let vocab = [
            "1": "<de-DE>", "2": "▁Hallo", "3": ".", "4": "▁Welt", "5": "▁Später", "6": "▁",
            "7": "क", "8": "ि",
        ]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("publication-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: vocab).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        return try NemotronMultilingualTokenizer(vocabPath: url, langTagTokenIds: [langTag])
    }

    private func assertPrefixExtensions(_ texts: [String], file: StaticString = #filePath, line: UInt = #line) {
        for (previous, current) in zip(texts, texts.dropFirst()) {
            XCTAssertTrue(
                current.unicodeScalars.starts(with: previous.unicodeScalars),
                "\(String(reflecting: current)) revised \(String(reflecting: previous))", file: file, line: line)
        }
    }
}

extension StreamingNemotronMultilingualAsrManager {
    fileprivate func setPublicationState(
        tokenizer: NemotronMultilingualTokenizer, ids: [Int], timings: [TokenTiming], openSpanStart: Int? = nil
    ) {
        self.tokenizer = tokenizer
        accumulatedTokenIds = ids
        accumulatedTokenTimings = timings
        rescueFrameCursor = 28
        rescueSpanOpen = openSpanStart != nil
        rescueSpanStartFrame = openSpanStart ?? 0
        rescueSpanLastSpeechFrame = openSpanStart ?? 0
        rescueSpanPreRollFrames = 3
        rescueSpanSpeechWindows = openSpanStart == nil ? 0 : 1
    }

    fileprivate func setRescueDecodeActive(_ active: Bool) {
        inBlankRescue = active
    }

    fileprivate func adjustPublicationSuppressionDepth(by delta: Int) {
        partialPublicationSuppressionDepth += delta
    }

    fileprivate func publishCurrentState(isFinal: Bool = false) {
        publishPartialTranscript(using: partialCallback, isFinal: isFinal)
    }
}
