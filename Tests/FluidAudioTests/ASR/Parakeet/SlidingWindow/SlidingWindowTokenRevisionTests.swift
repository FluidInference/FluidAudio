import AVFoundation
import Foundation
import XCTest
@testable import FluidAudio

final class SlidingWindowTokenRevisionTests: XCTestCase {
    private func token(_ piece: String, _ start: Double = 0) -> TokenTiming {
        TokenTiming(token: piece, tokenId: 1, startTime: start, endTime: start + 0.08, confidence: 0.99)
    }

    func testRetirementAndEmptyTerminalPreserveExplicitPositions() throws {
        var cursor = SlidingWindowTokenCursor()
        let first = try cursor.next(
            tokens: [token("▁We"), token("▁an", 0.2)],
            previous: 0, retained: 0, stable: 1, window: 0..<22400, center: 0..<16000, isConfirmed: true)
        let second = try cursor.next(
            tokens: [token("▁analyzing", 0.24), token("▁now.", 0.6)],
            previous: 2, retained: 1, stable: 2, window: 0..<38400, center: 16000..<32000)
        XCTAssertEqual(first.revision.ordinal, 1)
        XCTAssertTrue(first.isConfirmed)
        XCTAssertFalse(second.isConfirmed)
        XCTAssertEqual(second.revision.previousTokenCount, 2)
        XCTAssertEqual(second.revision.retainedTokenCount, 1)
        XCTAssertEqual(second.revision.jobID, first.revision.jobID)
        let final = try cursor.finish()
        XCTAssertTrue(final.tokens.isEmpty)
        XCTAssertEqual(final.revision.stableTokenCount, 3)
        XCTAssertTrue(final.revision.isTerminal)
        XCTAssertThrowsError(try cursor.finish())
    }

    func testInvalidPublicationCannotAdvanceCursorOrRetireStablePrefix() throws {
        var cursor = SlidingWindowTokenCursor()
        _ = try cursor.next(
            tokens: [token("▁yes"), token("▁yes", 0.1)],
            previous: 0, retained: 0, stable: 1, window: 0..<22400, center: 0..<16000)
        for (previous, retained, stable) in [(3, 2, 2), (2, 0, 1), (2, 2, 0), (2, 2, 3)] {
            XCTAssertThrowsError(
                try cursor.next(
                    tokens: [], previous: previous, retained: retained,
                    stable: stable, window: 0..<38400, center: 16000..<32000))
            XCTAssertEqual(cursor.ordinal, 1)
            XCTAssertEqual(cursor.count, 2)
        }
        _ = try cursor.finish()
    }

    func testEmptySessionAndIndependentJobIdentity() throws {
        var first = SlidingWindowTokenCursor()
        var second = SlidingWindowTokenCursor()
        let a = try first.finish()
        let b = try second.finish()
        XCTAssertNotEqual(a.revision.jobID, b.revision.jobID)
        XCTAssertEqual(a.revision.stableTokenCount, 0)
        XCTAssertEqual(a.revision.ordinal, 1)
        XCTAssertEqual(b.revision.ordinal, 1)
    }

    func testMalformedTokensAndRegressingRegionsFailAtomically() throws {
        var cursor = SlidingWindowTokenCursor()
        for tokens in [[token("x", .nan)], [token("x", 2), token("y", 1)]] {
            XCTAssertThrowsError(
                try cursor.next(
                    tokens: tokens, previous: 0, retained: 0, stable: 0,
                    window: 0..<22400, center: 0..<16000))
            XCTAssertEqual(cursor.ordinal, 0)
        }
        _ = try cursor.next(
            tokens: [], previous: 0, retained: 0, stable: 0,
            window: 0..<38400, center: 16000..<32000)
        XCTAssertThrowsError(
            try cursor.next(
                tokens: [], previous: 0, retained: 0, stable: 0,
                window: 0..<22400, center: 0..<16000))
        XCTAssertEqual(cursor.ordinal, 1)
    }

    /// Right context longer than the chunk: 1 s chunk, 2 s left and right, 3 s of
    /// audio. The regular window reads 0..<48000; the drain reads 0..<32000 while
    /// its center advances. That is valid lineage and still ends in one terminal.
    func testDrainMayReadLessRightContextWhileItsCenterAdvances() throws {
        var cursor = SlidingWindowTokenCursor()
        _ = try cursor.next(
            tokens: [], previous: 0, retained: 0, stable: 0, window: 0..<48000, center: 0..<16000)
        let drain = try cursor.next(
            tokens: [], previous: 0, retained: 0, stable: 0, window: 0..<32000, center: 16000..<32000)
        let terminal = try cursor.finish()

        XCTAssertEqual(drain.revision.window, 0..<32000)
        XCTAssertTrue(terminal.revision.isTerminal)
        XCTAssertEqual(terminal.revision.ordinal, 3)
    }

    func testRepeatedStartEndsTheStrictStreamInsteadOfRestartingItsLineage() async throws {
        let manager = SlidingWindowAsrManager()
        let updates = await manager.tokenUpdates
        await manager.beginStreamingJob()
        try await manager.publishTokenUpdate(
            tokens: [token("a", 1)], previous: 0, retained: 0, stable: 1,
            window: 0..<22400, center: 0..<16000)
        await manager.beginStreamingJob()
        try await manager.publishTokenUpdate(
            tokens: [token("b", 2)], previous: 0, retained: 0, stable: 1,
            window: 0..<22400, center: 0..<16000)

        var ordinals: [Int] = []
        do {
            for try await update in updates { ordinals.append(update.revision.ordinal) }
            XCTFail("a repeated start must end the strict stream")
        } catch SlidingWindowTokenStreamError.unavailableJob {}
        XCTAssertEqual(ordinals, [1], "no second ordinal 1 from a restarted lineage")
    }

    func testManagerCancellationIsErrorNotSuccessfulTerminal() async throws {
        let manager = SlidingWindowAsrManager()
        let updates = await manager.tokenUpdates
        await manager.cancel()
        do {
            for try await _ in updates { XCTFail("cancel cannot publish a token commit") }
            XCTFail("cancel must throw")
        } catch is CancellationError {}
    }

    func testSecondSubscriberFailsRatherThanStealingLineage() async throws {
        let manager = SlidingWindowAsrManager()
        let first = await manager.tokenUpdates
        let second = await manager.tokenUpdates
        do {
            for try await _ in second { XCTFail("second subscriber cannot receive updates") }
            XCTFail("second subscriber must fail")
        } catch SlidingWindowTokenStreamError.multipleSubscribers {}
        await manager.cancel()
        do { for try await _ in first {} } catch is CancellationError {}
    }
    func testManagerPublishesRetirementAndDrainsExactlyOnce() async throws {
        let manager = SlidingWindowAsrManager()
        let stream = await manager.tokenUpdates
        try await manager.publishTokenUpdate(
            tokens: [token("▁We"), token("▁an", 0.2)], previous: 0, retained: 0, stable: 1,
            window: 0..<22400, center: 0..<16000)
        try await manager.publishTokenUpdate(
            tokens: [token("▁analyzing", 0.24), token("▁now.", 0.6)], previous: 2, retained: 1, stable: 2,
            window: 0..<38400, center: 16000..<32000)
        _ = try await manager.finish()
        _ = try await manager.finish()
        var events: [SlidingWindowTokenUpdate] = []
        for try await event in stream { events.append(event) }
        XCTAssertEqual(events.map(\.revision.ordinal), [1, 2, 3])
        XCTAssertEqual(events.map(\.revision.retainedTokenCount), [0, 1, 3])
        XCTAssertEqual(events.map(\.revision.stableTokenCount), [1, 2, 3])
        XCTAssertTrue(events[2].revision.isTerminal)
    }

    func testManagerPublicationFailureCannotResumeOrEmitTerminal() async throws {
        let manager = SlidingWindowAsrManager()
        let stream = await manager.tokenUpdates
        do {
            try await manager.publishTokenUpdate(
                tokens: [], previous: 1, retained: 1, stable: 1,
                window: 0..<22400, center: 0..<16000)
            XCTFail("inconsistent native lineage must fail")
        } catch SlidingWindowTokenStreamError.invalidLineage {}
        try await manager.publishTokenUpdate(
            tokens: [token("▁hidden")], previous: 0, retained: 0, stable: 0,
            window: 0..<38400, center: 16000..<32000)
        _ = try await manager.finish()
        do {
            for try await _ in stream { XCTFail("failed lineage cannot resume") }
            XCTFail("failure must remain visible")
        } catch SlidingWindowTokenStreamError.invalidLineage {}
    }

    func testManagerEmptyDrainProducesOneTerminal() async throws {
        let manager = SlidingWindowAsrManager()
        let stream = await manager.tokenUpdates
        _ = try await manager.finish()
        var events: [SlidingWindowTokenUpdate] = []
        for try await event in stream { events.append(event) }
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.revision.stableTokenCount, 0)
        XCTAssertEqual(events.first?.revision.isTerminal, true)
    }

    func testLegacyResetCannotReopenStrictJob() async throws {
        let manager = SlidingWindowAsrManager()
        let first = await manager.tokenUpdates
        try await manager.reset()
        let second = await manager.tokenUpdates
        do {
            for try await _ in first { XCTFail("reset cannot commit") }
            XCTFail("reset must cancel")
        } catch is CancellationError {}
        do {
            for try await _ in second { XCTFail("reset cannot reopen lineage") }
            XCTFail("second job must fail")
        } catch SlidingWindowTokenStreamError.multipleSubscribers {}
    }

    func testLateSubscriberCannotClaimAlreadyAcceptedAudio() async throws {
        let manager = SlidingWindowAsrManager()
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1))
        buffer.frameLength = 1
        buffer.floatChannelData?[0][0] = 0
        await manager.streamAudio(buffer)
        let stream = await manager.tokenUpdates
        do {
            for try await _ in stream { XCTFail("late subscription cannot receive a prefix") }
            XCTFail("must reject")
        } catch SlidingWindowTokenStreamError.unavailableJob {}
        await manager.cancel()
    }

    func testCancelledOrFinishedLegacyJobCannotAcquireStrictSubscriber() async throws {
        for cancelled in [true, false] {
            let manager = SlidingWindowAsrManager()
            if cancelled { await manager.cancel() } else { _ = try await manager.finish() }
            let stream = await manager.tokenUpdates
            do {
                for try await _ in stream { XCTFail("closed job cannot reopen") }
                XCTFail("must reject")
            } catch SlidingWindowTokenStreamError.unavailableJob {}
        }
    }

}
