import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsSessionTests: XCTestCase {

    // MARK: - Manager Guard Tests

    func testMakeSessionFailsWithoutInitialization() async {
        let manager = PocketTtsManager()

        do {
            _ = try await manager.makeSession()
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testMakeSessionWithVoiceDataFailsWithoutInitialization() async {
        let manager = PocketTtsManager()
        let fakeVoiceData = PocketTtsVoiceData(audioPrompt: [], promptLength: 0)

        do {
            _ = try await manager.makeSession(voiceData: fakeVoiceData)
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - KV Cache Clone Tests

    func testCloneKVCacheStateProducesIndependentCopy() throws {
        let original = try PocketTtsSynthesizer.emptyKVCacheState(
            layers: PocketTtsLanguage.english.transformerLayers)

        // Write a known value into the original
        let ptr = original.caches[0].dataPointer.bindMemory(to: Float.self, capacity: 1)
        ptr[0] = 42.0
        original.positions[0][0] = NSNumber(value: Float(7.0))

        let clone = try PocketTtsSynthesizer.cloneKVCacheState(original)

        // Clone should have the same values
        let clonePtr = clone.caches[0].dataPointer.bindMemory(to: Float.self, capacity: 1)
        XCTAssertEqual(clonePtr[0], 42.0)
        XCTAssertEqual(clone.positions[0][0].floatValue, 7.0)

        // Mutating clone should not affect original
        clonePtr[0] = 99.0
        clone.positions[0][0] = NSNumber(value: Float(15.0))
        XCTAssertEqual(ptr[0], 42.0, "Original cache should be unaffected by clone mutation")
        XCTAssertEqual(
            original.positions[0][0].floatValue, 7.0,
            "Original position should be unaffected by clone mutation"
        )
    }

    func testCloneKVCacheStatePreservesShape() throws {
        let original = try PocketTtsSynthesizer.emptyKVCacheState(
            layers: PocketTtsLanguage.english.transformerLayers)
        let clone = try PocketTtsSynthesizer.cloneKVCacheState(original)

        XCTAssertEqual(clone.caches.count, original.caches.count)
        XCTAssertEqual(clone.positions.count, original.positions.count)

        for i in 0..<clone.caches.count {
            XCTAssertEqual(clone.caches[i].shape, original.caches[i].shape)
            XCTAssertEqual(clone.caches[i].dataType, original.caches[i].dataType)
        }
    }

    // MARK: - Producer-Task Lifetime Regression

    // With the old weak self capture in start(), a generation task scheduled
    // after the caller's session reference dropped exited silently and the
    // frames stream was never finished — the second one-shot synthesize hung
    // forever (0% CPU). These tests are bounded by explicit deadlines so a
    // recurrence fails instead of hanging the suite.

    func testConsecutiveOneShotSynthesesTerminate() async throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("MLState requires macOS 15+/iOS 18+")
        }
        guard Self.installedStateModelURL() != nil else {
            throw XCTSkip("pocket_state.mlmodelc not installed in the local cache")
        }
        let manager = PocketTtsManager(language: .english, placement: .aneState)
        do {
            try await manager.initialize()
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                throw XCTSkip("PocketTTS language pack not installed in the local cache")
            }
            throw error
        }
        for i in 0..<3 {
            let wav = try await withDeadline(30) {
                try await manager.synthesize(text: "Probe number \(i).")
            }
            XCTAssertFalse(wav.isEmpty, "one-shot synthesis \(i) produced no audio")
        }
    }

    func testCancelTerminatesFramesStream() async throws {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            throw XCTSkip("MLState requires macOS 15+/iOS 18+")
        }
        guard Self.installedStateModelURL() != nil else {
            throw XCTSkip("pocket_state.mlmodelc not installed in the local cache")
        }
        let manager = PocketTtsManager(language: .english, placement: .aneState)
        do {
            try await manager.initialize()
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                throw XCTSkip("PocketTTS language pack not installed in the local cache")
            }
            throw error
        }
        let session = try await manager.makeSession()
        let consumer = Task { () -> Int in
            var n = 0
            do {
                for try await _ in session.frames { n += 1 }
            } catch is CancellationError {
                // `cancel()` ends the stream by throwing CancellationError; ending is what counts.
            }
            return n
        }
        session.enqueue("A long enough utterance that cancellation arrives while it generates.")
        try await Task.sleep(nanoseconds: 200_000_000)
        await session.cancel()
        _ = try await withDeadline(10) { try await consumer.value }
    }

    // MARK: - Helpers

    private struct DeadlineExceeded: Error {}

    /// Bound an async body: a hang becomes a test failure, not a stuck suite.
    private func withDeadline<T: Sendable>(
        _ seconds: Double, _ body: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await body() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw DeadlineExceeded()
            }
            let first = try await group.next()!
            group.cancelAll()
            return first
        }
    }

    private static func installedStateModelURL() -> URL? {
        guard let cacheRoot = try? TtsCacheDirectory.ensure() else { return nil }
        let url =
            cacheRoot
            .appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(Repo.pocketTts.folderName)
            .appendingPathComponent(PocketTtsLanguage.english.repoSubdirectory)
            .appendingPathComponent(ModelNames.PocketTTS.pocketStateFile)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}
