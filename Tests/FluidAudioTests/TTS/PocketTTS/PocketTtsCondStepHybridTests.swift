@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the chunked `cond_step` dispatch mode added alongside the
/// existing chunk-1 (legacy) pipeline.
///
/// These tests are pure-logic: they verify the model store's mode plumbing
/// and accessor behaviour without requiring any downloaded CoreML artifacts.
/// End-to-end parity vs the legacy path is validated by the
/// `pocket-tts-cond-bench` CLI subcommand (which requires the chunk-16
/// `mlmodelc` file to be placed manually under the language root).
final class PocketTtsCondStepHybridTests: XCTestCase {

    // MARK: - PocketTtsCondStepMode

    func testCondStepModeEquatable() {
        XCTAssertEqual(PocketTtsCondStepMode.legacy, .legacy)
        XCTAssertEqual(PocketTtsCondStepMode.chunked(chunk: 16), .chunked(chunk: 16))
        XCTAssertNotEqual(PocketTtsCondStepMode.legacy, .chunked(chunk: 16))
        XCTAssertNotEqual(
            PocketTtsCondStepMode.chunked(chunk: 16),
            PocketTtsCondStepMode.chunked(chunk: 32)
        )
    }

    // MARK: - PocketTtsModelStore.condStepMode plumbing

    func testStoreDefaultModeIsLegacy() async {
        let store = PocketTtsModelStore(language: .english)
        let mode = await store.condStepMode
        XCTAssertEqual(mode, .legacy)
        let chunkSize = await store.condStepChunkSize()
        XCTAssertNil(chunkSize, "legacy mode must not expose a chunk size")
    }

    func testStoreChunkedModeExposesChunkSize() async {
        let store = PocketTtsModelStore(
            language: .english, condStepMode: .chunked(chunk: 16)
        )
        let mode = await store.condStepMode
        XCTAssertEqual(mode, .chunked(chunk: 16))
        let chunkSize = await store.condStepChunkSize()
        XCTAssertEqual(chunkSize, 16)
    }

    // MARK: - Accessor error semantics

    func testCondStepChunkModelThrowsWhenNotLoaded() async {
        // Legacy store never loads the chunk model — accessor must throw
        // a clean `modelNotFound` instead of returning a stale value.
        let store = PocketTtsModelStore(language: .english)
        do {
            _ = try await store.condStepChunkModel()
            XCTFail("expected condStepChunkModel() to throw in legacy mode")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("expected .modelNotFound, got \(error)")
            }
        } catch {
            XCTFail("expected PocketTTSError, got \(type(of: error)): \(error)")
        }
    }

    func testCondStepChunkModelThrowsWhenChunkedButUnloaded() async {
        // Chunked mode without `loadIfNeeded()` — accessor must throw
        // (we never run the loader because that would require network +
        // the unpublished chunk-16 file).
        let store = PocketTtsModelStore(
            language: .english, condStepMode: .chunked(chunk: 16)
        )
        do {
            _ = try await store.condStepChunkModel()
            XCTFail("expected condStepChunkModel() to throw before load")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("expected .modelNotFound, got \(error)")
            }
        } catch {
            XCTFail("expected PocketTTSError, got \(type(of: error)): \(error)")
        }
    }

    // MARK: - PocketTtsManager init plumbing

    func testManagerAcceptsCondStepMode() async {
        // Smoke test: verify the manager init compiles + threads the mode
        // through to the underlying store. We don't initialize() because
        // that would require network access.
        let manager = PocketTtsManager(
            defaultVoice: "alba",
            language: .english,
            condStepMode: .chunked(chunk: 16)
        )
        let isAvailable = await manager.isAvailable
        XCTAssertFalse(isAvailable, "manager should not be available before initialize()")
    }

    // MARK: - ModelNames

    func testCondStepChunk16FilenameMatchesConvention() {
        // The chunk-16 mlmodelc file lives next to cond_step.mlmodelc under
        // the v2/<lang>/ language root. Mismatching the filename here would
        // surface as a confusing modelNotFound at load time.
        XCTAssertEqual(ModelNames.PocketTTS.condStepChunk16, "cond_step_chunk16")
        XCTAssertEqual(ModelNames.PocketTTS.condStepChunk16File, "cond_step_chunk16.mlmodelc")
    }
}
