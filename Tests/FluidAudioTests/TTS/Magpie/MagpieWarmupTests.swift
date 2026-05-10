import XCTest

@testable import FluidAudio

/// Unit tests for `MagpieTtsManager.warmup()` covering the
/// no-model-files-needed contract (the success path requires real Magpie
/// CoreML artefacts and is left to integration tests / `fluidaudiocli
/// magpie bench`).
final class MagpieWarmupTests: XCTestCase {

    /// `warmup()` before `initialize()` must throw `MagpieError.notInitialized`,
    /// matching the `synthesize()` and `prepareLanguage()` contracts.
    func testWarmupBeforeInitializeThrowsNotInitialized() async throws {
        let manager = MagpieTtsManager()

        do {
            try await manager.warmup()
            XCTFail("warmup() should throw notInitialized when called pre-initialize")
        } catch let error as MagpieError {
            switch error {
            case .notInitialized:
                break  // expected
            default:
                XCTFail("expected .notInitialized; got \(error)")
            }
        } catch {
            XCTFail("expected MagpieError.notInitialized; got \(type(of: error))")
        }
    }

    /// `warmup()` after `cleanup()` must throw `MagpieError.notInitialized`.
    /// Without this, callers that cycle initialize → cleanup → warmup would
    /// silently dispatch into a torn-down synthesizer.
    func testWarmupAfterCleanupThrowsNotInitialized() async throws {
        let manager = MagpieTtsManager()
        // No initialize(); cleanup() should be a no-op on a fresh manager but
        // we explicitly null out the synthesizer to mirror the post-init path
        // a caller would hit if they called initialize() then cleanup().
        await manager.cleanup()

        do {
            try await manager.warmup()
            XCTFail("warmup() should throw notInitialized after cleanup()")
        } catch let error as MagpieError {
            switch error {
            case .notInitialized:
                break  // expected
            default:
                XCTFail("expected .notInitialized; got \(error)")
            }
        } catch {
            XCTFail("expected MagpieError.notInitialized; got \(type(of: error))")
        }
    }

    /// `isAvailable` reports `false` for a freshly-constructed manager;
    /// covered here because callers gate `warmup()` calls on it (e.g.
    /// `if await manager.isAvailable { try? await manager.warmup() }`).
    func testIsAvailableFalseBeforeInitialize() async {
        let manager = MagpieTtsManager()
        let available = await manager.isAvailable
        XCTAssertFalse(available)
    }

    // The warmup() success path (synthesizer.warmup() runs a 16-step
    // throwaway synthesis on a single "." input) requires real Magpie
    // CoreML artefacts and is exercised by:
    //   * MagpieTtsManager.initialize() — auto-warmup at load time
    //   * fluidaudiocli magpie bench — multi-shot warm-then-bench harness
    // Repeated warmup() invocations are documented as safe — there is no
    // internal "already warm" cache guard, so back-to-back calls each run
    // the dummy synthesis (cheap if ANE state is still warm; pays the
    // recompile cost if the cache was invalidated post-sleep).
}
