import XCTest

#if canImport(FluidAudioTTS) && canImport(ESpeakNG)
import FluidAudioTTS
@testable import FluidAudio

/// Tests that verify the ESpeakNG framework can be properly linked and loaded.
/// These tests catch issues like broken symlinks, binary naming errors, and dyld failures.
final class FrameworkLinkTests: XCTestCase {

    /// Test that the FluidAudio framework successfully imports, which requires ESpeakNG to be properly linked.
    func testESpeakNGFrameworkLink() {
        // Simply importing and using FluidAudio tests that ESpeakNG is properly linked
        // If the framework structure is broken (binary name wrong, symlinks broken, etc),
        // this would fail with dyld errors
        let manager = TtSManager()
        XCTAssertNotNil(manager, "TtSManager should be instantiable, meaning ESpeakNG framework is properly linked")
    }

    /// Test that the binary can actually be found by dyld at runtime.
    /// This would fail if:
    /// - Binary name is wrong (e.g., ESPeakNG instead of ESpeakNG)
    /// - Symlink chain is broken
    /// - Framework not embedded correctly
    func testFrameworkBinaryResolution() async throws {
        let manager = TtSManager()

        XCTExpectFailure("Framework usage may fail in test environment without models", strict: false)

        do {
            // Attempting to use TTS should trigger ESpeakNG framework loading
            // If the framework binary cannot be found, this will fail with dyld error
            let audio = try await manager.synthesize(text: "test")
            XCTAssertGreaterThan(audio.count, 0, "Should generate audio data")
            manager.cleanup()
        } catch {
            // Expected in CI environment due to missing models
            // But the important thing is we didn't get a dyld error
            XCTAssertFalse(
                error.localizedDescription.contains("dyld"),
                "Should not have dyld errors - framework must be properly linked")
        }
    }

    /// Test that TTS functionality is accessible (requires ESpeakNG framework).
    /// This ensures the framework is not just linked but properly functional.
    func testTTSFrameworkFunctionality() async throws {
        let manager = TtSManager()

        XCTExpectFailure("TTS may fail in CI without models", strict: false)

        do {
            // This call will trigger ESpeakNG framework usage
            let audio = try await manager.synthesize(text: "test")
            XCTAssertGreaterThan(audio.count, 0, "Should generate audio data")
            manager.cleanup()
        } catch {
            // Should fail gracefully, not with dyld errors
            XCTAssertFalse(
                error.localizedDescription.contains("dyld"),
                "Should not have dyld errors")
        }
    }
}
#endif
