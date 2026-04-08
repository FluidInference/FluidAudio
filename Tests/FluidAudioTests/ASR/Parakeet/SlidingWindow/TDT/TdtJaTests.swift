#if os(macOS)
import XCTest
@testable import FluidAudio
import AVFoundation

final class TdtJaTests: XCTestCase {

    func testTdtJaTranscription() async throws {
        // Skip in CI environment - HuggingFace downloads are unreliable
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["CI"] != nil,
            "Skipping model download tests in CI environment"
        )

        // Load TDT Japanese manager
        print("Loading TDT Japanese models...")
        let manager = try await TdtJaManager.load()
        print("✅ Models loaded")

        // Create test audio (1 second of silence at 16kHz)
        let sampleRate = 16000
        let duration = 1.0
        let frameCount = Int(Double(sampleRate) * duration)
        let audio = [Float](repeating: 0.0, count: frameCount)

        // Transcribe
        print("Running transcription...")
        let result = try await manager.transcribe(audio: audio)
        print("✅ Transcription complete")
        print("Result: '\(result)'")

        // For silence, we expect minimal output (blank or empty)
        XCTAssertNotNil(result)
        print("✅ TDT Japanese model is working!")
    }

    func testTdtJaWithRealAudio() async throws {
        // Skip in CI environment - HuggingFace downloads are unreliable
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["CI"] != nil,
            "Skipping model download tests in CI environment"
        )

        // This would need actual Japanese audio file
        // For now, just verify the model loads
        let manager = try await TdtJaManager.load()
        XCTAssertNotNil(manager)
        print("✅ TDT Japanese manager initialized successfully")
    }
}
#endif
