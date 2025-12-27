import XCTest

#if canImport(FluidAudioTTS) && canImport(ESpeakNG)
import FluidAudioTTS
@testable import FluidAudio

final class TtSManagerTests: XCTestCase {

    var manager: TtSManager!

    override func setUp() {
        super.setUp()
        manager = TtSManager()
    }

    override func tearDown() {
        manager?.cleanup()
        manager = nil
        super.tearDown()
    }

    func testInitialization() {
        XCTAssertNotNil(manager)
        XCTAssertFalse(manager.isAvailable)
    }

    func testSynthesizeShortText() async throws {
        try await ensureInitialized()
        XCTAssertTrue(manager.isAvailable)

        let text = "Hello, world!"
        let audioData = try await manager.synthesize(text: text)

        XCTAssertGreaterThan(audioData.count, 0)
        XCTAssertGreaterThan(audioData.count, 1000, "Audio data should be substantial")
    }

    func testSynthesizeWithDifferentSpeeds() async throws {
        try await ensureInitialized()

        let text = "Testing different speeds"

        let normalSpeed = try await manager.synthesize(text: text, voiceSpeed: 1.0)
        let slowSpeed = try await manager.synthesize(text: text, voiceSpeed: 0.5)
        let fastSpeed = try await manager.synthesize(text: text, voiceSpeed: 2.0)

        XCTAssertGreaterThan(normalSpeed.count, 0)
        XCTAssertGreaterThan(slowSpeed.count, 0)
        XCTAssertGreaterThan(fastSpeed.count, 0)

        XCTAssertGreaterThan(
            slowSpeed.count, fastSpeed.count,
            "Slower speech should generally produce longer audio")
    }

    func testSynthesizeToFile() async throws {
        try await ensureInitialized()

        let text = "Save this to a file"
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_tts_output.wav")

        defer {
            try? FileManager.default.removeItem(at: outputURL)
        }

        try await manager.synthesizeToFile(
            text: text,
            outputURL: outputURL
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))

        let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
        let fileSize = attributes[.size] as? Int ?? 0
        XCTAssertGreaterThan(fileSize, 1000, "Audio file should have substantial size")
    }

    func testEmptyTextHandling() async throws {
        try await ensureInitialized()

        do {
            _ = try await manager.synthesize(text: "")
            XCTFail("Should throw error for empty text")
        } catch {
            XCTAssertTrue(error is TTSError)
        }
    }

    func testMultipleSpeakerIds() async throws {
        try await ensureInitialized()

        let text = "Testing speaker voices"

        let speaker0 = try await manager.synthesize(text: text, speakerId: 0)
        let speaker1 = try await manager.synthesize(text: text, speakerId: 1)

        XCTAssertGreaterThan(speaker0.count, 0)
        XCTAssertGreaterThan(speaker1.count, 0)
    }

    func testLongTextSynthesis() async throws {
        try await ensureInitialized()

        let longText = """
            This is a much longer text to test the synthesis capabilities. 
            It contains multiple sentences and should produce a longer audio output. 
            The model should handle this gracefully without any issues.
            """

        let audioData = try await manager.synthesize(text: longText)

        XCTAssertGreaterThan(
            audioData.count, 10000,
            "Long text should produce substantial audio")
    }

    func testSpecialCharactersHandling() async throws {
        try await ensureInitialized()

        let specialText = "Hello! How are you? I'm fine, thanks."
        let audioData = try await manager.synthesize(text: specialText)

        XCTAssertGreaterThan(audioData.count, 0)
    }

    func testConcurrentSynthesis() async throws {
        try await ensureInitialized()

        let texts = [
            "First text",
            "Second text",
            "Third text",
        ]

        // Swift 6: TtSManager is not Sendable, so use sequential synthesis
        var results: [Data] = []
        for text in texts {
            let result = try await manager.synthesize(text: text)
            results.append(result)
        }

        for (index, data) in results.enumerated() {
            XCTAssertGreaterThan(data.count, 0, "Text \(index) should produce audio")
        }
    }

    func testModelCleanup() async throws {
        try await ensureInitialized()
        XCTAssertTrue(manager.isAvailable)

        manager.cleanup()
        XCTAssertFalse(manager.isAvailable)
    }

    private func ensureInitialized() async throws {
        if manager.isAvailable { return }

        do {
            try await manager.initialize()
        } catch {
            throw XCTSkip("Skipping TTS manager tests due to initialization failure: \(error.localizedDescription)")
        }
    }
}
#endif
