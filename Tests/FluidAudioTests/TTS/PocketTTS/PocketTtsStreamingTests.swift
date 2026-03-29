import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsStreamingTests: XCTestCase {

    // MARK: - AudioFrame Tests

    func testAudioFrameProperties() {
        let samples: [Float] = Array(repeating: 0.5, count: PocketTtsConstants.samplesPerFrame)
        let frame = PocketTtsSynthesizer.AudioFrame(
            samples: samples,
            frameIndex: 3,
            chunkIndex: 1,
            chunkCount: 4,
            utteranceIndex: nil
        )

        XCTAssertEqual(frame.samples.count, PocketTtsConstants.samplesPerFrame)
        XCTAssertEqual(frame.frameIndex, 3)
        XCTAssertEqual(frame.chunkIndex, 1)
        XCTAssertEqual(frame.chunkCount, 4)
        XCTAssertNil(frame.utteranceIndex)
    }

    func testAudioFrameIsSendable() {
        // Verify AudioFrame can be sent across concurrency boundaries
        let frame = PocketTtsSynthesizer.AudioFrame(
            samples: [1.0, 2.0, 3.0],
            frameIndex: 0,
            chunkIndex: 0,
            chunkCount: 1,
            utteranceIndex: nil
        )

        let expectation = expectation(description: "Frame sent across tasks")
        Task {
            let _: PocketTtsSynthesizer.AudioFrame = frame
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }

    // MARK: - PocketTtsManager Guard Tests

    func testSynthesizeStreamingFailsWithoutInitialization() async {
        let manager = PocketTtsManager()

        do {
            _ = try await manager.synthesizeStreaming(text: "Hello")
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

    func testSynthesizeStreamingWithVoiceDataFailsWithoutInitialization() async {
        let manager = PocketTtsManager()
        let fakeVoiceData = PocketTtsVoiceData(audioPrompt: [], promptLength: 0)

        do {
            _ = try await manager.synthesizeStreaming(text: "Hello", voiceData: fakeVoiceData)
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

    // MARK: - Text Normalization (used by streaming pipeline)

    func testNormalizeTextAddsTerminalPunctuation() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Hello world")
        XCTAssertTrue(text.hasSuffix("."), "Should add period when no terminal punctuation")
    }

    func testNormalizeTextPreservesExistingPunctuation() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Hello world!")
        XCTAssertTrue(text.hasSuffix("!"), "Should preserve existing punctuation")
        XCTAssertFalse(text.hasSuffix("!."), "Should not add extra period")
    }

    func testNormalizeTextCapitalizesFirstLetter() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("hello")
        XCTAssertTrue(text.contains("H"), "Should capitalize first letter")
    }

    func testNormalizeTextShortTextPadding() {
        // Short text (< 5 words) gets padding
        let (text, frames) = PocketTtsSynthesizer.normalizeText("Hi")
        XCTAssertTrue(text.hasPrefix(" "), "Short text should be padded")
        XCTAssertEqual(frames, PocketTtsConstants.shortTextPadFrames)
    }

    func testNormalizeTextLongTextNoExtraPadding() {
        let (_, frames) = PocketTtsSynthesizer.normalizeText(
            "This is a longer sentence with more than five words in it")
        XCTAssertEqual(frames, PocketTtsConstants.longTextExtraFrames)
    }
}
