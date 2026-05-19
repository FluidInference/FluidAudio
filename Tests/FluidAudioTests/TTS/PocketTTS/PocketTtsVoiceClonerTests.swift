import Foundation
import XCTest

@testable import FluidAudio

/// Pure-logic unit tests for `PocketTtsVoiceCloner`'s pad/truncate and
/// frame-trim helpers. The full `cloneVoice(from:using:)` entry point
/// needs an `MLModel`, so these tests drive the smaller internal
/// helpers (`makeEncoderInputBuffer`, `usableFrameCount`) which the
/// production path delegates to.
final class PocketTtsVoiceClonerTests: XCTestCase {

    // MARK: - makeEncoderInputBuffer

    func testEncoderInputBufferPadsShorterAudio() {
        // 7.5 s of audio @ 24 kHz = 180_000 samples; encoder wants 240_000.
        let realCount = 180_000
        let input = (0..<realCount).map { Float($0 % 17) - 8 }
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer(input)

        XCTAssertEqual(
            buffer.count, PocketTtsVoiceCloner.encoderInputSamples,
            "Buffer must always be encoderInputSamples long")
        XCTAssertEqual(
            Array(buffer.prefix(realCount)), input,
            "Real samples must be copied verbatim into the prefix")
        XCTAssertTrue(
            buffer.dropFirst(realCount).allSatisfy { $0 == 0 },
            "Padding region must be zero-filled")
    }

    func testEncoderInputBufferTruncatesLongerAudio() {
        // 15 s @ 24 kHz = 360_000 samples; must be truncated to 240_000.
        let oversize = PocketTtsVoiceCloner.encoderInputSamples + 120_000
        let input = (0..<oversize).map { Float($0 % 23) - 11 }
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer(input)

        XCTAssertEqual(
            buffer.count, PocketTtsVoiceCloner.encoderInputSamples,
            "Buffer must always be encoderInputSamples long, never longer")
        XCTAssertEqual(
            buffer, Array(input.prefix(PocketTtsVoiceCloner.encoderInputSamples)),
            "Truncation must keep the leading samples")
    }

    func testEncoderInputBufferHandlesExactLength() {
        // Exactly 240_000 samples → no padding, no truncation.
        let input = (0..<PocketTtsVoiceCloner.encoderInputSamples).map { Float($0) * 1e-6 }
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer(input)

        XCTAssertEqual(buffer, input)
    }

    func testEncoderInputBufferHandlesEmptyInput() {
        // Defensive: empty input shouldn't crash, just produce all zeros.
        let buffer = PocketTtsVoiceCloner.makeEncoderInputBuffer([])

        XCTAssertEqual(buffer.count, PocketTtsVoiceCloner.encoderInputSamples)
        XCTAssertTrue(buffer.allSatisfy { $0 == 0 })
    }

    // MARK: - usableFrameCount

    func testUsableFrameCountRoundsPartialFrameUp() {
        // 7.5 s @ 24 kHz = 180_000 samples. 180_000 / 1920 = 93.75 → 94 frames
        // (ceiling). Encoder always emits 125 frames for the full 10 s window,
        // so we use the ceiling rather than the full output.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: 180_000, availableFrames: 125)
        XCTAssertEqual(usable, 94)
    }

    func testUsableFrameCountCapsAtMaxVoiceFrames() {
        // Even with 10 s of real audio and a hypothetical bigger encoder
        // output, we never exceed `maxVoiceFrames` (KV cache budget).
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: PocketTtsVoiceCloner.encoderInputSamples,
            availableFrames: 200)
        XCTAssertEqual(usable, PocketTtsVoiceCloner.maxVoiceFrames)
    }

    func testUsableFrameCountCapsAtAvailableFrames() {
        // If the encoder somehow emits fewer frames than the real audio
        // implies, trust the encoder rather than over-reading its buffer.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: PocketTtsVoiceCloner.encoderInputSamples,
            availableFrames: 80)
        XCTAssertEqual(usable, 80)
    }

    func testUsableFrameCountHandlesExactFrameBoundary() {
        // 95 * 1920 = 182_400 samples — clean multiple, no rounding needed.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: 95 * PocketTtsVoiceCloner.frameSize,
            availableFrames: 125)
        XCTAssertEqual(usable, 95)
    }

    func testUsableFrameCountHandlesSubFrameAudio() {
        // < 1 frame of audio rounds up to 1 (the encoder still produces a
        // frame even for a tiny prefix). Below-minDurationSeconds inputs
        // are rejected upstream so this is mostly defensive.
        let usable = PocketTtsVoiceCloner.usableFrameCount(
            realSampleCount: 100, availableFrames: 125)
        XCTAssertEqual(usable, 1)
    }
}
