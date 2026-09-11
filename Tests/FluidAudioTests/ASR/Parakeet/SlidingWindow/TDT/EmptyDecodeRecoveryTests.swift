@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

/// Issue #909: a window the model decodes to nothing although it carries
/// speech is re-run with perturbed length declarations. These cover the pure
/// pieces of that ladder; the model-level behavior is pinned by the batch
/// benchmark and the streaming fixtures.
final class EmptyDecodeRecoveryTests: XCTestCase {

    private func tone(seconds: Double, amplitude: Float) -> [Float] {
        let count = Int(seconds * Double(ASRConstants.sampleRate))
        return (0..<count).map { amplitude * sin(Float($0) * 0.05) }
    }

    func testRecoveryGateRequiresSpeechEnergyAndEnoughAudio() {
        // Speech-level energy over 3 s: retry.
        XCTAssertTrue(
            AsrManager.shouldRecoverEmptyDecode(samples: tone(seconds: 3, amplitude: 0.05), actualLength: 48_000))
        // Digital silence and near-silence: an empty decode is the right answer.
        XCTAssertFalse(
            AsrManager.shouldRecoverEmptyDecode(samples: [Float](repeating: 0, count: 48_000), actualLength: 48_000))
        XCTAssertFalse(
            AsrManager.shouldRecoverEmptyDecode(samples: tone(seconds: 3, amplitude: 0.001), actualLength: 48_000))
        // Under 2 s of audio: too little to call a blank suspicious.
        XCTAssertFalse(
            AsrManager.shouldRecoverEmptyDecode(samples: tone(seconds: 3, amplitude: 0.05), actualLength: 16_000))
        // Only the actual length counts, not the zero padding behind it.
        var padded = tone(seconds: 3, amplitude: 0.05)
        padded += [Float](repeating: 0, count: 12 * ASRConstants.sampleRate)
        XCTAssertTrue(AsrManager.shouldRecoverEmptyDecode(samples: padded, actualLength: 48_000))
        // A silent window followed by padding does not retry because of the
        // padding's zeros either.
        XCTAssertFalse(
            AsrManager.shouldRecoverEmptyDecode(
                samples: [Float](repeating: 0, count: 240_000), actualLength: 208_000))
    }

    func testDeclaringFullMelLengthOverridesOnlyTheLength() throws {
        let mel = try MLMultiArray(shape: [1, 128, 1501], dataType: .float16)
        let melLength = try MLMultiArray(shape: [1], dataType: .int32)
        melLength[0] = 1301
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel), "mel_length": MLFeatureValue(multiArray: melLength),
        ])
        let full = try AsrManager.declaringFullMelLength(input)
        XCTAssertEqual(full.featureValue(for: "mel_length")?.multiArrayValue?[0].intValue, 1501)
        XCTAssertTrue(full.featureValue(for: "mel")?.multiArrayValue === mel, "the mel tensor is passed through")
        XCTAssertEqual(input.featureValue(for: "mel_length")?.multiArrayValue?[0].intValue, 1301, "input untouched")
        // A fused frontend without mel inputs is returned unchanged.
        let other = try MLDictionaryFeatureProvider(dictionary: ["audio_signal": MLFeatureValue(multiArray: mel)])
        XCTAssertEqual(try AsrManager.declaringFullMelLength(other).featureNames, other.featureNames)
    }

    /// A streaming re-decode whose new audio holds no speech has an empty
    /// visible sequence but tokens suppressed before the cutoff: the window
    /// decoded fine and those tokens are seam evidence. Not a blank.
    func testWindowWithOnlySuppressedTokensIsNotBlank() throws {
        var hypothesis = TdtHypothesis(decState: TdtDecoderState.make())
        XCTAssertTrue(AsrManager.isWholeWindowBlank(hypothesis))
        hypothesis.suppressedTokens = [506, 768]
        hypothesis.suppressedTimestamps = [140, 143]
        XCTAssertFalse(AsrManager.isWholeWindowBlank(hypothesis), "suppressed tokens mean the window decoded")
        hypothesis.ySequence = [575]
        hypothesis.timestamps = [150]
        XCTAssertFalse(AsrManager.isWholeWindowBlank(hypothesis))
    }

    /// The energy gate is a non-silence test, so what the ladder coaxes out of
    /// music or noise must clear a confidence bar before it replaces the empty
    /// decode. Genuine recoveries score about 0.9 over dozens of tokens.
    func testRecoveryMustBeCredible() {
        var hypothesis = TdtHypothesis(decState: TdtDecoderState.make())
        XCTAssertFalse(AsrManager.recoveryIsCredible(hypothesis), "nothing recovered")
        hypothesis.ySequence = [1, 2, 3, 4]
        hypothesis.tokenConfidences = [0.95, 0.9, 0.85, 0.99]
        XCTAssertTrue(AsrManager.recoveryIsCredible(hypothesis))
        hypothesis.tokenConfidences = [0.4, 0.5, 0.3, 0.6]
        XCTAssertFalse(AsrManager.recoveryIsCredible(hypothesis), "low mean confidence is a hallucination")
        hypothesis.ySequence = [1]
        hypothesis.tokenConfidences = [0.99]
        XCTAssertFalse(AsrManager.recoveryIsCredible(hypothesis), "a lone token is not a recovery")
        XCTAssertEqual(AsrManager.emptyDecodeRecoveryMinimumConfidence, 0.7)
    }

    func testRecoveryLadderOrderAndNames() {
        let ladder = AsrManager.emptyDecodeRecoveryPolicies
        XCTAssertEqual(
            ladder.map(\.description),
            [
                "encoderFull", "preprocessorFull", "trimmedTail", "encoderFull+trimmedTail",
                "preprocessorFull+trimmedTail",
            ])
        XCTAssertEqual(AsrManager.InferenceLengthPolicy([]).description, "actual")
        XCTAssertEqual(AsrManager.trimmedTailSamples, 3200)
        XCTAssertEqual(AsrManager.emptyDecodeRecoveryMinimumSamples, 32_000)
    }
}
