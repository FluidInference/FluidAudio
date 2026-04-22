import Accelerate
import Foundation
import XCTest

@testable import FluidAudio

final class CohereMelSpectrogramTests: XCTestCase {

    // MARK: - Basic Computation

    func testComputeWithEmptyAudioReturnsEmptyMel() {
        let melExtractor = CohereMelSpectrogram()
        let emptyAudio: [Float] = []

        let mel = melExtractor.compute(audio: emptyAudio)

        XCTAssertTrue(mel.isEmpty, "Empty audio should produce empty mel spectrogram")
    }

    func testComputeWithShortAudioReturnsCorrectDimensions() {
        let melExtractor = CohereMelSpectrogram()
        // 1 second of silence
        let audio = [Float](repeating: 0.0, count: CohereAsrConfig.sampleRate)

        let mel = melExtractor.compute(audio: audio)

        XCTAssertEqual(mel.count, CohereAsrConfig.numMelBins, "Should have 128 mel bins")
        XCTAssertGreaterThan(mel.first?.count ?? 0, 0, "Should have frames")
    }

    func testMelFrameCountMatchesExpectedValue() {
        let melExtractor = CohereMelSpectrogram()
        let nSamples = 16000  // 1 second
        let audio = [Float](repeating: 0.0, count: nSamples)

        let mel = melExtractor.compute(audio: audio)

        // Calculate expected frames:
        // With reflection padding: paddedLength = nSamples + 2 * (nFFT/2)
        // numFrames = 1 + (paddedLength - nFFT) / hopLength
        let nFFT = CohereAsrConfig.MelSpec.nFFT
        let hopLength = CohereAsrConfig.MelSpec.hopLength
        let padLength = nFFT / 2
        let paddedLength = nSamples + 2 * padLength
        let expectedFrames = 1 + (paddedLength - nFFT) / hopLength

        XCTAssertEqual(mel.first?.count, expectedFrames)
    }

    func testAllMelBinsArePopulated() {
        let melExtractor = CohereMelSpectrogram()
        let audio = [Float](repeating: 0.1, count: 16000)

        let mel = melExtractor.compute(audio: audio)

        XCTAssertEqual(mel.count, CohereAsrConfig.numMelBins)
        for (binIdx, bin) in mel.enumerated() {
            XCTAssertFalse(bin.isEmpty, "Mel bin \(binIdx) should not be empty")
        }
    }

    // MARK: - Edge Cases

    func testComputeWithVeryShortAudio() {
        // Reflection padding requires audio.count > nFFT/2 (= 512). Anything
        // shorter must return empty rather than crash.
        let melExtractor = CohereMelSpectrogram()
        let audio = [Float](repeating: 0.1, count: 100)

        let mel = melExtractor.compute(audio: audio)

        XCTAssertTrue(mel.isEmpty, "Audio shorter than nFFT/2 should return empty mel")
    }

    func testComputeWithSingleSample() {
        let melExtractor = CohereMelSpectrogram()
        let audio: [Float] = [0.5]

        let mel = melExtractor.compute(audio: audio)

        XCTAssertTrue(mel.isEmpty, "Single-sample audio should return empty mel")
    }

    func testComputeIsConsistent() {
        let melExtractor = CohereMelSpectrogram()
        let audio = (0..<16000).map { Float(sin(Double($0) * 0.01)) }

        let mel1 = melExtractor.compute(audio: audio)
        let mel2 = melExtractor.compute(audio: audio)

        XCTAssertEqual(mel1.count, mel2.count)
        XCTAssertEqual(mel1.first?.count, mel2.first?.count)

        // Check values are identical
        for (bin1, bin2) in zip(mel1, mel2) {
            XCTAssertEqual(bin1, bin2, "Repeated computation should give identical results")
        }
    }

    // MARK: - Known Input Values

    func testComputeWithSineWaveProducesNonZeroMel() {
        let melExtractor = CohereMelSpectrogram()
        // 440 Hz sine wave (A4 note)
        let frequency: Float = 440.0
        let duration: Float = 1.0  // 1 second
        let sampleRate = Float(CohereAsrConfig.sampleRate)
        let nSamples = Int(duration * sampleRate)

        let audio = (0..<nSamples).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / sampleRate)
        }

        let mel = melExtractor.compute(audio: audio)

        // Check that mel values are not all negative infinity (log of near-zero)
        let hasNonTrivialValues = mel.contains { bin in
            bin.contains { $0 > -20.0 }  // Reasonable log-scale threshold
        }

        XCTAssertTrue(hasNonTrivialValues, "Sine wave should produce non-trivial mel values")
    }

    func testComputeWithDCOffsetIsHandled() {
        let melExtractor = CohereMelSpectrogram()
        let audio = [Float](repeating: 1.0, count: 16000)  // DC signal

        let mel = melExtractor.compute(audio: audio)

        XCTAssertEqual(mel.count, CohereAsrConfig.numMelBins)
        XCTAssertGreaterThan(mel.first?.count ?? 0, 0)
    }

    // MARK: - Thread Safety (Non-Sendable Warning)

    func testMultipleInstancesCanBeUsedConcurrently() async {
        let audio = (0..<8000).map { Float(sin(Double($0) * 0.01)) }

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    // Each task uses its own instance (as documented)
                    let melExtractor = CohereMelSpectrogram()
                    let mel = melExtractor.compute(audio: audio)
                    XCTAssertEqual(mel.count, CohereAsrConfig.numMelBins)
                }
            }
        }
    }

    // MARK: - Preemphasis Filter

    func testPreemphasisIsApplied() {
        let melExtractor = CohereMelSpectrogram()
        // Two similar long-enough signals; preemphasis should differentiate them.
        // Use >= 2 * nFFT samples so we get several frames with room for the
        // filter to matter.
        let nSamples = CohereAsrConfig.MelSpec.nFFT * 2
        let audio1 = [Float](repeating: 1.0, count: nSamples)
        let audio2 = (0..<nSamples).map { i in Float(i) / Float(nSamples) }

        let mel1 = melExtractor.compute(audio: audio1)
        let mel2 = melExtractor.compute(audio: audio2)

        XCTAssertFalse(mel1.isEmpty)
        XCTAssertFalse(mel2.isEmpty)
        XCTAssertNotEqual(mel1, mel2, "Preemphasis should affect different input signals differently")
    }

    // MARK: - Mel Bins Sanity Checks

    func testMelBinValuesAreFinite() {
        let melExtractor = CohereMelSpectrogram()
        let audio = (0..<16000).map { Float(sin(Double($0) * 0.05)) }

        let mel = melExtractor.compute(audio: audio)

        for (binIdx, bin) in mel.enumerated() {
            for (frameIdx, value) in bin.enumerated() {
                XCTAssertTrue(
                    value.isFinite,
                    "Mel bin[\(binIdx)][\(frameIdx)] = \(value) is not finite"
                )
            }
        }
    }
}
