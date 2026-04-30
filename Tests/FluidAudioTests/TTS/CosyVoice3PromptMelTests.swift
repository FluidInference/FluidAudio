import XCTest

@testable import FluidAudio

final class CosyVoice3PromptMelTests: XCTestCase {

    func testFrameCountMatchesMatchaFormula() throws {
        // matcha/cosyvoice3: pad by 720 each side (reflect), center=False.
        // For 48000 samples: padded = 48000 + 1440 = 49440.
        // frames = (49440 - 1920) / 480 + 1 = 99 + 1 = 100.
        let mel = CosyVoice3PromptMel()
        let audio = [Float](repeating: 0.01, count: 48_000)
        let out = try mel.compute(audio: audio)
        XCTAssertEqual(out.frames, 100)
        XCTAssertEqual(out.mel.count, 100 * 80)
    }

    func testZeroAudioClampsToLogFloor() throws {
        // With audio of all zeros, mel values are 0 → clamped to 1e-5 → log = -11.5129...
        let mel = CosyVoice3PromptMel()
        let audio = [Float](repeating: 0, count: 24_000)
        let out = try mel.compute(audio: audio)
        let expected: Float = log(Float(1e-5))
        for v in out.mel {
            XCTAssertEqual(v, expected, accuracy: 1e-5)
        }
    }

    func testSinePeakInLowMelBins() throws {
        // 200 Hz sine at 24 kHz should light up one of the lowest mel bins
        // (fmin=0, the first few triangles cover 0..~200 Hz).
        let mel = CosyVoice3PromptMel()
        let sr: Float = 24_000
        let f: Float = 200
        let n = 12_000  // 0.5 s
        var audio = [Float](repeating: 0, count: n)
        for i in 0..<n {
            audio[i] = 0.5 * sin(2 * .pi * f * Float(i) / sr)
        }
        let out = try mel.compute(audio: audio)
        XCTAssertGreaterThan(out.frames, 0)
        // Average energy per mel bin
        var perBin = [Float](repeating: 0, count: 80)
        for frame in 0..<out.frames {
            for m in 0..<80 {
                perBin[m] += out.mel[frame * 80 + m]
            }
        }
        let argmax = perBin.enumerated().max(by: { $0.1 < $1.1 })!.offset
        // 200 Hz sits in the bottom ~10 Slaney-mel triangles with the 24 kHz
        // configuration (~linear below 1000 Hz). Accept any bin <20.
        XCTAssertLessThan(argmax, 20, "expected peak in low mel bins, got \(argmax)")
    }

    func testReflectPad() {
        // Manual reflect pad check mirroring PyTorch's F.pad(reflect).
        let y: [Float] = [1, 2, 3, 4, 5]
        let p = CosyVoice3PromptMel.reflectPad(y, pad: 2)
        // left: y[2], y[1] → 3, 2; core: 1,2,3,4,5; right: y[3], y[2] → 4, 3
        XCTAssertEqual(p, [3, 2, 1, 2, 3, 4, 5, 4, 3])
    }

    func testHannWindowPeriodicEndpoints() {
        let w = CosyVoice3PromptMel.hannWindowPeriodic(length: 4)
        // 0.5 * (1 - cos(2πi/4)) for i=0..3
        XCTAssertEqual(w[0], 0, accuracy: 1e-6)
        XCTAssertEqual(w[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(w[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(w[3], 0.5, accuracy: 1e-6)
    }

    func testMelBasisShape() {
        let basis = CosyVoice3PromptMel.buildSlaneyMelBasis(
            sampleRate: 24_000, nFFT: 1920, numMels: 80, fMin: 0, fMax: 12_000)
        XCTAssertEqual(basis.count, 80 * (1920 / 2 + 1))
        // Each triangle should integrate to >0.
        let numFreqBins = 1920 / 2 + 1
        for m in 0..<80 {
            var sum: Float = 0
            for f in 0..<numFreqBins {
                sum += basis[m * numFreqBins + f]
            }
            XCTAssertGreaterThan(sum, 0, "mel \(m) has zero sum")
        }
    }

    func testTrimToTokenRatio() throws {
        let mel = [Float](repeating: 1, count: 6 * 80)  // 6 frames
        let (trimmed, frames) = try CosyVoice3PromptMel.trimToTokenRatio(
            mel: mel, frames: 6, tokenCount: 2)
        XCTAssertEqual(frames, 4)
        XCTAssertEqual(trimmed.count, 4 * 80)
    }

    func testTrimToTokenRatioThrowsIfTooShort() {
        let mel = [Float](repeating: 1, count: 2 * 80)
        XCTAssertThrowsError(
            try CosyVoice3PromptMel.trimToTokenRatio(
                mel: mel, frames: 2, tokenCount: 2))
    }
}
