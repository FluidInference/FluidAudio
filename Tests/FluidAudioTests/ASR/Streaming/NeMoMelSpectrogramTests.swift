import Foundation
import XCTest

@testable import FluidAudio

final class NeMoMelSpectrogramTests: XCTestCase {

    private var mel: NeMoMelSpectrogram!

    override func setUp() {
        super.setUp()
        mel = NeMoMelSpectrogram()
    }

    override func tearDown() {
        mel = nil
        super.tearDown()
    }

    // MARK: - Empty / Edge Cases

    func testShortAudioProducesResult() {
        // Audio shorter than a single window (400 samples) should still produce output
        let shortAudio = [Float](repeating: 0.1, count: 800)
        let result = mel.compute(audio: shortAudio)

        XCTAssertGreaterThan(result.melLength, 0, "Short audio should still produce mel frames")
    }

    // MARK: - Output Shape

    func testComputeOutputShape() {
        // 1 second at 16kHz = 16000 samples
        let audio = [Float](repeating: 0, count: 16000)
        let result = mel.compute(audio: audio)

        // numFrames = 1 + (16000 - 400) / 160 = 1 + 97 = 98
        let expectedFrames = 1 + (16000 - 400) / 160
        XCTAssertEqual(result.melLength, expectedFrames, "Expected \(expectedFrames) frames for 1s audio")

        // Shape should be [1, 128, T]
        XCTAssertEqual(result.mel.count, 1, "Batch dimension should be 1")
        XCTAssertEqual(result.mel[0].count, 128, "Should have 128 mel bins")
        XCTAssertEqual(result.mel[0][0].count, expectedFrames, "Time dimension should match melLength")
    }

    func testComputeFlatOutputShape() {
        let audio = [Float](repeating: 0, count: 16000)
        let result = mel.computeFlat(audio: audio)

        XCTAssertGreaterThan(result.numFrames, 0)
        XCTAssertEqual(result.mel.count, 128 * result.numFrames, "Flat output should be nMels * numFrames")
    }

    // MARK: - Hann Window

    func testHannWindowSymmetry() {
        let window = mel.getHannWindow()

        XCTAssertEqual(window.count, 400, "Window length should be 400 (25ms at 16kHz)")

        for i in 0..<window.count / 2 {
            XCTAssertEqual(
                window[i], window[window.count - 1 - i],
                accuracy: 1e-6,
                "Window should be symmetric at index \(i)"
            )
        }
    }

    func testHannWindowEndpoints() {
        let window = mel.getHannWindow()
        // Symmetric Hann window: first and last values should be 0
        XCTAssertEqual(window[0], 0, accuracy: 1e-6, "First value should be 0")
        XCTAssertEqual(window[window.count - 1], 0, accuracy: 1e-6, "Last value should be 0")
    }

    func testHannWindowPeakAtCenter() {
        let window = mel.getHannWindow()
        let mid = window.count / 2
        // Peak should be near 1.0 at the center
        XCTAssertEqual(window[mid], 1.0, accuracy: 0.01, "Center should be near 1.0")
    }

    // MARK: - Mel Filterbank

    func testMelFilterbankShape() {
        let filterbank = mel.getFilterbank()
        XCTAssertEqual(filterbank.count, 128, "Should have 128 mel bins")
        XCTAssertEqual(filterbank[0].count, 257, "Each filter should have nFFT/2 + 1 = 257 freq bins")
    }

    func testMelFilterbankNonNegative() {
        let filterbank = mel.getFilterbank()
        for (melIdx, filter) in filterbank.enumerated() {
            for value in filter {
                XCTAssertGreaterThanOrEqual(
                    value, 0,
                    "Mel filterbank should be non-negative at mel bin \(melIdx)"
                )
            }
        }
    }

    // MARK: - Silence Behavior

    func testSilenceProducesLowMelValues() {
        let silence = [Float](repeating: 0, count: 16000)
        let result = mel.compute(audio: silence)

        guard !result.mel.isEmpty else {
            XCTFail("Expected non-empty mel output")
            return
        }

        for melBin in result.mel[0] {
            for value in melBin {
                // log(guard_value) ≈ -16.6 for 1e-10 floor, values should be very negative
                XCTAssertLessThan(value, 0, "Silence should produce negative log-mel values")
            }
        }
    }
}
