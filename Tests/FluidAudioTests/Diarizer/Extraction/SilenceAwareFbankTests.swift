import XCTest

@testable import FluidAudio

final class SilenceAwareFbankTests: XCTestCase {

    func testFramesOfExactZerosAreSilent() {
        // Frames start every 160 samples and span 400: speech in [0, 800), zeros after.
        let audio = [Float](repeating: 0.1, count: 800) + [Float](repeating: 0, count: 1_600)

        let silent = audio.withUnsafeBufferPointer { SilenceAwareFbank.silentFrames(audio: $0, frameCount: 13) }

        // Frame 2 covers [320, 720) and frame 3 covers [480, 880): only frames from 5 on are all zero.
        XCTAssertEqual(silent, [false, false, false, false, false, true, true, true, true, true, true, true, true])
    }

    func testRecenterMatchesAMeanThatExcludesSilentFrames() {
        // Two bands, four frames (band-major); frames 2 and 3 are silent at the log floor.
        let raw: [Float] = [1, 3, -13.8, -13.8, 2, 6, -13.8, -13.8]
        let modelMean: [Float] = [(1 + 3 - 27.6) / 4, (2 + 6 - 27.6) / 4]
        var modelOutput = (0..<8).map { raw[$0] - modelMean[$0 / 4] }

        let changed = modelOutput.withUnsafeMutableBufferPointer {
            SilenceAwareFbank.recenter(
                features: $0.baseAddress!, bandCount: 2, bandStride: 4, frameStride: 1,
                silent: [false, false, true, true])
        }

        XCTAssertTrue(changed)
        XCTAssertEqual(modelOutput[0], 1 - 2, accuracy: 1e-5)
        XCTAssertEqual(modelOutput[1], 3 - 2, accuracy: 1e-5)
        XCTAssertEqual(modelOutput[4], 2 - 4, accuracy: 1e-5)
        XCTAssertEqual(modelOutput[5], 6 - 4, accuracy: 1e-5)
    }

    func testWindowsWithoutSilenceOrWithOnlySilenceAreUntouched() {
        var features: [Float] = [1, 2, 3, 4]

        let withoutSilence = features.withUnsafeMutableBufferPointer {
            SilenceAwareFbank.recenter(
                features: $0.baseAddress!, bandCount: 1, bandStride: 4, frameStride: 1,
                silent: [false, false, false, false])
        }
        let allSilent = features.withUnsafeMutableBufferPointer {
            SilenceAwareFbank.recenter(
                features: $0.baseAddress!, bandCount: 1, bandStride: 4, frameStride: 1,
                silent: [true, true, true, true])
        }

        XCTAssertFalse(withoutSilence)
        XCTAssertFalse(allSilent)
        XCTAssertEqual(features, [1, 2, 3, 4])
    }
}
