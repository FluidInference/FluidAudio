import XCTest

@testable import FluidAudio

/// Pins `LuxTtsMelExtractor` against the Python `VocosFbank` fixture mel of
/// the prompt clip (24 kHz, n_fft 1024, hop 256, 100 mels, log + ×0.1).
final class LuxTtsMelExtractorTests: XCTestCase {

    func testFrameCountMatchesLhotse() {
        let extractor = LuxTtsMelExtractor()
        // lhotse compute_num_frames: (n + hop/2) / hop.
        XCTAssertEqual(extractor.frameCount(sampleCount: 103936), 406)
        XCTAssertEqual(extractor.frameCount(sampleCount: 256), 1)
        XCTAssertEqual(extractor.frameCount(sampleCount: 127), 0)
        XCTAssertEqual(extractor.frameCount(sampleCount: 128), 1)
    }

    func testPromptMelMatchesFixture() throws {
        let fixtures = try LuxTtsFixtures.load()
        let audio = try LuxTtsFixtures.loadFloats("prompt_24k_f32le.bin")
        XCTAssertEqual(audio.count, fixtures.prompt.wav24kSamples)

        let extractor = LuxTtsMelExtractor()
        let mel = extractor.extract(audio: audio)
        XCTAssertEqual(mel.count, fixtures.prompt.melFrames)
        XCTAssertEqual(mel.first?.count, fixtures.prompt.melDim)

        // Fixture mel is feat-scaled (×0.1); the extractor output is not.
        let reference = try LuxTtsFixtures.loadFloats("prompt_mel_f32le.bin")
        XCTAssertEqual(reference.count, mel.count * fixtures.prompt.melDim)

        var maxAbsDiff: Float = 0
        for frame in 0..<mel.count {
            for d in 0..<fixtures.prompt.melDim {
                let got = mel[frame][d] * LuxTtsConstants.featScale
                let expected = reference[frame * fixtures.prompt.melDim + d]
                maxAbsDiff = max(maxAbsDiff, abs(got - expected))
            }
        }
        XCTAssertLessThan(maxAbsDiff, 1e-3, "mel parity gate: max_abs must stay < 1e-3")
    }
}
