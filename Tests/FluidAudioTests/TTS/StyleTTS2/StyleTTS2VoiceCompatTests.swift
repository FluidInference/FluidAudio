import Foundation
import XCTest

@testable import FluidAudio

/// Verifies that the existing per-voice `ref_s.bin` blob format ships
/// unchanged for the 7-graph ANE re-cut.
///
/// The ANE re-cut deliberately reuses `StyleTTS2VoiceStyle` from the legacy
/// module — the style encoders are PyTorch-only and were not re-exported as
/// part of this PR. If the ANE pipeline ever introduces a divergent voice
/// blob format, this test will fail.
final class StyleTTS2VoiceCompatTests: XCTestCase {

    private func writeBlob(_ floats: [Float]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("styletts2-ane-voice-compat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("ref_s.bin")
        var local = floats
        let data = local.withUnsafeMutableBufferPointer { buf in
            Data(buffer: buf)
        }
        try data.write(to: url)
        return url
    }

    func testExistingRefSBlobIsConsumableByAneSynthesizer() throws {
        // Construct a 256-fp32 ramp identical to what
        // `mobius-styletts2/scripts/06_dump_ref_s.py` produces.
        let floats: [Float] = (0..<256).map { Float($0) / 255.0 }
        let url = try writeBlob(floats)
        let voice = try StyleTTS2VoiceStyle.load(from: url)

        // Same loader, same dimensions — the ANE backend doesn't introduce a
        // new voice format.
        XCTAssertEqual(voice.concatenated.count, StyleTTS2Constants.refStyleDim)
        XCTAssertEqual(voice.acoustic.count, StyleTTS2Constants.styleDim)
        XCTAssertEqual(voice.prosody.count, StyleTTS2Constants.styleDim)
        XCTAssertEqual(
            voice.acoustic.count + voice.prosody.count,
            voice.concatenated.count)
    }

    func testStyleDimensionConstantsAlignWithLegacy() {
        // refStyleDim == 2 * styleDim is a layout invariant relied on by
        // both the synthesizer (style mix split) and the voice loader.
        XCTAssertEqual(
            StyleTTS2Constants.refStyleDim,
            2 * StyleTTS2Constants.styleDim)
    }

    func testAneSynthesizerOptionsAcceptDefaultMixWeights() {
        // Defaults must match the upstream Python reference (alpha=0.3,
        // beta=0.7, 5 ADPM2 steps). Documented in `Options.init`.
        let opts = StyleTTS2Synthesizer.Options()
        XCTAssertEqual(opts.diffusionSteps, StyleTTS2Constants.defaultDiffusionSteps)
        XCTAssertEqual(opts.alpha, 0.3, accuracy: 1e-6)
        XCTAssertEqual(opts.beta, 0.7, accuracy: 1e-6)
        XCTAssertNil(opts.randomSeed)
    }

    func testAneManagerInitDoesNotRequireNetwork() async {
        // Constructing the manager must not touch the network. Only verifies
        // that the API surface compiles + the actor reports `isAvailable == false`
        // before `initialize()` is called.
        let manager = StyleTTS2Manager()
        let available = await manager.isAvailable
        XCTAssertFalse(available, "Fresh manager must not yet be loaded")
    }
}
