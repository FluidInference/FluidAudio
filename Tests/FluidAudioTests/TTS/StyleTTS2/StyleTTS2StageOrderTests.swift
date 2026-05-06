import Foundation
import XCTest

@testable import FluidAudio

/// Network-free structural tests for the `StyleTTS2Stage` enum and the
/// `StyleTTS2StageTimings` aggregate. Pins the 7-stage order so that
/// future re-cuts can't silently shuffle the chain.
final class StyleTTS2StageOrderTests: XCTestCase {

    func testAllStagesEnumerated() {
        XCTAssertEqual(
            StyleTTS2Stage.allCases.count, 7,
            "Pipeline must expose exactly 7 stages")
    }

    /// The order in `allCases` (= source order in the enum decl) is the
    /// chain's *call order*. Anything that breaks this ordering will silently
    /// rewire the model dispatch.
    func testStagesAreInExecutionOrder() {
        XCTAssertEqual(
            StyleTTS2Stage.allCases,
            [
                .plBert,
                .postBert,
                .alignment,
                .diffusionStep,
                .prosody,
                .noise,
                .vocoder,
            ])
    }

    func testStageRawValuesAreStableForLogging() {
        // Per-stage timings are reported by raw value in error paths and
        // benchmark JSON. Lock these strings.
        XCTAssertEqual(StyleTTS2Stage.plBert.rawValue, "plBert")
        XCTAssertEqual(StyleTTS2Stage.postBert.rawValue, "postBert")
        XCTAssertEqual(StyleTTS2Stage.alignment.rawValue, "alignment")
        XCTAssertEqual(StyleTTS2Stage.diffusionStep.rawValue, "diffusionStep")
        XCTAssertEqual(StyleTTS2Stage.prosody.rawValue, "prosody")
        XCTAssertEqual(StyleTTS2Stage.noise.rawValue, "noise")
        XCTAssertEqual(StyleTTS2Stage.vocoder.rawValue, "vocoder")
    }

    func testBundleNamesAreUnique() {
        let names = Set(StyleTTS2Stage.allCases.map(\.bundleName))
        XCTAssertEqual(
            names.count, StyleTTS2Stage.allCases.count,
            "Each stage must have a unique mlmodelc filename")
    }

    func testStageTimingsTotalIsLinearSum() {
        var t = StyleTTS2StageTimings()
        t.plBert = 1
        t.postBert = 2
        t.alignment = 4
        t.diffusionStep = 8
        t.prosody = 16
        t.noise = 32
        t.vocoder = 64
        XCTAssertEqual(t.totalMs, 127)
    }

    func testStageTimingsZeroByDefault() {
        let t = StyleTTS2StageTimings()
        XCTAssertEqual(t.totalMs, 0)
    }

    // MARK: - Compute-units mapping

    func testDefaultComputeUnitsPinAllStagesToANEExceptNoise() {
        // Empirical defaults from the conversion script: every stage on
        // .cpuAndNeuralEngine except Noise (fp32 phase precision -> .all).
        let cu = StyleTTS2ComputeUnits.default
        XCTAssertEqual(cu.units(for: .plBert), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .postBert), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .alignment), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .diffusionStep), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .prosody), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .noise), .all)
        XCTAssertEqual(cu.units(for: .vocoder), .cpuAndNeuralEngine)
    }

    func testCpuOnlyPresetSetsAllStagesCpuOnly() {
        let cu = StyleTTS2ComputeUnits.cpuOnly
        for stage in StyleTTS2Stage.allCases {
            XCTAssertEqual(
                cu.units(for: stage), .cpuOnly,
                "cpuOnly preset must pin \(stage) to .cpuOnly")
        }
    }

    func testAllAnePresetSetsAllStagesAne() {
        let cu = StyleTTS2ComputeUnits.allAne
        for stage in StyleTTS2Stage.allCases {
            XCTAssertEqual(
                cu.units(for: stage), .cpuAndNeuralEngine,
                "allAne preset must pin \(stage) to .cpuAndNeuralEngine")
        }
    }

    func testTtsPresetEnumMappingMatchesDirectInits() {
        XCTAssertEqual(
            StyleTTS2ComputeUnits(preset: .default), .default)
        XCTAssertEqual(
            StyleTTS2ComputeUnits(preset: .allAne), .allAne)
        XCTAssertEqual(
            StyleTTS2ComputeUnits(preset: .cpuAndGpu), .cpuAndGpu)
        XCTAssertEqual(
            StyleTTS2ComputeUnits(preset: .cpuOnly), .cpuOnly)
    }
}
