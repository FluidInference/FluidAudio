import Foundation
import XCTest

@testable import FluidAudio

/// Network-free structural tests for the `StyleTTS2AneStage` enum and the
/// `StyleTTS2AneStageTimings` aggregate. Pins the 7-stage order so that
/// future re-cuts can't silently shuffle the chain.
final class StyleTTS2AneStageOrderTests: XCTestCase {

    func testAllStagesEnumerated() {
        XCTAssertEqual(
            StyleTTS2AneStage.allCases.count, 7,
            "Pipeline must expose exactly 7 stages")
    }

    /// The order in `allCases` (= source order in the enum decl) is the
    /// chain's *call order*. Anything that breaks this ordering will silently
    /// rewire the model dispatch.
    func testStagesAreInExecutionOrder() {
        XCTAssertEqual(
            StyleTTS2AneStage.allCases,
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
        XCTAssertEqual(StyleTTS2AneStage.plBert.rawValue, "plBert")
        XCTAssertEqual(StyleTTS2AneStage.postBert.rawValue, "postBert")
        XCTAssertEqual(StyleTTS2AneStage.alignment.rawValue, "alignment")
        XCTAssertEqual(StyleTTS2AneStage.diffusionStep.rawValue, "diffusionStep")
        XCTAssertEqual(StyleTTS2AneStage.prosody.rawValue, "prosody")
        XCTAssertEqual(StyleTTS2AneStage.noise.rawValue, "noise")
        XCTAssertEqual(StyleTTS2AneStage.vocoder.rawValue, "vocoder")
    }

    func testBundleNamesAreUnique() {
        let names = Set(StyleTTS2AneStage.allCases.map(\.bundleName))
        XCTAssertEqual(
            names.count, StyleTTS2AneStage.allCases.count,
            "Each stage must have a unique mlmodelc filename")
    }

    func testStageTimingsTotalIsLinearSum() {
        var t = StyleTTS2AneStageTimings()
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
        let t = StyleTTS2AneStageTimings()
        XCTAssertEqual(t.totalMs, 0)
    }

    // MARK: - Compute-units mapping

    func testDefaultComputeUnitsPinAllStagesToANEExceptNoise() {
        // Empirical defaults from the conversion script: every stage on
        // .cpuAndNeuralEngine except Noise (fp32 phase precision -> .all).
        let cu = StyleTTS2AneComputeUnits.default
        XCTAssertEqual(cu.units(for: .plBert), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .postBert), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .alignment), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .diffusionStep), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .prosody), .cpuAndNeuralEngine)
        XCTAssertEqual(cu.units(for: .noise), .all)
        XCTAssertEqual(cu.units(for: .vocoder), .cpuAndNeuralEngine)
    }

    func testCpuOnlyPresetSetsAllStagesCpuOnly() {
        let cu = StyleTTS2AneComputeUnits.cpuOnly
        for stage in StyleTTS2AneStage.allCases {
            XCTAssertEqual(
                cu.units(for: stage), .cpuOnly,
                "cpuOnly preset must pin \(stage) to .cpuOnly")
        }
    }

    func testAllAnePresetSetsAllStagesAne() {
        let cu = StyleTTS2AneComputeUnits.allAne
        for stage in StyleTTS2AneStage.allCases {
            XCTAssertEqual(
                cu.units(for: stage), .cpuAndNeuralEngine,
                "allAne preset must pin \(stage) to .cpuAndNeuralEngine")
        }
    }

    func testTtsPresetEnumMappingMatchesDirectInits() {
        XCTAssertEqual(
            StyleTTS2AneComputeUnits(preset: .default), .default)
        XCTAssertEqual(
            StyleTTS2AneComputeUnits(preset: .allAne), .allAne)
        XCTAssertEqual(
            StyleTTS2AneComputeUnits(preset: .cpuAndGpu), .cpuAndGpu)
        XCTAssertEqual(
            StyleTTS2AneComputeUnits(preset: .cpuOnly), .cpuOnly)
    }
}
