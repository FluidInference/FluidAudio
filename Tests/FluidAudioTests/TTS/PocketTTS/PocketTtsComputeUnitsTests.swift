import CoreML
import XCTest

@testable import FluidAudio

/// Pure-logic tests for the per-stage compute-unit overrides (#881). No model
/// files or network access required.
final class PocketTtsComputeUnitsTests: XCTestCase {

    // MARK: - Defaults reproduce the measured-fastest routing

    func testDefaultGpuPlacementMatchesMeasuredRouting() {
        let r = PocketTtsComputeUnits.default.resolved(for: .gpu)
        XCTAssertEqual(r.conditioner, .all)
        XCTAssertEqual(r.flowLM, .all)
        XCTAssertEqual(r.flowDecoder, .all)
        XCTAssertEqual(r.mimiDecoder, .cpuOnly)
    }

    func testDefaultAnePlacementPinsFlowLMToNeuralEngine() {
        let r = PocketTtsComputeUnits.default.resolved(for: .ane)
        XCTAssertEqual(r.conditioner, .all)
        XCTAssertEqual(r.flowLM, .cpuAndNeuralEngine)
        XCTAssertEqual(r.flowDecoder, .all)
        XCTAssertEqual(r.mimiDecoder, .cpuOnly)
    }

    func testDefaultStatePipelineAndMimiEncoder() {
        XCTAssertEqual(PocketTtsComputeUnits.default.resolvedStatePipeline, .cpuAndNeuralEngine)
        XCTAssertEqual(PocketTtsComputeUnits.default.resolvedMimiEncoder, .cpuAndGPU)
    }

    // MARK: - Overrides

    /// #881: the failing stage is `flow_decoder_fused`, which no `placement`
    /// value moves off `.all`. A single-stage override must reach it while
    /// leaving the other stages on their defaults.
    func testSingleStageOverrideOnlyMovesThatStage() {
        let units = PocketTtsComputeUnits(flowDecoder: .cpuAndGPU)
        let r = units.resolved(for: .gpu)
        XCTAssertEqual(r.flowDecoder, .cpuAndGPU)
        XCTAssertEqual(r.conditioner, .all)
        XCTAssertEqual(r.flowLM, .all)
        XCTAssertEqual(r.mimiDecoder, .cpuOnly)
    }

    func testOverrideBeatsPlacementDefaultForFlowLM() {
        let units = PocketTtsComputeUnits(flowLM: .cpuAndGPU)
        XCTAssertEqual(units.resolved(for: .ane).flowLM, .cpuAndGPU)
        XCTAssertEqual(units.resolvedStatePipeline, .cpuAndGPU)
    }

    func testAvoidNeuralEngineSchedulesNoStageOnTheANE() {
        let units = PocketTtsComputeUnits.avoidNeuralEngine
        for placement in [PocketTtsModelPlacement.gpu, .ane] {
            let r = units.resolved(for: placement)
            for stage in [r.conditioner, r.flowLM, r.flowDecoder, r.mimiDecoder] {
                XCTAssertNotEqual(stage, .all, "\(placement)")
                XCTAssertNotEqual(stage, .cpuAndNeuralEngine, "\(placement)")
            }
            XCTAssertEqual(r.mimiDecoder, .cpuOnly, "mimi stays on the CPU, where it is fastest")
        }
        XCTAssertEqual(units.resolvedMimiEncoder, .cpuAndGPU)
    }

    func testUniformAppliesToEveryStage() {
        let units = PocketTtsComputeUnits.uniform(.cpuOnly)
        let r = units.resolved(for: .gpu)
        XCTAssertEqual([r.conditioner, r.flowLM, r.flowDecoder, r.mimiDecoder], Array(repeating: .cpuOnly, count: 4))
        XCTAssertEqual(units.resolvedStatePipeline, .cpuOnly)
        XCTAssertEqual(units.resolvedMimiEncoder, .cpuOnly)
    }

    func testDefaultIsAllNil() {
        let d = PocketTtsComputeUnits.default
        XCTAssertNil(d.conditioner)
        XCTAssertNil(d.flowLM)
        XCTAssertNil(d.flowDecoder)
        XCTAssertNil(d.mimiDecoder)
        XCTAssertNil(d.mimiEncoder)
        XCTAssertEqual(d, PocketTtsComputeUnits())
    }
}
