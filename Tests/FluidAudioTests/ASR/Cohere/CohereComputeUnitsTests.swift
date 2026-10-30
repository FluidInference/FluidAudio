import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Per-model compute-unit resolution for `CoherePipeline.loadModels`
/// (FluidAudio #1045).
@available(macOS 14, iOS 17, *)
final class CohereComputeUnitsTests: XCTestCase {

    // MARK: - Defaults

    func testEncoderDefaultsToNeuralEngine() {
        // Under `.all` Core ML may route the INT8 encoder to the GPU, where
        // the weights are materialised at full precision (~12 GB footprint).
        XCTAssertEqual(CoherePipeline.defaultEncoderComputeUnits, .cpuAndNeuralEngine)
    }

    func testDecoderDefaultsToNeuralEngine() {
        XCTAssertEqual(CoherePipeline.defaultDecoderComputeUnits, .cpuAndNeuralEngine)
    }

    func testNoArgumentsResolveToDefaults() {
        let units = CoherePipeline.resolveComputeUnits(shared: nil, encoder: nil, decoder: nil)
        XCTAssertEqual(units.encoder, CoherePipeline.defaultEncoderComputeUnits)
        XCTAssertEqual(units.decoder, CoherePipeline.defaultDecoderComputeUnits)
    }

    // MARK: - Shared override

    func testSharedUnitsApplyToBothModels() {
        let units = CoherePipeline.resolveComputeUnits(shared: .cpuAndGPU, encoder: nil, decoder: nil)
        XCTAssertEqual(units.encoder, .cpuAndGPU)
        XCTAssertEqual(units.decoder, .cpuAndGPU)
    }

    func testSharedAllIsHonouredWhenExplicit() {
        // A caller that asks for `.all` gets `.all`, memory cost and all.
        let units = CoherePipeline.resolveComputeUnits(shared: .all, encoder: nil, decoder: nil)
        XCTAssertEqual(units.encoder, .all)
        XCTAssertEqual(units.decoder, .all)
    }

    // MARK: - Per-model override

    func testPerModelUnitsWinOverShared() {
        let units = CoherePipeline.resolveComputeUnits(
            shared: .cpuOnly, encoder: .cpuAndNeuralEngine, decoder: .all)
        XCTAssertEqual(units.encoder, .cpuAndNeuralEngine)
        XCTAssertEqual(units.decoder, .all)
    }

    func testEncoderOnlyOverrideLeavesDecoderOnShared() {
        let units = CoherePipeline.resolveComputeUnits(shared: .cpuAndGPU, encoder: .cpuOnly, decoder: nil)
        XCTAssertEqual(units.encoder, .cpuOnly)
        XCTAssertEqual(units.decoder, .cpuAndGPU)
    }

    func testDecoderOnlyOverrideLeavesEncoderOnDefault() {
        let units = CoherePipeline.resolveComputeUnits(shared: nil, encoder: nil, decoder: .cpuAndGPU)
        XCTAssertEqual(units.encoder, CoherePipeline.defaultEncoderComputeUnits)
        XCTAssertEqual(units.decoder, .cpuAndGPU)
    }
}
