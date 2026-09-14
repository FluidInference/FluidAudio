import Foundation
import XCTest

@testable import FluidAudio

/// Output-capacity budget math and model-set selection (#924). The budgets
/// are what a caller can actually use: the static bucket shapes minus the
/// voice's own conditioning/prompt footprint.
final class ChatterboxNanoCapacityTests: XCTestCase {

    // MARK: - Bucket shapes

    func testBucketShapes() {
        XCTAssertEqual(ChatterboxNanoOutputCapacity.standard.flowTokenBucket, 500)
        XCTAssertEqual(ChatterboxNanoOutputCapacity.standard.melFrameBucket, 1000)
        XCTAssertEqual(ChatterboxNanoOutputCapacity.extended.flowTokenBucket, 1000)
        XCTAssertEqual(ChatterboxNanoOutputCapacity.extended.melFrameBucket, 2000)
    }

    func testStandardBucketMatchesConstants() {
        // The legacy constants describe the `.standard` capacity.
        XCTAssertEqual(
            ChatterboxNanoOutputCapacity.standard.flowTokenBucket,
            ChatterboxNanoConstants.flowTokenBucket)
        XCTAssertEqual(
            ChatterboxNanoOutputCapacity.standard.melFrameBucket,
            ChatterboxNanoConstants.melFrameBucket)
    }

    // MARK: - Generation budget (#924 numbers)

    func testGenerationBudgetForBuiltInVoice() {
        // The built-in voice ships 250 prompt tokens; 3 silence tokens are
        // appended before vocoding. 500 − 250 − 3 = 247 ≈ 9.9 s at 25 Hz.
        XCTAssertEqual(
            ChatterboxNanoOutputCapacity.standard.generationBudget(promptTokens: 250), 247)
        // 1000 − 250 − 3 = 747 ≈ 29.9 s.
        XCTAssertEqual(
            ChatterboxNanoOutputCapacity.extended.generationBudget(promptTokens: 250), 747)
    }

    func testGenerationBudgetClampsToZero() {
        XCTAssertEqual(
            ChatterboxNanoOutputCapacity.standard.generationBudget(promptTokens: 600), 0)
    }

    // MARK: - Model-set selection

    func testRequiredModelsPerCapacity() {
        XCTAssertEqual(
            ModelNames.ChatterboxNano.requiredModels(capacity: .standard),
            [
                ModelNames.ChatterboxNano.prefillFile,
                ModelNames.ChatterboxNano.decodeFile,
                ModelNames.ChatterboxNano.flowFile,
                ModelNames.ChatterboxNano.vocoderFile,
            ])
        XCTAssertEqual(
            ModelNames.ChatterboxNano.requiredModels(capacity: .extended),
            [
                ModelNames.ChatterboxNano.prefillFile,
                ModelNames.ChatterboxNano.decodeFile,
                ModelNames.ChatterboxNano.flowFileExtended,
                ModelNames.ChatterboxNano.vocoderFileExtended,
            ])
    }

    func testRepoVariantMapping() {
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .chatterboxNano, variant: nil),
            ModelNames.ChatterboxNano.requiredModels(capacity: .standard))
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .chatterboxNano, variant: "extended"),
            ModelNames.ChatterboxNano.requiredModels(capacity: .extended))
        // Unknown variant strings fall back to the standard set.
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .chatterboxNano, variant: "bogus"),
            ModelNames.ChatterboxNano.requiredModels(capacity: .standard))
    }

    func testExtendedFileNamesMatchPublishedArtifacts() {
        XCTAssertEqual(ModelNames.ChatterboxNano.flowFileExtended, "FlowMean-N1000-fp16.mlmodelc")
        XCTAssertEqual(ModelNames.ChatterboxNano.vocoderFileExtended, "HiFT-T2000-fp16.mlmodelc")
    }
}
