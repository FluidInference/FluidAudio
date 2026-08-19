import XCTest

@testable import FluidAudio

/// Pins `PocketTtsResourceDownloader.shouldSkipAsset(at:required:)` — the
/// download-time filter that keeps unused precision/placement variants off
/// the network (#853). The upstream `v2/<lang>/` directory ships every
/// FlowLM precision plus the `_ane`/`pocket_state` packages side by side;
/// only the bundles in `ModelNames.PocketTTS.requiredModels(precision:placement:)`
/// may come through.
final class PocketTtsDownloadFilterTests: XCTestCase {

    private func skips(
        _ path: String, precision: PocketTtsPrecision = .fp16,
        placement: PocketTtsModelPlacement = .gpu
    ) -> Bool {
        PocketTtsResourceDownloader.shouldSkipAsset(
            at: path,
            required: ModelNames.PocketTTS.requiredModels(
                precision: precision, placement: placement))
    }

    // MARK: - Precision (gpu placement)

    func testFp16KeepsDefaultFlowlmAndSkipsInt8Variant() {
        XCTAssertFalse(skips("v2.1/english/flowlm_step.mlmodelc", precision: .fp16))
        XCTAssertFalse(skips("v2.1/english/flowlm_step.mlmodelc/model.mil", precision: .fp16))
        XCTAssertTrue(skips("v2.1/english/flowlm_stepv2.mlmodelc", precision: .fp16))
        XCTAssertTrue(skips("v2.1/english/flowlm_stepv2.mlmodelc/weights/weight.bin", precision: .fp16))
    }

    func testInt8KeepsV2FlowlmAndSkipsDefaultVariant() {
        XCTAssertFalse(skips("v2.1/english/flowlm_stepv2.mlmodelc", precision: .int8))
        XCTAssertTrue(skips("v2.1/english/flowlm_step.mlmodelc", precision: .int8))
    }

    // MARK: - Placement

    func testGpuPlacementSkipsAneAndStateVariants() {
        XCTAssertTrue(skips("v2.1/english/flowlm_step_ane.mlmodelc"))
        XCTAssertTrue(skips("v2.1/english/cond_prefill_ane.mlmodelc"))
        XCTAssertTrue(skips("v2.1/english/pocket_state.mlmodelc"))
        XCTAssertFalse(skips("v2.1/english/cond_prefill.mlmodelc"))
        XCTAssertFalse(skips("v2.1/english/flow_decoder_fused.mlmodelc"))
        XCTAssertFalse(skips("v2.1/english/mimi_decoder.mlmodelc"))
    }

    func testAnePlacementKeepsAneVariantsAndSkipsGpuOnes() {
        XCTAssertFalse(skips("v2.1/english/flowlm_step_ane.mlmodelc", placement: .ane))
        XCTAssertFalse(skips("v2.1/english/cond_prefill_ane.mlmodelc", placement: .ane))
        XCTAssertFalse(skips("v2.1/english/flow_decoder_fused.mlmodelc", placement: .ane))
        XCTAssertTrue(skips("v2.1/english/cond_prefill.mlmodelc", placement: .ane))
        XCTAssertTrue(skips("v2.1/english/flowlm_step.mlmodelc", placement: .ane))
        XCTAssertTrue(skips("v2.1/english/flowlm_stepv2.mlmodelc", placement: .ane))
        XCTAssertTrue(skips("v2.1/english/pocket_state.mlmodelc", placement: .ane))
    }

    func testAneStatePlacementKeepsOnlyStateAndMimi() {
        XCTAssertFalse(skips("v2.1/english/pocket_state.mlmodelc", placement: .aneState))
        XCTAssertFalse(skips("v2.1/english/mimi_decoder.mlmodelc", placement: .aneState))
        XCTAssertTrue(skips("v2.1/english/cond_prefill.mlmodelc", placement: .aneState))
        XCTAssertTrue(skips("v2.1/english/flowlm_step.mlmodelc", placement: .aneState))
        XCTAssertTrue(skips("v2.1/english/flow_decoder_fused.mlmodelc", placement: .aneState))
    }

    // MARK: - Non-model assets are unaffected by the required set

    func testConstantsBinAndLooseFilesAreKept() {
        XCTAssertFalse(skips("v2.1/english/constants_bin"))
        XCTAssertFalse(skips("v2.1/english/constants_bin/tokenizer.model"))
        XCTAssertFalse(skips("v2.1/english/constants_bin/alba.safetensors"))
        XCTAssertFalse(skips("v2.1/english/manifest.json"))
    }

    func testPackLocalCloningEncoderIsSkippedInPackDownloads() {
        // The per-language voice-cloning encoder (`mimi_encoderv3.mlmodelc`,
        // #793) is not part of any required model set — it is fetched lazily
        // by `ensurePackMimiEncoder` on first clone, so pack downloads for
        // synthesis-only users stay lean.
        XCTAssertTrue(skips("v2.1/spanish_24l/mimi_encoderv3.mlmodelc"))
        XCTAssertTrue(skips("v2.1/spanish_24l/mimi_encoderv3.mlmodelc/weights/weight.bin"))
        XCTAssertTrue(skips("v2.1/spanish_24l/mimi_encoderv3.mlmodelc", placement: .ane))
    }

    func testHistoricalExclusionsStillApply() {
        // .mlpackage sources are skipped even for a required model name.
        XCTAssertTrue(skips("v2.1/english/cond_prefill.mlpackage"))
        XCTAssertTrue(skips("v2.1/english/cond_prefill.mlpackage/Data/weight.bin"))
        XCTAssertTrue(skips("v2.1/english/constants"))
        XCTAssertTrue(skips("v2.1/english/constants/embed.npy"))
        XCTAssertTrue(skips("v2.1/english/verify.wav"))
        XCTAssertTrue(skips("v2.1/english/.DS_Store"))
    }
}
