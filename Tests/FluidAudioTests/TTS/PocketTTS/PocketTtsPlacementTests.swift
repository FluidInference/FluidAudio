import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Pure-logic unit tests for the `.ane` placement plumbing (rank-4 split-KV
/// models, mobius Trials 19/20). No model files or network access required.
final class PocketTtsPlacementTests: XCTestCase {

    // MARK: - ModelNames.requiredModels(precision:placement:)

    func testRequiredModelsGpuMatchesLegacySet() {
        XCTAssertEqual(
            ModelNames.PocketTTS.requiredModels(precision: .fp16, placement: .gpu),
            ModelNames.PocketTTS.requiredModels(precision: .fp16)
        )
    }

    func testRequiredModelsAneSwapsConditionerAndFlowLM() {
        let ane = ModelNames.PocketTTS.requiredModels(precision: .fp16, placement: .ane)
        XCTAssertTrue(ane.contains(ModelNames.PocketTTS.flowlmStepAneFile))
        XCTAssertTrue(ane.contains(ModelNames.PocketTTS.condPrefillAneFile))
        XCTAssertFalse(ane.contains(ModelNames.PocketTTS.flowlmStepFile))
        XCTAssertFalse(ane.contains(ModelNames.PocketTTS.condPrefillFile))
        // The fused flow decoder and mimi are shared across placements.
        XCTAssertTrue(ane.contains(ModelNames.PocketTTS.flowDecoderFusedFile))
        XCTAssertTrue(ane.contains(ModelNames.PocketTTS.mimiDecoderFile))
    }

    func testRequiredModelsAneStateIsMultifunctionPlusMimi() {
        let state = ModelNames.PocketTTS.requiredModels(precision: .fp16, placement: .aneState)
        XCTAssertEqual(
            state,
            [
                ModelNames.PocketTTS.pocketStateFile,
                ModelNames.PocketTTS.mimiDecoderFile,
                ModelNames.PocketTTS.constantsBinDir,
            ]
        )
        // The multifunction package fuses conditioner + FlowLM + flow
        // decoder, so NONE of the IO model files belong in the set.
        XCTAssertFalse(state.contains(ModelNames.PocketTTS.flowDecoderFusedFile))
        XCTAssertFalse(state.contains(ModelNames.PocketTTS.flowlmStepAneFile))
        XCTAssertFalse(state.contains(ModelNames.PocketTTS.condPrefillAneFile))
        // Precision is ignored (fp16 only): the int8 FlowLM never appears.
        XCTAssertEqual(
            ModelNames.PocketTTS.requiredModels(precision: .int8, placement: .aneState),
            state
        )
    }

    func testAneStatePlacementRawValueRoundTrip() {
        // The CLI parses `--placement ane-state` via the raw value.
        XCTAssertEqual(PocketTtsModelPlacement(rawValue: "ane-state"), .aneState)
        XCTAssertEqual(PocketTtsModelPlacement.aneState.rawValue, "ane-state")
        XCTAssertEqual(
            ModelNames.PocketTTS.pocketStateFile, "pocket_state.mlmodelc")
        XCTAssertEqual(ModelNames.PocketTTS.StateFunction.writeState, "write_state")
        XCTAssertEqual(ModelNames.PocketTTS.StateFunction.prefill, "prefill")
        XCTAssertEqual(ModelNames.PocketTTS.StateFunction.generate, "generate")
    }

    // MARK: - PocketTtsLayerKeys.aneKeys

    func testAneKeysExplicitNamesAndOrdering() {
        let keys = PocketTtsLayerKeys.aneKeys(layers: 6, kind: .flowlmStep)
        XCTAssertTrue(keys.isSplitKV)
        XCTAssertEqual(keys.layerCount, 6)
        XCTAssertEqual(keys.cacheKeys, (0..<6).map { "new_k_cache\($0)" })
        XCTAssertEqual(keys.vCacheKeys, (0..<6).map { "new_v_cache\($0)" })
        XCTAssertEqual(keys.positionKeys, (0..<6).map { "new_position\($0)" })
        XCTAssertEqual(keys.transformerOut, "transformer_out")
        XCTAssertEqual(keys.eosLogit, "is_eos")
    }

    func testAneKeysCondKindHasNoFlowLMOutputs() {
        let keys = PocketTtsLayerKeys.aneKeys(layers: 6, kind: .condStep)
        XCTAssertTrue(keys.isSplitKV)
        XCTAssertNil(keys.transformerOut)
        XCTAssertNil(keys.eosLogit)
    }

    // MARK: - emptyKVCacheState

    func testEmptyKVCacheStateSplitShapes() throws {
        let maxLen = PocketTtsConstants.kvCacheMaxLen
        let split = try PocketTtsSynthesizer.emptyKVCacheState(layers: 2, splitKV: true)
        XCTAssertTrue(split.isSplitKV)
        XCTAssertEqual(split.caches.count, 2)
        XCTAssertEqual(split.vCaches?.count, 2)
        XCTAssertEqual(split.caches[0].shape.map(\.intValue), [1, maxLen, 16, 64])
        XCTAssertEqual(split.vCaches?[0].shape.map(\.intValue), [1, maxLen, 16, 64])

        let combined = try PocketTtsSynthesizer.emptyKVCacheState(layers: 2)
        XCTAssertFalse(combined.isSplitKV)
        XCTAssertNil(combined.vCaches)
        XCTAssertEqual(combined.caches[0].shape.map(\.intValue), [2, 1, maxLen, 16, 64])
    }

    // MARK: - kvCacheStateFromSnapshot (split)

    func testSnapshotSplitMatchesCombinedBlocks() throws {
        // One layer, tiny seq, deterministic values: K block = 1..<N,
        // V block = 1000..<1000+N (flat row-major [2, 1, seq, 16, 64]).
        let seqLen = 3
        let perKV = seqLen * 16 * 64
        var flat = [Float]()
        flat.append(contentsOf: (0..<perKV).map { Float($0) })
        flat.append(contentsOf: (0..<perKV).map { Float($0) + 1_000_000 })
        let snapshot = PocketTtsVoiceCacheSnapshot(
            layers: [.init(cache: flat, offset: seqLen)],
            cacheSeqLen: seqLen
        )

        let split = try PocketTtsSynthesizer.kvCacheStateFromSnapshot(
            snapshot, layers: 1, splitKV: true)
        let combined = try PocketTtsSynthesizer.kvCacheStateFromSnapshot(
            snapshot, layers: 1, splitKV: false)

        XCTAssertEqual(split.positions[0][0].floatValue, Float(seqLen))
        XCTAssertEqual(combined.positions[0][0].floatValue, Float(seqLen))

        let destSeq = PocketTtsConstants.kvCacheMaxLen
        let destPerKV = destSeq * 16 * 64
        let kSplit = split.caches[0].dataPointer.bindMemory(to: Float.self, capacity: destPerKV)
        let vSplit = split.vCaches![0].dataPointer.bindMemory(to: Float.self, capacity: destPerKV)
        let comb = combined.caches[0].dataPointer.bindMemory(
            to: Float.self, capacity: 2 * destPerKV)

        // Valid prefix matches the combined layout's K and V blocks exactly.
        for i in 0..<perKV {
            XCTAssertEqual(kSplit[i], comb[i])
            XCTAssertEqual(vSplit[i], comb[destPerKV + i])
        }
        // Slots beyond the prefix are zero (the `_ane` models' host contract).
        XCTAssertEqual(kSplit[perKV], 0)
        XCTAssertEqual(vSplit[perKV], 0)
    }

    // MARK: - BOS start sequence

    func testBosStartSequenceSplitUsesBosLatent() throws {
        let bos: [Float] = (0..<32).map { Float($0) * 0.5 }
        let seq = try PocketTtsSynthesizer.createBosStartSequence(
            bosEmbedding: bos, splitKV: true)
        XCTAssertEqual(seq.shape.map(\.intValue), [1, 1, 32])
        for i in 0..<32 {
            XCTAssertEqual(seq[i].floatValue, bos[i])
            XCTAssertFalse(seq[i].floatValue.isNaN)
        }
    }

    func testBosStartSequenceCombinedUsesNaNProtocol() throws {
        let bos: [Float] = (0..<32).map { Float($0) * 0.5 }
        let seq = try PocketTtsSynthesizer.createBosStartSequence(
            bosEmbedding: bos, splitKV: false)
        for i in 0..<32 {
            XCTAssertTrue(seq[i].floatValue.isNaN)
        }
    }
}
