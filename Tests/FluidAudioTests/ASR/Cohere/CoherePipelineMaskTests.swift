import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 14, iOS 17, *)
final class CoherePipelineMaskTests: XCTestCase {

    // MARK: - DecoderVariant

    func testDecoderVariantCompiledFileNames() {
        XCTAssertEqual(
            CoherePipeline.DecoderVariant.v2.compiledFileName,
            ModelNames.CohereTranscribe.decoderCacheExternalV2CompiledFile)
        XCTAssertEqual(
            CoherePipeline.DecoderVariant.v1.compiledFileName,
            ModelNames.CohereTranscribe.decoderCacheExternalCompiledFile)
        XCTAssertEqual(
            CoherePipeline.DecoderVariant.v2.compiledFileName,
            "cohere_decoder_cache_external_v2.mlmodelc")
        XCTAssertEqual(
            CoherePipeline.DecoderVariant.v1.compiledFileName,
            "cohere_decoder_cache_external.mlmodelc")
        XCTAssertEqual(
            CoherePipeline.DecoderVariant.v3.compiledFileName,
            ModelNames.CohereTranscribe.decoderCacheExternalV3CompiledFile)
        XCTAssertEqual(
            CoherePipeline.DecoderVariant.v3.compiledFileName,
            "cohere_decoder_cache_external_v3.mlmodelc")
        XCTAssertTrue(ModelNames.CohereTranscribe.requiredModels.contains("cohere_decoder_cache_external_v3.mlmodelc"))
        XCTAssertFalse(ModelNames.CohereTranscribe.requiredModels.contains("cohere_decoder_cache_external_v2.mlmodelc"))
    }

    func testDecoderVariantUsesStaticSelfMask() {
        XCTAssertTrue(CoherePipeline.DecoderVariant.v3.usesStaticSelfMask)
        XCTAssertTrue(CoherePipeline.DecoderVariant.v2.usesStaticSelfMask)
        XCTAssertFalse(CoherePipeline.DecoderVariant.v1.usesStaticSelfMask)
    }

    // MARK: - buildSelfAttentionMask: dynamic (v1) path

    func testDynamicMaskShapeAndZeroFillFloat32() throws {
        let step = 4
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: step, length: CohereAsrConfig.maxSeqLen, useStatic: false, dtype: .float32)

        XCTAssertEqual(mask.shape, [1, 1, 1, NSNumber(value: step + 1)])
        XCTAssertEqual(mask.dataType, .float32)

        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: step + 1)
        for i in 0...step {
            XCTAssertEqual(ptr[i], 0.0, "Dynamic mask must be zero-filled at \(i)")
        }
    }

    func testDynamicMaskShapeAndZeroFillFloat16() throws {
        let step = 7
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: step, length: CohereAsrConfig.maxSeqLen, useStatic: false, dtype: .float16)

        XCTAssertEqual(mask.shape, [1, 1, 1, NSNumber(value: step + 1)])
        XCTAssertEqual(mask.dataType, .float16)

        let ptr = mask.dataPointer.bindMemory(to: UInt16.self, capacity: step + 1)
        for i in 0...step {
            XCTAssertEqual(ptr[i], 0, "Dynamic FP16 mask must be zero-filled at \(i)")
        }
    }

    func testDynamicMaskAtStepZero() throws {
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: 0, length: CohereAsrConfig.maxSeqLen, useStatic: false, dtype: .float32)
        XCTAssertEqual(mask.shape, [1, 1, 1, 1])
        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: 1)
        XCTAssertEqual(ptr[0], 0.0)
    }

    // MARK: - buildSelfAttentionMask: static (v2) path

    func testStaticMaskShapeIsAlwaysMaxSeqLen() throws {
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: 0, length: CohereAsrConfig.maxSeqLen, useStatic: true, dtype: .float32)
        XCTAssertEqual(
            mask.shape, [1, 1, 1, NSNumber(value: CohereAsrConfig.maxSeqLen)])
    }

    func testStaticMaskCausalSplitFloat32() throws {
        let step = 10
        let length = CohereAsrConfig.maxSeqLen
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: step, length: length, useStatic: true, dtype: .float32)

        XCTAssertEqual(mask.shape, [1, 1, 1, NSNumber(value: length)])
        XCTAssertEqual(mask.dataType, .float32)

        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: length)
        for i in 0...step {
            XCTAssertEqual(ptr[i], 0.0, "Position \(i) (≤ step) must attend (0.0)")
        }
        for i in (step + 1)..<length {
            XCTAssertEqual(
                ptr[i], -1.0e4, "Position \(i) (> step) must be masked (-1e4)")
        }
    }

    func testStaticMaskCausalSplitFloat16() throws {
        let step = 5
        let length = CohereAsrConfig.maxSeqLen
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: step, length: length, useStatic: true, dtype: .float16)

        XCTAssertEqual(mask.shape, [1, 1, 1, NSNumber(value: length)])
        XCTAssertEqual(mask.dataType, .float16)

        let ptr = mask.dataPointer.bindMemory(to: UInt16.self, capacity: length)
        // FP16 zero is 0x0000; -1e4 is a nonzero sign-bit-set pattern. Assert
        // shape (zero vs. uniform nonzero) rather than hardcoding bits.
        for i in 0...step {
            XCTAssertEqual(
                ptr[i], 0, "Position \(i) (≤ step) must attend (FP16 zero)")
        }
        let blockedBits = ptr[step + 1]
        XCTAssertNotEqual(blockedBits, 0, "Blocked positions must be nonzero")
        XCTAssertNotEqual(
            blockedBits & 0x8000, 0, "Blocked positions must be negative (sign bit set)")
        for i in (step + 1)..<length {
            XCTAssertEqual(
                ptr[i], blockedBits,
                "All blocked positions must hold identical FP16 bits; mismatch at \(i)")
        }
    }

    func testStaticMaskAtFinalStepHasNoBlockedPositions() throws {
        // step == maxSeqLen - 1 → every position attends, none masked.
        let length = CohereAsrConfig.maxSeqLen
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: length - 1, length: length, useStatic: true, dtype: .float32)

        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: length)
        for i in 0..<length {
            XCTAssertEqual(ptr[i], 0.0, "Final step: every position attends, got \(ptr[i]) at \(i)")
        }
    }

    func testStaticMaskAtStepZeroBlocksAllExceptFirst() throws {
        let length = CohereAsrConfig.maxSeqLen
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: 0, length: length, useStatic: true, dtype: .float32)

        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: length)
        XCTAssertEqual(ptr[0], 0.0, "step=0 attends position 0")
        for i in 1..<length {
            XCTAssertEqual(ptr[i], -1.0e4, "step=0 must mask position \(i)")
        }
    }

    // MARK: - Cache length is a runtime value (#1013)

    func testStaticMaskHonorsProbedCacheLength() throws {
        // A 256-slot decoder must get a 256-wide mask, not the legacy 108.
        let length = 256
        let step = 150
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: step, length: length, useStatic: true, dtype: .float32)
        XCTAssertEqual(mask.shape, [1, 1, 1, NSNumber(value: length)])
        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: length)
        for i in 0...step { XCTAssertEqual(ptr[i], 0) }
        for i in (step + 1)..<length { XCTAssertEqual(ptr[i], -1.0e4) }
    }

    func testResolveMaxNewTokensDefaultsToCacheCapacity() {
        // The last prompt step yields the first output, so a 10-token prompt
        // costs 9 of the cache's steps.
        XCTAssertEqual(
            CoherePipeline.resolveMaxNewTokens(requested: nil, cacheLength: 256, promptCount: 10), 247)
        XCTAssertEqual(
            CoherePipeline.resolveMaxNewTokens(requested: nil, cacheLength: 108, promptCount: 10), 99)
    }

    func testResolveMaxNewTokensClampsRequestToCache() {
        XCTAssertEqual(
            CoherePipeline.resolveMaxNewTokens(requested: 50, cacheLength: 256, promptCount: 10), 50)
        XCTAssertEqual(
            CoherePipeline.resolveMaxNewTokens(requested: 1000, cacheLength: 256, promptCount: 10), 247)
        XCTAssertEqual(
            CoherePipeline.resolveMaxNewTokens(requested: 0, cacheLength: 256, promptCount: 10), 0)
        XCTAssertEqual(
            CoherePipeline.resolveMaxNewTokens(requested: -5, cacheLength: 256, promptCount: 10), 0)
        XCTAssertEqual(
            CoherePipeline.resolveMaxNewTokens(requested: nil, cacheLength: 5, promptCount: 10), 0)
    }

    func testDecodeStepCountEmitsExactlyTheBudget() {
        let prompt = CohereAsrConfig.Language.german.promptSequence.count
        for (budget, cache) in [(0, 108), (1, 108), (50, 108), (99, 108), (247, 256)] {
            let steps = CoherePipeline.decodeStepCount(
                maxNewTokens: budget, promptCount: prompt, cacheLength: cache)
            XCTAssertLessThanOrEqual(steps, cache, "budget \(budget) cache \(cache)")
            // Outputs appear from step promptCount-1 onward.
            XCTAssertEqual(max(0, steps - (prompt - 1)), budget, "budget \(budget) cache \(cache)")
        }
        // The full default budget uses every cache slot exactly once.
        let full = CoherePipeline.resolveMaxNewTokens(requested: nil, cacheLength: 256, promptCount: prompt)
        XCTAssertEqual(
            CoherePipeline.decodeStepCount(maxNewTokens: full, promptCount: prompt, cacheLength: 256), 256)
    }

    func testMaxExtentPrefersFlexibleUpperBound() {
        let shape: [NSNumber] = [1, 8, 1, 128]
        XCTAssertEqual(CoherePipeline.maxExtent(shape: shape, shapeConstraint: nil, dimension: 2), 1)
        XCTAssertEqual(CoherePipeline.maxExtent(shape: shape, shapeConstraint: nil, dimension: 7), 0)
        XCTAssertEqual(CoherePipeline.maxExtent(shape: [1, 8, 108, 128], shapeConstraint: nil, dimension: 2), 108)
    }

    func testRequiredModelsFollowDecoderVariant() {
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .cohereTranscribeCoreml, variant: nil),
            ModelNames.CohereTranscribe.requiredModels)
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .cohereTranscribeCoreml, variant: "v2"),
            ModelNames.CohereTranscribe.requiredModelsV2)
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .cohereTranscribeCoreml, variant: "v1"),
            ModelNames.CohereTranscribe.requiredModelsLegacy)
    }
}
