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
    }

    func testDecoderVariantUsesStaticSelfMask() {
        XCTAssertTrue(CoherePipeline.DecoderVariant.v2.usesStaticSelfMask)
        XCTAssertFalse(CoherePipeline.DecoderVariant.v1.usesStaticSelfMask)
    }

    // MARK: - buildSelfAttentionMask: dynamic (v1) path

    func testDynamicMaskShapeAndZeroFillFloat32() throws {
        let step = 4
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: step, useStatic: false, dtype: .float32)

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
            step: step, useStatic: false, dtype: .float16)

        XCTAssertEqual(mask.shape, [1, 1, 1, NSNumber(value: step + 1)])
        XCTAssertEqual(mask.dataType, .float16)

        let ptr = mask.dataPointer.bindMemory(to: UInt16.self, capacity: step + 1)
        for i in 0...step {
            XCTAssertEqual(ptr[i], 0, "Dynamic FP16 mask must be zero-filled at \(i)")
        }
    }

    func testDynamicMaskAtStepZero() throws {
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: 0, useStatic: false, dtype: .float32)
        XCTAssertEqual(mask.shape, [1, 1, 1, 1])
        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: 1)
        XCTAssertEqual(ptr[0], 0.0)
    }

    // MARK: - buildSelfAttentionMask: static (v2) path

    func testStaticMaskShapeIsAlwaysMaxSeqLen() throws {
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: 0, useStatic: true, dtype: .float32)
        XCTAssertEqual(
            mask.shape, [1, 1, 1, NSNumber(value: CohereAsrConfig.maxSeqLen)])
    }

    func testStaticMaskCausalSplitFloat32() throws {
        let step = 10
        let length = CohereAsrConfig.maxSeqLen
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: step, useStatic: true, dtype: .float32)

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
            step: step, useStatic: true, dtype: .float16)

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
            step: length - 1, useStatic: true, dtype: .float32)

        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: length)
        for i in 0..<length {
            XCTAssertEqual(ptr[i], 0.0, "Final step: every position attends, got \(ptr[i]) at \(i)")
        }
    }

    func testStaticMaskAtStepZeroBlocksAllExceptFirst() throws {
        let length = CohereAsrConfig.maxSeqLen
        let mask = try CoherePipeline.buildSelfAttentionMask(
            step: 0, useStatic: true, dtype: .float32)

        let ptr = mask.dataPointer.bindMemory(to: Float.self, capacity: length)
        XCTAssertEqual(ptr[0], 0.0, "step=0 attends position 0")
        for i in 1..<length {
            XCTAssertEqual(ptr[i], -1.0e4, "step=0 must mask position \(i)")
        }
    }
}
