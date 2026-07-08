import CoreML
import XCTest

@testable import FluidAudio

/// Wiring tests for the per-platform graph variant (`gpu/` on macOS,
/// `ane/` on iOS). These are model-independent: they pin the compute-unit
/// policy, the download file-set (so macOS never pulls `ane/` and iOS never
/// pulls `gpu/`), and the fixed I/O shapes the synthesizer packs — which the
/// `ane/` and `gpu/` FmDecoder graphs share byte-for-byte.
final class LuxTtsVariantTests: XCTestCase {

    // MARK: - Compute-unit policy

    func testAneVariantUsesNeuralEngine() {
        XCTAssertEqual(
            LuxTtsModelStore.encoderDecoderComputeUnits(
                variant: ModelNames.LuxTts.aneVariant, override: nil),
            .cpuAndNeuralEngine)
    }

    func testGpuVariantUsesGpu() {
        XCTAssertEqual(
            LuxTtsModelStore.encoderDecoderComputeUnits(
                variant: ModelNames.LuxTts.gpuVariant, override: nil),
            .cpuAndGPU)
    }

    func testExplicitOverrideWinsForBothVariants() {
        for variant in [ModelNames.LuxTts.gpuVariant, ModelNames.LuxTts.aneVariant] {
            XCTAssertEqual(
                LuxTtsModelStore.encoderDecoderComputeUnits(variant: variant, override: .cpuOnly),
                .cpuOnly,
                "override must win for variant \(variant)")
        }
    }

    // MARK: - Download file-set (variant isolation)

    func testAneVariantRequiresOnlyAneModels() {
        let files = ModelNames.LuxTts.requiredFiles(variant: ModelNames.LuxTts.aneVariant)
        XCTAssertTrue(files.contains("ane/TextEncoder.mlmodelc"))
        XCTAssertTrue(files.contains("ane/FmDecoder.mlmodelc"))
        // iOS must never drag in the gpu graph.
        XCTAssertFalse(files.contains { $0.hasPrefix("gpu/") })
    }

    func testGpuVariantRequiresOnlyGpuModels() {
        let files = ModelNames.LuxTts.requiredFiles(variant: ModelNames.LuxTts.gpuVariant)
        XCTAssertTrue(files.contains("gpu/TextEncoder.mlmodelc"))
        XCTAssertTrue(files.contains("gpu/FmDecoder.mlmodelc"))
        // macOS must never drag in the ane graph.
        XCTAssertFalse(files.contains { $0.hasPrefix("ane/") })
    }

    func testSharedAssetsAreRequiredForBothVariants() {
        let shared: Set<String> = [
            ModelNames.LuxTts.tokensFile,
            ModelNames.LuxTts.configFile,
            ModelNames.LuxTts.vocoder282File,
            ModelNames.LuxTts.vocoder555File,
        ]
        for variant in [ModelNames.LuxTts.gpuVariant, ModelNames.LuxTts.aneVariant] {
            let files = ModelNames.LuxTts.requiredFiles(variant: variant)
            XCTAssertTrue(
                shared.isSubset(of: files),
                "variant \(variant) is missing shared assets \(shared.subtracting(files))")
        }
    }

    // MARK: - FmDecoder I/O contract (shared ane/gpu external signature)

    /// The synthesizer packs the FmDecoder inputs at these fixed shapes; the
    /// compiled `ane/` and `gpu/` graphs both declare exactly these. A drift
    /// here (e.g. a bucket change) would silently mispack the ANE decoder.
    func testFmDecoderContractShapes() throws {
        let featDim = LuxTtsConstants.featDim
        let maxFrames = LuxTtsConstants.maxFrames
        let maxTokens = LuxTtsConstants.maxTokens
        XCTAssertEqual(featDim, 100)
        XCTAssertEqual(maxFrames, 1024)
        XCTAssertEqual(maxTokens, 256)

        // x / text_condition / speech_condition: (1, 1024, 100)
        let x = try MLMultiArray(shape: [1, maxFrames, featDim].map { NSNumber(value: $0) }, dataType: .float32)
        XCTAssertEqual(x.shape.map { $0.intValue }, [1, 1024, 100])
        // padding_mask: (1, 1024)
        let mask = try MLMultiArray(shape: [1, maxFrames].map { NSNumber(value: $0) }, dataType: .float32)
        XCTAssertEqual(mask.shape.map { $0.intValue }, [1, 1024])
        // t / guidance_scale: (1)
        let scalar = try MLMultiArray(shape: [NSNumber(value: 1)], dataType: .float32)
        XCTAssertEqual(scalar.shape.map { $0.intValue }, [1])
        // tokens: (1, 256) int32
        let tokens = try MLMultiArray(shape: [1, maxTokens].map { NSNumber(value: $0) }, dataType: .int32)
        XCTAssertEqual(tokens.shape.map { $0.intValue }, [1, 256])
        XCTAssertEqual(tokens.dataType, .int32)
    }

    /// The FmDecoder `v` output is `(1, 1024, 100)` fp32 row-major, and CoreML
    /// stride-pads the last dim on some backends. This mirrors the
    /// synthesizer's `copyRows` unpack: read `featuresLength` rows of
    /// `featDim` honoring the row stride, so an ANE stride-padded output is
    /// de-padded correctly. Regression guard for the ANE unpack path.
    func testStridePaddedOutputUnpack() throws {
        let featDim = LuxTtsConstants.featDim
        let rows = 8
        let paddedRowLen = 112  // e.g. ANE/GPU pad 100 → 112

        let backing = try MLMultiArray(
            shape: [1, NSNumber(value: rows), NSNumber(value: paddedRowLen)],
            dataType: .float32)
        // Fill each active [row, d] cell with a unique sentinel; pad region 0.
        backing.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for r in 0..<rows {
                for d in 0..<paddedRowLen {
                    buf[r * paddedRowLen + d] = d < featDim ? Float(r * 1000 + d) : -1
                }
            }
        }

        // Reproduce copyRows: honor the row stride (paddedRowLen), copy featDim.
        var out = [Float](repeating: 0, count: rows * featDim)
        let rowStride = backing.strides[1].intValue
        XCTAssertEqual(rowStride, paddedRowLen, "expected stride-padded rows for this fixture")
        backing.withUnsafeBufferPointer(ofType: Float.self) { src in
            out.withUnsafeMutableBufferPointer { dst in
                for r in 0..<rows {
                    dst.baseAddress!.advanced(by: r * featDim)
                        .update(from: src.baseAddress!.advanced(by: r * rowStride), count: featDim)
                }
            }
        }

        for r in 0..<rows {
            for d in 0..<featDim {
                XCTAssertEqual(
                    out[r * featDim + d], Float(r * 1000 + d),
                    "row \(r) dim \(d) mispacked — stride not honored")
            }
        }
    }
}
