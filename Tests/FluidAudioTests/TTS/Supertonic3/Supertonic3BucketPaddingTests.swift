import XCTest

@testable import FluidAudio

/// Verifies the channel-major pad/trim helpers used to fit a dynamic-length
/// latent into a fixed ANE bucket and trim it back for the vocoder.
final class Supertonic3BucketPaddingTests: XCTestCase {

    func testPadRowsZeroFillsEachChannelTail() {
        // 2 channels, length 3 → length 5. Layout is row-major [c*len + t].
        let flat: [Float] = [1, 2, 3, 4, 5, 6]  // c0:[1,2,3] c1:[4,5,6]
        let out = Supertonic3Synthesizer.padRows(flat, channels: 2, fromLen: 3, toLen: 5)
        XCTAssertEqual(out, [1, 2, 3, 0, 0, 4, 5, 6, 0, 0])
    }

    func testTrimRowsDropsEachChannelTail() {
        // 2 channels, length 5 → length 3.
        let flat: [Float] = [1, 2, 3, 9, 9, 4, 5, 6, 9, 9]
        let out = Supertonic3Synthesizer.trimRows(flat, channels: 2, fromLen: 5, toLen: 3)
        XCTAssertEqual(out, [1, 2, 3, 4, 5, 6])
    }

    func testPadThenTrimRoundTrips() {
        let channels = 4
        let trueLen = 7
        let bucket = 16
        let flat = (0..<(channels * trueLen)).map { Float($0) }
        let padded = Supertonic3Synthesizer.padRows(
            flat, channels: channels, fromLen: trueLen, toLen: bucket)
        XCTAssertEqual(padded.count, channels * bucket)
        let restored = Supertonic3Synthesizer.trimRows(
            padded, channels: channels, fromLen: bucket, toLen: trueLen)
        XCTAssertEqual(restored, flat)
    }

    func testPadRowsNoopWhenLengthsEqual() {
        let flat: [Float] = [1, 2, 3, 4]
        XCTAssertEqual(
            Supertonic3Synthesizer.padRows(flat, channels: 2, fromLen: 2, toLen: 2), flat)
    }

    func testPadTailZeroFillsMask() {
        let mask: [Float] = [1, 1, 1]
        XCTAssertEqual(Supertonic3Synthesizer.padTail(mask, toLen: 6), [1, 1, 1, 0, 0, 0])
        XCTAssertEqual(Supertonic3Synthesizer.padTail(mask, toLen: 3), [1, 1, 1])
    }

    func testVariantFileNaming() {
        // FP16 dynamic stays at repo root; variants live under the subdir.
        XCTAssertEqual(
            ModelNames.Supertonic3.vectorEstimatorFile(precisionSuffix: nil, bucket: nil),
            "VectorEstimator_v2.mlmodelc")
        XCTAssertEqual(
            ModelNames.Supertonic3.vectorEstimatorFile(precisionSuffix: "int4", bucket: nil),
            "VectorEstimatorVariants/VectorEstimator_v2_int4.mlmodelc")
        XCTAssertEqual(
            ModelNames.Supertonic3.vectorEstimatorFile(precisionSuffix: "int8", bucket: 256),
            "VectorEstimatorVariants/VectorEstimator_v2_L256_int8.mlmodelc")
    }

    func testRequiredFilesPerVariant() {
        let ane = ModelNames.Supertonic3.requiredFiles(veVariant: "ane-int4")
        XCTAssertTrue(ane.contains("VectorEstimatorVariants/VectorEstimator_v2_L128_int4.mlmodelc"))
        XCTAssertTrue(ane.contains("VectorEstimatorVariants/VectorEstimator_v2_L256_int4.mlmodelc"))
        XCTAssertTrue(ane.contains("VectorEstimatorVariants/VectorEstimator_v2_L512_int4.mlmodelc"))
        XCTAssertFalse(ane.contains("VectorEstimator_v2.mlmodelc"))
        XCTAssertTrue(ane.contains("TextEncoder_v2.mlmodelc"))  // shared module stays at root

        let dyn = ModelNames.Supertonic3.requiredFiles(veVariant: "dyn-int8")
        XCTAssertTrue(dyn.contains("VectorEstimatorVariants/VectorEstimator_v2_int8.mlmodelc"))
        XCTAssertFalse(dyn.contains("VectorEstimator_v2.mlmodelc"))

        let def = ModelNames.Supertonic3.requiredFiles(veVariant: nil)
        XCTAssertTrue(def.contains("VectorEstimator_v2.mlmodelc"))
    }

    func testV2EdgePadBundlesReplaceV1() {
        // TE / DP / VE are the edge-pad-fixed v2 builds; the causal Vocoder is unchanged.
        let files = ModelNames.Supertonic3.requiredFiles(veVariant: "ane-int4")
        XCTAssertTrue(files.contains("TextEncoder_v2.mlmodelc"))
        XCTAssertTrue(files.contains("DurationPredictor_v2.mlmodelc"))
        XCTAssertTrue(files.contains("Vocoder.mlmodelc"))
        XCTAssertFalse(files.contains("TextEncoder.mlmodelc"))
        XCTAssertFalse(files.contains("DurationPredictor.mlmodelc"))
        XCTAssertFalse(files.contains { $0.hasPrefix("VectorEstimatorVariants/VectorEstimator_L") })
    }
}
