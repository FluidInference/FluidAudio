import Foundation
import XCTest

@testable import FluidAudio

final class CanaryConfigTests: XCTestCase {

    // MARK: - Repo registration

    func testRepoRegistered() {
        XCTAssertEqual(Repo.canary1bV2.rawValue, "FluidInference/canary-1b-v2-coreml")
        XCTAssertEqual(Repo.canary1bV2.name, "canary-1b-v2-coreml")
        XCTAssertEqual(Repo.canary1bV2.folderName, "canary-1b-v2")
        XCTAssertTrue(Repo.canary1bV2.remotePath.contains("FluidInference/"))
    }

    func testRequiredModelsByPrecision() {
        let int4 = ModelNames.getRequiredModelNames(for: .canary1bV2, variant: "int4")
        XCTAssertTrue(int4.contains("EncoderInt4.mlmodelc"))
        XCTAssertTrue(int4.contains("DecoderInt4.mlmodelc"))
        XCTAssertTrue(int4.contains("Preprocessor.mlmodelc"))
        XCTAssertTrue(int4.contains("Projection.mlmodelc"))
        XCTAssertTrue(int4.contains("vocab.json"))

        let fp16 = ModelNames.getRequiredModelNames(for: .canary1bV2, variant: "fp16")
        XCTAssertTrue(fp16.contains("Encoder.mlmodelc"))
        XCTAssertTrue(fp16.contains("Decoder.mlmodelc"))

        // default (nil variant) falls back to int4
        let def = ModelNames.getRequiredModelNames(for: .canary1bV2, variant: nil)
        XCTAssertEqual(def, int4)
    }

    // MARK: - Precision → model name / compute units

    func testPrecisionModelNames() {
        XCTAssertEqual(CanaryPrecision.int4.encoderName, ModelNames.Canary.encoderInt4)
        XCTAssertEqual(CanaryPrecision.int4.decoderName, ModelNames.Canary.decoderInt4)
        XCTAssertEqual(CanaryPrecision.fp16.encoderName, ModelNames.Canary.encoder)
        XCTAssertEqual(CanaryPrecision.int8.encoderName, ModelNames.Canary.encoderInt8)
    }

    func testPrecisionComputeUnits() {
        // int8 crashes the GPU/ANE MPSGraph backend → CPU only; int4/fp16 run on ANE.
        XCTAssertEqual(CanaryPrecision.int8.computeUnits, .cpuOnly)
        XCTAssertEqual(CanaryPrecision.int4.computeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(CanaryPrecision.fp16.computeUnits, .cpuAndNeuralEngine)
    }

    // MARK: - Config contract

    func testConfigContract() {
        XCTAssertEqual(CanaryConfig.sampleRate, 16000)
        XCTAssertEqual(CanaryConfig.maxSamples, 240_000)  // 15 s
        XCTAssertEqual(CanaryConfig.encoderHidden, 1024)
        XCTAssertEqual(CanaryConfig.vocabSize, 16384)
        XCTAssertEqual(CanaryConfig.eosId, 3)  // <|endoftext|>
        XCTAssertEqual(CanaryConfig.padId, 2)
        XCTAssertEqual(CanaryConfig.bosId, 4)
        // canary2 English transcribe+pnc prompt
        XCTAssertEqual(CanaryConfig.promptEnTranscribePnc, [16053, 7, 4, 16, 64, 64, 5, 9, 11, 13])
    }
}
