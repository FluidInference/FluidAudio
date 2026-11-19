import XCTest

@testable import FluidAudio

/// Opt-in Encoder_v2 (int8-linear, issue #760) precision selection.
final class AsrModelsEncoderV2Tests: XCTestCase {

    func testEncoderFileNames() {
        XCTAssertEqual(ParakeetEncoderPrecision.int8.encoderFileName, "Encoder.mlmodelc")
        XCTAssertEqual(ParakeetEncoderPrecision.int8V2.encoderFileName, "Encoder_v2.mlmodelc")
        XCTAssertEqual(ParakeetEncoderPrecision.int4.encoderFileName, "EncoderInt4.mlmodelc")
    }

    func testVariantStringRoundTrip() {
        // The download variant string must map back to the same precision so
        // ModelHub's required-model set matches the file AsrModels loads.
        for precision in ParakeetEncoderPrecision.allCases {
            XCTAssertEqual(ParakeetEncoderPrecision(rawValue: precision.rawValue), precision)
        }
    }

    func testRequiredModelsV3PerPrecision() {
        // int8 stays the default and keeps requiring the original encoder;
        // int8-v2 is strictly opt-in.
        let int8 = ModelNames.ASR.requiredModelsV3(precision: .int8)
        XCTAssertTrue(int8.contains("Encoder.mlmodelc"))
        XCTAssertFalse(int8.contains("Encoder_v2.mlmodelc"))

        let int8V2 = ModelNames.ASR.requiredModelsV3(precision: .int8V2)
        XCTAssertTrue(int8V2.contains("Encoder_v2.mlmodelc"))
        XCTAssertFalse(int8V2.contains("Encoder.mlmodelc"))
    }

    func testModelsExistChecksPrecisionSpecificEncoder() throws {
        let parentDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsEncoderV2Tests-\(UUID().uuidString)")
        let repoDir = parentDir.appendingPathComponent(Repo.parakeetV3.folderName)
        defer { try? FileManager.default.removeItem(at: parentDir) }

        for file in [
            ModelNames.ASR.preprocessorFile,
            ModelNames.ASR.decoderFile,
            ModelNames.ASR.jointV3File,
            ModelNames.ASR.encoderV2File,
        ] {
            let bundle = repoDir.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
            try Data().write(to: bundle.appendingPathComponent("coremldata.bin"))
        }
        try Data("{}".utf8).write(
            to: repoDir.appendingPathComponent(ModelNames.ASR.vocabularyFile))

        // Only Encoder_v2 on disk: satisfies an explicit int8-v2 request but
        // NOT the int8 default (which still requires Encoder.mlmodelc).
        XCTAssertTrue(
            AsrModels.modelsExist(at: repoDir, version: .v3, encoderPrecision: .int8V2))
        XCTAssertFalse(
            AsrModels.modelsExist(at: repoDir, version: .v3, encoderPrecision: .int8))
    }

    /// The #1067 shape: an interrupted `.int8V2` download leaves an `Encoder_v2.mlmodelc`
    /// directory with only `analytics/coremldata.bin` and `weights/weight.bin.partial`.
    /// A bare existence check called that cache present, so `download` returned early
    /// and the load purged the whole folder; it must count as not present.
    func testModelsExistRejectsInterruptedEncoderDownload() throws {
        let parentDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsEncoderV2Tests-\(UUID().uuidString)")
        let repoDir = parentDir.appendingPathComponent(Repo.parakeetV3.folderName)
        defer { try? FileManager.default.removeItem(at: parentDir) }

        for file in ModelNames.ASR.requiredModelsV3(precision: .int8) {
            let bundle = repoDir.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
            try Data().write(to: bundle.appendingPathComponent("coremldata.bin"))
        }
        try Data("{}".utf8).write(
            to: repoDir.appendingPathComponent(ModelNames.ASR.vocabularyFile))
        let interrupted = repoDir.appendingPathComponent(ModelNames.ASR.encoderV2File)
        for dir in ["analytics", "weights"] {
            try FileManager.default.createDirectory(
                at: interrupted.appendingPathComponent(dir), withIntermediateDirectories: true)
        }
        try Data().write(to: interrupted.appendingPathComponent("analytics/coremldata.bin"))
        try Data("w".utf8).write(to: interrupted.appendingPathComponent("weights/weight.bin.partial"))

        XCTAssertFalse(
            AsrModels.modelsExist(at: repoDir, version: .v3, encoderPrecision: .int8V2))
        XCTAssertTrue(
            AsrModels.modelsExist(at: repoDir, version: .v3, encoderPrecision: .int8))
    }
}
