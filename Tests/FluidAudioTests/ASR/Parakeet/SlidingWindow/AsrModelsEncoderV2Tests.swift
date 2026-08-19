import XCTest

@testable import FluidAudio

/// Encoder_v2 (int8-linear, issue #760) file resolution.
final class AsrModelsEncoderV2Tests: XCTestCase {

    private var parentDir: URL!
    private var repoDir: URL!
    /// Directory handed to AsrModels APIs; its parent must contain the repo folder.
    private var modelsDir: URL!

    override func setUpWithError() throws {
        parentDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsEncoderV2Tests-\(UUID().uuidString)")
        repoDir = parentDir.appendingPathComponent(Repo.parakeetV3.folderName)
        modelsDir = repoDir
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: parentDir)
    }

    private func place(_ fileName: String) throws {
        try FileManager.default.createDirectory(
            at: repoDir.appendingPathComponent(fileName), withIntermediateDirectories: true)
    }

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

    func testRequiredModelsV3IncludesV2Encoder() {
        let required = ModelNames.ASR.requiredModelsV3(precision: .int8V2)
        XCTAssertTrue(required.contains("Encoder_v2.mlmodelc"))
        XCTAssertFalse(required.contains("Encoder.mlmodelc"))
    }

    func testResolvePrefersV2WhenBothPresent() throws {
        try place(ModelNames.ASR.encoderFile)
        try place(ModelNames.ASR.encoderV2File)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8, version: .v3, directory: modelsDir),
            .int8V2)
    }

    func testResolveKeepsOriginalWhenOnlyOldPresent() throws {
        // Existing caches keep working without a surprise re-download.
        try place(ModelNames.ASR.encoderFile)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8, version: .v3, directory: modelsDir),
            .int8)
    }

    func testResolvePicksV2WhenOnlyV2Present() throws {
        try place(ModelNames.ASR.encoderV2File)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8, version: .v3, directory: modelsDir),
            .int8V2)
    }

    func testResolveFreshInstallTriesV2First() {
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8, version: .v3, directory: modelsDir),
            .int8V2)
    }

    func testResolvePassesThroughExplicitPrecisions() throws {
        try place(ModelNames.ASR.encoderV2File)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int4, version: .v3, directory: modelsDir),
            .int4)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8V2, version: .v3, directory: modelsDir),
            .int8V2)
    }

    func testResolvePassesThroughNonV3Versions() throws {
        try place(ModelNames.ASR.encoderV2File)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8, version: .v2, directory: modelsDir),
            .int8)
    }

    func testModelsExistAcceptsEitherEncoderFile() throws {
        for file in [
            ModelNames.ASR.preprocessorFile,
            ModelNames.ASR.decoderFile,
            ModelNames.ASR.jointV3File,
        ] {
            try place(file)
        }
        try Data("{}".utf8).write(
            to: repoDir.appendingPathComponent(ModelNames.ASR.vocabularyFile))

        XCTAssertFalse(AsrModels.modelsExist(at: modelsDir, version: .v3, encoderPrecision: .int8))

        try place(ModelNames.ASR.encoderFile)
        XCTAssertTrue(AsrModels.modelsExist(at: modelsDir, version: .v3, encoderPrecision: .int8))

        try FileManager.default.removeItem(
            at: repoDir.appendingPathComponent(ModelNames.ASR.encoderFile))
        try place(ModelNames.ASR.encoderV2File)
        XCTAssertTrue(AsrModels.modelsExist(at: modelsDir, version: .v3, encoderPrecision: .int8))
    }
}
