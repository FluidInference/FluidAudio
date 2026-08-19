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

    /// Create a load-ready compiled bundle: resolution requires a complete
    /// `.mlmodelc` (layout check + no `.partial` staging files), so a bare
    /// directory must NOT count as a usable encoder.
    private func place(_ fileName: String) throws {
        let bundle = repoDir.appendingPathComponent(fileName)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data().write(to: bundle.appendingPathComponent("coremldata.bin"))
    }

    private func placeIncomplete(_ fileName: String, partial: Bool) throws {
        let bundle = repoDir.appendingPathComponent(fileName)
        try FileManager.default.createDirectory(
            at: bundle.appendingPathComponent("weights"), withIntermediateDirectories: true)
        if partial {
            try Data().write(to: bundle.appendingPathComponent("coremldata.bin"))
            try Data().write(
                to: bundle.appendingPathComponent("weights/weight.bin.partial"))
        }
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

    func testIncompleteV2DoesNotOutrankValidOriginal() throws {
        // An interrupted v2 download leaves a partial bundle for resume; it
        // must not be selected over a load-ready original encoder.
        try place(ModelNames.ASR.encoderFile)
        try placeIncomplete(ModelNames.ASR.encoderV2File, partial: false)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8, version: .v3, directory: modelsDir),
            .int8)

        try FileManager.default.removeItem(
            at: repoDir.appendingPathComponent(ModelNames.ASR.encoderV2File))
        try placeIncomplete(ModelNames.ASR.encoderV2File, partial: true)
        XCTAssertEqual(
            AsrModels.resolveEncoderPrecision(.int8, version: .v3, directory: modelsDir),
            .int8)
    }

    func testBothIncompleteResolvesToV2ForResume() throws {
        // Neither bundle is usable: resolve like a fresh install so the v2
        // download runs (and resumes its partial file).
        try placeIncomplete(ModelNames.ASR.encoderFile, partial: false)
        try placeIncomplete(ModelNames.ASR.encoderV2File, partial: true)
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
