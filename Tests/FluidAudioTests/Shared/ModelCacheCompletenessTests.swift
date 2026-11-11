import XCTest

@testable import FluidAudio

/// Cache-completeness checks behind `ModelHub.loadWithRecovery` (issue #819):
/// an interrupted download leaves a `.mlmodelc` directory that passes a bare
/// existence check but can never load (no root `coremldata.bin`, a
/// `weights/weight.bin.partial` instead of the real weights). These tests
/// exercise the on-disk layout logic with plain filesystem fixtures — no
/// CoreML model is ever loaded.
final class ModelCacheCompletenessTests: XCTestCase {

    private var repoPath: URL!

    override func setUpWithError() throws {
        repoPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-completeness-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: repoPath)
        ModelHub.offlineMode = false
        super.tearDown()
    }

    /// Lay down a `.mlmodelc` fixture directory (filesystem shape only).
    private func makeBundle(
        _ name: String, coremldata: Bool = true, partialWeights: Bool = false
    ) throws {
        let bundle = repoPath.appendingPathComponent(name)
        let weights = bundle.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        try CompiledBundleFixture.writeProgram(in: bundle)
        if coremldata {
            try Data("x".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))
        }
        if partialWeights {
            try Data("y".utf8).write(to: weights.appendingPathComponent("weight.bin.partial"))
            try Data("z".utf8).write(to: weights.appendingPathComponent("weight.bin.partial.etag"))
        } else {
            try Data("w".utf8).write(to: weights.appendingPathComponent("weight.bin"))
        }
    }

    private func makeFile(_ name: String) throws {
        try Data("{}".utf8).write(to: repoPath.appendingPathComponent(name))
    }

    // MARK: - incompleteFiles / isCacheComplete

    func testCompleteCacheReportsNoIncompleteFiles() throws {
        try makeBundle("encoder.mlmodelc")
        try makeFile("vocab.json")

        let required: Set<String> = ["encoder.mlmodelc", "vocab.json"]
        XCTAssertEqual(ModelCache.incompleteFiles(at: repoPath, requiredFiles: required), [])
        XCTAssertTrue(ModelCache.isCacheComplete(at: repoPath, requiredFiles: required))
    }

    func testBundleMissingRootCoremldataIsIncomplete() throws {
        // The exact #819 shape: directory exists, weights partial, no root
        // coremldata.bin. A bare fileExists check passes; this must not.
        try makeBundle("encoder.mlmodelc", coremldata: false, partialWeights: true)

        XCTAssertEqual(
            ModelCache.incompleteFiles(at: repoPath, requiredFiles: ["encoder.mlmodelc"]),
            ["encoder.mlmodelc"])
    }

    func testBundleWithPartialWeightsButValidLayoutIsIncomplete() throws {
        // coremldata.bin downloaded before the weights were interrupted:
        // layout validation alone passes, the partial marker must fail it.
        try makeBundle("encoder.mlmodelc", coremldata: true, partialWeights: true)

        XCTAssertEqual(
            ModelCache.incompleteFiles(at: repoPath, requiredFiles: ["encoder.mlmodelc"]),
            ["encoder.mlmodelc"])
    }

    func testMissingBundleAndMissingPlainFileAreIncomplete() throws {
        try makeBundle("decoder.mlmodelc")

        XCTAssertEqual(
            ModelCache.incompleteFiles(
                at: repoPath,
                requiredFiles: ["decoder.mlmodelc", "encoder.mlmodelc", "vocab.json"]),
            ["encoder.mlmodelc", "vocab.json"])
    }

    func testNestedBundlePathIsValidated() throws {
        // Nemotron keeps its encoder at encoder/encoder_int8.mlmodelc.
        try makeBundle("encoder/encoder_int8.mlmodelc", coremldata: false)

        XCTAssertEqual(
            ModelCache.incompleteFiles(
                at: repoPath, requiredFiles: ["encoder/encoder_int8.mlmodelc"]),
            ["encoder/encoder_int8.mlmodelc"])

        try Data("x".utf8).write(
            to: repoPath.appendingPathComponent("encoder/encoder_int8.mlmodelc/coremldata.bin"))
        XCTAssertTrue(
            ModelCache.isCacheComplete(
                at: repoPath, requiredFiles: ["encoder/encoder_int8.mlmodelc"]))
    }

    func testPartialOutsideRequiredBundlesDoesNotFlagCache() throws {
        // A leftover partial from a bundle the caller does not need must not
        // mark the cache incomplete (it would trigger a listing round-trip on
        // every load).
        try makeBundle("encoder.mlmodelc")
        try makeBundle("encoder_fp16.mlmodelc", partialWeights: true)

        XCTAssertTrue(
            ModelCache.isCacheComplete(at: repoPath, requiredFiles: ["encoder.mlmodelc"]))
    }

    // MARK: - Program files (issue #1147)

    /// The issue #1147 shape: an interrupted download left `MelSpectrogram.mlmodelc`
    /// with its weights and `coremldata.bin` but no `model.mil`. Core ML segfaults on
    /// that bundle on the Neural Engine path, so the cache check must refuse it.
    func testBundleWithoutProgramIsIncomplete() throws {
        let bundle = repoPath.appendingPathComponent("MelSpectrogram.mlmodelc")
        try CompiledBundleFixture.writeComplete(at: bundle)
        try FileManager.default.removeItem(at: bundle.appendingPathComponent("model.mil"))

        XCTAssertEqual(ModelCache.missingProgramFile(in: bundle), "model.mil")
        XCTAssertFalse(ModelCache.isCacheComplete(at: repoPath, requiredFiles: ["MelSpectrogram.mlmodelc"]))
        XCTAssertThrowsError(try ModelCache.validateCompiledModelLayout(at: bundle, name: "MelSpectrogram.mlmodelc"))
    }

    func testBundleMissingOrEmptyReferencedWeightsIsIncomplete() throws {
        let bundle = repoPath.appendingPathComponent("encoder.mlmodelc")
        try CompiledBundleFixture.writeComplete(at: bundle)
        XCTAssertNil(ModelCache.missingProgramFile(in: bundle))

        let weights = bundle.appendingPathComponent("weights/weight.bin")
        try Data().write(to: weights)
        XCTAssertEqual(ModelCache.missingProgramFile(in: bundle), "weights/weight.bin")

        try FileManager.default.removeItem(at: weights)
        XCTAssertEqual(ModelCache.missingProgramFile(in: bundle), "weights/weight.bin")
        XCTAssertFalse(ModelCache.isCacheComplete(at: repoPath, requiredFiles: ["encoder.mlmodelc"]))
    }

    /// A `model.mil` cut short in a download segfaults Core ML like a missing one.
    func testTruncatedProgramIsIncomplete() throws {
        let bundle = repoPath.appendingPathComponent("encoder.mlmodelc")
        try CompiledBundleFixture.writeComplete(at: bundle)
        let text = CompiledBundleFixture.programText
        try Data(text.prefix(text.count / 2).utf8).write(to: bundle.appendingPathComponent("model.mil"))

        XCTAssertEqual(ModelCache.missingProgramFile(in: bundle), "model.mil")
    }

    /// Files behind a symbolic link count by their target; a broken link is missing.
    func testSymlinkedWeightsCountByTheirTarget() throws {
        let bundle = repoPath.appendingPathComponent("encoder.mlmodelc")
        try CompiledBundleFixture.writeComplete(at: bundle)
        let weights = bundle.appendingPathComponent("weights/weight.bin")
        let target = repoPath.appendingPathComponent("blobs/weight")
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("w".utf8).write(to: target)
        try FileManager.default.removeItem(at: weights)
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: target)
        XCTAssertNil(ModelCache.missingProgramFile(in: bundle))

        try FileManager.default.removeItem(at: target)
        XCTAssertEqual(ModelCache.missingProgramFile(in: bundle), "weights/weight.bin")
    }

    func testNeuralNetworkBundleNeedsItsEspressoNet() throws {
        let bundle = repoPath.appendingPathComponent("legacy.mlmodelc")
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))
        XCTAssertEqual(ModelCache.missingProgramFile(in: bundle), "model.mil")

        try Data("{}".utf8).write(to: bundle.appendingPathComponent("model.espresso.net"))
        XCTAssertNil(ModelCache.missingProgramFile(in: bundle))
    }

    /// A pipeline (Sortformer) keeps one program per stage in `model0/`, `model1/`,
    /// each naming its own weights relative to that folder.
    func testPipelineBundleChecksEachStage() throws {
        let bundle = repoPath.appendingPathComponent("Sortformer.mlmodelc")
        try FileManager.default.createDirectory(
            at: bundle.appendingPathComponent("analytics"), withIntermediateDirectories: true)
        try Data("x".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))
        try Data("x".utf8).write(to: bundle.appendingPathComponent("analytics/coremldata.bin"))
        for stage in ["model0", "model1"] {
            try CompiledBundleFixture.writeComplete(at: bundle.appendingPathComponent(stage))
        }
        XCTAssertNil(ModelCache.missingProgramFile(in: bundle))

        try FileManager.default.removeItem(at: bundle.appendingPathComponent("model1/weights/weight.bin"))
        XCTAssertEqual(ModelCache.missingProgramFile(in: bundle), "model1/weights/weight.bin")
    }

    func testModelPathReferencesListsEveryNamedFile() {
        let mil = Data(
            #"""
            a = const()[val = tensor<fp16, [2]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
            b = const()[val = tensor<fp16, [2]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(192)))];
            c = const()[val = tensor<fp16, [2]>(BLOBFILE(path = string("@model_path/weights/0-weight.bin"), offset = uint64(64)))];
            """#.utf8)
        XCTAssertEqual(ModelCache.modelPathReferences(in: mil), ["weights/weight.bin", "weights/0-weight.bin"])
    }

    /// `CtcModels.modelsExist` and `ModelHub` judge a CTC cache by the same files, so
    /// a missing vocabulary or program makes both re-run the download (issue #1147).
    func testCtcModelsExistMatchesTheLoadCheck() throws {
        for name in ModelNames.CTC.requiredModels {
            try CompiledBundleFixture.writeComplete(at: repoPath.appendingPathComponent(name))
        }
        XCTAssertTrue(
            ModelNames.getRequiredModelNames(for: .parakeetCtc110m, variant: nil)
                .contains(ModelNames.CTC.vocabularyPath))
        XCTAssertFalse(CtcModels.modelsExist(at: repoPath), "no vocab.json")

        try makeFile(ModelNames.CTC.vocabularyPath)
        XCTAssertTrue(CtcModels.modelsExist(at: repoPath))

        try FileManager.default.removeItem(
            at: repoPath.appendingPathComponent(ModelNames.CTC.melSpectrogramPath).appendingPathComponent("model.mil"))
        XCTAssertFalse(CtcModels.modelsExist(at: repoPath), "MelSpectrogram without model.mil")
    }

    // MARK: - Revision identity

    func testHistoricalMainCacheDoesNotRequireRevisionMarker() throws {
        try makeFile("weights.bin")

        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: "main"))
        XCTAssertFalse(
            ModelCache.matchesRevision(
                at: repoPath, revision: String(repeating: "a", count: 40)))
    }

    func testPreparingPinnedRevisionPurgesStaleCacheAndWritesMarker() throws {
        let revision = String(repeating: "a", count: 40)
        try makeFile("stale.bin")

        try ModelCache.prepareForDownload(at: repoPath, revision: revision)

        XCTAssertFalse(FileManager.default.fileExists(atPath: repoPath.appendingPathComponent("stale.bin").path))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
        XCTAssertFalse(
            ModelCache.matchesRevision(
                at: repoPath, revision: String(repeating: "b", count: 40)))
    }

    func testPreparingSamePinnedRevisionPreservesResumableFiles() throws {
        let revision = String(repeating: "a", count: 40)
        try ModelCache.prepareForDownload(at: repoPath, revision: revision)
        try makeFile("weights.bin.partial")

        try ModelCache.prepareForDownload(at: repoPath, revision: revision)

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: repoPath.appendingPathComponent("weights.bin.partial").path))
    }

    // MARK: - SenseVoice / Paraformer public gates

    func testSenseVoiceModelsExistRejectsInterruptedDownload() throws {
        try makeBundle(ModelNames.SenseVoice.preprocessorFile)
        try makeBundle(
            ModelNames.SenseVoice.encoderFile, coremldata: false, partialWeights: true)
        try makeFile(ModelNames.SenseVoice.vocabularyFile)

        XCTAssertFalse(SenseVoiceModels.modelsExist(at: repoPath, precision: .fp16))

        // Complete the encoder bundle: gate opens.
        let encoder = repoPath.appendingPathComponent(ModelNames.SenseVoice.encoderFile)
        try Data("x".utf8).write(to: encoder.appendingPathComponent("coremldata.bin"))
        try FileManager.default.moveItem(
            at: encoder.appendingPathComponent("weights/weight.bin.partial"),
            to: encoder.appendingPathComponent("weights/weight.bin"))
        XCTAssertTrue(SenseVoiceModels.modelsExist(at: repoPath, precision: .fp16))
        // Other precisions still gated on their own encoder.
        XCTAssertFalse(SenseVoiceModels.modelsExist(at: repoPath, precision: .int8))
    }

    func testParaformerModelsExistRejectsInterruptedDownload() throws {
        try makeBundle(ModelNames.ParaformerZh.preprocessorFile)
        try makeBundle(ModelNames.ParaformerZh.encoderFile, coremldata: false)
        try makeBundle(ModelNames.ParaformerZh.cifAlphasFile)
        try makeBundle(ModelNames.ParaformerZh.decoderFile)
        try makeFile(ModelNames.ParaformerZh.vocabularyFile)

        XCTAssertFalse(ParaformerModels.modelsExist(at: repoPath, precision: .fp16))

        try Data("x".utf8).write(
            to: repoPath.appendingPathComponent(ModelNames.ParaformerZh.encoderFile)
                .appendingPathComponent("coremldata.bin"))
        XCTAssertTrue(ParaformerModels.modelsExist(at: repoPath, precision: .fp16))
    }

    // MARK: - undersizedFiles (truncated vs incompatible diagnosis)

    func testUndersizedFilesFlagsTruncatedAndMissingFiles() throws {
        try makeBundle("encoder.mlmodelc")  // weights/weight.bin is 1 byte
        let remote = [
            RemoteFile(path: "encoder.mlmodelc/weights/weight.bin", size: 540),
            RemoteFile(path: "encoder.mlmodelc/coremldata.bin", size: 1),
            RemoteFile(path: "vocab.json", size: 100),  // missing locally
        ]

        XCTAssertEqual(
            ModelCache.undersizedFiles(remote: remote, at: repoPath, subPath: nil),
            [
                "encoder.mlmodelc/weights/weight.bin (1/540 bytes)",
                "vocab.json (0/100 bytes)",
            ])
    }

    func testUndersizedFilesEmptyWhenSizesMatch() throws {
        try makeBundle("encoder.mlmodelc")
        let remote = [
            RemoteFile(path: "encoder.mlmodelc/coremldata.bin", size: 1),
            RemoteFile(path: "encoder.mlmodelc/weights/weight.bin", size: 1),
            // Unreported sizes must not be flagged.
            RemoteFile(path: "encoder.mlmodelc/metadata.json", size: -1),
        ]

        XCTAssertEqual(
            ModelCache.undersizedFiles(remote: remote, at: repoPath, subPath: nil), [])
    }

    func testUndersizedFilesStripsSubPath() throws {
        try makeFile("vocab.json")  // 2 bytes ("{}")
        let remote = [
            RemoteFile(path: "160ms/vocab.json", size: 2),
            RemoteFile(path: "160ms/missing.bin", size: 10),
        ]

        XCTAssertEqual(
            ModelCache.undersizedFiles(remote: remote, at: repoPath, subPath: "160ms"),
            ["missing.bin (0/10 bytes)"])
    }

    // MARK: - loadWithRecovery (no-network paths)

    /// A unique models root per test: parallel CI runs other suites in
    /// sibling processes that also create/purge repo caches directly under
    /// the shared temp directory, so the assertions here must never share
    /// paths with them.
    private func makeUniqueBaseDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("recovery-\(UUID().uuidString)")
    }

    func testLoadWithRecoveryRunsLoadOnCompleteCache() async throws {
        let baseDir = makeUniqueBaseDir()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: baseDir) }
        try CompiledBundleFixture.writeComplete(
            at: vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile))

        // Offline mode proves no download is attempted for a complete cache.
        ModelHub.offlineMode = true
        let result = try await ModelHub.loadWithRecovery(
            .vad, directory: baseDir,
            requiredFiles: [ModelNames.VAD.sileroVadFile]
        ) { "loaded" }
        XCTAssertEqual(result, "loaded")
    }

    func testLoadWithRecoveryOfflineIncompleteCacheThrowsTypedError() async {
        ModelHub.offlineMode = true
        let baseDir = makeUniqueBaseDir()
        defer { try? FileManager.default.removeItem(at: baseDir) }

        do {
            _ = try await ModelHub.loadWithRecovery(
                .vad, directory: baseDir,
                requiredFiles: [ModelNames.VAD.sileroVadFile]
            ) {
                XCTFail("load must not run when the cache is incomplete")
            }
            XCTFail("expected DownloadError.modelMissing")
        } catch let DownloadError.modelMissing(repo, missing) {
            XCTAssertEqual(repo, Repo.vad.folderName)
            XCTAssertEqual(missing, [ModelNames.VAD.sileroVadFile])
        } catch {
            XCTFail("expected DownloadError.modelMissing, got: \(error)")
        }
    }

    func testLoadWithRecoveryOfflineLoadFailureDoesNotPurgeCache() async throws {
        struct LoadFailure: Error {}
        let baseDir = makeUniqueBaseDir()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: baseDir) }
        let bundle = vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
        try CompiledBundleFixture.writeComplete(at: bundle)

        ModelHub.offlineMode = true
        do {
            _ = try await ModelHub.loadWithRecovery(
                .vad, directory: baseDir,
                requiredFiles: [ModelNames.VAD.sileroVadFile]
            ) { throw LoadFailure() }
            XCTFail("expected LoadFailure to surface")
        } catch is LoadFailure {
            // Pinned: offline mode rethrows the original error and must not
            // delete the cache behind the app's back.
            XCTAssertTrue(FileManager.default.fileExists(atPath: bundle.path))
        } catch {
            XCTFail("expected LoadFailure, got: \(error)")
        }
    }

    func testLoadWithRecoveryCachePreservingErrorKeepsCache() async throws {
        struct LowDisk: ModelCachePreservingError {
            var preservesModelCache: Bool { true }
        }
        let baseDir = makeUniqueBaseDir()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: baseDir) }
        let bundle = vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
        try CompiledBundleFixture.writeComplete(at: bundle)

        // Online: a purge-and-redownload would delete the bundle before any network attempt.
        do {
            _ = try await ModelHub.loadWithRecovery(
                .vad, directory: baseDir,
                requiredFiles: [ModelNames.VAD.sileroVadFile]
            ) { throw LowDisk() }
            XCTFail("expected LowDisk to surface")
        } catch is LowDisk {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: bundle.path),
                "a failure outside the cache must not purge it")
        } catch {
            XCTFail("expected LowDisk, got: \(error)")
        }
    }

    func testLoadWithRecoveryCancellationPreservesCache() async throws {
        let baseDir = makeUniqueBaseDir()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: baseDir) }
        let bundle = vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
        try CompiledBundleFixture.writeComplete(at: bundle)

        // offlineMode stays false: if cancellation wrongly fell through to
        // purge-and-redownload, the purge assertion below would catch it
        // before any network attempt.
        do {
            _ = try await ModelHub.loadWithRecovery(
                .vad, directory: baseDir,
                requiredFiles: [ModelNames.VAD.sileroVadFile]
            ) { throw CancellationError() }
            XCTFail("expected CancellationError to surface")
        } catch is CancellationError {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: bundle.path),
                "cancellation is not corruption — cache must survive")
        } catch {
            XCTFail("expected CancellationError, got: \(error)")
        }
    }

    // MARK: - loadModels (#1067)

    func testLoadModelsTreatsInterruptedBundleAsIncomplete() async throws {
        // The #819 shape in a `loadModels` repo: the bundle directory exists
        // with only partial weights. A bare existence check skipped the
        // download, the load failed and the whole folder was purged; it must
        // re-enter the download instead (offline mode surfaces that as the
        // typed missing-model error, without touching the files).
        let baseDir = makeUniqueBaseDir()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: baseDir) }
        let bundle = vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
        try FileManager.default.createDirectory(
            at: bundle.appendingPathComponent("weights"), withIntermediateDirectories: true)
        try Data("y".utf8).write(to: bundle.appendingPathComponent("weights/weight.bin.partial"))

        ModelHub.offlineMode = true
        do {
            _ = try await ModelHub.loadModels(
                .vad, modelNames: [ModelNames.VAD.sileroVadFile], directory: baseDir)
            XCTFail("expected DownloadError.modelMissing")
        } catch let DownloadError.modelMissing(repo, missing) {
            XCTAssertEqual(repo, Repo.vad.folderName)
            XCTAssertEqual(missing, [ModelNames.VAD.sileroVadFile])
            XCTAssertTrue(
                FileManager.default.fileExists(
                    atPath: bundle.appendingPathComponent("weights/weight.bin.partial").path),
                "the partial file must survive so the download can resume it")
        } catch {
            XCTFail("expected DownloadError.modelMissing, got: \(error)")
        }
    }

    func testPurgeCorruptedFilesKeepsSiblingVariantsAndMarker() throws {
        // A Parakeet v3 folder holding both encoder variants (#1067): purging
        // the `.int8V2` encoder that failed must keep `.int8`'s encoder, the
        // shared bundles and the revision marker.
        try ModelCache.prepareForDownload(at: repoPath, revision: "abc123")
        try makeBundle(ModelNames.ASR.encoderFile)
        try makeBundle(ModelNames.ASR.encoderV2File, coremldata: false, partialWeights: true)
        try makeBundle(ModelNames.ASR.decoderFile)
        try makeFile(ModelNames.ASR.vocabularyFile)
        try Data("y".utf8).write(to: repoPath.appendingPathComponent("config.json.partial"))

        ModelCache.purgeCorruptedFiles(
            [ModelNames.ASR.encoderV2File, "config.json", "missing.mlmodelc"], at: repoPath)

        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: repoPath.appendingPathComponent(ModelNames.ASR.encoderV2File).path))
        XCTAssertFalse(fm.fileExists(atPath: repoPath.appendingPathComponent("config.json.partial").path))
        for kept in [ModelNames.ASR.encoderFile, ModelNames.ASR.decoderFile, ModelNames.ASR.vocabularyFile] {
            XCTAssertTrue(fm.fileExists(atPath: repoPath.appendingPathComponent(kept).path), kept)
        }
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: "abc123"))
    }
}
