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

    // MARK: - loadWithRecovery (no-network paths)

    func testLoadWithRecoveryRunsLoadOnCompleteCache() async throws {
        let baseDir = repoPath.deletingLastPathComponent()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: vadRepoPath) }
        try FileManager.default.createDirectory(
            at: vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile),
            withIntermediateDirectories: true)
        try Data("x".utf8).write(
            to: vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
                .appendingPathComponent("coremldata.bin"))

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
        let baseDir = repoPath.deletingLastPathComponent()
            .appendingPathComponent("recovery-offline-\(UUID().uuidString)")
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
        let baseDir = repoPath.deletingLastPathComponent()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: vadRepoPath) }
        let bundle = vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))

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

    func testLoadWithRecoveryCancellationPreservesCache() async throws {
        let baseDir = repoPath.deletingLastPathComponent()
        let vadRepoPath = baseDir.appendingPathComponent(Repo.vad.folderName)
        defer { try? FileManager.default.removeItem(at: vadRepoPath) }
        let bundle = vadRepoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))

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
}
