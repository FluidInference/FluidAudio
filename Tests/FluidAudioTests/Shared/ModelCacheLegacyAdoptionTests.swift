import Foundation
import XCTest

@testable import FluidAudio

/// Filesystem-only cache fixtures; no CoreML model is created or loaded.
final class ModelCacheLegacyAdoptionTests: XCTestCase {

    private var repoPath: URL!
    private let revision = Repo.diarizer.revision

    override func setUpWithError() throws {
        repoPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-legacy-adoption-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
        TreeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        TreeStubURLProtocol.reset()
        try FileManager.default.removeItem(at: repoPath)
    }

    private func makeFile(_ path: String, contents: String = "local") throws {
        let destination = repoPath.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(contents.utf8).write(to: destination)
    }

    private var stubConfiguration: URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [TreeStubURLProtocol.self]
        return configuration
    }

    private func listedFile(path: String, size: Int) -> RemoteFile {
        RemoteFile(
            path: path, size: size,
            contentID: .lfsSHA256("25bf8e1a2393f1108d37029b3df5593236c755742ec93465bbafa9b290bddcf6"))
    }

    private func prepare(_ files: [RemoteFile], revision: String? = nil, subPath: String? = nil) throws {
        let revision = revision ?? self.revision
        try ModelCache.adoptLegacyCache(at: repoPath, revision: revision, files: files, subPath: subPath)
        try ModelCache.prepareForDownload(at: repoPath, revision: revision)
    }

    private func exists(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: repoPath.appendingPathComponent(path).path)
    }

    func testMatchingLegacyFilesArePreservedAndMarkerWritten() throws {
        try makeFile("config.json")
        try makeFile("nested/extra.json")
        try makeFile("unlisted.json")

        try prepare([
            listedFile(path: "config.json", size: 5),
            listedFile(path: "nested/extra.json", size: 5),
        ])

        XCTAssertEqual(try Data(contentsOf: repoPath.appendingPathComponent("config.json")), Data("local".utf8))
        XCTAssertEqual(try Data(contentsOf: repoPath.appendingPathComponent("nested/extra.json")), Data("local".utf8))
        XCTAssertFalse(exists("unlisted.json"))
        XCTAssertEqual(
            try String(contentsOf: repoPath.appendingPathComponent(".fluidaudio-revision"), encoding: .utf8),
            revision + "\n")
    }

    func testOnlyMismatchedLegacyFileIsRemoved() throws {
        try makeFile("config.json")
        try makeFile("nested/extra.json", contents: "truncated")

        try prepare([
            listedFile(path: "config.json", size: 5),
            listedFile(path: "nested/extra.json", size: 5),
        ])

        XCTAssertTrue(exists("config.json"))
        XCTAssertFalse(exists("nested/extra.json"))
        XCTAssertTrue(exists("nested"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testUnknownAndZeroSizedLegacyFilesWithoutIdentityAreRemoved() throws {
        try makeFile("unknown.json")
        try makeFile("empty.json", contents: "")
        try makeFile("config.json")

        try prepare([
            RemoteFile(path: "unknown.json", size: -1),
            RemoteFile(path: "empty.json", size: 0),
            listedFile(path: "config.json", size: 5),
        ])

        XCTAssertFalse(exists("unknown.json"))
        XCTAssertFalse(exists("empty.json"))
        XCTAssertTrue(exists("config.json"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testMissingLegacyFileRemainsMissing() throws {
        try makeFile("config.json")

        try prepare([
            listedFile(path: "config.json", size: 5),
            listedFile(path: "missing.json", size: 5),
        ])

        XCTAssertTrue(exists("config.json"))
        XCTAssertFalse(exists("missing.json"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testSubPathIsStrippedButRootAuxiliaryFileIsPreserved() throws {
        try makeFile("nested/config.json")
        try makeFile("vocab.json")

        try prepare(
            [
                listedFile(path: "q8/nested/config.json", size: 5),
                listedFile(path: "vocab.json", size: 5),
            ], subPath: "q8")

        XCTAssertTrue(exists("nested/config.json"))
        XCTAssertTrue(exists("vocab.json"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testDifferentRevisionMarkerStillWipesWholeCache() throws {
        try makeFile(".fluidaudio-revision", contents: String(repeating: "b", count: 40))
        try makeFile("nested/config.json")
        try makeFile("unlisted.json")

        try prepare([listedFile(path: "nested/config.json", size: 5)])

        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: repoPath.path), [".fluidaudio-revision"])
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testMatchingRevisionMarkerPreservesFilesWithoutSizeChecks() throws {
        try makeFile(".fluidaudio-revision", contents: revision + "\n")
        try makeFile("config.json")
        try makeFile("weights.bin.partial")

        try prepare([listedFile(path: "config.json", size: 100)])

        XCTAssertTrue(exists("config.json"))
        XCTAssertTrue(exists("weights.bin.partial"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testUnmarkedMainCacheIsUnchanged() throws {
        try makeFile("config.json")
        try makeFile("unknown.json")

        try prepare(
            [listedFile(path: "config.json", size: 100), listedFile(path: "unknown.json", size: -1)],
            revision: "main")

        XCTAssertTrue(exists("config.json"))
        XCTAssertTrue(exists("unknown.json"))
        XCTAssertFalse(exists(".fluidaudio-revision"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: "main"))
    }

    func testRepoDownloadAdoptsMatchingLegacyFiles() async throws {
        let repo = Repo.diarizer
        let required = ModelNames.getRequiredModelNames(for: repo, variant: nil).sorted()
        var trees: [String: [[String: Any]]] = ["": []]
        for name in required {
            // Directory layout only, matching the existing cache-completeness fixtures.
            try makeFile("\(repo.folderName)/\(name)/config.json")
            trees["", default: []].append(["path": name, "type": "directory"])
            trees[name] = [TreeStubURLProtocol.fileEntry("\(name)/config.json")]
        }
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(repo, to: repoPath, configuration: stubConfiguration)

        for name in required {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/\(name)/config.json")),
                Data("local".utf8))
        }
        XCTAssertTrue(
            ModelCache.matchesRevision(at: repoPath.appendingPathComponent(repo.folderName), revision: revision))
    }

    private func downloadMetadata() async throws {
        TreeStubURLProtocol.trees = [
            "metadata": [
                TreeStubURLProtocol.fileEntry("metadata/config.json"),
                ["path": "metadata/nested", "type": "directory"],
            ],
            "metadata/nested": [TreeStubURLProtocol.fileEntry("metadata/nested/extra.json")],
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)
        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath,
            configuration: stubConfiguration)
    }

    func testSubdirectoryDownloadAdoptsMatchingLegacyFiles() async throws {
        try makeFile("metadata/config.json")
        try makeFile("metadata/nested/extra.json")

        try await downloadMetadata()

        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("metadata/config.json")), Data("local".utf8))
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("metadata/nested/extra.json")), Data("local".utf8))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath.appendingPathComponent("metadata"), revision: revision))
    }

    func testSubdirectoryDownloadRefetchesOnlyMismatchedLegacyFile() async throws {
        try makeFile("metadata/config.json")
        try makeFile("metadata/nested/extra.json", contents: "truncated")

        try await downloadMetadata()

        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("metadata/config.json")), Data("local".utf8))
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("metadata/nested/extra.json")), Data("fresh".utf8))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath.appendingPathComponent("metadata"), revision: revision))
    }

    func testStreamingDownloadValidatesStaleOfflineBundle() async throws {
        let repo = Repo.diarizer
        let streaming = ModelNames.getRequiredModelNames(for: repo, variant: nil).sorted()
        let offline = try XCTUnwrap(
            ModelNames.getRequiredModelNames(for: repo, variant: "offline").sorted().first { $0.hasSuffix(".mlmodelc") }
        )
        var trees: [String: [[String: Any]]] = ["": []]
        for name in streaming + [offline] {
            try makeFile("\(repo.folderName)/\(name)/config.json", contents: name == offline ? "truncated" : "local")
            trees["", default: []].append(["path": name, "type": "directory"])
            trees[name] = [TreeStubURLProtocol.fileEntry("\(name)/config.json")]
        }
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(repo, to: repoPath, configuration: stubConfiguration)

        XCTAssertFalse(exists("\(repo.folderName)/\(offline)"))
        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 0, "variant A must not refetch variant B")
        XCTAssertFalse(
            ModelCache.allModelsExist(at: repoPath.appendingPathComponent(repo.folderName), models: [offline]),
            "variant B's next load must enter download")
        for name in streaming {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/\(name)/config.json")),
                Data("local".utf8))
        }
        XCTAssertTrue(
            ModelCache.matchesRevision(at: repoPath.appendingPathComponent(repo.folderName), revision: revision))
    }

    func testWrongInnerFileRemovesWholeBundleBeforeWritingMarker() throws {
        try makeFile("encoder.mlmodelc/config.json")
        try makeFile("encoder.mlmodelc/weights/weight.bin", contents: "truncated")
        try makeFile("encoder.mlmodelc/weights/weight.bin.partial", contents: "par")
        try makeFile("encoder.mlmodelc/weights/weight.bin.partial.etag", contents: "\"etag\"")
        try makeFile("decoder.mlmodelc/config.json")

        try prepare([
            listedFile(path: "encoder.mlmodelc/config.json", size: 5),
            listedFile(path: "encoder.mlmodelc/weights/weight.bin", size: 5),
            listedFile(path: "decoder.mlmodelc/config.json", size: 5),
        ])

        XCTAssertFalse(exists("encoder.mlmodelc"))
        XCTAssertFalse(exists("encoder.mlmodelc/weights/weight.bin.partial"))
        XCTAssertFalse(exists("encoder.mlmodelc/weights/weight.bin.partial.etag"))
        XCTAssertTrue(exists("decoder.mlmodelc/config.json"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
        // This is the exact existence gate used by loadModelsOnce before any CoreML loading.
        XCTAssertFalse(ModelCache.allModelsExist(at: repoPath, models: ["encoder.mlmodelc", "decoder.mlmodelc"]))
    }

    func testMissingInnerFileRemovesWholeBundleBeforeWritingMarker() throws {
        try makeFile("encoder.mlmodelc/config.json")

        try prepare([
            listedFile(path: "encoder.mlmodelc/config.json", size: 5),
            listedFile(path: "encoder.mlmodelc/weights/weight.bin", size: 5),
        ])

        XCTAssertFalse(exists("encoder.mlmodelc"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
        XCTAssertFalse(ModelCache.allModelsExist(at: repoPath, models: ["encoder.mlmodelc"]))
    }

    func testKeptRejectedAndMissingFilesLoseStaleSidecars() throws {
        for file in ["kept.json", "removed.json", "missing.json"] {
            if file != "missing.json" { try makeFile(file, contents: file == "kept.json" ? "local" : "stale") }
            try makeFile(file + ".partial", contents: "stale")
            try makeFile(file + ".partial.etag", contents: "\"etag\"")
        }

        try prepare([
            listedFile(path: "kept.json", size: 5),
            listedFile(path: "removed.json", size: 5),
            listedFile(path: "missing.json", size: 5),
        ])

        XCTAssertTrue(exists("kept.json"))
        XCTAssertFalse(exists("kept.json.partial"))
        XCTAssertFalse(exists("kept.json.partial.etag"))
        XCTAssertFalse(exists("removed.json"))
        XCTAssertFalse(exists("missing.json"))
        for file in ["removed.json", "missing.json"] {
            XCTAssertFalse(exists(file + ".partial"))
            XCTAssertFalse(exists(file + ".partial.etag"))
        }
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testRejectedAndMissingLegacyFilesCannotReuseFinishedStalePartials() async throws {
        try makeFile("metadata/rejected.json", contents: "stale")
        for file in ["rejected.json", "missing.json"] {
            try makeFile("metadata/" + file + ".partial", contents: "stale")
            try makeFile("metadata/" + file + ".partial.etag", contents: "\"old-etag\"")
        }
        TreeStubURLProtocol.trees = [
            "metadata": [
                TreeStubURLProtocol.fileEntry("metadata/rejected.json", contents: "fresh", lfs: false),
                TreeStubURLProtocol.fileEntry("metadata/missing.json", contents: "fresh", lfs: false),
            ]
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 2, "both paths must fetch fresh bytes")
        for file in ["rejected.json", "missing.json"] {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("metadata/" + file)), Data("fresh".utf8))
            XCTAssertFalse(exists("metadata/" + file + ".partial"))
            XCTAssertFalse(exists("metadata/" + file + ".partial.etag"))
        }
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath.appendingPathComponent("metadata"), revision: revision))
    }

    func testRepoRefetchesRequestedFileWithoutFetchingRejectedOtherVariant() async throws {
        let repo = Repo.diarizer
        let streaming = ModelNames.getRequiredModelNames(for: repo, variant: nil).sorted()
        let offline = try XCTUnwrap(
            ModelNames.getRequiredModelNames(for: repo, variant: "offline").sorted().first { $0.hasSuffix(".mlmodelc") }
        )
        var trees: [String: [[String: Any]]] = ["": []]
        for name in streaming + [offline] {
            try makeFile("\(repo.folderName)/\(name)/config.json", contents: name == offline ? "stale" : "local")
            trees["", default: []].append(["path": name, "type": "directory"])
            trees[name] = [TreeStubURLProtocol.fileEntry("\(name)/config.json")]
        }
        let requested = "plda-parameters.json"
        try makeFile("\(repo.folderName)/\(requested)", contents: "stale")
        try makeFile("\(repo.folderName)/\(requested).partial", contents: "stale")
        try makeFile("\(repo.folderName)/\(requested).partial.etag", contents: "\"old-etag\"")
        trees["", default: []].append(TreeStubURLProtocol.fileEntry(requested, contents: "fresh", lfs: false))
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            repo, to: repoPath, additionalModelNames: [requested], configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 1, "only the caller's rejected file must be fetched")
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/\(requested)")), Data("fresh".utf8)
        )
        XCTAssertFalse(exists("\(repo.folderName)/\(requested).partial"))
        XCTAssertFalse(exists("\(repo.folderName)/\(requested).partial.etag"))
        XCTAssertFalse(exists("\(repo.folderName)/\(offline)"))
        for name in streaming {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/\(name)/config.json")),
                Data("local".utf8))
        }
        XCTAssertTrue(
            ModelCache.matchesRevision(at: repoPath.appendingPathComponent(repo.folderName), revision: revision))
    }

    func testConcurrentAdoptersTolerateDisappearingFiles() async throws {
        let files = (0..<128).map { listedFile(path: "file\($0).json", size: 5) }
        for file in files {
            try makeFile(file.path, contents: "truncated")
        }
        let path = try XCTUnwrap(repoPath)
        let revision = revision

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<32 {
                group.addTask {
                    try ModelCache.adoptLegacyCache(at: path, revision: revision, files: files)
                }
            }
            try await group.waitForAll()
        }

        XCTAssertTrue(ModelCache.matchesRevision(at: path, revision: revision))
        for file in files { XCTAssertFalse(exists(file.path)) }
    }

    func testMissingFileErrorClassificationIncludesCocoaAndPOSIX() {
        XCTAssertTrue(ModelCache.isMissingFile(CocoaError(.fileNoSuchFile)))
        XCTAssertTrue(ModelCache.isMissingFile(CocoaError(.fileReadNoSuchFile)))
        XCTAssertTrue(ModelCache.isMissingFile(POSIXError(.ENOENT)))
        XCTAssertFalse(ModelCache.isMissingFile(CocoaError(.fileReadNoPermission)))
        XCTAssertFalse(ModelCache.isMissingFile(POSIXError(.EACCES)))
    }

    func testKeptBundleHasNoStalePartialDownloads() throws {
        try makeFile("encoder.mlmodelc/config.json")
        try makeFile("encoder.mlmodelc/config.json.partial", contents: "par")
        try makeFile("encoder.mlmodelc/config.json.partial.etag", contents: "\"etag\"")

        try prepare([listedFile(path: "encoder.mlmodelc/config.json", size: 5)])

        XCTAssertTrue(exists("encoder.mlmodelc/config.json"))
        XCTAssertFalse(ModelCache.containsPartialDownload(at: repoPath.appendingPathComponent("encoder.mlmodelc")))
    }

    func testPinnedSubPathDownloadValidatesOtherBundlesAndRootAuxiliaryFiles() async throws {
        let repo = Repo.parakeetEou160
        let sub = try XCTUnwrap(repo.subPath)
        let oldOverrides = ModelRegistry.revisionOverrides
        ModelRegistry.revisionOverrides[repo.remotePath] = revision
        defer { ModelRegistry.revisionOverrides = oldOverrides }
        let bundles = ModelNames.getRequiredModelNames(for: repo, variant: nil).sorted().filter {
            $0.hasSuffix(".mlmodelc")
        }
        var trees: [String: [[String: Any]]] = [sub: []]
        for bundle in bundles + ["other.mlmodelc"] {
            try makeFile(
                "\(repo.folderName)/\(bundle)/config.json", contents: bundle == "other.mlmodelc" ? "truncated" : "local"
            )
            trees[sub, default: []].append(["path": "\(sub)/\(bundle)", "type": "directory"])
            trees["\(sub)/\(bundle)"] = [TreeStubURLProtocol.fileEntry("\(sub)/\(bundle)/config.json")]
        }
        for file in ["vocab.json", "LICENSE"] { try makeFile("\(repo.folderName)/\(file)") }
        trees[""] = [
            TreeStubURLProtocol.fileEntry("vocab.json", lfs: false),
            TreeStubURLProtocol.fileEntry("LICENSE", lfs: false),
        ]
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(repo, to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 0)
        XCTAssertFalse(exists("\(repo.folderName)/other.mlmodelc"))
        for file in ["vocab.json", "LICENSE"] {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/\(file)")), Data("local".utf8))
        }
        XCTAssertTrue(
            ModelCache.matchesRevision(at: repoPath.appendingPathComponent(repo.folderName), revision: revision))
    }

    func testSubdirectoryLegacyFileNeverUsesRepoRootListing() async throws {
        try makeFile("metadata/config.json")
        try makeFile("metadata/LICENSE")
        TreeStubURLProtocol.trees = [
            "metadata": [TreeStubURLProtocol.fileEntry("metadata/config.json")],
            "": [TreeStubURLProtocol.fileEntry("LICENSE")],
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath, configuration: stubConfiguration)

        XCTAssertFalse(TreeStubURLProtocol.treeRequests.contains(""), "subdirectory adoption must not list repo root")
        XCTAssertFalse(exists("LICENSE"), "repo-root files must not be downloaded into repoDirectory")
        XCTAssertFalse(exists("metadata/LICENSE"), "an unrelated root entry must not validate a subdirectory file")
        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 0)
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath.appendingPathComponent("metadata"), revision: revision))
    }

    func testSameSizeDifferentContentIsRefetched() async throws {
        try makeFile("metadata/config.json", contents: "stale")
        TreeStubURLProtocol.trees = [
            "metadata": [TreeStubURLProtocol.fileEntry("metadata/config.json", contents: "fresh")]
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 1)
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("metadata/config.json")), Data("fresh".utf8))
    }

    func testMatchingLFSAndGitBlobIdentitiesAreKept() async throws {
        try makeFile("metadata/weight.bin")
        try makeFile("metadata/config.json")
        TreeStubURLProtocol.trees = [
            "metadata": [
                TreeStubURLProtocol.fileEntry("metadata/weight.bin"),
                TreeStubURLProtocol.fileEntry("metadata/config.json", lfs: false),
            ]
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 0)
        for file in ["weight.bin", "config.json"] {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("metadata/" + file)), Data("local".utf8))
        }
    }

    func testMatchingEmptyAndUnknownSizeIdentitiesAreKept() async throws {
        try makeFile("metadata/empty.json", contents: "")
        try makeFile("metadata/empty.json.partial", contents: "par")
        try makeFile("metadata/empty.json.partial.etag", contents: "etag")
        try makeFile("metadata/unknown.bin")
        TreeStubURLProtocol.trees = [
            "metadata": [
                TreeStubURLProtocol.fileEntry("metadata/empty.json", size: 0, contents: "", lfs: false),
                TreeStubURLProtocol.fileEntry("metadata/unknown.bin", size: -1),
            ]
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 0)
        XCTAssertEqual(try Data(contentsOf: repoPath.appendingPathComponent("metadata/empty.json")), Data())
        XCTAssertFalse(exists("metadata/empty.json.partial"))
        XCTAssertFalse(exists("metadata/empty.json.partial.etag"))
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("metadata/unknown.bin")), Data("local".utf8))
    }

    func testListedFileWithoutIdentityIsRefetched() async throws {
        try makeFile("metadata/config.json")
        TreeStubURLProtocol.trees = [
            "metadata": [["path": "metadata/config.json", "type": "file", "size": 5]]
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 1)
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("metadata/config.json")), Data("fresh".utf8))
    }

    func testMalformedContentIDsAreRejected() throws {
        let files = [
            RemoteFile(path: "short.bin", size: 5, contentID: .lfsSHA256("abc")),
            RemoteFile(path: "nonhex.bin", size: 5, contentID: .lfsSHA256(String(repeating: "z", count: 64))),
            RemoteFile(path: "short.json", size: 5, contentID: .gitBlobSHA1("abc")),
            RemoteFile(path: "nonhex.json", size: 5, contentID: .gitBlobSHA1(String(repeating: "z", count: 40))),
        ]
        for file in files { try makeFile(file.path) }

        try prepare(files)

        for file in files { XCTAssertFalse(exists(file.path)) }
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testStreamingHashesMatchAcrossChunkBoundariesWithUnknownSizes() async throws {
        // Multibyte text exercises Git's byte-count header as well as multiple reads.
        let contents = String(repeating: "cache-é", count: 150_000)
        try makeFile("metadata/weight.bin", contents: contents)
        try makeFile("metadata/config.json", contents: contents)
        TreeStubURLProtocol.trees = [
            "metadata": [
                TreeStubURLProtocol.fileEntry("metadata/weight.bin", size: -1, contents: contents),
                TreeStubURLProtocol.fileEntry("metadata/config.json", size: -1, contents: contents, lfs: false),
            ]
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 0)
        for file in ["weight.bin", "config.json"] {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("metadata/" + file)), Data(contents.utf8))
        }
    }

    func testSkippedLegacyFilesAreValidatedBeforeMarkerWithoutRefetching() async throws {
        try makeFile("metadata/config.json")
        try makeFile("metadata/skipped.bin", contents: "stale")
        TreeStubURLProtocol.trees = [
            "metadata": [
                TreeStubURLProtocol.fileEntry("metadata/config.json"),
                TreeStubURLProtocol.fileEntry("metadata/skipped.bin"),
            ]
        ]
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(
            .diarizer, subdirectory: "metadata", to: repoPath,
            shouldSkip: { $0.hasSuffix("skipped.bin") }, configuration: stubConfiguration)

        XCTAssertFalse(exists("metadata/skipped.bin"))
        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 0)
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath.appendingPathComponent("metadata"), revision: revision))
    }

    func testRepoAdoptionVerifiesNestedPlainFilesAndInvalidatesNestedBundles() async throws {
        let repo = Repo.diarizer
        let required = ModelNames.getRequiredModelNames(for: repo, variant: nil).sorted()
        var trees: [String: [[String: Any]]] = ["": [["path": "voices", "type": "directory"]]]
        for name in required {
            try makeFile("\(repo.folderName)/\(name)/config.json")
            trees["", default: []].append(["path": name, "type": "directory"])
            trees[name] = [TreeStubURLProtocol.fileEntry("\(name)/config.json")]
        }
        for file in [
            "voices/x.bin", "voices/kept.bin", "voices/unlisted.bin", "voices/deep/other.mlmodelc/config.json",
        ] {
            try makeFile("\(repo.folderName)/" + file, contents: file.hasSuffix("kept.bin") ? "local" : "stale")
        }
        trees["voices"] = [
            TreeStubURLProtocol.fileEntry("voices/x.bin"),
            TreeStubURLProtocol.fileEntry("voices/kept.bin"),
            ["path": "voices/deep", "type": "directory"],
        ]
        trees["voices/deep"] = [["path": "voices/deep/other.mlmodelc", "type": "directory"]]
        trees["voices/deep/other.mlmodelc"] = [TreeStubURLProtocol.fileEntry("voices/deep/other.mlmodelc/config.json")]
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(repo, to: repoPath, configuration: stubConfiguration)

        XCTAssertFalse(exists("\(repo.folderName)/voices/x.bin"))
        XCTAssertFalse(exists("\(repo.folderName)/voices/unlisted.bin"))
        XCTAssertFalse(exists("\(repo.folderName)/voices/deep/other.mlmodelc"))
        XCTAssertTrue(exists("\(repo.folderName)/voices"))
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/voices/kept.bin")),
            Data("local".utf8))
        XCTAssertEqual(
            TreeStubURLProtocol.fileRequestCount, 0, "validation-only files must not enter the download loop")
        XCTAssertTrue(
            ModelCache.matchesRevision(at: repoPath.appendingPathComponent(repo.folderName), revision: revision))
    }

}
