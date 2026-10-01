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
            RemoteFile(path: "config.json", size: 5),
            RemoteFile(path: "nested/extra.json", size: 5),
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
            RemoteFile(path: "config.json", size: 5),
            RemoteFile(path: "nested/extra.json", size: 5),
        ])

        XCTAssertTrue(exists("config.json"))
        XCTAssertFalse(exists("nested/extra.json"))
        XCTAssertTrue(exists("nested"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testUnknownAndZeroSizedLegacyFilesAreRemoved() throws {
        try makeFile("unknown.json")
        try makeFile("empty.json", contents: "")
        try makeFile("config.json")

        try prepare([
            RemoteFile(path: "unknown.json", size: -1),
            RemoteFile(path: "empty.json", size: 0),
            RemoteFile(path: "config.json", size: 5),
        ])

        XCTAssertFalse(exists("unknown.json"))
        XCTAssertFalse(exists("empty.json"))
        XCTAssertTrue(exists("config.json"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testMissingLegacyFileRemainsMissing() throws {
        try makeFile("config.json")

        try prepare([
            RemoteFile(path: "config.json", size: 5),
            RemoteFile(path: "missing.json", size: 5),
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
                RemoteFile(path: "q8/nested/config.json", size: 5),
                RemoteFile(path: "vocab.json", size: 5),
            ], subPath: "q8")

        XCTAssertTrue(exists("nested/config.json"))
        XCTAssertTrue(exists("vocab.json"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testDifferentRevisionMarkerStillWipesWholeCache() throws {
        try makeFile(".fluidaudio-revision", contents: String(repeating: "b", count: 40))
        try makeFile("nested/config.json")
        try makeFile("unlisted.json")

        try prepare([RemoteFile(path: "nested/config.json", size: 5)])

        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: repoPath.path), [".fluidaudio-revision"])
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testMatchingRevisionMarkerPreservesFilesWithoutSizeChecks() throws {
        try makeFile(".fluidaudio-revision", contents: revision + "\n")
        try makeFile("config.json")
        try makeFile("weights.bin.partial")

        try prepare([RemoteFile(path: "config.json", size: 100)])

        XCTAssertTrue(exists("config.json"))
        XCTAssertTrue(exists("weights.bin.partial"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
    }

    func testUnmarkedMainCacheIsUnchanged() throws {
        try makeFile("config.json")
        try makeFile("unknown.json")

        try prepare(
            [RemoteFile(path: "config.json", size: 100), RemoteFile(path: "unknown.json", size: -1)],
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
            trees[name] = [["path": "\(name)/config.json", "type": "file", "size": 5]]
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
                ["path": "metadata/config.json", "type": "file", "size": 5],
                ["path": "metadata/nested", "type": "directory"],
            ],
            "metadata/nested": [["path": "metadata/nested/extra.json", "type": "file", "size": 5]],
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
            trees[name] = [["path": "\(name)/config.json", "type": "file", "size": 5]]
        }
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(repo, to: repoPath, configuration: stubConfiguration)

        let offlineFile = repoPath.appendingPathComponent("\(repo.folderName)/\(offline)/config.json")
        if FileManager.default.fileExists(atPath: offlineFile.path) {
            XCTAssertEqual(try Data(contentsOf: offlineFile), Data("fresh".utf8))
        }
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
            RemoteFile(path: "encoder.mlmodelc/config.json", size: 5),
            RemoteFile(path: "encoder.mlmodelc/weights/weight.bin", size: 5),
            RemoteFile(path: "decoder.mlmodelc/config.json", size: 5),
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
            RemoteFile(path: "encoder.mlmodelc/config.json", size: 5),
            RemoteFile(path: "encoder.mlmodelc/weights/weight.bin", size: 5),
        ])

        XCTAssertFalse(exists("encoder.mlmodelc"))
        XCTAssertTrue(ModelCache.matchesRevision(at: repoPath, revision: revision))
        XCTAssertFalse(ModelCache.allModelsExist(at: repoPath, models: ["encoder.mlmodelc"]))
    }

    func testKeptFileLosesStaleSidecarsButRemovedFileKeepsThem() throws {
        for file in ["kept.json", "removed.json"] {
            try makeFile(file, contents: file == "kept.json" ? "local" : "truncated")
            try makeFile(file + ".partial", contents: "par")
            try makeFile(file + ".partial.etag", contents: "\"etag\"")
        }

        try prepare([RemoteFile(path: "kept.json", size: 5), RemoteFile(path: "removed.json", size: 5)])

        XCTAssertTrue(exists("kept.json"))
        XCTAssertFalse(exists("kept.json.partial"))
        XCTAssertFalse(exists("kept.json.partial.etag"))
        XCTAssertFalse(exists("removed.json"))
        XCTAssertTrue(exists("removed.json.partial"))
        XCTAssertTrue(exists("removed.json.partial.etag"))
    }

    func testConcurrentAdoptersTolerateDisappearingFiles() async throws {
        let files = (0..<128).map { RemoteFile(path: "file\($0).json", size: 5) }
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

        try prepare([RemoteFile(path: "encoder.mlmodelc/config.json", size: 5)])

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
            trees["\(sub)/\(bundle)"] = [["path": "\(sub)/\(bundle)/config.json", "type": "file", "size": 5]]
        }
        for file in ["vocab.json", "LICENSE"] { try makeFile("\(repo.folderName)/\(file)") }
        trees[""] = [
            ["path": "vocab.json", "type": "file", "size": 5],
            ["path": "LICENSE", "type": "file", "size": 5],
        ]
        TreeStubURLProtocol.trees = trees
        TreeStubURLProtocol.fileBody = Data("fresh".utf8)

        try await ModelHub.download(repo, to: repoPath, configuration: stubConfiguration)

        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, 1)
        XCTAssertEqual(
            try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/other.mlmodelc/config.json")),
            Data("fresh".utf8))
        for file in ["vocab.json", "LICENSE"] {
            XCTAssertEqual(
                try Data(contentsOf: repoPath.appendingPathComponent("\(repo.folderName)/\(file)")), Data("local".utf8))
        }
        XCTAssertTrue(
            ModelCache.matchesRevision(at: repoPath.appendingPathComponent(repo.folderName), revision: revision))
    }
}
