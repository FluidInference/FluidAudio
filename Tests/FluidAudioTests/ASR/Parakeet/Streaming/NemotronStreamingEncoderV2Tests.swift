import Foundation
import XCTest

@testable import FluidAudio

/// English `encoder_int8_v2.mlmodelc` (#1170). Filesystem fixtures and a stub
/// transport only; no CoreML model is loaded and nothing is downloaded.
final class NemotronStreamingEncoderV2Tests: XCTestCase {

    private typealias Names = ModelNames.NemotronStreaming
    private typealias Manager = StreamingNemotronAsrManager

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("NemotronStreamingEncoderV2-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        TreeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        TreeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    private var stubConfiguration: URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [TreeStubURLProtocol.self]
        return configuration
    }

    private func stubEncoderV2Tree(chunkMs: Int) {
        let subdirectory = Names.encoderV2Subdirectory(chunkMs: chunkMs)
        let remote = "\(subdirectory)/encoder_int8_v2.mlmodelc"
        TreeStubURLProtocol.trees = [
            subdirectory: [["path": remote, "type": "directory"]],
            remote: [
                ["path": "\(remote)/coremldata.bin", "type": "file", "size": 3],
                ["path": "\(remote)/model.mil", "type": "file", "size": 3],
                ["path": "\(remote)/weights", "type": "directory"],
            ],
            "\(remote)/weights": [["path": "\(remote)/weights/weight.bin", "type": "file", "size": 3]],
        ]
        // Every file gets this body; as a `model.mil` it is a closed program.
        TreeStubURLProtocol.fileBody = Data("{ }".utf8)
    }

    func testEncoderV2IsPublishedOutsideTheTierFolders() {
        XCTAssertEqual(Names.encoderV2Subdirectory(chunkMs: 1120), "encoder-v2/1120ms")
        for chunkSize in NemotronChunkSize.allCases {
            let subdirectory = Names.encoderV2Subdirectory(chunkMs: chunkSize.rawValue)
            XCTAssertFalse(subdirectory.hasPrefix(chunkSize.subdirectory + "/"))
        }
        // `ModelHub.download(repo)` keeps fetching encoder_int8; the manager's
        // variant skips it because it fetches encoder_int8_v2 itself.
        for chunkSize in NemotronChunkSize.allCases {
            let byDefault = ModelNames.getRequiredModelNames(for: chunkSize.repo, variant: nil)
            let forManager = ModelNames.getRequiredModelNames(
                for: chunkSize.repo, variant: Names.withoutEncoderVariant)
            XCTAssertTrue(byDefault.contains(Names.encoderInt8File))
            XCTAssertEqual(byDefault.subtracting(forManager), [Names.encoderInt8File])
            XCTAssertFalse(byDefault.union(forManager).contains(Names.encoderV2File))
        }
    }

    private func makeCompiledBundle(_ name: String) throws {
        let bundle = workDir.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data("c".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))
        try Data("{ }".utf8).write(to: bundle.appendingPathComponent("model.mil"))
    }

    func testLoaderPrefersEncoderV2OnlyWhenItIsComplete() throws {
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderInt8File)
        // An incomplete copy (no coremldata.bin) is not loaded.
        try FileManager.default.createDirectory(
            at: workDir.appendingPathComponent(Names.encoderV2File), withIntermediateDirectories: true)
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderInt8File)
        try makeCompiledBundle(Names.encoderV2File)
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderV2File)
    }

    func testDownloadMovesEncoderV2IntoTheEncoderFolder() async throws {
        stubEncoderV2Tree(chunkMs: 2240)
        // An interrupted copy at the destination is replaced, not merged.
        let destination = workDir.appendingPathComponent(Names.encoderV2File)
        let staleWeights = destination.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: staleWeights, withIntermediateDirectories: true)
        try Data("w".utf8).write(to: staleWeights.appendingPathComponent("weight.bin.partial"))

        try await Manager.downloadEncoderV2(
            chunkSize: .ms2240, into: workDir, progressHandler: nil, configuration: stubConfiguration)

        XCTAssertTrue(ModelCache.isCacheComplete(at: workDir, requiredFiles: [Names.encoderV2File]))
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("weights/weight.bin")).count, 3)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: staleWeights.appendingPathComponent("weight.bin.partial").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: workDir.appendingPathComponent("encoder-v2").path))
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderV2File)
    }

    func testCompleteEncoderV2IsNotFetchedAgain() async throws {
        stubEncoderV2Tree(chunkMs: 560)
        try await Manager.ensureEncoderV2(
            chunkSize: .ms560, in: workDir, progressHandler: nil, configuration: stubConfiguration)
        let requests = TreeStubURLProtocol.fileRequestCount
        XCTAssertGreaterThan(requests, 0)

        try await Manager.ensureEncoderV2(
            chunkSize: .ms560, in: workDir, progressHandler: nil, configuration: stubConfiguration)
        XCTAssertEqual(TreeStubURLProtocol.fileRequestCount, requests)
    }

    func testFailedDownloadKeepsACachedInt8Encoder() async throws {
        // No tree is stubbed, so listing encoder-v2/1120ms fails.
        do {
            try await Manager.ensureEncoderV2(
                chunkSize: .ms1120, in: workDir, progressHandler: nil, configuration: stubConfiguration)
            XCTFail("Without a cached encoder the failed download must throw")
        } catch {}

        try makeCompiledBundle(Names.encoderInt8File)
        // An interrupted v2 copy next to it must not win over the complete int8.
        try FileManager.default.createDirectory(
            at: workDir.appendingPathComponent(Names.encoderV2File).appendingPathComponent("weights"),
            withIntermediateDirectories: true)
        try await Manager.ensureEncoderV2(
            chunkSize: .ms1120, in: workDir, progressHandler: nil, configuration: stubConfiguration)
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderInt8File)
    }
}
