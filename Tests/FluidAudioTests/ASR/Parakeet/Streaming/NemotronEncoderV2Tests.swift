import Foundation
import XCTest

@testable import FluidAudio

/// `encoder_v2.mlmodelc` (#1141). Filesystem fixtures and a stub transport
/// only; no CoreML model is loaded and nothing is downloaded.
final class NemotronEncoderV2Tests: XCTestCase {

    private typealias Names = ModelNames.NemotronMultilingualStreaming
    private typealias Manager = StreamingNemotronMultilingualAsrManager

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("NemotronEncoderV2-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        TreeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        TreeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    func testEncoderV2IsPublishedOutsideTheTierFolders() {
        XCTAssertEqual(Names.encoderV2ChunkMs, [560, 1120, 2240])
        XCTAssertEqual(Names.encoderV2Subdirectory(chunkMs: 2240), "encoder-v2/2240ms")
        for chunkMs in Names.encoderV2ChunkMs {
            let subdirectory = Names.encoderV2Subdirectory(chunkMs: chunkMs)
            XCTAssertFalse(subdirectory.hasPrefix("latin/"))
            XCTAssertFalse(subdirectory.hasPrefix("multilingual/"))
        }
    }

    func testVariantDownloadSkipsTheTierEncoderOnlyWhereV2ReplacesIt() {
        let skip = Manager.skipsVariantPath
        let encoderWeights = "multilingual/2240ms/encoder.mlmodelc/weights/weight.bin"
        XCTAssertTrue(skip(encoderWeights, true, 2240))
        XCTAssertTrue(skip("latin/560ms/encoder.mlmodelc/model.mil", true, 560))
        XCTAssertFalse(skip("multilingual/4480ms/encoder.mlmodelc/weights/weight.bin", true, 4480))
        XCTAssertTrue(skip("multilingual/4480ms/encoder.mlmodelc/weights/weight.bin", false, 4480))
        XCTAssertFalse(skip("multilingual/2240ms/decoder_joint.mlmodelc/weights/weight.bin", true, 2240))
        XCTAssertFalse(skip("multilingual/2240ms/metadata.json", true, 2240))
        XCTAssertTrue(skip("multilingual/2240ms/encoder.mlpackage/Manifest.json", true, 4480))
    }

    func testLoaderPrefersEncoderV2WhenTheDirectoryHasIt() throws {
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderFile)
        try FileManager.default.createDirectory(
            at: workDir.appendingPathComponent(Names.encoderV2File), withIntermediateDirectories: true)
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderV2File)
    }

    func testDownloadMovesEncoderV2NextToTheOtherModels() async throws {
        let remote = "encoder-v2/2240ms/\(Names.encoderV2File)"
        TreeStubURLProtocol.trees = [
            "encoder-v2/2240ms": [["path": remote, "type": "directory"]],
            remote: [
                ["path": "\(remote)/coremldata.bin", "type": "file", "size": 4],
                ["path": "\(remote)/model.mil", "type": "file", "size": 4],
                ["path": "\(remote)/weights", "type": "directory"],
            ],
            "\(remote)/weights": [["path": "\(remote)/weights/weight.bin", "type": "file", "size": 4]],
        ]
        TreeStubURLProtocol.fileBody = Data("xxxx".utf8)
        // An interrupted copy at the destination is replaced, not merged.
        let destination = workDir.appendingPathComponent(Names.encoderV2File)
        let staleWeights = destination.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: staleWeights, withIntermediateDirectories: true)
        try Data("w".utf8).write(to: staleWeights.appendingPathComponent("weight.bin.partial"))

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [TreeStubURLProtocol.self]
        try await Manager.downloadEncoderV2(
            chunkMs: 2240, into: workDir, progressHandler: nil, configuration: configuration)

        XCTAssertTrue(ModelCache.isCacheComplete(at: workDir, requiredFiles: [Names.encoderV2File]))
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("weights/weight.bin")).count, 4)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: staleWeights.appendingPathComponent("weight.bin.partial").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: workDir.appendingPathComponent("encoder-v2").path))
        XCTAssertEqual(Manager.compiledEncoderName(in: workDir), Names.encoderV2File)
    }
}
