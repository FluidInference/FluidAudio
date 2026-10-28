#if os(macOS)
import Foundation
import XCTest

@testable import FluidAudioCLI

final class EnhanceBenchmarkDatasetTests: XCTestCase {
    private let header = "fileid,ser,is_farend_noisy,is_nearend_noisy,nearend_scale"

    func testLoadsCompleteManifestInNumericFileIDOrder() throws {
        let metadata = [
            header,
            "10,-2,1,0,0.5",
            "2,3,0,1,0.8",
        ].joined(separator: "\n")
        let directory = try metadataDirectory(metadata)
        let expectedFiles = Set(
            [2, 10].flatMap { id in
                ["fileid_\(id)_mic.wav", "fileid_\(id)_lpb.wav", "fileid_\(id)_clean.wav"]
            })

        let examples = try EnhanceBenchmarkDataset.loadExamples(from: directory) {
            expectedFiles.contains(URL(fileURLWithPath: $0).lastPathComponent)
        }

        XCTAssertEqual(examples.map(\.fileID), ["2", "10"])
        XCTAssertEqual(examples.map(\.ser), [3, -2])
        XCTAssertEqual(examples.map(\.farendNoisy), [false, true])
        XCTAssertEqual(examples.map(\.nearendNoisy), [true, false])
    }

    func testShardsAreContiguousAndCoverEveryItemOnce() throws {
        let items = Array(1...11)
        let shards = try (0..<3).map { try EnhanceBenchmarkDataset.shard(items, index: $0, count: 3) }
        XCTAssertEqual(shards, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]])
        XCTAssertEqual(try EnhanceBenchmarkDataset.shard(items, index: 0, count: 1), items)
        XCTAssertEqual(try EnhanceBenchmarkDataset.shard(items, index: 11, count: 12), [])
        XCTAssertThrowsError(try EnhanceBenchmarkDataset.shard(items, index: 3, count: 3))
        XCTAssertThrowsError(try EnhanceBenchmarkDataset.shard(items, index: -1, count: 3))
        XCTAssertThrowsError(try EnhanceBenchmarkDataset.shard(items, index: 0, count: 0))
    }

    func testRejectsDuplicateFileID() throws {
        let metadata = [header, "1,0,0,0,1", "1,1,0,0,1"].joined(separator: "\n")
        let directory = try metadataDirectory(metadata)

        XCTAssertThrowsError(try EnhanceBenchmarkDataset.loadExamples(from: directory, fileExists: { _ in true })) {
            XCTAssertEqual($0 as? EnhanceBenchmarkDataset.DatasetError, .duplicateFileID("1"))
        }
    }

    func testRejectsMalformedInteger() throws {
        let metadata = [header, "1,not-an-int,0,0,1"].joined(separator: "\n")
        let directory = try metadataDirectory(metadata)

        XCTAssertThrowsError(try EnhanceBenchmarkDataset.loadExamples(from: directory, fileExists: { _ in true })) {
            XCTAssertEqual(
                $0 as? EnhanceBenchmarkDataset.DatasetError,
                .invalidInteger(line: 2, column: "ser", value: "not-an-int"))
        }
    }

    func testRejectsMissingAudioInsteadOfSilentlySkippingRow() throws {
        let directory = try metadataDirectory([header, "7,0,0,0,1"].joined(separator: "\n"))

        XCTAssertThrowsError(
            try EnhanceBenchmarkDataset.loadExamples(from: directory) {
                !URL(fileURLWithPath: $0).lastPathComponent.hasSuffix("_lpb.wav")
            }
        ) {
            XCTAssertEqual(
                $0 as? EnhanceBenchmarkDataset.DatasetError,
                .missingAudio(fileID: "7", path: "fileid_7_lpb.wav"))
        }
    }

    func testStreamingSHA256() throws {
        let directory = try metadataDirectory("abc")
        let url = directory.appendingPathComponent("meta.csv")

        XCTAssertEqual(
            try EnhanceBenchmarkDataset.sha256(of: url),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    func testRejectsNonCanonicalFileIDs() throws {
        for id in ["01", "-1", "abc", "../other"] {
            let directory = try metadataDirectory([header, "\(id),0,0,0,1"].joined(separator: "\n"))
            XCTAssertThrowsError(try EnhanceBenchmarkDataset.loadExamples(from: directory, fileExists: { _ in true }))
        }
    }

    func testRejectsInvalidBooleanFlags() throws {
        let directory = try metadataDirectory([header, "1,0,2,0,1"].joined(separator: "\n"))
        XCTAssertThrowsError(try EnhanceBenchmarkDataset.loadExamples(from: directory, fileExists: { _ in true })) {
            XCTAssertEqual(
                $0 as? EnhanceBenchmarkDataset.DatasetError,
                .invalidInteger(line: 2, column: "is_farend_noisy", value: "2"))
        }
    }

    private func metadataDirectory(_ metadata: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try metadata.write(to: directory.appendingPathComponent("meta.csv"), atomically: true, encoding: .utf8)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }
}
#endif
