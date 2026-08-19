import Foundation
import XCTest

@testable import FluidAudio

/// Drives `ModelHub.download(_:subdirectory:...)` through the
/// `configuration:` seam with a stub transport. Pins the behavior the #853
/// concurrent rewrite must preserve: every listed file lands on disk,
/// `shouldSkip` prunes subtrees, already-present files are not refetched,
/// and progress stays monotonic in [0, 1] with accurate file counters even
/// though files now finish out of order.
final class SubdirectoryDownloadTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("SubdirDownload-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        TreeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        TreeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    private var stubConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [TreeStubURLProtocol.self]
        return config
    }

    /// A pack with more files than the concurrency width (default 4), so the
    /// task-group refill path is exercised, plus a nested directory and a
    /// zero-byte file (created locally, never fetched).
    private func stubPack(fileCount: Int, bodySize: Int) {
        var rootItems: [[String: Any]] = (0..<fileCount).map {
            ["path": "pack/file\($0).bin", "type": "file", "size": bodySize]
        }
        rootItems.append(["path": "pack/sub", "type": "directory"])
        rootItems.append(["path": "pack/empty.bin", "type": "file", "size": 0])
        TreeStubURLProtocol.trees = [
            "pack": rootItems,
            "pack/sub": [["path": "pack/sub/nested.bin", "type": "file", "size": bodySize]],
        ]
        TreeStubURLProtocol.fileBody = Data(String(repeating: "x", count: bodySize).utf8)
    }

    func testDownloadsEveryListedFile() async throws {
        stubPack(fileCount: 9, bodySize: 32)

        try await ModelHub.download(
            .vad, subdirectory: "pack", to: workDir,
            configuration: stubConfiguration)

        for index in 0..<9 {
            let file = workDir.appendingPathComponent("pack/file\(index).bin")
            XCTAssertEqual(try Data(contentsOf: file).count, 32, "missing or truncated \(file.path)")
        }
        XCTAssertEqual(
            try Data(contentsOf: workDir.appendingPathComponent("pack/sub/nested.bin")).count, 32)
        XCTAssertEqual(
            try Data(contentsOf: workDir.appendingPathComponent("pack/empty.bin")).count, 0)
    }

    func testProgressIsMonotonicAndReachesAllFiles() async throws {
        stubPack(fileCount: 9, bodySize: 32)

        let recorder = ProgressStreamRecorder()
        try await ModelHub.download(
            .vad, subdirectory: "pack", to: workDir,
            progressHandler: { recorder.append($0) },
            configuration: stubConfiguration)

        let events = recorder.snapshot()
        XCTAssertFalse(events.isEmpty)

        guard case .listing = events[0].phase else {
            return XCTFail("first emission should be .listing, got \(events[0].phase)")
        }
        XCTAssertEqual(events[0].fractionCompleted, 0.0)

        // Subdirectory downloads span the full 0-1 range (no compile phase).
        var previousFraction = -Double.infinity
        var previousCompleted = -1
        for event in events {
            XCTAssertGreaterThanOrEqual(event.fractionCompleted, previousFraction)
            XCTAssertLessThanOrEqual(event.fractionCompleted, 1.0)
            previousFraction = event.fractionCompleted
            if case .downloading(let completed, let total) = event.phase {
                XCTAssertEqual(total, 11)  // 9 root + 1 nested + 1 empty
                XCTAssertLessThanOrEqual(completed, total)
                // Completed-file counts never regress across emissions: the
                // aggregator updates and emits under one lock.
                XCTAssertGreaterThanOrEqual(completed, previousCompleted)
                previousCompleted = completed
            }
        }

        guard case .downloading(let completed, let total) = events.last!.phase else {
            return XCTFail("last emission should be .downloading, got \(events.last!.phase)")
        }
        XCTAssertEqual(completed, total)
        XCTAssertEqual(events.last!.fractionCompleted, 1.0, accuracy: 0.0001)
    }

    func testShouldSkipPrunesFilesAndSubtrees() async throws {
        stubPack(fileCount: 3, bodySize: 16)

        try await ModelHub.download(
            .vad, subdirectory: "pack", to: workDir,
            shouldSkip: { path in
                path == "pack/sub" || path.hasSuffix("file1.bin")
            },
            configuration: stubConfiguration)

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: workDir.appendingPathComponent("pack/file0.bin").path))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: workDir.appendingPathComponent("pack/file1.bin").path))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: workDir.appendingPathComponent("pack/sub/nested.bin").path))
    }

    func testAlreadyPresentFileIsNotRefetched() async throws {
        stubPack(fileCount: 2, bodySize: 16)
        let existing = workDir.appendingPathComponent("pack/file0.bin")
        try FileManager.default.createDirectory(
            at: existing.deletingLastPathComponent(), withIntermediateDirectories: true)
        let localContent = Data("local-content".utf8)
        try localContent.write(to: existing)

        try await ModelHub.download(
            .vad, subdirectory: "pack", to: workDir,
            configuration: stubConfiguration)

        XCTAssertEqual(try Data(contentsOf: existing), localContent)
        XCTAssertEqual(
            try Data(contentsOf: workDir.appendingPathComponent("pack/file1.bin")).count, 16)
    }
}

/// Unit tests for `ConcurrentProgress` — the counter object the concurrent
/// loop shares across tasks. Exercises the orderings the end-to-end stub
/// tests can't force deterministically.
final class ConcurrentProgressTests: XCTestCase {

    func testRetryByteResetIsLiftedToHighWaterMark() throws {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(handler: { recorder.append($0) }, downloadPhaseWeight: 1.0)
        let progress = ConcurrentProgress(reporter: reporter, totalBytes: 100, totalFiles: 2)

        let onBytes = try XCTUnwrap(progress.liveBytesCallback(fileIndex: 0))
        onBytes(50, 100)
        // A retry that restarts the file from byte 0 reports fewer cumulative
        // bytes; the emission must hold the high-water mark, not regress.
        onBytes(10, 100)

        let fractions = recorder.snapshot().map(\.fractionCompleted)
        XCTAssertEqual(fractions, [0.5, 0.5])
    }

    func testFileCompletedCountsAndFinalFraction() {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(handler: { recorder.append($0) }, downloadPhaseWeight: 1.0)
        let progress = ConcurrentProgress(reporter: reporter, totalBytes: 100, totalFiles: 2)

        // Files complete out of order relative to their indices.
        XCTAssertEqual(progress.fileCompleted(fileIndex: 1, size: 60), 1)
        XCTAssertEqual(progress.fileCompleted(fileIndex: 0, size: 40), 2)

        let events = recorder.snapshot()
        XCTAssertEqual(events.count, 2)
        guard case .downloading(let completed, let total) = events.last!.phase else {
            return XCTFail("expected .downloading, got \(events.last!.phase)")
        }
        XCTAssertEqual(completed, 2)
        XCTAssertEqual(total, 2)
        XCTAssertEqual(events.last!.fractionCompleted, 1.0, accuracy: 0.0001)
    }

    func testUnknownTotalBytesFallsBackToFileCounts() {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(handler: { recorder.append($0) }, downloadPhaseWeight: 1.0)
        // All sizes unknown: totalBytes 0, fractions come from file counts.
        let progress = ConcurrentProgress(reporter: reporter, totalBytes: 0, totalFiles: 4)

        XCTAssertEqual(progress.fileCompleted(fileIndex: 2, size: -1), 1)
        let fractions = recorder.snapshot().map(\.fractionCompleted)
        XCTAssertEqual(fractions, [0.25])
    }
}
