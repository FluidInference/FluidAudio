import Foundation
import XCTest
import os

@testable import FluidAudio

/// Characterization tests for `downloadRepo`'s progress emissions (#765
/// Wave 1). Pins the CURRENT convention that UIs depend on: the download
/// phase of a repo operation occupies `fractionCompleted` 0.0–0.5 (compile
/// occupies 0.5–1.0 and needs real models, so it is pinned by inspection,
/// not here), fractions are monotonic and byte-weighted, and phases arrive
/// in listing → downloading order with accurate file counters.
final class ProgressSequenceTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProgressSequence-\(UUID().uuidString)", isDirectory: true)
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

    func testRepoDownloadProgressIsMonotonicWithinZeroToHalf() async throws {
        let model = ModelNames.VAD.sileroVadFile
        TreeStubURLProtocol.trees = [
            "": [["path": model, "type": "directory"]],
            model: [
                ["path": "\(model)/coremldata.bin", "type": "file", "size": 40],
                ["path": "\(model)/weights/weight.bin", "type": "file", "size": 40],
            ],
        ]
        // Same body served for every file; sizes above must match its length.
        TreeStubURLProtocol.fileBody = Data(String(repeating: "x", count: 40).utf8)

        let recorder = ProgressStreamRecorder()
        try await DownloadUtils.downloadRepo(
            .vad, to: workDir,
            progressHandler: { recorder.append($0) },
            configuration: stubConfiguration)

        let events = recorder.snapshot()
        XCTAssertFalse(events.isEmpty)

        // Pinned: the stream opens with a `.listing` emission at fraction 0.
        guard case .listing = events[0].phase else {
            return XCTFail("first emission should be .listing, got \(events[0].phase)")
        }
        XCTAssertEqual(events[0].fractionCompleted, 0.0)

        // Pinned: download-phase fractions live in [0, 0.5] and never regress.
        var previous = -Double.infinity
        for event in events {
            XCTAssertGreaterThanOrEqual(event.fractionCompleted, previous)
            XCTAssertLessThanOrEqual(event.fractionCompleted, 0.5)
            previous = event.fractionCompleted
        }

        // Pinned: the final emission reports all files completed at 0.5.
        guard case .downloading(let completed, let total) = events.last!.phase else {
            return XCTFail("last emission should be .downloading, got \(events.last!.phase)")
        }
        XCTAssertEqual(completed, total)
        XCTAssertEqual(events.last!.fractionCompleted, 0.5, accuracy: 0.0001)
    }

    func testFileCountersNeverExceedTotalsAndReachTotal() async throws {
        let model = ModelNames.VAD.sileroVadFile
        TreeStubURLProtocol.trees = [
            "": [["path": model, "type": "directory"]],
            model: [
                ["path": "\(model)/a.bin", "type": "file", "size": 10],
                ["path": "\(model)/b.bin", "type": "file", "size": 10],
                ["path": "\(model)/coremldata.bin", "type": "file", "size": 10],
            ],
        ]
        TreeStubURLProtocol.fileBody = Data(String(repeating: "x", count: 10).utf8)

        let recorder = ProgressStreamRecorder()
        try await DownloadUtils.downloadRepo(
            .vad, to: workDir,
            progressHandler: { recorder.append($0) },
            configuration: stubConfiguration)

        var sawFinal = false
        for event in recorder.snapshot() {
            if case .downloading(let completed, let total) = event.phase {
                XCTAssertLessThanOrEqual(completed, total)
                XCTAssertEqual(total, 3)
                if completed == total { sawFinal = true }
            }
        }
        XCTAssertTrue(sawFinal, "stream must report completedFiles == totalFiles at the end")
    }
}

/// Lock-guarded recorder for progress emissions (tests only).
final class ProgressStreamRecorder: Sendable {
    private let events = OSAllocatedUnfairLock<[DownloadUtils.DownloadProgress]>(initialState: [])

    func append(_ event: DownloadUtils.DownloadProgress) {
        events.withLock { $0.append(event) }
    }

    func snapshot() -> [DownloadUtils.DownloadProgress] {
        events.withLock { $0 }
    }
}
