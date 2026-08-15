import Foundation
import XCTest

@testable import FluidAudio

/// Network-gated smoke test (#765 Wave 1): cold-download the smallest real
/// repo end-to-end through the live HF CDN — listing → filtering → retry →
/// transport → validation → cache layout → CoreML load. Run by
/// `download-smoke.yml` on every PR touching download code; skipped unless
/// `FLUID_NETWORK_TESTS=1` so default `swift test` stays offline.
final class DownloadSmokeTests: XCTestCase {

    func testColdDownloadOfSmallestRepoProducesLoadableModel() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUID_NETWORK_TESTS"] == "1",
            "network-gated; set FLUID_NETWORK_TESTS=1 to run")

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("download-smoke-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        // Cold: no cache exists at `dir`, so this exercises the full pipeline.
        let models = try await ModelHub.loadModels(
            .vad,
            modelNames: [ModelNames.VAD.sileroVadFile],
            directory: dir,
            computeUnits: .cpuOnly
        )

        // The model compiled and loaded — the strongest end-to-end assertion
        // that the downloaded bytes are complete and uncorrupted.
        XCTAssertNotNil(models[ModelNames.VAD.sileroVadFile])

        // And the on-disk layout is where the cache contract says it is.
        let repoPath = dir.appendingPathComponent(Repo.vad.folderName)
        let modelPath = repoPath.appendingPathComponent(ModelNames.VAD.sileroVadFile)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: modelPath.appendingPathComponent("coremldata.bin").path))
    }
}
