import XCTest

@testable import FluidAudio

/// The published bundles are keyed by preset name, not by weights, so a cache from an
/// earlier checkpoint passes every presence check in `loadFromHuggingFace`. Without a
/// version marker a client would keep serving superseded models with no error — which is
/// exactly what happened when NVIDIA's general-access checkpoint replaced the preview.
final class Nemotron3CacheVersionTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron3-cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func seedBundle() throws {
        let bundle = root.appendingPathComponent("monolithic/Nemotron3Diarizer_low.mlmodelc")
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: bundle.appendingPathComponent("coremldata.bin"))
    }

    private func writeMarker(_ value: String) throws {
        try Data((value + "\n").utf8).write(
            to: root.appendingPathComponent(ModelNames.Nemotron3.weightsVersionFile))
    }

    /// A cache written by the current release is kept.
    func testCurrentVersionCacheIsKept() throws {
        try seedBundle()
        try writeMarker(ModelNames.Nemotron3.weightsVersion)

        try Nemotron3Models.discardStaleCache(at: root)

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.path))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: root.appendingPathComponent("monolithic/Nemotron3Diarizer_low.mlmodelc/coremldata.bin").path))
    }

    /// A cache from a different checkpoint is removed so the models are re-fetched.
    func testDifferentVersionCacheIsDiscarded() throws {
        try seedBundle()
        try writeMarker("preview-2026-08-28")

        try Nemotron3Models.discardStaleCache(at: root)

        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    /// Caches predating the marker could be from any checkpoint, so they are discarded too.
    func testUnmarkedCacheIsDiscarded() throws {
        try seedBundle()

        try Nemotron3Models.discardStaleCache(at: root)

        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    /// Trailing whitespace in the marker must not force a spurious re-download.
    func testMarkerIsComparedAfterTrimming() throws {
        try seedBundle()
        try Data(("  " + ModelNames.Nemotron3.weightsVersion + " \n\n").utf8).write(
            to: root.appendingPathComponent(ModelNames.Nemotron3.weightsVersionFile))

        try Nemotron3Models.discardStaleCache(at: root)

        XCTAssertTrue(FileManager.default.fileExists(atPath: root.path))
    }

    /// No cache at all is not an error.
    func testMissingCacheIsANoOp() throws {
        let absent = root.appendingPathComponent("does-not-exist")
        XCTAssertNoThrow(try Nemotron3Models.discardStaleCache(at: absent))
    }

    func testMonolithicPresetsLoadFromV2AndSplitPresetsFromSplit() {
        for config in [Nemotron3Config.low, .offline, .fast, .fast32, .fast128] {
            XCTAssertEqual(config.hubSubdirectory, "monolithic/v2", config.modelFileName)
        }
        XCTAssertEqual(Nemotron3Config.preset(named: "fast32-split-w8a8")?.hubSubdirectory, "split")
    }
}
