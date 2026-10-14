import XCTest

@testable import FluidAudio

/// Issue #1023: the BART / CharsiuG2P models and the Kokoro text-frontend
/// assets must resolve under the caller's `directory`, not always under the
/// default TTS cache root.
final class G2PModelDirectoryTests: XCTestCase {

    private func makeTempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("g2p-dir-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // MARK: - G2PModel

    func testG2PModelResolvesKokoroDirectoryUnderCustomRoot() throws {
        let root = try makeTempRoot()
        let model = G2PModel(directory: root)
        XCTAssertEqual(model.modelsDirectory, root)
        XCTAssertEqual(
            try model.kokoroDirectory().standardizedFileURL,
            root.appendingPathComponent(Repo.kokoro.folderName).standardizedFileURL)
    }

    func testG2PModelDefaultRootIsSharedTtsCache() throws {
        let expected = try TtsCacheDirectory.ensure()
            .appendingPathComponent("Models")
            .appendingPathComponent(Repo.kokoro.folderName)
        XCTAssertNil(G2PModel.shared.modelsDirectory)
        XCTAssertEqual(try G2PModel.shared.kokoroDirectory().standardizedFileURL, expected.standardizedFileURL)
    }

    func testG2PModelSharedRegistryIsPerDirectory() throws {
        let root = try makeTempRoot()
        let other = try makeTempRoot()
        XCTAssertTrue(G2PModel.shared(directory: root) === G2PModel.shared(directory: root))
        XCTAssertTrue(G2PModel.shared === G2PModel.shared(directory: nil))
        XCTAssertFalse(G2PModel.shared(directory: root) === G2PModel.shared)
        XCTAssertFalse(G2PModel.shared(directory: root) === G2PModel.shared(directory: other))
        // Same location spelled differently (trailing slash) → same instance.
        let slashed = URL(fileURLWithPath: root.path + "/", isDirectory: true)
        XCTAssertTrue(G2PModel.shared(directory: root) === G2PModel.shared(directory: slashed))
    }

    func testG2PModelLoadsFromCustomRootNotDefaultCache() async throws {
        // An empty custom root must fail with a path inside that root: proof
        // the loader never silently falls back to ~/.cache/fluidaudio.
        let root = try makeTempRoot()
        let model = G2PModel(directory: root)
        do {
            try await model.ensureModelsAvailable()
            XCTFail("expected vocabLoadFailed from an empty root")
        } catch let error as G2PModel.G2PModelError {
            guard case .vocabLoadFailed(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains(root.path), "detail should point into the custom root: \(detail)")
        }
    }

    // MARK: - MultilingualG2PModel

    func testMultilingualG2PModelResolvesKokoroDirectoryUnderCustomRoot() throws {
        let root = try makeTempRoot()
        let model = MultilingualG2PModel(directory: root)
        XCTAssertEqual(model.modelsDirectory, root)
        XCTAssertEqual(
            try model.kokoroDirectory().standardizedFileURL,
            root.appendingPathComponent(Repo.kokoro.folderName).standardizedFileURL)
        XCTAssertTrue(MultilingualG2PModel.shared(directory: root) === MultilingualG2PModel.shared(directory: root))
        XCTAssertTrue(MultilingualG2PModel.shared === MultilingualG2PModel.shared(directory: nil))
        XCTAssertFalse(MultilingualG2PModel.shared(directory: root) === MultilingualG2PModel.shared)
    }

    func testMultilingualG2PModelLoadsFromCustomRootNotDefaultCache() async throws {
        let root = try makeTempRoot()
        let model = MultilingualG2PModel(directory: root)
        do {
            try await model.ensureModelsAvailable()
            XCTFail("expected modelLoadFailed from an empty root")
        } catch let error as MultilingualG2PError {
            guard case .modelLoadFailed(let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains(root.path), "detail should point into the custom root: \(detail)")
        }
    }

    // MARK: - KokoroAne wiring

    func testKokoroAneManagerBindsEnglishG2PToStoreDirectory() throws {
        let root = try makeTempRoot()
        let manager = KokoroAneManager(directory: root)
        XCTAssertEqual(manager.englishG2P.modelsDirectory, root)
        XCTAssertTrue(manager.englishG2P === G2PModel.shared(directory: root))

        let store = KokoroAneModelStore(directory: root, variant: .english)
        XCTAssertEqual(store.directory, root)
        let fromStore = KokoroAneManager(modelStore: store)
        XCTAssertTrue(fromStore.englishG2P === G2PModel.shared(directory: root))

        XCTAssertTrue(KokoroAneManager().englishG2P === G2PModel.shared)
    }
}
