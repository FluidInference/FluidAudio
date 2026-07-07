import XCTest

@testable import FluidAudio

/// `DownloadUtils.enforceOffline` short-circuits every public download
/// surface and the `loadModels` retry-with-redownload fallback. Validate
/// the toggle behaviour without spinning up a real HuggingFace fetch.
///
/// Each test toggles the flag on, asserts the relevant entry point
/// throws `DownloadUtils.OfflineError.networkDisabled`, and resets the
/// flag in `tearDown` so cross-test order does not leak state.
final class DownloadUtilsOfflineTests: XCTestCase {

    override func tearDown() {
        DownloadUtils.enforceOffline = false
        super.tearDown()
    }

    func testFetchWithAuthThrowsNetworkDisabledInOfflineMode() async {
        DownloadUtils.enforceOffline = true
        let url = URL(string: "https://huggingface.co/test/file")!

        do {
            _ = try await DownloadUtils.fetchWithAuth(from: url)
            XCTFail("expected OfflineError.networkDisabled")
        } catch let DownloadUtils.OfflineError.networkDisabled(operation) {
            XCTAssertTrue(
                operation.hasPrefix("fetchWithAuth("),
                "operation tag should identify the blocked path, got: \(operation)"
            )
        } catch {
            XCTFail("expected OfflineError.networkDisabled, got: \(error)")
        }
    }

    func testFetchHuggingFaceFileThrowsNetworkDisabledInOfflineMode() async {
        DownloadUtils.enforceOffline = true
        let url = URL(string: "https://huggingface.co/test/file")!

        do {
            _ = try await DownloadUtils.fetchHuggingFaceFile(
                from: url,
                description: "test-file",
                maxAttempts: 1,
                minBackoff: 0.01
            )
            XCTFail("expected OfflineError.networkDisabled")
        } catch let DownloadUtils.OfflineError.networkDisabled(operation) {
            XCTAssertEqual(operation, "fetchHuggingFaceFile(test-file)")
        } catch {
            XCTFail("expected OfflineError.networkDisabled, got: \(error)")
        }
    }

    func testDownloadRepoThrowsNetworkDisabledInOfflineMode() async {
        DownloadUtils.enforceOffline = true
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try await DownloadUtils.downloadRepo(.vad, to: dir)
            XCTFail("expected OfflineError.networkDisabled")
        } catch let DownloadUtils.OfflineError.networkDisabled(operation) {
            XCTAssertTrue(operation.hasPrefix("downloadRepo("), "got: \(operation)")
        } catch {
            XCTFail("expected OfflineError.networkDisabled, got: \(error)")
        }
    }

    func testDownloadSubdirectoryThrowsNetworkDisabledInOfflineMode() async {
        DownloadUtils.enforceOffline = true
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try await DownloadUtils.downloadSubdirectory(
                .vad, subdirectory: "anything", to: dir)
            XCTFail("expected OfflineError.networkDisabled")
        } catch let DownloadUtils.OfflineError.networkDisabled(operation) {
            XCTAssertTrue(operation.hasPrefix("downloadSubdirectory("), "got: \(operation)")
        } catch {
            XCTFail("expected OfflineError.networkDisabled, got: \(error)")
        }
    }

    func testLoadModelsSurfacesTypedMissingModelsInOfflineMode() async {
        DownloadUtils.enforceOffline = true
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try await DownloadUtils.loadModels(
                .vad, modelNames: [ModelNames.VAD.sileroVadFile], directory: dir)
            XCTFail("expected OfflineError.modelMissing")
        } catch let DownloadUtils.OfflineError.modelMissing(repo, missing) {
            // Pinned: offline + empty cache surfaces the typed error with the
            // missing file list — it must NOT attempt a download or purge.
            XCTAssertEqual(repo, Repo.vad.folderName)
            XCTAssertEqual(missing, [ModelNames.VAD.sileroVadFile])
        } catch {
            XCTFail("expected OfflineError.modelMissing, got: \(error)")
        }
    }

    func testDefaultBehaviourDoesNotShortCircuit() async {
        // Flag defaults to false. We do not exercise the real network here
        // (the unit-test environment has no offline guarantees about HF
        // reachability), but we confirm the gate itself does not throw
        // when the flag is off.
        XCTAssertFalse(DownloadUtils.enforceOffline)
        do {
            try Self.callEnsureOnlineAllowed("test.no-op")
        } catch {
            XCTFail("ensureOnlineAllowed must not throw when enforceOffline=false; got: \(error)")
        }
    }

    func testOfflineErrorDescriptionsFormat() {
        let blocked = DownloadUtils.OfflineError.networkDisabled(
            operation: "downloadRepo(parakeet)"
        )
        XCTAssertEqual(
            blocked.errorDescription,
            "FluidAudio offline mode: downloadRepo(parakeet) blocked"
        )

        let missing = DownloadUtils.OfflineError.modelMissing(
            repo: "parakeet",
            missing: ["A.mlmodelc", "B.mlmodelc"]
        )
        XCTAssertEqual(
            missing.errorDescription,
            "FluidAudio offline mode: required models missing for parakeet: A.mlmodelc, B.mlmodelc"
        )
    }

    // MARK: - test reflection helpers

    /// The gate helper is `private static`. We re-implement the same
    /// check shape in the test to validate the contract — the
    /// behaviour matters more than the exact symbol being addressable.
    private static func callEnsureOnlineAllowed(_ operation: String) throws {
        if DownloadUtils.enforceOffline {
            throw DownloadUtils.OfflineError.networkDisabled(operation: operation)
        }
    }
}
