import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the unified tree lister (#765 Wave 3): recursive walking,
/// include-based pruning, Link-header pagination (confirmed against the live
/// HF API in Wave 0), and typed errors for rate-limit/HTML/malformed pages.
/// The repo-specific filter *rules* are pinned separately by
/// `DownloadFilterCharacterizationTests` through `downloadRepo`.
final class HFTreeListerTests: XCTestCase {

    private static let repo = "FluidInference/test-repo"

    /// Canned fetch: pages keyed by absolute URL; records request order.
    private final class PageServer {
        private var pages: [String: (Data, HTTPURLResponse)] = [:]
        private(set) var requested: [String] = []

        func addPage(
            url: String, items: [[String: Any]], status: Int = 200, nextPage: String? = nil
        ) throws {
            var headers: [String: String] = [:]
            if let nextPage {
                headers["Link"] = "<\(nextPage)>; rel=\"next\""
            }
            let response = HTTPURLResponse(
                url: URL(string: url)!, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: headers)!
            pages[url] = (try JSONSerialization.data(withJSONObject: items), response)
        }

        func addRawPage(url: String, body: Data, status: Int = 200) {
            let response = HTTPURLResponse(
                url: URL(string: url)!, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: [:])!
            pages[url] = (body, response)
        }

        var fetch: HFTreeLister.Fetch {
            { url in
                self.requested.append(url.absoluteString)
                guard let page = self.pages[url.absoluteString] else {
                    throw HFDownload.DownloadError.invalidResponse
                }
                return page
            }
        }
    }

    private func treeURL(_ path: String = "") -> String {
        let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
        // Mirrors ModelRegistry.apiModels(repo, apiPath)
        return (try! ModelRegistry.apiModels(Self.repo, apiPath)).absoluteString
    }

    // MARK: - Walking + pruning

    func testRecursiveWalkWithPruningAndFileExclusion() async throws {
        let server = PageServer()
        try server.addPage(
            url: treeURL(),
            items: [
                ["path": "keep.mlmodelc", "type": "directory"],
                ["path": "prune.mlmodelc", "type": "directory"],
                ["path": "root.json", "type": "file", "size": 7],
                ["path": "excluded.txt", "type": "file", "size": 9],
            ])
        try server.addPage(
            url: treeURL("keep.mlmodelc"),
            items: [
                ["path": "keep.mlmodelc/coremldata.bin", "type": "file", "size": 42],
                ["path": "keep.mlmodelc/weights", "type": "directory"],
            ])
        try server.addPage(
            url: treeURL("keep.mlmodelc/weights"),
            items: [["path": "keep.mlmodelc/weights/weight.bin", "type": "file"]])

        let files = try await HFTreeLister.listTree(
            repoRemotePath: Self.repo,
            include: { path, isDirectory in
                if isDirectory { return path != "prune.mlmodelc" }
                return path != "excluded.txt"
            },
            fetch: server.fetch
        )

        XCTAssertEqual(
            files,
            [
                RemoteFile(path: "keep.mlmodelc/coremldata.bin", size: 42),
                // Size defaults to -1 when the API omits it.
                RemoteFile(path: "keep.mlmodelc/weights/weight.bin", size: -1),
                RemoteFile(path: "root.json", size: 7),
            ])
        XCTAssertFalse(
            server.requested.contains(treeURL("prune.mlmodelc")),
            "a pruned directory must not be fetched at all")
    }

    // MARK: - Pagination

    func testFollowsLinkCursorAcrossPagesWithinADirectory() async throws {
        let server = PageServer()
        let cursorURL = treeURL() + "?cursor=abc123&limit=2"
        try server.addPage(
            url: treeURL(),
            items: [
                ["path": "a.bin", "type": "file", "size": 1],
                ["path": "sub", "type": "directory"],
            ],
            nextPage: cursorURL)
        try server.addPage(
            url: cursorURL,
            items: [["path": "z.bin", "type": "file", "size": 3]])
        try server.addPage(
            url: treeURL("sub"),
            items: [["path": "sub/b.bin", "type": "file", "size": 2]])

        let files = try await HFTreeLister.listTree(
            repoRemotePath: Self.repo, include: { _, _ in true }, fetch: server.fetch)

        XCTAssertEqual(
            files,
            [
                RemoteFile(path: "a.bin", size: 1),
                RemoteFile(path: "sub/b.bin", size: 2),
                RemoteFile(path: "z.bin", size: 3),
            ],
            "page-2 entries must be walked; pre-pagination listers silently dropped them")
        XCTAssertEqual(server.requested.last, cursorURL)
    }

    func testNextPageURLParsesLinkHeaderShapes() {
        func next(_ link: String?) -> String? {
            var headers: [String: String] = [:]
            if let link { headers["Link"] = link }
            let response = HTTPURLResponse(
                url: URL(string: "https://example.test")!, statusCode: 200,
                httpVersion: "HTTP/1.1", headerFields: headers)!
            return HFClient.nextPageURL(from: response)?.absoluteString
        }

        XCTAssertNil(next(nil))
        XCTAssertEqual(next("<https://hf.co/api/x?cursor=abc>; rel=\"next\""), "https://hf.co/api/x?cursor=abc")
        XCTAssertEqual(
            next("<https://hf.co/first>; rel=\"first\", <https://hf.co/n>; rel=\"next\""),
            "https://hf.co/n")
        XCTAssertNil(next("<https://hf.co/p>; rel=\"prev\""))
        // RFC 8288 shapes a naive split breaks on:
        XCTAssertEqual(
            next("<https://hf.co/x?cursor=a,b;c=d>; rel=\"next\""),
            "https://hf.co/x?cursor=a,b;c=d",
            "commas/semicolons are legal inside the <target>")
        XCTAssertEqual(next("<https://hf.co/n>; rel=next"), "https://hf.co/n", "unquoted rel token")
        XCTAssertEqual(
            next("<https://hf.co/n>; rel=\"next last\""), "https://hf.co/n", "multi-value rel")
        XCTAssertNil(next("<https://hf.co/n>; rel=\"nexterior\""), "no substring false-positives")
    }

    func testPaginationCursorCycleTerminates() async throws {
        let server = PageServer()
        // Server echoes the SAME page URL as its own next cursor.
        try server.addPage(
            url: treeURL(),
            items: [["path": "a.bin", "type": "file", "size": 1]],
            nextPage: treeURL())

        let files = try await HFTreeLister.listTree(
            repoRemotePath: Self.repo, include: { _, _ in true }, fetch: server.fetch)

        XCTAssertEqual(files, [RemoteFile(path: "a.bin", size: 1)])
        XCTAssertEqual(server.requested.count, 1, "a repeating cursor must not refetch forever")
    }

    // MARK: - Pagination through the real fetch path

    /// End-to-end: downloadRepo → configuration seam → HFTreeLister.fetch →
    /// URLSession → Link cursor. The unit tests above bypass the session via
    /// an injected Fetch; this pins the production wiring.
    func testDownloadRepoFollowsPaginationThroughRealFetchPath() async throws {
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lister-e2e-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workDir) }
        LinkedTreeStubURLProtocol.reset()

        let model = ModelNames.VAD.sileroVadFile
        let root = try ModelRegistry.apiModels(Repo.vad.remotePath, "tree/main").absoluteString
        let cursor = root + "?cursor=page2"
        let modelDir = try ModelRegistry.apiModels(Repo.vad.remotePath, "tree/main/\(model)")
            .absoluteString

        LinkedTreeStubURLProtocol.addJSONPage(
            url: root,
            items: [["path": model, "type": "directory"]],
            linkNext: cursor)
        LinkedTreeStubURLProtocol.addJSONPage(
            url: cursor,
            items: [["path": "config.json", "type": "file", "size": 10]])
        LinkedTreeStubURLProtocol.addJSONPage(
            url: modelDir,
            items: [["path": "\(model)/coremldata.bin", "type": "file", "size": 10]])
        LinkedTreeStubURLProtocol.fileBody = Data(String(repeating: "x", count: 10).utf8)

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [LinkedTreeStubURLProtocol.self]
        try await DownloadUtils.downloadRepo(.vad, to: workDir, configuration: config)

        let repoPath = workDir.appendingPathComponent(Repo.vad.folderName)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: repoPath.appendingPathComponent("config.json").path),
            "the page-2 file must arrive through the real session-backed fetch")
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: repoPath.appendingPathComponent("\(model)/coremldata.bin").path))
    }

    // MARK: - Typed errors

    func testRateLimitedPageThrowsTypedError() async throws {
        let server = PageServer()
        try server.addPage(url: treeURL(), items: [], status: 429)

        do {
            _ = try await HFTreeLister.listTree(
                repoRemotePath: Self.repo, include: { _, _ in true }, fetch: server.fetch)
            XCTFail("expected rateLimited")
        } catch HFDownload.DownloadError.rateLimited(let statusCode, _) {
            XCTAssertEqual(statusCode, 429)
        }
    }

    func testHTMLErrorPageThrowsTypedError() async throws {
        let server = PageServer()
        server.addRawPage(url: treeURL(), body: Data("<!DOCTYPE html><html>err</html>".utf8))

        do {
            _ = try await HFTreeLister.listTree(
                repoRemotePath: Self.repo, include: { _, _ in true }, fetch: server.fetch)
            XCTFail("expected htmlErrorResponse")
        } catch HFDownload.DownloadError.htmlErrorResponse {
            // expected
        }
    }

    func testMalformedJSONThrowsInvalidResponse() async throws {
        let server = PageServer()
        server.addRawPage(url: treeURL(), body: Data("{\"not\": \"an array\"}".utf8))

        do {
            _ = try await HFTreeLister.listTree(
                repoRemotePath: Self.repo, include: { _, _ in true }, fetch: server.fetch)
            XCTFail("expected invalidResponse")
        } catch HFDownload.DownloadError.invalidResponse {
            // expected
        }
    }
}

// MARK: - Link-capable URLProtocol stub

/// Like `TreeStubURLProtocol` but with per-URL responses that can carry a
/// `Link` pagination header — the piece the Wave 1 stub can't express, needed
/// to test pagination through the real session-backed fetch path.
final class LinkedTreeStubURLProtocol: URLProtocol {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var pages: [String: (data: Data, headers: [String: String])] = [:]
    nonisolated(unsafe) private static var _fileBody = Data()

    static var fileBody: Data {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _fileBody
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _fileBody = newValue
        }
    }

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        pages = [:]
        _fileBody = Data()
    }

    static func addJSONPage(url: String, items: [[String: Any]], linkNext: String? = nil) {
        guard let data = try? JSONSerialization.data(withJSONObject: items) else { return }
        var headers: [String: String] = [:]
        if let linkNext {
            headers["Link"] = "<\(linkNext)>; rel=\"next\""
        }
        lock.lock()
        defer { lock.unlock() }
        pages[url] = (data, headers)
    }

    private static func page(for url: String) -> (data: Data, headers: [String: String])? {
        lock.lock()
        defer { lock.unlock() }
        return pages[url]
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }

        let payload: Data
        var headers: [String: String] = [:]
        if let page = Self.page(for: url.absoluteString) {
            payload = page.data
            headers = page.headers
        } else if url.path.contains("/resolve/main/") {
            payload = Self.fileBody
        } else {
            payload = Data("[]".utf8)
        }
        headers["Content-Length"] = String(payload.count)

        let response = HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: payload)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
