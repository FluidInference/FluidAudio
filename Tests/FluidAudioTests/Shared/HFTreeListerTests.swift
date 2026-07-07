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
            repoRemotePath: Self.repo, fetch: server.fetch)

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

    func testNextPageURLParsesMultiEntryLinkHeaders() {
        func response(link: String?) -> HTTPURLResponse {
            var headers: [String: String] = [:]
            if let link { headers["Link"] = link }
            return HTTPURLResponse(
                url: URL(string: "https://example.test")!, statusCode: 200,
                httpVersion: "HTTP/1.1", headerFields: headers)!
        }

        XCTAssertNil(HFTreeLister.nextPageURL(from: response(link: nil)))
        XCTAssertEqual(
            HFTreeLister.nextPageURL(
                from: response(link: "<https://hf.co/api/x?cursor=abc>; rel=\"next\""))?
                .absoluteString,
            "https://hf.co/api/x?cursor=abc")
        XCTAssertEqual(
            HFTreeLister.nextPageURL(
                from: response(
                    link: "<https://hf.co/first>; rel=\"first\", <https://hf.co/n>; rel=\"next\""))?
                .absoluteString,
            "https://hf.co/n")
        XCTAssertNil(
            HFTreeLister.nextPageURL(from: response(link: "<https://hf.co/p>; rel=\"prev\"")))
    }

    // MARK: - Typed errors

    func testRateLimitedPageThrowsTypedError() async throws {
        let server = PageServer()
        try server.addPage(url: treeURL(), items: [], status: 429)

        do {
            _ = try await HFTreeLister.listTree(repoRemotePath: Self.repo, fetch: server.fetch)
            XCTFail("expected rateLimited")
        } catch HFDownload.DownloadError.rateLimited(let statusCode, _) {
            XCTAssertEqual(statusCode, 429)
        }
    }

    func testHTMLErrorPageThrowsTypedError() async throws {
        let server = PageServer()
        server.addRawPage(url: treeURL(), body: Data("<!DOCTYPE html><html>err</html>".utf8))

        do {
            _ = try await HFTreeLister.listTree(repoRemotePath: Self.repo, fetch: server.fetch)
            XCTFail("expected htmlErrorResponse")
        } catch HFDownload.DownloadError.htmlErrorResponse {
            // expected
        }
    }

    func testMalformedJSONThrowsInvalidResponse() async throws {
        let server = PageServer()
        server.addRawPage(url: treeURL(), body: Data("{\"not\": \"an array\"}".utf8))

        do {
            _ = try await HFTreeLister.listTree(repoRemotePath: Self.repo, fetch: server.fetch)
            XCTFail("expected invalidResponse")
        } catch HFDownload.DownloadError.invalidResponse {
            // expected
        }
    }
}
