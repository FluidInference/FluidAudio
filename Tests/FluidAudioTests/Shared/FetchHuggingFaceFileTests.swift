import Foundation
import XCTest
import os

@testable import FluidAudio

/// Wave 5 behavior tests (#765): `fetchFile` converges onto the
/// shared retry policy and artifact validation. These pin the DELIBERATE
/// behavior changes this wave introduces:
///   - permanent errors (404) fail fast instead of consuming the backoff budget
///   - 5xx retries, then succeeds
///   - HTML and empty 200 bodies throw instead of being returned as content
///   - `Retry-After` paces the backoff, and the envelope never escapes
final class FetchHuggingFaceFileTests: XCTestCase {

    private var stubConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FetchStubURLProtocol.self]
        return config
    }

    private let url = URL(string: "https://stub.test/repo/resolve/main/vocab.json")!

    override func setUp() {
        super.setUp()
        FetchStubURLProtocol.reset()
    }

    override func tearDown() {
        FetchStubURLProtocol.reset()
        super.tearDown()
    }

    private func fetch(maxAttempts: Int = 4) async throws -> Data {
        try await ModelHub.fetchFile(
            from: url, description: "vocab.json",
            maxAttempts: maxAttempts, minBackoff: 0.01,
            configuration: stubConfiguration)
    }

    func testSuccessReturnsBody() async throws {
        FetchStubURLProtocol.enqueue(status: 200, body: Data("{\"vocab\": 1}".utf8))
        let data = try await fetch()
        XCTAssertEqual(data, Data("{\"vocab\": 1}".utf8))
        XCTAssertEqual(FetchStubURLProtocol.requestCount, 1)
    }

    func testPermanent404FailsFastOnFirstAttempt() async {
        FetchStubURLProtocol.enqueue(status: 404, body: Data())
        do {
            _ = try await fetch()
            XCTFail("expected downloadFailed")
        } catch DownloadError.downloadFailed(let path, let underlying) {
            XCTAssertEqual(path, "vocab.json")
            XCTAssertEqual((underlying as NSError).code, 404)
        } catch {
            XCTFail("expected downloadFailed, got \(error)")
        }
        XCTAssertEqual(
            FetchStubURLProtocol.requestCount, 1,
            "Wave 5 delta: a hard 404 must not consume the backoff budget")
    }

    func testTransient5xxRetriesThenSucceeds() async throws {
        FetchStubURLProtocol.enqueue(status: 502, body: Data())
        FetchStubURLProtocol.enqueue(status: 200, body: Data("payload".utf8))
        let data = try await fetch()
        XCTAssertEqual(data, Data("payload".utf8))
        XCTAssertEqual(FetchStubURLProtocol.requestCount, 2)
    }

    func testHTMLBodyIsRejectedNotReturned() async {
        // Wave 5 delta: previously a 200 HTML error page was returned as-is
        // and could be cached as vocab.json (#748). invalidArtifact is
        // classified retryable, so with the queue exhausted the fetch fails.
        FetchStubURLProtocol.enqueue(
            status: 200, body: Data("<!DOCTYPE html><html>rate limited</html>".utf8))
        do {
            _ = try await fetch(maxAttempts: 1)
            XCTFail("expected invalidArtifact")
        } catch DownloadError.invalidArtifact(_, let reason) {
            XCTAssertTrue(reason.contains("HTML"), "got: \(reason)")
        } catch {
            XCTFail("expected invalidArtifact, got \(error)")
        }
    }

    func testEmptyBodyIsRejected() async {
        // Wave 5 delta: previously an empty 200 body was returned as Data().
        FetchStubURLProtocol.enqueue(status: 200, body: Data())
        do {
            _ = try await fetch(maxAttempts: 1)
            XCTFail("expected invalidArtifact")
        } catch DownloadError.invalidArtifact(_, let reason) {
            XCTAssertEqual(reason, "empty file")
        } catch {
            XCTFail("expected invalidArtifact, got \(error)")
        }
    }

    func testHTMLPageIsRetryableAndRecovers() async throws {
        // Pinned: invalidArtifact classifies as transient — an HTML body is
        // usually a passing rate-limit/proxy page, so the next attempt can
        // succeed. (A persistent page still burns the budget; accepted.)
        FetchStubURLProtocol.enqueue(
            status: 200, body: Data("<!DOCTYPE html><html>moment</html>".utf8))
        FetchStubURLProtocol.enqueue(status: 200, body: Data("real content".utf8))

        let data = try await fetch(maxAttempts: 2)
        XCTAssertEqual(data, Data("real content".utf8))
        XCTAssertEqual(FetchStubURLProtocol.requestCount, 2)
    }

    func testTextHTMLContentTypeIsRejectedWithoutSniffablePrefix() async {
        // A 200 error page whose body lacks a doctype/html prefix is still
        // caught via the Content-Type header (mirrors the file path's check).
        FetchStubURLProtocol.enqueue(
            status: 200, body: Data("<h1>502 Bad Gateway</h1>".utf8),
            headers: ["Content-Type": "text/html; charset=utf-8"])
        do {
            _ = try await fetch(maxAttempts: 1)
            XCTFail("expected invalidArtifact")
        } catch DownloadError.invalidArtifact(_, let reason) {
            XCTAssertTrue(reason.contains("text/html"), "got: \(reason)")
        } catch {
            XCTFail("expected invalidArtifact, got \(error)")
        }
    }

    func testRetryAfterPacesBackoffAndEnvelopeNeverEscapes() async {
        // 429 with a 0.3s Retry-After (>> the 0.01s minBackoff), then success.
        FetchStubURLProtocol.enqueue(
            status: 429, body: Data(), headers: ["Retry-After": "0.3"])
        FetchStubURLProtocol.enqueue(status: 200, body: Data("ok".utf8))

        let start = Date()
        do {
            let data = try await fetch()
            XCTAssertEqual(data, Data("ok".utf8))
        } catch {
            XCTFail("expected success after paced retry, got \(error)")
        }
        XCTAssertGreaterThanOrEqual(
            Date().timeIntervalSince(start), 0.3,
            "backoff must honor the server's Retry-After over the 0.01s exponential floor")
    }

    func testExhaustedRateLimitSurfacesPlainRateLimitedError() async {
        // Every attempt 429s; the surfaced error must be the plain typed
        // rateLimited — never the internal RetryAfterHint envelope.
        for _ in 0..<2 {
            FetchStubURLProtocol.enqueue(
                status: 429, body: Data(), headers: ["Retry-After": "0.05"])
        }
        do {
            _ = try await fetch(maxAttempts: 2)
            XCTFail("expected rateLimited")
        } catch DownloadError.rateLimited(let statusCode, let message) {
            XCTAssertEqual(statusCode, 429)
            XCTAssertTrue(message.contains("fetching vocab.json"), "got: \(message)")
        } catch {
            XCTFail("the RetryAfterHint envelope escaped: \(error)")
        }
        XCTAssertEqual(FetchStubURLProtocol.requestCount, 2)
    }
}

// MARK: - Simple FIFO data-response stub

final class FetchStubURLProtocol: URLProtocol {

    private struct Response {
        let status: Int
        let body: Data
        let headers: [String: String]
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var queue: [Response] = []
    nonisolated(unsafe) private static var _requestCount = 0

    static var requestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _requestCount
    }

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        queue = []
        _requestCount = 0
    }

    static func enqueue(status: Int, body: Data, headers: [String: String] = [:]) {
        lock.lock()
        defer { lock.unlock() }
        queue.append(Response(status: status, body: body, headers: headers))
    }

    private static func dequeue() -> Response? {
        lock.lock()
        defer { lock.unlock() }
        _requestCount += 1
        return queue.isEmpty ? nil : queue.removeFirst()
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let next = Self.dequeue() else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        var headers = next.headers
        headers["Content-Length"] = String(next.body.count)
        let response = HTTPURLResponse(
            url: url, statusCode: next.status, httpVersion: "HTTP/1.1",
            headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: next.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
