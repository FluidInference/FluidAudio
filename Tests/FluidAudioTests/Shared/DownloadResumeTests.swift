import Foundation
import XCTest
import os

@testable import FluidAudio

/// Tests for HTTP Range resume in `downloadFileWithRetry` (#757): partial
/// bytes survive mid-transfer drops, retries send `Range`/`If-Range`, a 200
/// after a range request replaces (never splices) the partial file, and a
/// fully-downloaded partial from a previous run is reused without network.
final class DownloadResumeTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("DownloadResumeTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        ResumeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        ResumeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    // MARK: - Helpers

    private var stubConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ResumeStubURLProtocol.self]
        return config
    }

    private var request: URLRequest {
        URLRequest(url: URL(string: "https://stub.test/repo/resolve/main/model.bin")!)
    }

    private func partialURL(_ name: String = "model.bin") -> URL {
        workDir.appendingPathComponent(name).appendingPathExtension("partial")
    }

    /// Body content that passes artifact validation (not empty, not HTML).
    private static let fullBody = Data("BINARYCONTENT-0123456789-ABCDEFGHIJ".utf8)

    private func download(
        expectedSize: Int = fullBody.count,
        partial: URL? = nil,
        maxAttempts: Int = 3
    ) async throws -> Data {
        let url = try await DownloadUtils.downloadFileWithRetry(
            request: request,
            path: "model.bin",
            expectedSize: expectedSize,
            partialFileURL: partial ?? partialURL(),
            onProgress: nil,
            maxAttempts: maxAttempts,
            minBackoff: 0.01,
            configuration: stubConfiguration
        )
        return try Data(contentsOf: url)
    }

    private func seedPartial(_ data: Data, validator: String?) throws {
        try data.write(to: partialURL())
        if let validator {
            try validator.write(
                to: DownloadUtils.resumeValidatorURL(for: partialURL()),
                atomically: true, encoding: .utf8)
        }
    }

    // MARK: - Tests

    func testFreshDownloadWritesFullBody() async throws {
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertNil(requests[0].value(forHTTPHeaderField: "Range"))
        // Validator sidecar is cleaned up after a completed download.
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: DownloadUtils.resumeValidatorURL(for: partialURL()).path))
    }

    func testMidStreamDropResumesWithRangeAndAppends() async throws {
        let splitAt = 10
        let head = Self.fullBody.prefix(splitAt)
        let tail = Self.fullBody.dropFirst(splitAt)

        // Attempt 1: 200 with ETag, drops after `head` bytes.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200, headers: ["ETag": "\"v1\""],
                chunks: [Data(head), Data(tail)], failAfterChunks: 1))
        // Attempt 2: server honors the range with a 206 for the remainder.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "ETag": "\"v1\"",
                    "Content-Range": "bytes \(splitAt)-\(Self.fullBody.count - 1)/\(Self.fullBody.count)",
                ],
                chunks: [Data(tail)]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody, "resumed file must equal head + tail with no gap")
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertNil(requests[0].value(forHTTPHeaderField: "Range"))
        XCTAssertEqual(requests[1].value(forHTTPHeaderField: "Range"), "bytes=\(splitAt)-")
        XCTAssertEqual(requests[1].value(forHTTPHeaderField: "If-Range"), "\"v1\"")
    }

    func testServerIgnoringRangeReplacesPartialInsteadOfSplicing() async throws {
        // Stale partial from a different file version.
        try seedPartial(Data("STALEBYTES".utf8), validator: "\"v0\"")
        // Server replies 200 (range ignored / file changed) with the new body.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody, "a 200 must replace the partial, never splice")
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 1)
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "Range"), "bytes=10-")
        XCTAssertEqual(requests[0].value(forHTTPHeaderField: "If-Range"), "\"v0\"")
    }

    func testRangeNotSatisfiableClearsPartialAndRetriesFresh() async throws {
        try seedPartial(Data("BOGUSPARTIALBYTES".utf8), validator: "\"v0\"")
        ResumeStubURLProtocol.enqueue(.init(status: 416, headers: [:], chunks: []))
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertNotNil(requests[0].value(forHTTPHeaderField: "Range"))
        XCTAssertNil(
            requests[1].value(forHTTPHeaderField: "Range"),
            "after a 416 the partial is cleared and the retry starts from byte 0")
    }

    func testNoValidatorMeansNoResume() async throws {
        let splitAt = 10
        // Attempt 1: 200 with only a weak ETag (unusable for If-Range), drops mid-body.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200, headers: ["ETag": "W/\"weak\""],
                chunks: [Data(Self.fullBody.prefix(splitAt)), Data(Self.fullBody.dropFirst(splitAt))],
                failAfterChunks: 1))
        // Attempt 2 must be a fresh full download.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: [:], chunks: [Self.fullBody]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertNil(
            requests[1].value(forHTTPHeaderField: "Range"),
            "without a strong validator the partial must be discarded, not resumed")
    }

    func testErrorBodyDoesNotClobberPartialBytes() async throws {
        let splitAt = 10
        try seedPartial(Data(Self.fullBody.prefix(splitAt)), validator: "\"v1\"")
        // Attempt 1: 503 with an HTML error page body — must not touch the partial.
        ResumeStubURLProtocol.enqueue(
            .init(status: 503, headers: [:], chunks: [Data("<html>err</html>".utf8)]))
        // Attempt 2: resume succeeds.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes \(splitAt)-\(Self.fullBody.count - 1)/\(Self.fullBody.count)"
                ],
                chunks: [Data(Self.fullBody.dropFirst(splitAt))]))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertEqual(requests[1].value(forHTTPHeaderField: "Range"), "bytes=\(splitAt)-")
    }

    func testCompletedPartialFromPreviousRunIsReusedWithoutNetwork() async throws {
        try seedPartial(Self.fullBody, validator: "\"v1\"")
        // No scripts enqueued — any network request would fail the test.

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody)
        XCTAssertTrue(
            ResumeStubURLProtocol.recordedRequests().isEmpty,
            "a complete, valid partial must be reused without hitting the network")
    }

    func testProgressIncludesResumedOffsetSoItNeverDips() async throws {
        let splitAt = 10
        try seedPartial(Data(Self.fullBody.prefix(splitAt)), validator: "\"v1\"")
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 206,
                headers: [
                    "Content-Range": "bytes \(splitAt)-\(Self.fullBody.count - 1)/\(Self.fullBody.count)"
                ],
                chunks: [Data(Self.fullBody.dropFirst(splitAt))]))

        let recorded = ProgressRecorder()
        _ = try await DownloadUtils.downloadFileWithRetry(
            request: request,
            path: "model.bin",
            expectedSize: Self.fullBody.count,
            partialFileURL: partialURL(),
            onProgress: { written, expected in recorded.append((written, expected)) },
            maxAttempts: 1,
            minBackoff: 0.01,
            configuration: stubConfiguration
        )

        let events = recorded.snapshot()
        XCTAssertFalse(events.isEmpty)
        XCTAssertGreaterThan(
            events[0].written, Int64(splitAt) - 1,
            "first progress callback must already include the resumed offset")
        XCTAssertEqual(events.last?.written, Int64(Self.fullBody.count))
        XCTAssertEqual(events.last?.expected, Int64(Self.fullBody.count))
    }
}

// MARK: - Scripted URLProtocol stub

/// Serves a FIFO queue of scripted responses and records every request,
/// including mid-body connection drops (`failAfterChunks`).
final class ResumeStubURLProtocol: URLProtocol {

    struct Script {
        let status: Int
        let headers: [String: String]
        let chunks: [Data]
        /// Deliver this many chunks, then fail with `.networkConnectionLost`.
        var failAfterChunks: Int? = nil
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var scripts: [Script] = []
    nonisolated(unsafe) private static var requests: [URLRequest] = []

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        scripts = []
        requests = []
    }

    static func enqueue(_ script: Script) {
        lock.lock()
        defer { lock.unlock() }
        scripts.append(script)
    }

    static func recordedRequests() -> [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return requests
    }

    private static func dequeue(recording request: URLRequest) -> Script? {
        lock.lock()
        defer { lock.unlock() }
        requests.append(request)
        return scripts.isEmpty ? nil : scripts.removeFirst()
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let script = Self.dequeue(recording: request), let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }

        var headers = script.headers
        let bodyBytes = script.chunks.reduce(0) { $0 + $1.count }
        if headers["Content-Length"] == nil {
            headers["Content-Length"] = String(bodyBytes)
        }
        guard
            let response = HTTPURLResponse(
                url: url, statusCode: script.status, httpVersion: "HTTP/1.1",
                headerFields: headers)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        for (index, chunk) in script.chunks.enumerated() {
            if let failAfter = script.failAfterChunks, index >= failAfter {
                client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
                return
            }
            client?.urlProtocol(self, didLoad: chunk)
        }
        if let failAfter = script.failAfterChunks, failAfter >= script.chunks.count {
            client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
            return
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Lock-guarded accumulator for progress events (tests only).
private final class ProgressRecorder: Sendable {
    private let values = OSAllocatedUnfairLock<[(written: Int64, expected: Int64)]>(initialState: [])

    func append(_ value: (written: Int64, expected: Int64)) {
        values.withLock { $0.append(value) }
    }

    func snapshot() -> [(written: Int64, expected: Int64)] {
        values.withLock { $0 }
    }
}
