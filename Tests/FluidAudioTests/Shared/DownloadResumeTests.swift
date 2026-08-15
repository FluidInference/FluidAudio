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
        let url = try await FileDownloader.download(
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
                to: FileDownloader.resumeValidatorURL(for: partialURL()),
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
                atPath: FileDownloader.resumeValidatorURL(for: partialURL()).path))
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
        // Attempt 2: a range-capable server. Whether the head bytes survived
        // the drop is racy at the URLProtocol layer (data delivery may be
        // discarded on failure), so the stub answers both shapes: 206 tail if
        // the client resumed, 200 full if it restarted. Either way the final
        // file must be complete and unspliced. The deterministic
        // resume-append path is covered by the seeded-partial tests below.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [], rangeAwareBody: Self.fullBody))

        let body = try await download()

        XCTAssertEqual(body, Self.fullBody, "post-drop download must produce the complete file")
        let requests = ResumeStubURLProtocol.recordedRequests()
        XCTAssertEqual(requests.count, 2)
        XCTAssertNil(requests[0].value(forHTTPHeaderField: "Range"))
        if let range = requests[1].value(forHTTPHeaderField: "Range") {
            // Bytes reached disk before the drop → a true resume from whatever
            // offset survived, guarded by the validator captured on attempt 1.
            XCTAssertTrue(range.hasPrefix("bytes="), "got: \(range)")
            XCTAssertEqual(requests[1].value(forHTTPHeaderField: "If-Range"), "\"v1\"")
        }
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

    // MARK: - Stall watchdog (#810)

    /// A frozen mid-transfer connection is cancelled by the watchdog and the
    /// retry resumes to completion — instead of hanging on the idle `timeout`.
    func testStalledTransferIsCancelledAndRetried() async throws {
        // Attempt 1: 200 + ETag, delivers a small head, then freezes. The head
        // is far below the 1 MiB threshold so the watchdog trips on its first
        // (sub-second) window.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200,
                headers: ["ETag": "\"v1\"", "Content-Length": String(Self.fullBody.count)],
                chunks: [Data(Self.fullBody.prefix(10))],
                stall: true))
        // Attempt 2: a range-capable server completes the file.
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [], rangeAwareBody: Self.fullBody))

        let fastWatchdog = DownloadConfig(minStallBytes: 1 << 20, stallWindow: 0.2)
        let url = try await FileDownloader.download(
            request: request,
            path: "model.bin",
            expectedSize: Self.fullBody.count,
            partialFileURL: partialURL(),
            onProgress: nil,
            maxAttempts: 3,
            minBackoff: 0.01,
            config: fastWatchdog,
            configuration: stubConfiguration
        )

        XCTAssertEqual(try Data(contentsOf: url), Self.fullBody)
        XCTAssertEqual(
            ResumeStubURLProtocol.recordedRequests().count, 2,
            "the stalled attempt must be cancelled and retried")
    }

    /// A disabled watchdog (`minStallBytes == 0`) must not cancel a healthy,
    /// progressing download.
    func testDisabledWatchdogDoesNotCancelHealthyDownload() async throws {
        ResumeStubURLProtocol.enqueue(
            .init(status: 200, headers: ["ETag": "\"v1\""], chunks: [Self.fullBody]))

        let noWatchdog = DownloadConfig(minStallBytes: 0, stallWindow: 0)
        let url = try await FileDownloader.download(
            request: request,
            path: "model.bin",
            expectedSize: Self.fullBody.count,
            partialFileURL: partialURL(),
            onProgress: nil,
            maxAttempts: 1,
            minBackoff: 0.01,
            config: noWatchdog,
            configuration: stubConfiguration
        )

        XCTAssertEqual(try Data(contentsOf: url), Self.fullBody)
        XCTAssertEqual(ResumeStubURLProtocol.recordedRequests().count, 1)
    }

    func testStalledErrorIsClassifiedRetryable() {
        XCTAssertTrue(RetryPolicy.isRetryable(DownloadError.stalled(path: "model.bin", window: 120)))
        XCTAssertFalse(RetryPolicy.isCancellation(DownloadError.stalled(path: "model.bin", window: 120)))
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
        _ = try await FileDownloader.download(
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
        /// Deliver the chunks, then go silent forever — never finish, never
        /// fail. Simulates a frozen CDN connection so the stall watchdog fires.
        var stall: Bool = false
        /// When set, respond like a real range-capable server instead of using
        /// `status`/`chunks`: 206 + the slice from the request's `Range` offset
        /// when the header is present, else 200 + the full body. Lets tests
        /// stay deterministic when a prior scripted drop races URLProtocol
        /// data delivery (the partial may or may not contain the bytes).
        var rangeAwareBody: Data? = nil
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

        if let body = script.rangeAwareBody {
            respondRangeAware(url: url, body: body, headers: script.headers)
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
                failAfterFlush()
                return
            }
            client?.urlProtocol(self, didLoad: chunk)
        }
        // Frozen connection: bytes stop flowing but the request stays open, so
        // only the stall watchdog (not the idle timeout) can end it.
        if script.stall { return }
        if let failAfter = script.failAfterChunks, failAfter >= script.chunks.count {
            failAfterFlush()
            return
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    /// Deliver the scripted connection drop only after already-loaded chunks
    /// have had time to reach the data task's delegate. Failing immediately
    /// after `didLoad` races the URL loading system's internal buffer and can
    /// drop the delivered bytes, which breaks resume tests that rely on the
    /// partial surviving the drop (seen flaky on CI runners).
    private func failAfterFlush() {
        let client = self.client
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) { [self] in
            client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
        }
    }

    /// Behave like a real range-capable server for `body`: 206 + slice when
    /// the request carries `Range: bytes=N-`, else 200 + the full body.
    private func respondRangeAware(url: URL, body: Data, headers: [String: String]) {
        var offset = 0
        if let range = request.value(forHTTPHeaderField: "Range"),
            range.hasPrefix("bytes="), range.hasSuffix("-"),
            let start = Int(range.dropFirst("bytes=".count).dropLast()),
            start > 0, start < body.count
        {
            offset = start
        }

        var responseHeaders = headers
        let payload = body.suffix(from: offset)
        responseHeaders["Content-Length"] = String(payload.count)
        let status: Int
        if offset > 0 {
            status = 206
            responseHeaders["Content-Range"] = "bytes \(offset)-\(body.count - 1)/\(body.count)"
        } else {
            status = 200
        }
        guard
            let response = HTTPURLResponse(
                url: url, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: responseHeaders)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(payload))
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
