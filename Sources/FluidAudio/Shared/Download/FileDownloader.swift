import Foundation
import os

/// The one per-file download unit for the download stack (#765 Wave 4):
/// bounded retry (RetryPolicy) around the streaming byte-range-resume
/// transport (#757), artifact validation, and atomic placement into the
/// cache. Used by both `downloadRepo` and `downloadSubdirectory`.
enum FileDownloader {

    /// Historical category kept so existing log predicates keep capturing the
    /// whole download trail across the #765 refactor; Wave 6 renames it
    /// deliberately alongside the API cutover.
    private static let logger = AppLogger(category: "DownloadUtils")

    /// What `ensure(file:from:at:)` did for a file.
    enum Outcome {
        /// Destination already existed; nothing fetched.
        case alreadyPresent
        /// Zero-byte file created locally (HF serves 500s for empty files).
        case createdEmpty
        /// Fetched from the network, validated, and moved into place.
        case downloaded
    }

    /// Ensure `file` from `repoRemotePath` exists at `destination`: skip when
    /// present, create empties locally, otherwise download with retry +
    /// byte-range resume (via `<destination>.partial`), validate, and move
    /// atomically into place.
    ///
    /// - Parameter recoveringBlockedPaths: When true, a regular file blocking
    ///   a parent-directory component is DELETED and replaced (the repo-cache
    ///   corrupt-recovery behavior). When false, a blocked path fails loudly
    ///   with the thrown filesystem error — callers outside the managed model
    ///   cache must not silently destroy user files.
    @discardableResult
    static func ensure(
        file: RemoteFile,
        from repoRemotePath: String,
        at destination: URL,
        recoveringBlockedPaths: Bool,
        configuration: URLSessionConfiguration? = nil,
        onBytes: (@Sendable (Int64, Int64) -> Void)? = nil
    ) async throws -> Outcome {
        if FileManager.default.fileExists(atPath: destination.path) {
            return .alreadyPresent
        }

        let parentDir = destination.deletingLastPathComponent()
        if recoveringBlockedPaths {
            // Create parent directory, removing any conflicting files in the path.
            try ModelCache.createDirectoryRobustly(at: parentDir)
        } else {
            try FileManager.default.createDirectory(
                at: parentDir, withIntermediateDirectories: true)
        }

        // HuggingFace returns 500 for 0-byte files — create empty file locally.
        if file.size == 0 {
            FileManager.default.createFile(atPath: destination.path, contents: Data())
            return .createdEmpty
        }

        let encodedPath =
            file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
        let fileURL = try ModelRegistry.resolveModel(repoRemotePath, encodedPath)
        let request = HFClient.authorizedRequest(url: fileURL)

        let tempURL = try await download(
            request: request,
            path: file.path,
            expectedSize: file.size,
            partialFileURL: destination.appendingPathExtension("partial"),
            onProgress: onBytes,
            configuration: configuration
        )

        if FileManager.default.fileExists(atPath: destination.path) {
            try? FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
        return .downloaded
    }

    /// `response` is nil when re-validating a fully-downloaded partial file from
    /// a previous run (no live response to check Content-Type against).
    static func validateDownloadedArtifact(
        at tempURL: URL,
        response: HTTPURLResponse?,
        path: String,
        expectedSize: Int
    ) throws {
        if let contentType = response?.value(forHTTPHeaderField: "Content-Type")?.lowercased(),
            contentType.contains("text/html")
        {
            throw HFDownload.DownloadError.invalidArtifact(
                path: path, reason: "server returned Content-Type: \(contentType)")
        }

        let actualSize =
            ((try? FileManager.default.attributesOfItem(atPath: tempURL.path))?[.size] as? Int) ?? 0
        if actualSize == 0 {
            throw HFDownload.DownloadError.invalidArtifact(path: path, reason: "empty file")
        }

        if let handle = try? FileHandle(forReadingFrom: tempURL) {
            defer { try? handle.close() }
            if HFClient.looksLikeHTML(handle.readData(ofLength: 512)) {
                throw HFDownload.DownloadError.invalidArtifact(
                    path: path, reason: "response body begins with HTML markup")
            }
        }

        // HuggingFace reports the exact (LFS-resolved) object size; a short body is truncation.
        if expectedSize > 0 && actualSize != expectedSize {
            throw HFDownload.DownloadError.invalidArtifact(
                path: path,
                reason: "size mismatch (expected \(expectedSize) bytes, got \(actualSize))")
        }
    }

    // MARK: - Streaming download to a persistent file

    /// Stream a download directly into `destination` so received bytes survive
    /// a mid-transfer drop (#757). Pure transport: the caller sets resume
    /// headers and validates the HTTP status. Body bytes are written only for
    /// 2xx — appended after `resumeOffset` on 206, from byte 0 otherwise.
    ///
    /// - Parameters:
    ///   - resumeOffset: Byte count a 206 appends after; also the base for
    ///     progress callbacks so resumed progress never dips.
    ///   - configuration: Session configuration override for tests.
    ///   - onProgress: `(totalBytesWritten, totalBytesExpected)`, both
    ///     including `resumeOffset`. Delegate-driven byte progress (#756).
    ///   - onResponse: Fires before any body byte is written — the resume path
    ///     persists the validator here so it survives a drop mid-body.
    /// Coverage flows through the `DownloadUtils.downloadFileWithRetry`
    /// forward (DownloadResumeTests); no test drives this directly.
    static func streamDownload(
        request: URLRequest,
        to destination: URL,
        resumeOffset: Int64 = 0,
        configuration: URLSessionConfiguration? = nil,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil,
        onResponse: (@Sendable (HTTPURLResponse) -> Void)? = nil
    ) async throws -> HTTPURLResponse {
        let delegate = StreamingDownloadDelegate(
            destination: destination,
            resumeOffset: resumeOffset,
            onProgress: onProgress,
            onResponse: onResponse
        )
        // Dedicated session with delegate — one per download to avoid cross-talk.
        // DownloadUtils still owns the shared session (public API); ownership
        // moves to the new surface in the Wave 6 cutover.
        let session = URLSession(
            configuration: configuration ?? DownloadUtils.sharedSession.configuration,
            delegate: delegate,
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let task = session.dataTask(with: request)
                delegate.attach(continuation: continuation, task: task)
                task.resume()
            }
        } onCancel: {
            delegate.cancel()
        }
    }

    // MARK: - Per-file download with bounded retry and byte-range resume

    /// Sidecar storing the HTTP validator (ETag/Last-Modified) a partial file
    /// came from; written at response time so it survives a drop mid-body.
    static func resumeValidatorURL(for partialFileURL: URL) -> URL {
        partialFileURL.appendingPathExtension("etag")
    }

    private static func fileSize(at url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64) ?? 0
    }

    /// Remove a partial download and its validator sidecar.
    private static func clearPartialDownload(_ partialFileURL: URL) {
        try? FileManager.default.removeItem(at: partialFileURL)
        try? FileManager.default.removeItem(at: resumeValidatorURL(for: partialFileURL))
    }

    /// Resume validator for `If-Range`: a strong ETag, else Last-Modified.
    /// Weak ETags are unusable for `If-Range` (RFC 9110 §13.1.5); nil means
    /// resume is unsafe and the file restarts from 0.
    private static func resumeValidator(from response: HTTPURLResponse) -> String? {
        if let etag = response.value(forHTTPHeaderField: "ETag"), !etag.hasPrefix("W/") {
            return etag
        }
        return response.value(forHTTPHeaderField: "Last-Modified")
    }

    /// Download a single repo file with bounded exponential-backoff retry on
    /// transient failures (timeout/TLS/connectivity, HTTP 429/503/5xx; 4xx and
    /// non-network errors fail fast — see `RetryPolicy.isRetryable`).
    ///
    /// Resume (#757): bytes stream into `partialFileURL`, so a retry — or a new
    /// process — continues with `Range`/`If-Range` instead of restarting from
    /// byte 0. A 200 (range ignored / remote changed) restarts the file rather
    /// than splicing mixed-version bytes.
    ///
    /// - Parameters:
    ///   - partialFileURL: Where in-flight bytes live (e.g. `<dest>.partial`).
    ///     Pass nil for a unique temp location (in-run resume only).
    ///   - configuration: Session configuration override for tests.
    /// - Returns: The URL of a validated (2xx) fully-written download; the
    ///   caller moves it into the cache.
    /// Pinned via the `DownloadUtils.downloadFileWithRetry` forward
    /// (DownloadResumeTests drives resume/splice/416 behavior through it).
    static func download(
        request: URLRequest,
        path: String,
        expectedSize: Int,
        partialFileURL: URL? = nil,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0,
        configuration: URLSessionConfiguration? = nil
    ) async throws -> URL {
        let defaultPartialURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("partial")
        let partialURL = partialFileURL ?? defaultPartialURL
        let validatorURL = resumeValidatorURL(for: partialURL)

        return try await RetryPolicy.withRetry(
            label: path, maxAttempts: maxAttempts, minBackoff: minBackoff, logger: logger
        ) { attempt in
            // Re-check per attempt, matching HFTreeLister.fetch: flipping
            // enforceOffline mid-operation stops the walk at the next request.
            guard !DownloadUtils.enforceOffline else {
                throw DownloadUtils.OfflineError.networkDisabled(operation: "download(\(path))")
            }
            var attemptRequest = request
            var resumeOffset: Int64 = 0

            let partialSize = fileSize(at: partialURL)
            let validator = (try? String(contentsOf: validatorURL, encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if partialSize > 0, let validator, !validator.isEmpty {
                if expectedSize > 0 && partialSize >= Int64(expectedSize) {
                    // A previous run finished the body but didn't get to move
                    // the file. Validate and reuse it instead of re-fetching.
                    if (try? validateDownloadedArtifact(
                        at: partialURL, response: nil, path: path, expectedSize: expectedSize))
                        != nil
                    {
                        try? FileManager.default.removeItem(at: validatorURL)
                        return partialURL
                    }
                    clearPartialDownload(partialURL)
                } else {
                    attemptRequest.setValue("bytes=\(partialSize)-", forHTTPHeaderField: "Range")
                    attemptRequest.setValue(validator, forHTTPHeaderField: "If-Range")
                    resumeOffset = partialSize
                    logger.info(
                        "Resuming \(path) from byte \(partialSize) (attempt \(attempt))")
                }
            } else if partialSize > 0 {
                // Partial bytes with no validator can't be safely resumed.
                clearPartialDownload(partialURL)
            }

            let httpResponse = try await streamDownload(
                request: attemptRequest,
                to: partialURL,
                resumeOffset: resumeOffset,
                configuration: configuration,
                onProgress: onProgress,
                onResponse: { response in
                    // Persist the validator as soon as the fresh body starts, so
                    // a drop mid-body still leaves it for the next attempt. A 206
                    // keeps the validator of the response it is extending.
                    guard response.statusCode != 206, (200..<300).contains(response.statusCode)
                    else { return }
                    if let validator = resumeValidator(from: response) {
                        try? validator.write(to: validatorURL, atomically: true, encoding: .utf8)
                    } else {
                        try? FileManager.default.removeItem(at: validatorURL)
                    }
                }
            )

            try HFClient.checkRateLimit(httpResponse, context: "downloading \(path)")

            if httpResponse.statusCode == 416 {
                // Range not satisfiable — the partial is bogus (or the remote
                // shrank). Restart from 0 on the next attempt.
                clearPartialDownload(partialURL)
                throw HFDownload.DownloadError.invalidArtifact(
                    path: path, reason: "server rejected resume range (416)")
            }

            guard (200..<300).contains(httpResponse.statusCode) else {
                throw HFDownload.DownloadError.downloadFailed(
                    path: path,
                    underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                )
            }

            // Validate before the caller moves the file into the cache.
            do {
                try validateDownloadedArtifact(
                    at: partialURL, response: httpResponse, path: path,
                    expectedSize: expectedSize)
            } catch {
                // A completed-but-invalid body must not be resumed from.
                clearPartialDownload(partialURL)
                throw error
            }

            try? FileManager.default.removeItem(at: validatorURL)
            return partialURL
        }
    }
}

// MARK: - URLSession download delegate for byte-level progress

/// `URLSessionDownloadTask` → async/await bridge that forwards byte progress (#756).
/// Streams response bytes straight into a destination file so partial content
/// survives connection drops (#757). Appends after `resumeOffset` on HTTP 206;
/// truncates and writes from 0 on any other 2xx; discards the body otherwise.
private final class StreamingDownloadDelegate: NSObject, URLSessionDataDelegate, Sendable {
    private struct State {
        var continuation: CheckedContinuation<HTTPURLResponse, Error>?
        var task: URLSessionDataTask?
        var handle: FileHandle?
        var bytesWritten: Int64 = 0
        var expectedTotal: Int64 = -1
        var writeError: Error?
        var finished = false
    }

    private let destination: URL
    private let resumeOffset: Int64
    private let onProgress: (@Sendable (Int64, Int64) -> Void)?
    private let onResponse: (@Sendable (HTTPURLResponse) -> Void)?
    // Holds the non-Sendable continuation and file handle; the lock (not
    // `@unchecked`) makes the delegate Sendable as URLSession requires.
    private let state = OSAllocatedUnfairLock<State>(uncheckedState: State())

    init(
        destination: URL,
        resumeOffset: Int64,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        onResponse: (@Sendable (HTTPURLResponse) -> Void)?
    ) {
        self.destination = destination
        self.resumeOffset = resumeOffset
        self.onProgress = onProgress
        self.onResponse = onResponse
    }

    /// Register the continuation and task before the task starts.
    func attach(
        continuation: CheckedContinuation<HTTPURLResponse, Error>,
        task: URLSessionDataTask
    ) {
        state.withLockUnchecked {
            $0.continuation = continuation
            $0.task = task
        }
    }

    func cancel() {
        state.withLockUnchecked { $0.task }?.cancel()
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        onResponse?(http)

        // Only 2xx bodies are file content worth keeping; error bodies are
        // discarded so a 503 page can never overwrite resumable bytes.
        if (200..<300).contains(http.statusCode) {
            let appending = http.statusCode == 206
            do {
                if !FileManager.default.fileExists(atPath: destination.path) {
                    FileManager.default.createFile(atPath: destination.path, contents: nil)
                }
                let handle = try FileHandle(forWritingTo: destination)
                if appending {
                    try handle.seekToEnd()
                } else {
                    try handle.truncate(atOffset: 0)
                }
                let base = appending ? resumeOffset : 0
                // Content-Range carries the full object size on a 206; a plain
                // 200 reports the full size via expectedContentLength.
                let expectedTotal: Int64
                if let contentRange = http.value(forHTTPHeaderField: "Content-Range"),
                    let totalPart = contentRange.split(separator: "/").last,
                    let total = Int64(totalPart)
                {
                    expectedTotal = total
                } else if http.expectedContentLength >= 0 {
                    expectedTotal = base + http.expectedContentLength
                } else {
                    expectedTotal = -1
                }
                state.withLockUnchecked {
                    $0.handle = handle
                    $0.bytesWritten = base
                    $0.expectedTotal = expectedTotal
                }
            } catch {
                state.withLockUnchecked { $0.writeError = error }
                completionHandler(.cancel)
                return
            }
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let progress: (written: Int64, expected: Int64)? = state.withLockUnchecked { st in
            guard let handle = st.handle, st.writeError == nil else { return nil }
            do {
                try handle.write(contentsOf: data)
                st.bytesWritten += Int64(data.count)
                return (st.bytesWritten, st.expectedTotal)
            } catch {
                st.writeError = error
                return nil
            }
        }
        if let progress {
            onProgress?(progress.written, progress.expected)
        } else if state.withLockUnchecked({ $0.writeError }) != nil {
            cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let response = task.response as? HTTPURLResponse
        // Extract the resume action under the lock, run it after releasing.
        let resume = state.withLockUnchecked { st -> (() -> Void)? in
            guard !st.finished, let continuation = st.continuation else { return nil }
            st.finished = true
            st.continuation = nil
            st.task = nil
            try? st.handle?.close()
            st.handle = nil
            let writeError = st.writeError
            return {
                if let writeError {
                    continuation.resume(throwing: writeError)
                } else if let error {
                    continuation.resume(throwing: error)
                } else if let response {
                    continuation.resume(returning: response)
                } else {
                    continuation.resume(throwing: HFDownload.DownloadError.invalidResponse)
                }
            }
        }
        resume?()
    }
}
