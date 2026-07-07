import Foundation

/// The single retry policy for the download stack: bounded exponential
/// backoff on transient failures (paced by a server `Retry-After` when one
/// is provided), fail-fast on permanent ones. Every network path —
/// `ModelHub.loadModels`/`download`/`fetchFile` — retries through here.
enum RetryPolicy {

    private static let defaultLogger = AppLogger(category: "RetryPolicy")

    /// Longest server-requested pause `withRetry` will honor. Rate-limited
    /// endpoints occasionally send hour-scale `Retry-After` values; waiting
    /// that long inside a download call would look like a hang.
    static let maxHonoredRetryAfter: TimeInterval = 30

    /// Internal envelope carrying a typed `Retry-After` hint alongside the
    /// real error (#765 Wave 5). Thrown ONLY inside `withRetry` operations
    /// (see `HFClient.checkRateLimitForRetry`); `withRetry` unwraps it for
    /// pacing and always rethrows the underlying error, so the envelope never
    /// escapes to callers or tests.
    struct RetryAfterHint: Error {
        let underlying: Error
        let retryAfter: TimeInterval
    }

    /// Run `operation`, retrying transient failures (per `isRetryable`) with
    /// exponential backoff. Permanent errors and the final attempt's error are
    /// rethrown unchanged. The closure receives the 1-based attempt number
    /// (for logging/resume diagnostics).
    ///
    /// - Parameters:
    ///   - maxAttempts: Total attempts including the first; must be >= 1.
    ///   - logger: Where per-attempt retry warnings go. Pass the calling
    ///     component's logger so one operation's log trail stays in one
    ///     category.
    static func withRetry<T>(
        label: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0,
        logger: AppLogger = defaultLogger,
        operation: (_ attempt: Int) async throws -> T
    ) async throws -> T {
        precondition(maxAttempts >= 1, "maxAttempts must be >= 1 (got \(maxAttempts))")

        for attempt in 1...maxAttempts {
            do {
                return try await operation(attempt)
            } catch {
                // Unwrap a Retry-After envelope: the hint paces the sleep, the
                // underlying error drives classification and is what callers
                // see — the envelope never escapes this function.
                let hint = error as? RetryAfterHint
                let underlying = hint?.underlying ?? error

                guard attempt < maxAttempts, isRetryable(underlying) else {
                    throw underlying
                }
                let backoffSeconds = pow(2.0, Double(attempt - 1)) * minBackoff
                let serverPause = min(hint?.retryAfter ?? 0, maxHonoredRetryAfter)
                let pauseSeconds = max(backoffSeconds, serverPause)
                let pacedNote = serverPause > backoffSeconds ? " (server Retry-After)" : ""
                logger.warning(
                    "Download attempt \(attempt) for \(label) failed: \(underlying.localizedDescription). Retrying in \(String(format: "%.1f", pauseSeconds))s\(pacedNote)."
                )
                try await Task.sleep(nanoseconds: UInt64(pauseSeconds * 1_000_000_000))
            }
        }

        // The final attempt's catch always rethrows (attempt < maxAttempts
        // fails), so the loop cannot complete normally.
        preconditionFailure("unreachable: the final attempt rethrows in the catch guard")
    }

    /// Classify an error as transient (worth retrying) or permanent.
    /// Transient: URLSession timeout / TLS / connectivity failures and HTTP
    /// 429/503/5xx. Everything else (404/other 4xx, invalid response,
    /// non-network errors) is permanent.
    static func isRetryable(_ error: Error) -> Bool {
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut, .cannotConnectToHost, .cannotFindHost,
                .networkConnectionLost, .notConnectedToInternet,
                .dnsLookupFailed, .secureConnectionFailed,
                .resourceUnavailable:
                return true
            default:
                return false
            }
        }

        switch error {
        case DownloadError.rateLimited:
            return true
        case DownloadError.invalidArtifact:
            // Usually a transient unhealthy network path (proxy, mirror 5xx) — retry.
            return true
        case DownloadError.downloadFailed(_, let underlying):
            let nsError = underlying as NSError
            return nsError.domain == "HTTP" && (500...599).contains(nsError.code)
        default:
            return false
        }
    }

    /// `true` when `error` represents cancellation (Swift `CancellationError`,
    /// `NSURLErrorCancelled`, or `NSUserCancelledError`, at any depth of the
    /// underlying-error chain) rather than a real failure.
    ///
    /// The `NSUnderlyingErrorKey` chain is walked to its end. Visited errors
    /// are tracked by identity so a self-referential chain terminates without
    /// an arbitrary depth cap.
    static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }

        var current: NSError? = error as NSError
        var visited: Set<ObjectIdentifier> = []
        while let nsError = current, visited.insert(ObjectIdentifier(nsError)).inserted {
            if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
                return true
            }
            if nsError.domain == NSCocoaErrorDomain, nsError.code == NSUserCancelledError {
                return true
            }
            current = nsError.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return false
    }
}
