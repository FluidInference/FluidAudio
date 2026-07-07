import Foundation

/// The single retry policy for download paths (#765 Wave 2): bounded
/// exponential backoff on transient failures, fail-fast on permanent ones.
enum RetryPolicy {

    private static let logger = AppLogger(category: "RetryPolicy")

    /// Run `operation`, retrying transient failures (per `isRetryable`) with
    /// exponential backoff. Permanent errors and the final attempt's error are
    /// rethrown unchanged. The closure receives the 1-based attempt number
    /// (for logging/resume diagnostics).
    static func withRetry<T>(
        label: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0,
        operation: (_ attempt: Int) async throws -> T
    ) async throws -> T {
        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                return try await operation(attempt)
            } catch {
                lastError = error
                guard attempt < maxAttempts, isRetryable(error) else {
                    throw error
                }
                let backoffSeconds = pow(2.0, Double(attempt - 1)) * minBackoff
                logger.warning(
                    "Download attempt \(attempt) for \(label) failed: \(error.localizedDescription). Retrying in \(String(format: "%.1f", backoffSeconds))s."
                )
                try await Task.sleep(nanoseconds: UInt64(backoffSeconds * 1_000_000_000))
            }
        }
        throw lastError ?? DownloadUtils.HuggingFaceDownloadError.invalidResponse
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
        case DownloadUtils.HuggingFaceDownloadError.rateLimited:
            return true
        case DownloadUtils.HuggingFaceDownloadError.invalidArtifact:
            // Usually a transient unhealthy network path (proxy, mirror 5xx) — retry.
            return true
        case DownloadUtils.HuggingFaceDownloadError.downloadFailed(_, let underlying):
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
