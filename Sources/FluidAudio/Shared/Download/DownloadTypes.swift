import Foundation

/// Namespace for the canonical download-stack types (#765 Wave 2).
///
/// These live outside `DownloadUtils` so the Shared/Download primitives
/// (HFClient, RetryPolicy, and later extractions) don't depend back on the
/// class they were extracted from. The long-standing nested spellings
/// (`DownloadUtils.HuggingFaceDownloadError`, `DownloadUtils.DownloadConfig`)
/// remain source-compatible via typealiases.
public enum HFDownload {

    /// Errors thrown by the HuggingFace download stack.
    public enum DownloadError: LocalizedError {
        case invalidResponse
        case rateLimited(statusCode: Int, message: String)
        case downloadFailed(path: String, underlying: Error)
        case modelNotFound(path: String)
        case htmlErrorResponse(path: String, snippet: String)
        case invalidArtifact(path: String, reason: String)

        public var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "Received an invalid response from Hugging Face."
            case .rateLimited(_, let message):
                return "Hugging Face rate limit encountered: \(message)"
            case .downloadFailed(let path, let underlying):
                return "Failed to download \(path): \(underlying.localizedDescription)"
            case .htmlErrorResponse(let path, let snippet):
                return "HuggingFace returned HTML instead of JSON for \(path) (rate limit or server issue): \(snippet)"
            case .modelNotFound(let path):
                return "Model file not found: \(path)"
            case .invalidArtifact(let path, let reason):
                return "Downloaded artifact for \(path) is invalid (\(reason)); refusing to cache it."
            }
        }
    }

    /// Download configuration shared by the download stack.
    public struct Config: Sendable {
        public let timeout: TimeInterval

        public init(timeout: TimeInterval = 1800) {  // 30 minutes for large models
            self.timeout = timeout
        }

        public static let `default` = Config()
    }
}
