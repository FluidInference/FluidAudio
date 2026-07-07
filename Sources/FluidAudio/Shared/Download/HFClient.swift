import Foundation

/// HuggingFace HTTP plumbing shared by every download path (#765 Wave 2):
/// token resolution, authorized request building, and the single place where
/// rate-limit (429/503) responses become typed errors.
enum HFClient {

    /// HuggingFace token from the environment, if available. Supports the env
    /// vars used by the official CLI (`HF_TOKEN`), the Python `huggingface_hub`
    /// library (`HUGGING_FACE_HUB_TOKEN`), and LangChain/older integrations
    /// (`HUGGINGFACEHUB_API_TOKEN`).
    static var huggingFaceToken: String? {
        ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGINGFACEHUB_API_TOKEN"]
    }

    /// Create a URLRequest with optional auth header and timeout.
    static func authorizedRequest(
        url: URL, timeout: TimeInterval = DownloadUtils.DownloadConfig.default.timeout
    ) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        if let token = huggingFaceToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// The one place 429/503 responses are turned into
    /// `HuggingFaceDownloadError.rateLimited`. A `Retry-After` header, when
    /// present, is included in the message for diagnostics.
    static func checkRateLimit(_ response: HTTPURLResponse, context: String) throws {
        guard response.statusCode == 429 || response.statusCode == 503 else { return }
        var message = "Rate limited while \(context)"
        if let retryAfter = response.value(forHTTPHeaderField: "Retry-After") {
            message += " (Retry-After: \(retryAfter))"
        }
        throw DownloadUtils.HuggingFaceDownloadError.rateLimited(
            statusCode: response.statusCode, message: message)
    }

    /// Reject 200-OK responses whose body is an HTML error page instead of the
    /// JSON the HF API was asked for (seen during rate limiting/timeouts).
    static func validateJSONResponse(_ data: Data, path: String) throws {
        if let responseString = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        {
            if responseString.hasPrefix("<") || responseString.lowercased().contains("<!doctype html") {
                let snippet = String(responseString.prefix(100))
                throw DownloadUtils.HuggingFaceDownloadError.htmlErrorResponse(
                    path: path, snippet: snippet)
            }
        }
    }
}
