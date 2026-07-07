import Foundation

/// HuggingFace HTTP plumbing for the DownloadUtils download paths, converging
/// here across the #765 waves: token resolution, authorized request building,
/// and the single place where rate-limit (429/503) responses become typed
/// errors. (`fetchHuggingFaceFile` keeps its divergent retry loop until
/// Wave 5; `AssetDownloader` and the CLI's `DatasetDownloader` are not yet in
/// scope.)
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
        url: URL, timeout: TimeInterval = HFDownload.Config.default.timeout
    ) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        if let token = huggingFaceToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// The one place 429/503 responses are turned into
    /// `HFDownload.DownloadError.rateLimited`. The message is deterministic
    /// ("Rate limited while <context> (HTTP <code>)"); the machine-readable
    /// `Retry-After` hint stays available via `retryAfter(from:)` for the
    /// retry layer to honor (Wave 5) rather than being flattened into text.
    static func checkRateLimit(
        _ response: HTTPURLResponse, context: @autoclosure () -> String
    ) throws {
        guard response.statusCode == 429 || response.statusCode == 503 else { return }
        throw HFDownload.DownloadError.rateLimited(
            statusCode: response.statusCode,
            message: "Rate limited while \(context()) (HTTP \(response.statusCode))")
    }

    /// Typed `Retry-After` from a response: delay-seconds, or an HTTP-date
    /// converted to seconds-from-now. Nil when absent or unparsable.
    static func retryAfter(from response: HTTPURLResponse) -> TimeInterval? {
        guard let value = response.value(forHTTPHeaderField: "Retry-After") else { return nil }
        if let seconds = TimeInterval(value) {
            return max(0, seconds)
        }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
        if let date = formatter.date(from: value) {
            return max(0, date.timeIntervalSinceNow)
        }
        return nil
    }

    /// `true` when `data` begins with HTML/XML markup — the single HTML
    /// sniffer for the download stack (error pages served with 200s).
    static func looksLikeHTML(_ data: Data) -> Bool {
        let prefix = data.prefix(512)
        let text = String(data: prefix, encoding: .utf8) ?? String(decoding: prefix, as: UTF8.self)
        let lowered = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        return lowered.hasPrefix("<!doctype html") || lowered.hasPrefix("<html") || lowered.hasPrefix("<?xml")
    }

    /// Reject 200-OK responses whose body is an HTML error page instead of the
    /// JSON the HF API was asked for (seen during rate limiting/timeouts).
    /// Delegates markup detection to `looksLikeHTML`, plus the JSON-specific
    /// check that a JSON document can never open with `<`.
    static func validateJSONResponse(_ data: Data, path: String) throws {
        let trimmed = String(data: data.prefix(512), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed?.hasPrefix("<") == true || looksLikeHTML(data) {
            let snippet = String((trimmed ?? "").prefix(100))
            throw HFDownload.DownloadError.htmlErrorResponse(path: path, snippet: snippet)
        }
    }
}
