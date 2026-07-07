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

    /// `checkRateLimit` for call sites inside `RetryPolicy.withRetry`: when
    /// the response carries `Retry-After`, the thrown `rateLimited` error is
    /// wrapped in `RetryPolicy.RetryAfterHint` so backoff honors the server's
    /// pacing (#765 Wave 5). Never use outside a retry operation — the
    /// envelope must not escape to callers (unpaced sites like the tree
    /// lister keep plain `checkRateLimit`).
    static func checkRateLimitForRetry(
        _ response: HTTPURLResponse, context: @autoclosure () -> String
    ) throws {
        do {
            try checkRateLimit(response, context: context())
        } catch {
            if let delay = retryAfter(from: response) {
                throw RetryPolicy.RetryAfterHint(underlying: error, retryAfter: delay)
            }
            throw error
        }
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

    /// Parse a pagination cursor from a `Link` header's `rel="next"` entry
    /// (the HF APIs paginate this way; verified live in #765 Wave 0).
    ///
    /// Tolerates the RFC 8288 shapes a naive split breaks on: target URLs
    /// containing `,`/`;` (legal inside `<…>`), unquoted `rel=next`, and
    /// multi-value `rel="next last"`. Returns nil when there is no next page.
    static func nextPageURL(from response: HTTPURLResponse) -> URL? {
        guard let link = response.value(forHTTPHeaderField: "Link") else { return nil }

        var rest = Substring(link)
        while let start = rest.firstIndex(of: "<") {
            guard let end = rest[start...].firstIndex(of: ">") else { return nil }
            let target = rest[rest.index(after: start)..<end]

            // Parameters run from after the target to the next top-level comma
            // (commas inside a later <…> can't be reached: we stop at the first).
            let afterTarget = rest[rest.index(after: end)...]
            let paramsEnd = afterTarget.firstIndex(of: ",") ?? afterTarget.endIndex
            let params = afterTarget[..<paramsEnd].lowercased()

            if let relRange = params.range(of: "rel=") {
                var value = params[relRange.upperBound...]
                if let paramBreak = value.firstIndex(of: ";") {
                    value = value[..<paramBreak]
                }
                let relations =
                    value
                    .trimmingCharacters(in: .whitespaces)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                    .split(separator: " ")
                if relations.contains("next") {
                    return URL(string: String(target))
                }
            }

            rest = afterTarget[paramsEnd...].dropFirst()
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
