import Foundation

/// A file discovered in a HuggingFace repository tree listing.
struct RemoteFile: Equatable, Sendable {
    let path: String
    /// Size in bytes as reported by the tree API; `-1` when not reported.
    let size: Int
}

/// The one tree-listing implementation for the download stack (#765 Wave 3),
/// replacing the two divergent recursive walkers in `downloadRepo` and
/// `downloadSubdirectory`. Follows the HF tree API's `Link` pagination cursor,
/// so directories with more entries than one API page (1,000 by default) no
/// longer silently drop files.
enum HFTreeLister {

    /// Executes one listing request; injectable so fixtures can serve canned
    /// pages. Returns the final HTTP response (headers carry the cursor).
    typealias Fetch = (URL) async throws -> (Data, HTTPURLResponse)

    /// Production fetch: authorized request on `session`. Non-HTTP responses
    /// are rejected as `invalidResponse` — an explicit decision; the old
    /// listers silently skipped the rate-limit check on non-HTTP responses.
    static func fetch(using session: URLSession) -> Fetch {
        { url in
            let request = HFClient.authorizedRequest(url: url)
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw HFDownload.DownloadError.invalidResponse
            }
            return (data, httpResponse)
        }
    }

    /// Recursively list files under `startingAt` (empty = repo root),
    /// depth-first in server order, following pagination cursors within each
    /// directory.
    ///
    /// `include` is consulted for every item: for directories, returning
    /// false prunes the whole subtree without recursing; for files, false
    /// excludes the file.
    static func listTree(
        repoRemotePath: String,
        startingAt path: String = "",
        include: (_ itemPath: String, _ isDirectory: Bool) -> Bool = { _, _ in true },
        fetch: Fetch
    ) async throws -> [RemoteFile] {
        var files: [RemoteFile] = []
        let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
        var pageURL: URL? = try ModelRegistry.apiModels(repoRemotePath, apiPath)

        while let url = pageURL {
            let (data, response) = try await fetch(url)
            try HFClient.checkRateLimit(
                response,
                context: path.isEmpty ? "listing files" : "listing files in \(path)")
            try HFClient.validateJSONResponse(data, path: path)
            guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                throw HFDownload.DownloadError.invalidResponse
            }

            for item in items {
                guard let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String
                else { continue }

                if itemType == "directory" {
                    guard include(itemPath, true) else { continue }
                    files += try await listTree(
                        repoRemotePath: repoRemotePath,
                        startingAt: itemPath,
                        include: include,
                        fetch: fetch
                    )
                } else if itemType == "file" {
                    guard include(itemPath, false) else { continue }
                    files.append(RemoteFile(path: itemPath, size: item["size"] as? Int ?? -1))
                }
            }

            pageURL = nextPageURL(from: response)
        }
        return files
    }

    /// Parse the HF tree API's pagination cursor from a
    /// `Link: <url>; rel="next"` header (verified against the live API in
    /// #765 Wave 0). Returns nil when there is no next page.
    static func nextPageURL(from response: HTTPURLResponse) -> URL? {
        guard let link = response.value(forHTTPHeaderField: "Link") else { return nil }
        // A Link header may carry multiple comma-separated entries.
        for entry in link.split(separator: ",") {
            let parts = entry.split(separator: ";")
            guard
                parts.dropFirst().contains(where: {
                    $0.trimmingCharacters(in: .whitespaces) == "rel=\"next\""
                }),
                let target = parts.first,
                let start = target.firstIndex(of: "<"),
                let end = target.firstIndex(of: ">"),
                start < end
            else { continue }
            return URL(string: String(target[target.index(after: start)..<end]))
        }
        return nil
    }
}
