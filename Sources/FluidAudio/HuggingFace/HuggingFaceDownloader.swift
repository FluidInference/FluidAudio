import CoreML
import Foundation

public enum HuggingFaceError: Error {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
    case fileSystemError(Error)
    case compilationError(Error)
    case noData
}

public struct HuggingFaceFile: Codable {
    public let type: String  // "file" or "directory"
    public let path: String
    public let size: Int?
    public let lfs: LfsInfo?

    public struct LfsInfo: Codable {
        public let oid: String
        public let size: Int
        public let pointerSize: Int?
    }
}

public class HuggingFaceDownloader {
    private let session: URLSession
    private let fileManager: FileManager

    public init() {
        self.session = URLSession.shared
        self.fileManager = FileManager.default
    }

    /// List files in a HuggingFace repository
    /// - Parameters:
    ///   - repoId: The repository ID (e.g. "alexwengg/parakeet-realtime-eou-120m-coreml")
    ///   - revision: The branch or commit hash (default: "main")
    ///   - path: The subdirectory path (optional)
    public func listFiles(
        repoId: String, revision: String = "main", path: String? = nil
    ) async throws -> [HuggingFaceFile] {
        var urlString = "https://huggingface.co/api/models/\(repoId)/tree/\(revision)"
        if let path = path {
            urlString += "/\(path)"
        }

        guard let url = URL(string: urlString) else {
            throw HuggingFaceError.invalidURL
        }

        let (data, _) = try await session.data(from: url)
        let files = try JSONDecoder().decode([HuggingFaceFile].self, from: data)
        return files
    }

    /// Download a file from HuggingFace
    /// - Parameters:
    ///   - repoId: The repository ID
    ///   - filePath: The path of the file within the repo
    ///   - destinationUrl: The local destination URL
    ///   - revision: The branch or commit hash
    public func downloadFile(
        repoId: String, filePath: String, destinationUrl: URL, revision: String = "main"
    ) async throws {
        // Construct raw download URL
        // https://huggingface.co/{repo_id}/resolve/{revision}/{path}
        let urlString = "https://huggingface.co/\(repoId)/resolve/\(revision)/\(filePath)"

        guard let url = URL(string: urlString) else {
            throw HuggingFaceError.invalidURL
        }

        print("Downloading \(filePath)...")
        let (tempUrl, _) = try await session.download(from: url)

        // Ensure directory exists
        let directory = destinationUrl.deletingLastPathComponent()
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

        // Remove existing file if needed
        if fileManager.fileExists(atPath: destinationUrl.path) {
            try fileManager.removeItem(at: destinationUrl)
        }

        try fileManager.moveItem(at: tempUrl, to: destinationUrl)
        print("Downloaded to \(destinationUrl.path)")
    }

    /// Download all files in a repo to a local directory
    public func downloadRepo(repoId: String, destinationDir: URL, revision: String = "main") async throws {
        try await downloadRecursive(
            repoId: repoId, currentPath: nil, stripPrefix: nil, localBase: destinationDir, revision: revision)
    }

    /// Download a subdirectory, stripping the prefix so contents go directly into destinationDir
    public func downloadSubdirectory(
        repoId: String, subPath: String, destinationDir: URL, revision: String = "main"
    ) async throws {
        try await downloadRecursive(
            repoId: repoId, currentPath: subPath, stripPrefix: subPath, localBase: destinationDir, revision: revision)
    }

    private func downloadRecursive(
        repoId: String, currentPath: String?, stripPrefix: String?, localBase: URL, revision: String
    ) async throws {
        let items = try await listFiles(repoId: repoId, revision: revision, path: currentPath)

        for item in items {
            if item.type == "file" {
                // Strip prefix if provided (for subdirectory downloads)
                var relativePath = item.path
                if let prefix = stripPrefix {
                    relativePath = String(item.path.dropFirst(prefix.count + 1))
                }
                let localUrl = localBase.appendingPathComponent(relativePath)
                if !fileManager.fileExists(atPath: localUrl.path) {
                    try await downloadFile(
                        repoId: repoId, filePath: item.path, destinationUrl: localUrl, revision: revision)
                }
            } else if item.type == "directory" {
                try await downloadRecursive(
                    repoId: repoId, currentPath: item.path, stripPrefix: stripPrefix, localBase: localBase,
                    revision: revision)
            }
        }
    }

    public func compileModel(at url: URL) throws -> URL {
        print("Compiling \(url.lastPathComponent)...")
        let compiledUrl = try MLModel.compileModel(at: url)

        // The compiled model is in a temp directory. We should move it to be next to the source.
        let destinationUrl = url.deletingPathExtension().appendingPathExtension("mlmodelc")

        if fileManager.fileExists(atPath: destinationUrl.path) {
            try fileManager.removeItem(at: destinationUrl)
        }

        try fileManager.moveItem(at: compiledUrl, to: destinationUrl)
        print("Compiled to \(destinationUrl.path)")
        return destinationUrl
    }
}
