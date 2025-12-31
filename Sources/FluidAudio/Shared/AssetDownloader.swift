import Foundation
import OSLog

/// Generic helper for downloading and caching remote assets.
/// Provides reusable logic for simple file/data transfers used across FluidAudio modules.
public enum AssetDownloader {

    public typealias DataWriter = @Sendable (Data, URL) throws -> Void
    public typealias FileMover = @Sendable (URL, URL) throws -> Void

    public static let defaultDataWriter: DataWriter = { data, destination in
        try data.write(to: destination, options: [.atomic])
    }

    public static let defaultFileMover: FileMover = { tempURL, destination in
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    public enum TransferMode {
        case data(DataWriter = AssetDownloader.defaultDataWriter)
        case file(FileMover = AssetDownloader.defaultFileMover)
    }

    public struct Descriptor {
        public let description: String
        public let remoteURL: URL
        public let destinationURL: URL
        public let skipIfExists: Bool
        public let transferMode: TransferMode

        public init(
            description: String,
            remoteURL: URL,
            destinationURL: URL,
            skipIfExists: Bool = true,
            transferMode: TransferMode = .data()
        ) {
            self.description = description
            self.remoteURL = remoteURL
            self.destinationURL = destinationURL
            self.skipIfExists = skipIfExists
            self.transferMode = transferMode
        }
    }

    public enum Error: LocalizedError {
        case invalidResponse(description: String, statusCode: Int)

        public var errorDescription: String? {
            switch self {
            case .invalidResponse(let description, let statusCode):
                return "Unexpected response (status \(statusCode)) while downloading \(description)"
            }
        }
    }

    public static func ensure(
        _ descriptor: Descriptor,
        session: URLSession = DownloadUtils.sharedSession,
        logger: AppLogger = AppLogger(category: "AssetDownloader")
    ) async throws -> URL {
        if descriptor.skipIfExists,
            FileManager.default.fileExists(atPath: descriptor.destinationURL.path)
        {
            return descriptor.destinationURL
        }

        try FileManager.default.createDirectory(
            at: descriptor.destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        logger.info("Downloading \(descriptor.description)…")

        switch descriptor.transferMode {
        case .data(let writer):
            let (data, response) = try await session.data(from: descriptor.remoteURL)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            guard (200..<300).contains(status) else {
                throw Error.invalidResponse(description: descriptor.description, statusCode: status)
            }
            try writer(data, descriptor.destinationURL)
        case .file(let handler):
            let (tempURL, response) = try await session.download(from: descriptor.remoteURL)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            guard (200..<300).contains(status) else {
                throw Error.invalidResponse(description: descriptor.description, statusCode: status)
            }
            try handler(tempURL, descriptor.destinationURL)
        }

        logger.info("Cached \(descriptor.description) at \(descriptor.destinationURL.path)")
        return descriptor.destinationURL
    }

    public static func fetchData(
        from remoteURL: URL,
        description: String,
        session: URLSession = DownloadUtils.sharedSession,
        logger: AppLogger = AppLogger(category: "AssetDownloader")
    ) async throws -> Data {
        logger.debug("Fetching \(description) from \(remoteURL.absoluteString)")

        let (data, response) = try await session.data(from: remoteURL)
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard (200..<300).contains(status) else {
            throw Error.invalidResponse(description: description, statusCode: status)
        }
        return data
    }

}
