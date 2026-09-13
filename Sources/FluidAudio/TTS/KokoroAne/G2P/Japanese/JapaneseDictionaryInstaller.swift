import CryptoKit
import Foundation

/// Serializes dictionary installation across Kokoro manager instances and
/// publishes the extracted directory only after validation succeeds.
actor JapaneseDictionaryInstaller {
    static let shared = JapaneseDictionaryInstaller()

    func ensureInstalled(
        repoDirectory: URL,
        downloader: @Sendable (URL, URL) async throws -> Void
    ) async throws -> URL {
        let dictionaryURL = repoDirectory.appendingPathComponent(
            KokoroAneConstants.japaneseG2PSubdir)
        if Self.isComplete(dictionaryURL) { return dictionaryURL }

        let g2pDirectory = dictionaryURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: g2pDirectory, withIntermediateDirectories: true)
        let archiveURL = g2pDirectory.appendingPathComponent(
            KokoroAneConstants.japaneseDictionaryArchiveFile)

        if FileManager.default.fileExists(atPath: archiveURL.path),
            try !Self.hasExpectedDigest(archiveURL)
        {
            try FileManager.default.removeItem(at: archiveURL)
        }
        if !FileManager.default.fileExists(atPath: archiveURL.path) {
            try await downloader(KokoroAneConstants.japaneseDictionaryArchiveURL, archiveURL)
        }

        guard try Self.hasExpectedDigest(archiveURL) else {
            try? FileManager.default.removeItem(at: archiveURL)
            throw KokoroAneError.downloadFailed(
                "OpenJTalk dictionary archive failed SHA-256 verification.")
        }

        let stagingDirectory = g2pDirectory.appendingPathComponent(
            ".open-jtalk-extract-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: stagingDirectory) }
        try GzipTarExtractor.extract(archiveURL: archiveURL, to: stagingDirectory)

        let extracted = stagingDirectory.appendingPathComponent(
            "open_jtalk_dic_utf_8-1.11")
        guard Self.isComplete(extracted) else {
            throw KokoroAneError.downloadFailed(
                "OpenJTalk dictionary archive is missing required files.")
        }

        if FileManager.default.fileExists(atPath: dictionaryURL.path) {
            try FileManager.default.removeItem(at: dictionaryURL)
        }
        try FileManager.default.moveItem(at: extracted, to: dictionaryURL)
        try? FileManager.default.removeItem(at: archiveURL)
        return dictionaryURL
    }

    static func isComplete(_ directory: URL) -> Bool {
        ["char.bin", "matrix.bin", "sys.dic", "unk.dic"].allSatisfy { name in
            let url = directory.appendingPathComponent(name)
            guard
                let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
                let size = attributes[.size] as? NSNumber
            else { return false }
            return size.intValue > 0
        }
    }

    private static func hasExpectedDigest(_ archiveURL: URL) throws -> Bool {
        let attributes = try FileManager.default.attributesOfItem(atPath: archiveURL.path)
        guard
            let size = attributes[.size] as? NSNumber,
            size.intValue == KokoroAneConstants.japaneseDictionaryArchiveBytes
        else { return false }

        let data = try Data(contentsOf: archiveURL, options: .mappedIfSafe)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return digest == KokoroAneConstants.japaneseDictionaryArchiveSHA256
    }
}
