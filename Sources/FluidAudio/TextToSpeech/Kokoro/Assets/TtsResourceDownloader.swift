import Foundation

/// Kokoro TTS resource downloader (lexicons, voice embeddings, eSpeak data)
public enum TtsResourceDownloader {

    private static let logger = AppLogger(category: "TtsResourceDownloader")
    private static let kokoroBaseURL = "https://huggingface.co/\(Repo.kokoro.remotePath)/resolve/main"

    private static func packagedEspeakBundleURL() -> URL? {
        #if SWIFT_PACKAGE
        return Bundle.module.url(
            forResource: "espeak-ng-data",
            withExtension: "bundle",
            subdirectory: "espeak-ng"
        )
        #else
        return nil
        #endif
    }

    private static func stagePackagedEspeakBundle(
        from bundleURL: URL,
        into repoPath: URL,
        voicesPath: URL
    ) throws {
        let targetBundle = repoPath.appendingPathComponent("Resources/espeak-ng/espeak-ng-data.bundle")
        let parentDir = targetBundle.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: targetBundle.path) {
            try FileManager.default.removeItem(at: targetBundle)
        }
        do {
            try FileManager.default.copyItem(at: bundleURL, to: targetBundle)
        } catch {
            throw TTSError.downloadFailed(
                "Failed to stage packaged eSpeak NG bundle: \(error.localizedDescription)"
            )
        }

        guard FileManager.default.fileExists(atPath: voicesPath.path) else {
            throw TTSError.downloadFailed(
                "Packaged eSpeak NG data bundle missing 'voices' directory after copy"
            )
        }
    }

    /// Download a voice embedding JSON file from HuggingFace
    public static func downloadVoiceEmbedding(voice: String) async throws -> Data {
        let jsonURL = "\(kokoroBaseURL)/voices/\(voice).json"

        guard let url = URL(string: jsonURL) else {
            throw TTSError.modelNotFound("Invalid URL for voice embedding: \(voice)")
        }

        do {
            let data = try await AssetDownloader.fetchData(
                from: url,
                description: "\(voice) voice embedding JSON",
                logger: logger
            )
            logger.info("Downloaded voice embedding JSON for \(voice)")
            return data
        } catch {
            throw TTSError.modelNotFound("Voice embedding JSON unavailable for \(voice): \(error.localizedDescription)")
        }
    }

    /// Ensure a voice embedding is available in cache
    public static func ensureVoiceEmbedding(voice: String = TtsConstants.recommendedVoice) async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")

        // Create directory if needed
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        let jsonFile = "\(voice).json"
        let jsonURL = voicesDir.appendingPathComponent(jsonFile)

        // Skip if already cached
        if FileManager.default.fileExists(atPath: jsonURL.path) {
            return
        }

        // Try to download
        let data = try await downloadVoiceEmbedding(voice: voice)
        try data.write(to: jsonURL, options: [.atomic])
        logger.info("Voice embedding cached: \(voice)")
    }

    /// Ensure a Kokoro lexicon file exists locally (e.g. `us_gold.json`).
    @discardableResult
    public static func ensureLexiconFile(named filename: String) async throws -> URL {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let localURL = kokoroDir.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        guard let remoteURL = URL(string: "\(kokoroBaseURL)/\(filename)") else {
            throw TTSError.modelNotFound("Invalid URL for \(filename)")
        }

        do {
            let descriptor = AssetDownloader.Descriptor(
                description: filename,
                remoteURL: remoteURL,
                destinationURL: localURL
            )
            return try await AssetDownloader.ensure(
                descriptor,
                logger: logger
            )
        } catch {
            throw TTSError.modelNotFound("Failed to download \(filename): \(error.localizedDescription)")
        }
    }

    /// Ensure the eSpeak NG data bundle is available locally for Kokoro G2P
    @available(macOS 13.0, iOS 16.0, *)
    public static func ensureEspeakDataBundle(in modelsDirectory: URL) async throws -> URL {
        let repo = Repo.kokoro
        let repoPath = modelsDirectory.appendingPathComponent(repo.folderName)
        let bundleRoot = repoPath.appendingPathComponent(
            "Resources/espeak-ng/espeak-ng-data.bundle/espeak-ng-data"
        )
        let voices = bundleRoot.appendingPathComponent("voices")

        if FileManager.default.fileExists(atPath: voices.path) {
            return bundleRoot
        }

        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        if let bundledBundle = packagedEspeakBundleURL() {
            try stagePackagedEspeakBundle(from: bundledBundle, into: repoPath, voicesPath: voices)
            logger.info("Using packaged espeak-ng-data bundle shipped with FluidAudio resources")
            return bundleRoot
        }

        #if os(macOS)
        logger.info("Downloading eSpeak NG data bundle from HuggingFace…")

        let zipPath = repoPath.appendingPathComponent("espeak-ng.zip")
        let zipURL = URL(string: "https://huggingface.co/\(repo.remotePath)/resolve/main/espeak-ng.zip")!

        if !FileManager.default.fileExists(atPath: zipPath.path) {
            let descriptor = AssetDownloader.Descriptor(
                description: "espeak-ng.zip",
                remoteURL: zipURL,
                destinationURL: zipPath,
                transferMode: .file()
            )
            _ = try await AssetDownloader.ensure(descriptor, logger: logger)
        }

        let resourcesDir = repoPath.appendingPathComponent("Resources")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", zipPath.path, "-d", resourcesDir.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw TTSError.downloadFailed(
                "Failed to extract eSpeak NG data bundle via unzip (status=\(process.terminationStatus))")
        }
        logger.info("Extracted espeak-ng-data successfully")
        #else
        throw TTSError.downloadFailed(
            "Packaged eSpeak NG data bundle missing from FluidAudio resources"
        )
        #endif

        guard FileManager.default.fileExists(atPath: voices.path) else {
            throw TTSError.downloadFailed("eSpeak NG data bundle missing 'voices' after extraction")
        }
        return bundleRoot
    }
}
