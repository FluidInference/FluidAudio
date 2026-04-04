#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Chinese dataset downloading functionality for THCHS-30
extension DatasetDownloader {

    // MARK: - THCHS-30

    /// Download THCHS-30 dataset from HuggingFace
    static func downloadTHCHS30(force: Bool, maxSamples: Int? = nil) async {
        let cacheDir = getTHCHS30CacheDirectory()

        logger.info("📥 Downloading THCHS-30 to \(cacheDir.path)")

        // Create directories
        do {
            try FileManager.default.createDirectory(
                at: cacheDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create directories: \(error)")
            return
        }

        // Check if already downloaded
        let metadataPath = cacheDir.appendingPathComponent("metadata.jsonl")
        if !force && FileManager.default.fileExists(atPath: metadataPath.path) {
            let audioDir = cacheDir.appendingPathComponent("audio", isDirectory: true)
            let existingFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: audioDir, includingPropertiesForKeys: nil)) ?? []
            let wavCount = existingFiles.filter { $0.pathExtension == "wav" }.count
            if wavCount > 0 {
                logger.info("📂 THCHS-30 exists (\(wavCount) WAV files)")
                return
            }
        }

        // Download using huggingface-cli
        do {
            try await downloadTHCHS30Dataset(to: cacheDir)
            logger.info("✅ THCHS-30 download complete")
        } catch {
            logger.error("Failed to download THCHS-30: \(error)")
        }
    }

    /// Get cache directory for THCHS-30 dataset
    static func getTHCHS30CacheDirectory() -> URL {
        #if os(macOS)
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(
            "Library/Application Support/FluidAudio/Datasets/THCHS-30")
        #else
        return FileManager.default.temporaryDirectory
            .appendingPathComponent("FluidAudio/Datasets/THCHS-30")
        #endif
    }

    /// Internal download function using huggingface-cli
    private static func downloadTHCHS30Dataset(to directory: URL) async throws {
        // Download using huggingface-cli
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "huggingface-cli",
            "download",
            "FluidInference/THCHS-30-tests",
            "--repo-type", "dataset",
            "--local-dir", directory.path,
        ]

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "ChineseDatasetDownloader",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        """
                    Failed to download THCHS-30 dataset from HuggingFace.
                    Make sure huggingface-cli is installed: pip install huggingface_hub
                    """
                ]
            )
        }
    }
}

// MARK: - Dataset Loading

/// Error types for Chinese dataset loading
enum ChineseDatasetError: Error, LocalizedError {
    case datasetNotFound(String)
    case invalidMetadata(String)

    var errorDescription: String? {
        switch self {
        case .datasetNotFound(let message): return message
        case .invalidMetadata(let message): return message
        }
    }
}

/// Sample structure for Chinese benchmark
struct ChineseBenchmarkSample {
    let audioPath: URL
    let transcript: String
    let sampleId: Int
}

/// Loader for Chinese datasets
enum ChineseDatasetLoader {

    private static let logger = AppLogger(category: "ChineseDatasetLoader")

    /// Load THCHS-30 samples
    static func loadTHCHS30Samples(
        maxSamples: Int,
        datasetPath: String? = nil
    ) async throws -> [ChineseBenchmarkSample] {
        let baseDir: URL

        if let path = datasetPath {
            // Use provided path
            baseDir = URL(fileURLWithPath: path)
        } else {
            // Use cache directory
            baseDir = DatasetDownloader.getTHCHS30CacheDirectory()

            // Check if exists
            let metadataPath = baseDir.appendingPathComponent("metadata.jsonl")
            guard FileManager.default.fileExists(atPath: metadataPath.path) else {
                throw ChineseDatasetError.datasetNotFound(
                    """
                    THCHS-30 dataset not found at: \(baseDir.path)

                    Options:
                    1. Use --auto-download to download from HuggingFace
                    2. Use --dataset-path <path> to specify local dataset directory

                    Expected directory structure:
                        <path>/
                        ├── audio/           # WAV files
                        └── metadata.jsonl   # Transcripts
                    """
                )
            }
        }

        // Load metadata.jsonl
        let metadataPath = baseDir.appendingPathComponent("metadata.jsonl")
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw ChineseDatasetError.invalidMetadata(
                "metadata.jsonl not found at: \(metadataPath.path)")
        }

        let metadataContent = try String(contentsOf: metadataPath, encoding: .utf8)
        var samples: [ChineseBenchmarkSample] = []

        struct MetadataEntry: Codable {
            let file_name: String
            let text: String
        }

        for (index, line) in metadataContent.components(separatedBy: .newlines).enumerated() {
            guard !line.isEmpty else { continue }
            guard samples.count < maxSamples else { break }

            let decoder = JSONDecoder()
            guard let data = line.data(using: .utf8),
                let entry = try? decoder.decode(MetadataEntry.self, from: data)
            else {
                logger.warning("Failed to decode line \(index): \(line)")
                continue
            }

            let audioPath = baseDir.appendingPathComponent(entry.file_name)
            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                logger.warning("Audio file not found: \(audioPath.path)")
                continue
            }

            samples.append(
                ChineseBenchmarkSample(
                    audioPath: audioPath,
                    transcript: entry.text,
                    sampleId: index
                ))
        }

        return samples
    }
}

#endif
