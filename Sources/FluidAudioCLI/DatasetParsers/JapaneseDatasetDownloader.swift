#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Japanese dataset downloading functionality for JSUT and Common Voice
extension DatasetDownloader {

    // MARK: - JSUT-basic5000

    /// Download JSUT-basic5000 dataset from HuggingFace
    static func downloadJSUTBasic5000(force: Bool, maxSamples: Int? = nil) async {
        let cacheDir = getJSUTCacheDirectory()
        let audioDir = cacheDir.appendingPathComponent("audio", isDirectory: true)

        logger.info("📥 Downloading JSUT-basic5000 to \(cacheDir.path)")

        // Create directories
        do {
            try FileManager.default.createDirectory(
                at: audioDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create directories: \(error)")
            return
        }

        // Check if already downloaded
        let metadataPath = cacheDir.appendingPathComponent("metadata.jsonl")
        if !force && FileManager.default.fileExists(atPath: metadataPath.path) {
            let existingFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: audioDir, includingPropertiesForKeys: nil)) ?? []
            let wavCount = existingFiles.filter { $0.pathExtension == "wav" }.count
            if wavCount > 0 {
                logger.info("📂 JSUT-basic5000 exists (\(wavCount) WAV files)")
                return
            }
        }

        // Download metadata and audio from HuggingFace
        let dataset = "FluidInference/JSUT-basic5000"

        do {
            // Download metadata.jsonl
            logger.info("📄 Downloading metadata...")
            let metadataURL = try ModelRegistry.resolveDataset(dataset, "metadata.jsonl")
            _ = try await downloadAudioFile(from: metadataURL.absoluteString, to: metadataPath)

            // Parse metadata to get file list
            let metadataContent = try String(contentsOf: metadataPath, encoding: .utf8)
            var audioFiles: [String] = []

            for line in metadataContent.components(separatedBy: .newlines) {
                guard !line.isEmpty else { continue }
                if let data = line.data(using: .utf8),
                    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let fileName = json["file_name"] as? String
                {
                    audioFiles.append(fileName)
                }
                // Respect maxSamples limit
                if let max = maxSamples, audioFiles.count >= max {
                    break
                }
            }

            logger.info("📄 Found \(audioFiles.count) audio files in metadata")

            // Download audio files
            var downloadedCount = 0
            for (index, fileName) in audioFiles.enumerated() {
                let audioURL = try ModelRegistry.resolveDataset(dataset, "audio/\(fileName)")
                let destination = audioDir.appendingPathComponent(fileName)

                // Skip if already exists
                if !force && FileManager.default.fileExists(atPath: destination.path) {
                    downloadedCount += 1
                    continue
                }

                do {
                    _ = try await downloadAudioFile(from: audioURL.absoluteString, to: destination)
                    downloadedCount += 1

                    if (index + 1) % 100 == 0 {
                        logger.info("  Downloaded \(index + 1)/\(audioFiles.count) files...")
                    }
                } catch {
                    logger.warning("Failed to download \(fileName): \(error)")
                }
            }

            logger.info("JSUT-basic5000 ready: \(downloadedCount)/\(audioFiles.count) files")

        } catch {
            logger.error("Failed to download JSUT-basic5000: \(error)")
        }
    }

    /// Get JSUT cache directory
    static func getJSUTCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent(
            "FluidAudio/Datasets/JSUT-basic5000", isDirectory: true)
    }

    // MARK: - Common Voice Japanese (cv-corpus-25.0-ja)

    /// Download Common Voice Japanese corpus from HuggingFace
    static func downloadCommonVoiceJapanese(
        force: Bool,
        maxSamples: Int? = nil,
        split: CVSplit = .train
    ) async {
        let cacheDir = getCommonVoiceCacheDirectory()
        let splitDir = cacheDir.appendingPathComponent(split.rawValue, isDirectory: true)
        let audioDir = splitDir.appendingPathComponent("audio", isDirectory: true)

        logger.info("📥 Downloading Common Voice Japanese (\(split.displayName)) to \(cacheDir.path)")

        // Create directories
        do {
            try FileManager.default.createDirectory(
                at: audioDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create directories: \(error)")
            return
        }

        // Check if already downloaded
        let metadataPath = splitDir.appendingPathComponent("metadata.jsonl")
        if !force && FileManager.default.fileExists(atPath: metadataPath.path) {
            let existingFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: audioDir, includingPropertiesForKeys: nil)) ?? []
            let mp3Count = existingFiles.filter { $0.pathExtension == "mp3" }.count
            if mp3Count > 0 {
                logger.info("📂 Common Voice Japanese \(split.displayName) exists (\(mp3Count) MP3 files)")
                return
            }
        }

        // Download metadata and audio from HuggingFace
        let dataset = "FluidInference/cv-corpus-25.0-ja"

        do {
            // Download metadata.jsonl for the split
            logger.info("📄 Downloading metadata...")
            let metadataURL = try ModelRegistry.resolveDataset(
                dataset, "\(split.rawValue)/metadata.jsonl")
            _ = try await downloadAudioFile(from: metadataURL.absoluteString, to: metadataPath)

            // Parse metadata to get file list
            let metadataContent = try String(contentsOf: metadataPath, encoding: .utf8)
            var audioEntries: [(fileName: String, path: String)] = []

            for line in metadataContent.components(separatedBy: .newlines) {
                guard !line.isEmpty else { continue }
                if let data = line.data(using: .utf8),
                    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    let path = json["path"] as? String
                {
                    let fileName = URL(fileURLWithPath: path).lastPathComponent
                    audioEntries.append((fileName: fileName, path: path))
                }
                // Respect maxSamples limit
                if let max = maxSamples, audioEntries.count >= max {
                    break
                }
            }

            logger.info("📄 Found \(audioEntries.count) audio files in metadata")

            // Download audio files
            var downloadedCount = 0
            for (index, entry) in audioEntries.enumerated() {
                let audioURL = try ModelRegistry.resolveDataset(
                    dataset, "\(split.rawValue)/audio/\(entry.path)")
                let destination = audioDir.appendingPathComponent(entry.fileName)

                // Skip if already exists
                if !force && FileManager.default.fileExists(atPath: destination.path) {
                    downloadedCount += 1
                    continue
                }

                do {
                    _ = try await downloadAudioFile(from: audioURL.absoluteString, to: destination)
                    downloadedCount += 1

                    if (index + 1) % 100 == 0 {
                        logger.info("  Downloaded \(index + 1)/\(audioEntries.count) files...")
                    }
                } catch {
                    logger.warning("Failed to download \(entry.fileName): \(error)")
                }
            }

            logger.info(
                "Common Voice Japanese \(split.displayName) ready: \(downloadedCount)/\(audioEntries.count) files")

        } catch {
            logger.error("Failed to download Common Voice Japanese: \(error)")
        }
    }

    /// Common Voice dataset splits
    enum CVSplit: String, CaseIterable {
        case train = "train"
        case validation = "validation"
        case test = "test"

        var displayName: String {
            switch self {
            case .train: return "Train"
            case .validation: return "Validation"
            case .test: return "Test"
            }
        }
    }

    /// Get Common Voice Japanese cache directory
    static func getCommonVoiceCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent(
            "FluidAudio/Datasets/cv-corpus-25.0-ja", isDirectory: true)
    }

    // MARK: - Metadata Types

    /// JSUT metadata entry
    struct JSUTMetadataEntry: Codable {
        let fileName: String
        let text: String
        let speakerId: String?

        enum CodingKeys: String, CodingKey {
            case fileName = "file_name"
            case text
            case speakerId = "speaker_id"
        }
    }

    /// Common Voice metadata entry
    struct CommonVoiceMetadataEntry: Codable {
        let path: String
        let text: String
        let clientId: String?
        let sentenceId: String?

        enum CodingKeys: String, CodingKey {
            case path
            case text = "sentence"
            case clientId = "client_id"
            case sentenceId
        }
    }
}

// MARK: - Japanese Dataset Loader

/// Loads and parses Japanese datasets for benchmarking
struct JapaneseDatasetLoader {
    private static let logger = AppLogger(category: "JapaneseDatasetLoader")

    /// Load JSUT-basic5000 samples
    static func loadJSUTSamples(maxSamples: Int? = nil) async throws -> [JapaneseBenchmarkSample] {
        let cacheDir = DatasetDownloader.getJSUTCacheDirectory()
        let audioDir = cacheDir.appendingPathComponent("audio")
        let metadataPath = cacheDir.appendingPathComponent("metadata.jsonl")

        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw JapaneseDatasetError.datasetNotFound(
                "JSUT-basic5000 not found. Run: fluidaudio download --dataset jsut-basic5000")
        }

        let metadataContent = try String(contentsOf: metadataPath, encoding: .utf8)
        var samples: [JapaneseBenchmarkSample] = []

        for (index, line) in metadataContent.components(separatedBy: .newlines).enumerated() {
            guard !line.isEmpty else { continue }
            guard let data = line.data(using: .utf8) else { continue }

            let decoder = JSONDecoder()
            guard let entry = try? decoder.decode(DatasetDownloader.JSUTMetadataEntry.self, from: data) else {
                continue
            }

            let audioPath = audioDir.appendingPathComponent(entry.fileName)
            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                continue
            }

            samples.append(
                JapaneseBenchmarkSample(
                    audioPath: audioPath,
                    transcript: entry.text,
                    sampleId: index,
                    speakerId: entry.speakerId ?? "jsut"
                ))

            if let max = maxSamples, samples.count >= max {
                break
            }
        }

        logger.info("Loaded \(samples.count) JSUT samples")
        return samples
    }

    /// Load Common Voice Japanese samples
    static func loadCommonVoiceSamples(
        split: DatasetDownloader.CVSplit = .test,
        maxSamples: Int? = nil
    ) async throws -> [JapaneseBenchmarkSample] {
        let cacheDir = DatasetDownloader.getCommonVoiceCacheDirectory()
        let splitDir = cacheDir.appendingPathComponent(split.rawValue)
        let audioDir = splitDir.appendingPathComponent("audio")
        let metadataPath = splitDir.appendingPathComponent("metadata.jsonl")

        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw JapaneseDatasetError.datasetNotFound(
                "Common Voice Japanese \(split.displayName) not found. Run: fluidaudio download --dataset cv-corpus-ja-\(split.rawValue)"
            )
        }

        let metadataContent = try String(contentsOf: metadataPath, encoding: .utf8)
        var samples: [JapaneseBenchmarkSample] = []

        for (index, line) in metadataContent.components(separatedBy: .newlines).enumerated() {
            guard !line.isEmpty else { continue }
            guard let data = line.data(using: .utf8) else { continue }

            let decoder = JSONDecoder()
            guard let entry = try? decoder.decode(DatasetDownloader.CommonVoiceMetadataEntry.self, from: data) else {
                continue
            }

            let fileName = URL(fileURLWithPath: entry.path).lastPathComponent
            let audioPath = audioDir.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                continue
            }

            samples.append(
                JapaneseBenchmarkSample(
                    audioPath: audioPath,
                    transcript: entry.text,
                    sampleId: index,
                    speakerId: entry.clientId ?? "unknown"
                ))

            if let max = maxSamples, samples.count >= max {
                break
            }
        }

        logger.info("Loaded \(samples.count) Common Voice \(split.displayName) samples")
        return samples
    }
}

/// Japanese benchmark sample
struct JapaneseBenchmarkSample {
    let audioPath: URL
    let transcript: String
    let sampleId: Int
    let speakerId: String
}

/// Japanese dataset errors
enum JapaneseDatasetError: Error, LocalizedError {
    case datasetNotFound(String)

    var errorDescription: String? {
        switch self {
        case .datasetNotFound(let message):
            return message
        }
    }
}

#endif
