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
            // Download transcript_utf8.txt (format: "FILENAME:transcription text")
            logger.info("📄 Downloading transcripts...")
            let transcriptURL = try ModelRegistry.resolveDataset(dataset, "basic5000/transcript_utf8.txt")
            let (transcriptData, _) = try await DownloadUtils.sharedSession.data(from: transcriptURL)
            let transcriptContent = String(data: transcriptData, encoding: .utf8) ?? ""

            // Parse transcripts and build metadata
            var entries: [(fileName: String, text: String)] = []
            for line in transcriptContent.components(separatedBy: .newlines) {
                guard !line.isEmpty else { continue }

                // Format: "BASIC5000_4501:transcription text"
                let parts = line.components(separatedBy: ":")
                guard parts.count >= 2 else { continue }

                let fileName = parts[0].trimmingCharacters(in: .whitespaces) + ".wav"
                let text = parts[1...].joined(separator: ":").trimmingCharacters(in: .whitespaces)

                entries.append((fileName: fileName, text: text))

                // Respect maxSamples limit
                if let max = maxSamples, entries.count >= max {
                    break
                }
            }

            logger.info("📄 Found \(entries.count) transcripts")

            // Download audio files from wav/ directory
            var downloadedCount = 0
            var metadataLines: [String] = []

            for (index, entry) in entries.enumerated() {
                let audioURL = try ModelRegistry.resolveDataset(dataset, "basic5000/wav/\(entry.fileName)")
                let destination = audioDir.appendingPathComponent(entry.fileName)

                // Skip if already exists
                if !force && FileManager.default.fileExists(atPath: destination.path) {
                    downloadedCount += 1

                    // Add to metadata
                    let metadataEntry: [String: String] = [
                        "file_name": entry.fileName,
                        "text": entry.text,
                        "speaker_id": "jsut"
                    ]
                    if let jsonData = try? JSONSerialization.data(withJSONObject: metadataEntry),
                       let jsonString = String(data: jsonData, encoding: .utf8) {
                        metadataLines.append(jsonString)
                    }
                    continue
                }

                do {
                    _ = try await downloadAudioFile(from: audioURL.absoluteString, to: destination)
                    downloadedCount += 1

                    // Add to metadata after successful download
                    let metadataEntry: [String: String] = [
                        "file_name": entry.fileName,
                        "text": entry.text,
                        "speaker_id": "jsut"
                    ]
                    if let jsonData = try? JSONSerialization.data(withJSONObject: metadataEntry),
                       let jsonString = String(data: jsonData, encoding: .utf8) {
                        metadataLines.append(jsonString)
                    }

                    if (index + 1) % 100 == 0 {
                        logger.info("  Downloaded \(index + 1)/\(entries.count) files...")
                    }
                } catch {
                    logger.warning("Failed to download \(entry.fileName): \(error)")
                }
            }

            // Write metadata.jsonl
            let metadataContent = metadataLines.joined(separator: "\n")
            try metadataContent.write(to: metadataPath, atomically: true, encoding: .utf8)

            logger.info("JSUT-basic5000 ready: \(downloadedCount)/\(entries.count) files")

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
            let mp3Count = existingFiles.filter { $0.pathExtension == "mp3" || $0.pathExtension == "wav" }.count
            if mp3Count > 0 {
                logger.info("📂 Common Voice Japanese \(split.displayName) exists (\(mp3Count) audio files)")
                return
            }
        }

        // Download metadata and audio from HuggingFace
        let dataset = "FluidInference/cv-corpus-25.0-ja"

        do {
            // Download TSV metadata (format: client_id\tpath\tsentence_id\tsentence\t...)
            logger.info("📄 Downloading metadata TSV...")
            let tsvURL = try ModelRegistry.resolveDataset(dataset, "ja/\(split.rawValue).tsv")
            let (tsvData, _) = try await DownloadUtils.sharedSession.data(from: tsvURL)
            let tsvContent = String(data: tsvData, encoding: .utf8) ?? ""

            // Parse TSV to extract entries
            var entries: [(fileName: String, text: String, clientId: String)] = []
            let lines = tsvContent.components(separatedBy: .newlines)

            // Skip header line
            for line in lines.dropFirst() {
                guard !line.isEmpty else { continue }

                let columns = line.components(separatedBy: "\t")
                guard columns.count >= 4 else { continue }

                let clientId = columns[0]
                let fileName = columns[1]
                let text = columns[3]

                entries.append((fileName: fileName, text: text, clientId: clientId))

                // Respect maxSamples limit
                if let max = maxSamples, entries.count >= max {
                    break
                }
            }

            logger.info("📄 Found \(entries.count) entries in TSV")

            // Download audio files from ja/clips/
            var downloadedCount = 0
            var metadataLines: [String] = []

            for (index, entry) in entries.enumerated() {
                let audioURL = try ModelRegistry.resolveDataset(dataset, "ja/clips/\(entry.fileName)")
                let destination = audioDir.appendingPathComponent(entry.fileName)

                // Skip if already exists
                if !force && FileManager.default.fileExists(atPath: destination.path) {
                    downloadedCount += 1

                    // Add to metadata
                    let metadataEntry: [String: String] = [
                        "path": entry.fileName,
                        "sentence": entry.text,
                        "client_id": entry.clientId
                    ]
                    if let jsonData = try? JSONSerialization.data(withJSONObject: metadataEntry),
                       let jsonString = String(data: jsonData, encoding: .utf8) {
                        metadataLines.append(jsonString)
                    }
                    continue
                }

                do {
                    _ = try await downloadAudioFile(from: audioURL.absoluteString, to: destination)
                    downloadedCount += 1

                    // Add to metadata after successful download
                    let metadataEntry: [String: String] = [
                        "path": entry.fileName,
                        "sentence": entry.text,
                        "client_id": entry.clientId
                    ]
                    if let jsonData = try? JSONSerialization.data(withJSONObject: metadataEntry),
                       let jsonString = String(data: jsonData, encoding: .utf8) {
                        metadataLines.append(jsonString)
                    }

                    if (index + 1) % 100 == 0 {
                        logger.info("  Downloaded \(index + 1)/\(entries.count) files...")
                    }
                } catch {
                    logger.warning("Failed to download \(entry.fileName): \(error)")
                }
            }

            // Write metadata.jsonl
            let metadataContent = metadataLines.joined(separator: "\n")
            try metadataContent.write(to: metadataPath, atomically: true, encoding: .utf8)

            logger.info(
                "Common Voice Japanese \(split.displayName) ready: \(downloadedCount)/\(entries.count) files")

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
