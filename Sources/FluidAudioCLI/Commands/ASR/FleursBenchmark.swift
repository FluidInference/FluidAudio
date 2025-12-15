#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation
import OSLog

/// FLEURS multilingual dataset benchmark for ASR evaluation
public class FLEURSBenchmark {

    private let logger = AppLogger(category: "FLEURSBenchmark")

    // Language codes mapped to Parakeet TDT v3 supported languages
    // Based on the model's training data with reported WER performance
    let supportedLanguages: [String: String] = [
        // Best performing languages (WER < 5%) - Using FLEURS language codes
        "en_us": "English (US)",  // 4.85% WER
        "es_419": "Spanish (Spain)",  // 3.45% WER (FLEURS code)
        "it_it": "Italian (Italy)",  // 3.00% WER
        "fr_fr": "French (France)",  // 5.15% WER
        "de_de": "German (Germany)",  // 5.04% WER

        // Good performance (WER 5-10%)
        "ru_ru": "Russian (Russia)",  // 5.51% WER
        "nl_nl": "Dutch (Netherlands)",  // 7.48% WER
        "pl_pl": "Polish (Poland)",  // 7.31% WER
        "uk_ua": "Ukrainian (Ukraine)",  // 6.79% WER
        "sk_sk": "Slovak (Slovakia)",  // 8.82% WER

        // Moderate performance (WER 10-15%)
        "cs_cz": "Czech (Czechia)",  // 11.01% WER
        "bg_bg": "Bulgarian (Bulgaria)",  // 12.64% WER
        "hr_hr": "Croatian (Croatia)",  // 12.46% WER
        "ro_ro": "Romanian (Romania)",  // 12.44% WER
        "fi_fi": "Finnish (Finland)",  // 13.21% WER

        // Lower performance (WER > 15%)
        "hu_hu": "Hungarian (Hungary)",  // 15.72% WER
        "sv_se": "Swedish (Sweden)",  // 15.08% WER
        "et_ee": "Estonian (Estonia)",  // 17.73% WER
        "da_dk": "Danish (Denmark)",  // 18.41% WER
        "lt_lt": "Lithuanian (Lithuania)",  // 20.35% WER
        "el_gr": "Greek (Greece)",  // 20.70% WER
        "mt_mt": "Maltese (Malta)",  // 20.46% WER
        "lv_lv": "Latvian (Latvia)",  // 22.84% WER
        "sl_si": "Slovenian (Slovenia)",  // 24.03% WER
    ]

    public struct FLEURSConfig {
        let languages: [String]
        let samplesPerLanguage: Int
        let outputFile: String
        let cacheDir: String
        let debugMode: Bool
    }

    public struct FLEURSSample {
        let audioPath: String
        let transcription: String
        let language: String
        let sampleId: String
    }

    public struct LanguageResults {
        let language: String
        let wer: Double
        let cer: Double
        let rtfx: Double
        let samplesProcessed: Int
        let samplesSkipped: Int
        let totalDuration: Double
        let processingTime: Double
    }

    public struct HighWERCase {
        let language: String
        let sampleId: String
        let reference: String
        let hypothesis: String
        let normalizedRef: String
        let normalizedHyp: String
        let wer: Double
        let duration: Double
        let audioPath: String
    }

    private let config: FLEURSConfig

    public init(config: FLEURSConfig) {
        self.config = config
    }

    /// Download FLEURS dataset for specified languages
    public func downloadFLEURS(languages: [String]) async throws {
        let cacheDir = URL(fileURLWithPath: config.cacheDir)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        for language in languages {
            guard supportedLanguages.keys.contains(language) else {
                logger.warning("Unsupported language: \(language)")
                continue
            }

            let languageDir = cacheDir.appendingPathComponent(language)

            // Check if already downloaded
            if FileManager.default.fileExists(atPath: languageDir.path) {
                do {
                    let contents = try FileManager.default.contentsOfDirectory(
                        at: languageDir, includingPropertiesForKeys: nil)
                    let audioFiles = contents.filter {
                        let ext = $0.pathExtension.lowercased()
                        return ext == "wav" || ext == "flac"
                    }
                    var validAudioFiles: [URL] = []
                    var corruptedFiles: [URL] = []
                    validAudioFiles.reserveCapacity(audioFiles.count)
                    corruptedFiles.reserveCapacity(audioFiles.count)

                    for file in audioFiles {
                        if isValidAudioFile(file) {
                            validAudioFiles.append(file)
                        } else {
                            corruptedFiles.append(file)
                        }
                    }

                    if !corruptedFiles.isEmpty {
                        logger.warning(
                            "Detected \(corruptedFiles.count) corrupted audio files for \(language); removing and re-downloading."
                        )
                        for file in corruptedFiles {
                            try? FileManager.default.removeItem(at: file)
                        }
                    }

                    // Determine how many samples we expect based on available transcripts
                    var expectedSamples = config.samplesPerLanguage
                    let transcriptPath = languageDir.appendingPathComponent("\(language).trans.txt")
                    if FileManager.default.fileExists(atPath: transcriptPath.path) {
                        let transcriptData = try String(contentsOf: transcriptPath)
                        let transcriptLines =
                            transcriptData.components(separatedBy: .newlines).filter { !$0.isEmpty }
                        if !transcriptLines.isEmpty {
                            expectedSamples =
                                config.samplesPerLanguage == Int.max
                                ? transcriptLines.count
                                : min(transcriptLines.count, config.samplesPerLanguage)
                        }
                    }

                    if corruptedFiles.isEmpty && validAudioFiles.count >= expectedSamples {
                        logger.info("FLEURS \(language) already downloaded")
                        continue
                    }

                    if corruptedFiles.isEmpty {
                        let expectedDescription =
                            expectedSamples == Int.max ? "the full dataset" : "\(expectedSamples)"
                        logger.warning(
                            "Found \(validAudioFiles.count) valid audio files for \(language); expected at least \(expectedDescription). Downloading remaining files."
                        )
                    }
                } catch {
                    // Directory exists but empty, re-download
                }
            }

            logger.info("Downloading FLEURS dataset for \(supportedLanguages[language]!)...")

            // Create language directory
            try FileManager.default.createDirectory(at: languageDir, withIntermediateDirectories: true)

            // Download sample metadata and audio files
            // Note: In a real implementation, you would fetch from Hugging Face API
            // For now, we'll create a structure for local testing
            try await downloadLanguageSamples(language: language, targetDir: languageDir)

            logger.info("Downloaded FLEURS \(language)")
        }
    }

    private func isValidAudioFile(_ url: URL) -> Bool {
        do {
            _ = try AVAudioFile(forReading: url)
            return true
        } catch {
            return false
        }
    }

    /// Download samples for a specific language
    private func downloadLanguageSamples(language: String, targetDir: URL) async throws {
        logger.info("Downloading FLEURS test set for \(language)...")

        // Check if already downloaded (look for .trans.txt file)
        let transFile = targetDir.appendingPathComponent("\(language).trans.txt")
        if FileManager.default.fileExists(atPath: transFile.path) {
            do {
                let contents = try String(contentsOf: transFile)
                let lines = contents.components(separatedBy: .newlines).filter { !$0.isEmpty }
                if lines.count > 10 {
                    let expectedCount =
                        config.samplesPerLanguage == Int.max
                        ? lines.count
                        : min(lines.count, config.samplesPerLanguage)
                    let existingAudio =
                        ((try? FileManager.default.contentsOfDirectory(
                            at: targetDir, includingPropertiesForKeys: nil
                        )) ?? [])
                        .filter {
                            let ext = $0.pathExtension.lowercased()
                            return (ext == "wav" || ext == "flac") && isValidAudioFile($0)
                        }
                    if existingAudio.count >= expectedCount {
                        logger.info("Found existing data with \(lines.count) samples")
                        return
                    } else {
                        logger.warning(
                            "Transcript lists \(lines.count) samples but only \(existingAudio.count) valid audio files found. Re-downloading."
                        )
                    }
                }
            } catch {
                // File exists but can't read, re-download
            }
        }

        // Download from Hugging Face dataset: FluidInference/fleurs
        logger.info("Downloading from HuggingFace: FluidInference/fleurs/\(language)...")

        let datasetRepo = "FluidInference/fleurs"

        do {
            // List files in the language directory using HuggingFace API (registry-aware with auth)
            let apiURL = try ModelRegistry.apiDatasets(datasetRepo, "tree/main/\(language)")
            let (listData, _) = try await DownloadUtils.fetchWithAuth(from: apiURL)

            guard let items = try JSONSerialization.jsonObject(with: listData) as? [[String: Any]] else {
                throw NSError(
                    domain: "FLEURSBenchmark",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Could not parse file list from HuggingFace"]
                )
            }

            // Find transcript file and audio files
            var audioFiles: [String] = []

            for item in items {
                guard let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String,
                    itemType == "file"
                else { continue }

                let fileName = URL(fileURLWithPath: itemPath).lastPathComponent

                if fileName == "\(language).trans.txt" {
                    // Download transcript file
                    let downloadURL = try ModelRegistry.resolveDataset(datasetRepo, itemPath)
                    let transData = try await DownloadUtils.fetchHuggingFaceFile(
                        from: downloadURL,
                        description: "\(language) transcript"
                    )
                    try transData.write(to: transFile, options: .atomic)

                    let transcriptContent = String(data: transData, encoding: .utf8) ?? ""
                    let lines = transcriptContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
                    logger.info("Downloaded \(lines.count) transcriptions")
                } else if fileName.hasSuffix(".wav") {
                    audioFiles.append(itemPath)
                }
            }

            // Download audio files based on samplesPerLanguage config
            let maxDownload =
                config.samplesPerLanguage == Int.max
                ? audioFiles.count : min(config.samplesPerLanguage, audioFiles.count)
            var downloadedCount = 0

            for audioPath in audioFiles.prefix(maxDownload) {
                let fileName = URL(fileURLWithPath: audioPath).lastPathComponent
                let audioFile = targetDir.appendingPathComponent(fileName)

                // Skip if already exists
                if FileManager.default.fileExists(atPath: audioFile.path) {
                    if isValidAudioFile(audioFile) {
                        downloadedCount += 1
                        continue
                    } else {
                        logger.warning("Detected corrupted placeholder for \(fileName). Re-downloading.")
                        try? FileManager.default.removeItem(at: audioFile)
                    }
                }

                // Download audio file
                let downloadURL = try ModelRegistry.resolveDataset(datasetRepo, audioPath)

                do {
                    let audioData = try await DownloadUtils.fetchHuggingFaceFile(
                        from: downloadURL,
                        description: "\(language)/\(fileName)"
                    )
                    try audioData.write(to: audioFile, options: .atomic)

                    guard isValidAudioFile(audioFile) else {
                        try? FileManager.default.removeItem(at: audioFile)
                        throw NSError(
                            domain: "FLEURSBenchmark",
                            code: 2,
                            userInfo: [
                                NSLocalizedDescriptionKey:
                                    "Downloaded data for \(fileName) is not valid audio. "
                                    + "The server likely returned a rate-limit placeholder."
                            ]
                        )
                    }
                    downloadedCount += 1

                    if downloadedCount % 10 == 0 {
                        logger.info("Downloaded \(downloadedCount)/\(maxDownload) audio files...")
                    }
                } catch {
                    logger.warning("Could not download \(fileName): \(error.localizedDescription)")
                }
            }

            logger.info("Downloaded \(downloadedCount) audio files")
            return

        } catch {
            logger.warning("Could not download from HuggingFace: \(error)")

            // Try fallback: Check if user has manually downloaded data
            let audioDir = targetDir.appendingPathComponent("audio")

            if FileManager.default.fileExists(atPath: audioDir.path) {
                // User has audio files, create a basic transcript file
                do {
                    let audioFiles = try FileManager.default.contentsOfDirectory(
                        at: audioDir,
                        includingPropertiesForKeys: [.isRegularFileKey]
                    ).filter { $0.pathExtension == "wav" || $0.pathExtension == "flac" }

                    if !audioFiles.isEmpty {
                        // Create a minimal transcript file with empty transcriptions
                        var transcriptLines: [String] = []
                        for audioFile in audioFiles {
                            let fileId = audioFile.deletingPathExtension().lastPathComponent
                            transcriptLines.append("\(fileId) ")  // Empty transcription
                        }

                        let transcriptContent = transcriptLines.joined(separator: "")
                        try transcriptContent.write(to: transFile, atomically: true, encoding: .utf8)

                        logger.info("Found \(audioFiles.count) audio files (no transcriptions)")
                        return
                    }
                } catch {
                    logger.warning("Error reading audio directory: \(error)")
                }
            }

            // No data available - throw error
            throw NSError(
                domain: "FLEURSBenchmark",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to download FLEURS data for \(language)"]
            )
        }
    }

    /// Load FLEURS samples for benchmarking
    public func loadFLEURSSamples(languages: [String]) throws -> [FLEURSSample] {
        var allSamples: [FLEURSSample] = []
        let cacheDir = URL(fileURLWithPath: config.cacheDir)

        for language in languages {
            let languageDir = cacheDir.appendingPathComponent(language)

            guard FileManager.default.fileExists(atPath: languageDir.path) else {
                logger.warning("No data found for \(language). Please download first.")
                continue
            }

            // Load transcriptions from .trans.txt file (LibriSpeech format)
            let transFile = languageDir.appendingPathComponent("\(language).trans.txt")
            var transcriptions: [String: String] = [:]

            if FileManager.default.fileExists(atPath: transFile.path) {
                do {
                    let content = try String(contentsOf: transFile)
                    let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }

                    for line in lines {
                        let parts = line.split(separator: " ", maxSplits: 1)
                        if parts.count >= 1 {
                            let fileId = String(parts[0])
                            let transcription = parts.count > 1 ? String(parts[1]) : ""
                            transcriptions[fileId] = transcription
                        }
                    }
                } catch {
                    logger.warning("Could not read transcriptions for \(language): \(error)")
                }
            }

            // Load audio files and match with transcriptions
            let audioFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: languageDir, includingPropertiesForKeys: nil
                )) ?? []

            let filteredAudioFiles =
                audioFiles
                .filter { $0.pathExtension == "wav" || $0.pathExtension == "flac" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
                .prefix(config.samplesPerLanguage == Int.max ? Int.max : config.samplesPerLanguage)

            for audioFile in filteredAudioFiles {
                let fileId = audioFile.deletingPathExtension().lastPathComponent
                let transcription = transcriptions[fileId] ?? ""

                let sample = FLEURSSample(
                    audioPath: audioFile.path,
                    transcription: transcription,
                    language: language,
                    sampleId: fileId
                )
                allSamples.append(sample)
            }

            if !filteredAudioFiles.isEmpty {
                logger.info("Loaded \(filteredAudioFiles.count) samples for \(language)")
            }
        }

        return allSamples
    }

    /// Load a single FLEURS sample by file path
    public func loadSingleFLEURSSample(filePath: String, language: String) throws -> FLEURSSample? {
        let cacheDir = URL(fileURLWithPath: config.cacheDir)
        let languageDir = cacheDir.appendingPathComponent(language)

        // Check if the file exists
        guard FileManager.default.fileExists(atPath: filePath) else {
            return nil
        }

        let audioFile = URL(fileURLWithPath: filePath)
        let fileId = audioFile.deletingPathExtension().lastPathComponent

        // Load transcriptions from .trans.txt file
        let transFile = languageDir.appendingPathComponent("\(language).trans.txt")
        var transcription = ""

        if FileManager.default.fileExists(atPath: transFile.path) {
            do {
                let content = try String(contentsOf: transFile)
                let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }

                for line in lines {
                    let parts = line.split(separator: " ", maxSplits: 1)
                    if parts.count >= 1 {
                        let lineFileId = String(parts[0])
                        if lineFileId == fileId {
                            transcription = parts.count > 1 ? String(parts[1]) : ""
                            break
                        }
                    }
                }
            } catch {
                logger.warning("⚠️ Could not read transcriptions for \(language): \(error)")
            }
        }

        return FLEURSSample(
            audioPath: filePath,
            transcription: transcription,
            language: language,
            sampleId: fileId
        )
    }

    /// Run multilingual benchmark
    public func runMultilingualBenchmark(
        asrManager: AsrManager
    ) async throws -> (results: [LanguageResults], allHighWERCases: [HighWERCase]) {
        logger.info("Starting FLEURS Multilingual ASR Benchmark")
        logger.info(String(repeating: "=", count: 50))

        var results: [LanguageResults] = []
        var allHighWERCases: [HighWERCase] = []

        // Download datasets if needed
        try await downloadFLEURS(languages: config.languages)

        // Load samples
        let samples = try loadFLEURSSamples(languages: config.languages)

        if samples.isEmpty {
            logger.warning("No samples found. Please ensure FLEURS data is available.")
            return ([], [])
        }

        logger.info("Processing \(samples.count) samples across \(config.languages.count) languages")

        // Group samples by language
        let languageGroups = Dictionary(grouping: samples, by: { $0.language })

        for (language, languageSamples) in languageGroups {
            logger.info("Processing \(supportedLanguages[language] ?? language)...")

            let (languageResult, highWERCases) = try await processLanguageSamples(
                samples: languageSamples,
                language: language,
                asrManager: asrManager
            )

            results.append(languageResult)
            allHighWERCases.append(contentsOf: highWERCases)

            // Print language summary
            let skippedInfo = languageResult.samplesSkipped > 0 ? ", \(languageResult.samplesSkipped) skipped" : ""
            logger.info(
                "\(language): WER=\(String(format: "%.1f", languageResult.wer * 100))%, CER=\(String(format: "%.1f", languageResult.cer * 100))%, RTFx=\(String(format: "%.1f", languageResult.rtfx))x (\(languageResult.samplesProcessed) processed\(skippedInfo))"
            )
        }

        return (results, allHighWERCases)
    }

    /// Process samples for a specific language
    private func processLanguageSamples(
        samples: [FLEURSSample],
        language: String,
        asrManager: AsrManager
    ) async throws -> (LanguageResults, [HighWERCase]) {
        var totalWER = 0.0
        var totalCER = 0.0
        var totalDuration = 0.0
        var totalProcessingTime = 0.0
        var processedCount = 0
        var skippedCount = 0

        // Track high WER cases for analysis
        var highWERCases: [HighWERCase] = []

        for (_, sample) in samples.enumerated() {
            // Skip if audio file doesn't exist
            guard FileManager.default.fileExists(atPath: sample.audioPath) else {
                logger.warning("Audio file not found: \(sample.audioPath)")
                continue
            }

            do {
                // Load audio first
                let audioSamples: [Float]

                do {
                    audioSamples = try AudioConverter().resampleAudioFile(path: sample.audioPath)
                } catch {
                    // Continue to next sample instead of failing the entire benchmark
                    skippedCount += 1
                    continue
                }

                let audioDuration = Double(audioSamples.count) / 16000.0
                logger.debug(
                    "\t Processing \(sample.audioPath) Duration: \(String(format: "%.2f", audioDuration))s with samples: \(audioSamples.count)"
                )
                // Measure only inference time for accurate RTFx calculation
                let url = URL(fileURLWithPath: sample.audioPath)
                let inferenceStartTime = Date()
                let result = try await asrManager.transcribe(url)
                let processingTime = Date().timeIntervalSince(inferenceStartTime)

                // Calculate metrics if reference transcription is available
                if !sample.transcription.isEmpty {
                    let metrics = calculateMetrics(
                        hypothesis: result.text,
                        reference: sample.transcription
                    )
                    totalWER += metrics.wer
                    totalCER += metrics.cer

                    // Track cases with high WER for analysis
                    if metrics.wer > ASRConstants.highWERThreshold {
                        let normalizedRef = TextNormalizer.normalize(sample.transcription)
                        let normalizedHyp = TextNormalizer.normalize(result.text)
                        highWERCases.append(
                            HighWERCase(
                                language: language,
                                sampleId: sample.sampleId,
                                reference: sample.transcription,
                                hypothesis: result.text,
                                normalizedRef: normalizedRef,
                                normalizedHyp: normalizedHyp,
                                wer: metrics.wer,
                                duration: audioDuration,
                                audioPath: sample.audioPath
                            ))
                    }
                }

                totalDuration += audioDuration
                totalProcessingTime += processingTime
                processedCount += 1

                if config.debugMode {
                    logger.debug("    Hypothesis: \(result.text)")
                    if !sample.transcription.isEmpty {
                        logger.debug("    Reference:  \(sample.transcription)")
                    }
                }

            } catch {
                logger.warning("Transcription error for \(sample.sampleId): \(error.localizedDescription)")
            }
        }

        // Calculate averages
        let avgWER = processedCount > 0 ? totalWER / Double(processedCount) : 0.0
        let avgCER = processedCount > 0 ? totalCER / Double(processedCount) : 0.0
        let rtfx = totalProcessingTime > 0 ? totalDuration / totalProcessingTime : 0.0

        return (
            LanguageResults(
                language: language,
                wer: avgWER,
                cer: avgCER,
                rtfx: rtfx,
                samplesProcessed: processedCount,
                samplesSkipped: skippedCount,
                totalDuration: totalDuration,
                processingTime: totalProcessingTime
            ), highWERCases
        )
    }

    /// Calculate WER and CER metrics
    private func calculateMetrics(hypothesis: String, reference: String) -> (wer: Double, cer: Double) {
        let metrics = WERCalculator.calculateWERAndCER(hypothesis: hypothesis, reference: reference)
        return (metrics.wer, metrics.cer)
    }

    /// Generate inline diff with full lines and highlighted differences
    private func generateInlineDiff(reference: [String], hypothesis: [String]) -> (String, String) {
        let m = reference.count
        let n = hypothesis.count

        // Handle empty hypothesis or reference
        if n == 0 {
            let supportsColor = ProcessInfo.processInfo.environment["TERM"] != nil
            let redColor = supportsColor ? "\u{001B}[31m" : "["
            let resetColor = supportsColor ? "\u{001B}[0m" : "]"
            let refString = reference.map { "\(redColor)\($0)\(resetColor)" }.joined(separator: " ")
            let hypString = ""
            return (refString, hypString)
        }
        if m == 0 {
            let supportsColor = ProcessInfo.processInfo.environment["TERM"] != nil
            let greenColor = supportsColor ? "\u{001B}[32m" : "["
            let resetColor = supportsColor ? "\u{001B}[0m" : "]"
            let refString = ""
            let hypString = hypothesis.map { "\(greenColor)\($0)\(resetColor)" }.joined(separator: " ")
            return (refString, hypString)
        }

        // Create DP table for edit distance with backtracking
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        // Initialize base cases
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        // Fill DP table
        for i in 1...m {
            for j in 1...n {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] =
                        1
                        + min(
                            dp[i - 1][j],  // deletion
                            dp[i][j - 1],  // insertion
                            dp[i - 1][j - 1]  // substitution
                        )
                }
            }
        }

        // Check if terminal supports colors
        let supportsColor = ProcessInfo.processInfo.environment["TERM"] != nil
        let redColor = supportsColor ? "\u{001B}[31m" : "["
        let greenColor = supportsColor ? "\u{001B}[32m" : "["
        let resetColor = supportsColor ? "\u{001B}[0m" : "]"

        // Backtrack to identify differences
        var i = m
        var j = n
        var refDiffWords: [(String, Bool)] = []  // (word, isDifferent)
        var hypDiffWords: [(String, Bool)] = []  // (word, isDifferent)

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] {
                // Match
                refDiffWords.insert((reference[i - 1], false), at: 0)
                hypDiffWords.insert((hypothesis[j - 1], false), at: 0)
                i -= 1
                j -= 1
            } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
                // Substitution
                refDiffWords.insert((reference[i - 1], true), at: 0)
                hypDiffWords.insert((hypothesis[j - 1], true), at: 0)
                i -= 1
                j -= 1
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
                // Deletion (word in reference but not in hypothesis)
                refDiffWords.insert((reference[i - 1], true), at: 0)
                i -= 1
            } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
                // Insertion (word in hypothesis but not in reference)
                hypDiffWords.insert((hypothesis[j - 1], true), at: 0)
                j -= 1
            } else {
                break
            }
        }

        // Build the formatted strings
        var refString = ""
        var hypString = ""

        for (word, isDifferent) in refDiffWords {
            if !refString.isEmpty { refString += " " }
            if isDifferent {
                refString += "\(redColor)\(word)\(resetColor)"
            } else {
                refString += word
            }
        }

        for (word, isDifferent) in hypDiffWords {
            if !hypString.isEmpty { hypString += " " }
            if isDifferent {
                hypString += "\(greenColor)\(word)\(resetColor)"
            } else {
                hypString += word
            }
        }

        return (refString, hypString)
    }

    /// Print all high WER cases collected across all languages, sorted by WER descending
    public func printAllHighWERCases(_ allHighWERCases: [HighWERCase]) {
        guard !allHighWERCases.isEmpty else {
            logger.info("No high WER cases (> \(Int(ASRConstants.highWERThreshold * 100))%) detected.")
            return
        }

        logger.info(
            "All High WER Cases (>\(Int(ASRConstants.highWERThreshold * 100))%) Across Languages (sorted by WER):")
        logger.info(String(repeating: "=", count: 80))

        // Sort all cases by WER descending, then by language
        let sortedCases = allHighWERCases.sorted {
            if $0.wer != $1.wer {
                return $0.wer > $1.wer
            } else {
                return $0.language < $1.language
            }
        }

        for sample in sortedCases {
            let langName = supportedLanguages[sample.language] ?? sample.language
            let werPercent = sample.wer * 100
            logger.info(
                "Language: \(langName) | File: \(sample.sampleId) (WER: \(String(format: "%.1f", werPercent))%, Duration: \(String(format: "%.2f", sample.duration))s)"
            )
            logger.info("Path: \(sample.audioPath)")
            logger.info(String(repeating: "-", count: 40))

            // Normalize the texts for comparison
            let normalizedReference = sample.normalizedRef
            let normalizedHypothesis = sample.normalizedHyp

            let refWords = normalizedReference.components(separatedBy: .whitespacesAndNewlines).filter {
                !$0.isEmpty
            }
            let hypWords = normalizedHypothesis.components(separatedBy: .whitespacesAndNewlines).filter {
                !$0.isEmpty
            }

            // Generate inline diff
            let (referenceDiff, hypothesisDiff) = generateInlineDiff(reference: refWords, hypothesis: hypWords)

            logger.info("Normalized Reference:\t\(referenceDiff)")
            logger.info("Normalized Hypothesis:\t\(hypothesisDiff)")
            logger.info("Original Hypothesis:\t\(sample.hypothesis)")
            logger.info(String(repeating: "-", count: 40))
        }
        logger.info(String(repeating: "=", count: 80))
    }

    /// Save results to JSON
    public func saveResults(_ results: [LanguageResults], to outputPath: String) throws {
        // Helper function to sanitize NaN and Infinity values
        func sanitizeDouble(_ value: Double) -> Double {
            if value.isNaN { return 0.0 }
            if value.isInfinite { return 0.0 }
            return value
        }

        let output: [String: Any] = [
            "benchmark": "FLEURS Multilingual ASR",
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "config": [
                "languages": config.languages,
                "samplesPerLanguage": config.samplesPerLanguage,
            ],
            "results": results.map { result in
                [
                    "language": result.language,
                    "languageName": supportedLanguages[result.language] ?? result.language,
                    "wer": sanitizeDouble(result.wer),
                    "cer": sanitizeDouble(result.cer),
                    "rtfx": sanitizeDouble(result.rtfx),
                    "samplesProcessed": result.samplesProcessed,
                    "samplesSkipped": result.samplesSkipped,
                    "totalDuration": result.totalDuration,
                    "processingTime": result.processingTime,
                ]
            },
            "summary": [
                "averageWER": sanitizeDouble(results.reduce(0.0) { $0 + $1.wer } / Double(results.count)),
                "averageCER": sanitizeDouble(results.reduce(0.0) { $0 + $1.cer } / Double(results.count)),
                "averageRTFx": sanitizeDouble(results.reduce(0.0) { $0 + $1.rtfx } / Double(results.count)),
                "totalSamples": results.reduce(0) { $0 + $1.samplesProcessed },
                "totalSkipped": results.reduce(0) { $0 + $1.samplesSkipped },
                "totalDuration": sanitizeDouble(results.reduce(0.0) { $0 + $1.totalDuration }),
                "totalProcessingTime": sanitizeDouble(results.reduce(0.0) { $0 + $1.processingTime }),
            ],
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: outputPath))
    }
}

/// CLI entry point for FLEURS benchmark
extension FLEURSBenchmark {

    public static func runCLI(arguments: [String]) async {
        // Get instance to access supportedLanguages
        let tempBenchmark = FLEURSBenchmark(
            config: FLEURSConfig(
                languages: [], samplesPerLanguage: 0, outputFile: "", cacheDir: "", debugMode: false))

        var languages: [String]? = nil  // Will be set to all languages if not specified
        var samplesPerLanguage = Int.max  // Default to all samples
        var outputFile = "fleurs_benchmark_results.json"
        var cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/FluidAudio/FLEURS").path
        var debugMode = false
        var singleFile: String? = nil

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--languages":
                if i + 1 < arguments.count {
                    let languageArg = arguments[i + 1].lowercased()
                    if languageArg == "all" {
                        languages = nil  // Will be set to all languages below
                    } else {
                        languages = arguments[i + 1].split(separator: ",").map(String.init)
                    }
                    i += 1
                }
            case "--samples":
                if i + 1 < arguments.count {
                    let samplesArg = arguments[i + 1].lowercased()
                    if samplesArg == "all" {
                        samplesPerLanguage = Int.max  // Will process all available files
                    } else {
                        samplesPerLanguage = Int(arguments[i + 1]) ?? 10
                    }
                    i += 1
                }
            case "--single-file":
                if i + 1 < arguments.count {
                    singleFile = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--cache-dir":
                if i + 1 < arguments.count {
                    cacheDir = arguments[i + 1]
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                let logger = AppLogger(category: "FLEURSBenchmark")
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Handle single file processing
        if let singleFileName = singleFile {
            await runSingleFile(
                fileName: singleFileName,
                cacheDir: cacheDir,
                outputFile: outputFile,
                debugMode: debugMode,
                supportedLanguages: tempBenchmark.supportedLanguages
            )
            return
        }

        // If no languages specified, use all supported languages
        let finalLanguages = languages ?? Array(tempBenchmark.supportedLanguages.keys).sorted()

        let cliLogger = AppLogger(category: "FLEURSBenchmark")
        cliLogger.info("FLEURS Multilingual ASR Benchmark")
        cliLogger.info(String(repeating: "=", count: 50))
        cliLogger.info(
            "Languages: \(finalLanguages.count == tempBenchmark.supportedLanguages.count ? "all (\(finalLanguages.count) languages)" : finalLanguages.joined(separator: ", "))"
        )
        cliLogger.info("Samples per language: \(samplesPerLanguage == Int.max ? "all" : String(samplesPerLanguage))")
        cliLogger.info("Output file: \(outputFile)")
        cliLogger.info("Cache directory: \(cacheDir)")

        // Create configuration
        let config = FLEURSConfig(
            languages: finalLanguages,
            samplesPerLanguage: samplesPerLanguage,
            outputFile: outputFile,
            cacheDir: cacheDir,
            debugMode: debugMode
        )

        let benchmark = FLEURSBenchmark(config: config)

        // Initialize ASR manager
        let asrConfig = ASRConfig(
            tdtConfig: TdtConfig()  // Uses default config
        )

        let asrManager = AsrManager(config: asrConfig)

        do {
            cliLogger.info("Initializing ASR system...")
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)
            cliLogger.info("ASR system initialized")

            // Run benchmark
            let (results, allHighWERCases) = try await benchmark.runMultilingualBenchmark(asrManager: asrManager)

            // Save results
            try benchmark.saveResults(results, to: outputFile)

            benchmark.printAllHighWERCases(allHighWERCases)
            cliLogger.info("Results saved to \(outputFile)")
            // Print summary
            cliLogger.info(String(repeating: "=", count: 80))
            cliLogger.info("FLEURS BENCHMARK SUMMARY")
            cliLogger.info(String(repeating: "=", count: 80))

            // Check if we have results to display
            guard !results.isEmpty else {
                cliLogger.warning("No results to display - benchmark produced no valid results")
                return
            }

            // Print table header
            print("")
            print(
                "Language".padding(toLength: 25, withPad: " ", startingAt: 0) + " | "
                    + "WER%".padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                    + "CER%".padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                    + "RTFx".padding(toLength: 7, withPad: " ", startingAt: 0) + " | "
                    + "Duration".padding(toLength: 8, withPad: " ", startingAt: 0) + " | "
                    + "Processed".padding(toLength: 9, withPad: " ", startingAt: 0) + " | "
                    + "Skipped".padding(toLength: 7, withPad: " ", startingAt: 0))
            print(String(repeating: "-", count: 89))

            for result in results.sorted(by: { lhs, rhs in
                let lhsName = benchmark.supportedLanguages[lhs.language] ?? lhs.language
                let rhsName = benchmark.supportedLanguages[rhs.language] ?? rhs.language
                return lhsName < rhsName
            }) {
                let langName = benchmark.supportedLanguages[result.language] ?? result.language
                let truncatedName = String(langName.prefix(24))
                let werStr = String(format: "%.1f", result.wer * 100)
                let cerStr = String(format: "%.1f", result.cer * 100)
                let rtfxStr = String(format: "%.1f", result.rtfx)
                let durationStr = String(format: "%.1fs", result.totalDuration)
                let processedStr = String(result.samplesProcessed)
                let skippedStr = result.samplesSkipped > 0 ? String(result.samplesSkipped) : "-"

                print(
                    truncatedName.padding(toLength: 25, withPad: " ", startingAt: 0) + " | "
                        + werStr.padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                        + cerStr.padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                        + rtfxStr.padding(toLength: 7, withPad: " ", startingAt: 0) + " | "
                        + durationStr.padding(toLength: 8, withPad: " ", startingAt: 0) + " | "
                        + processedStr.padding(toLength: 9, withPad: " ", startingAt: 0) + " | "
                        + skippedStr.padding(toLength: 7, withPad: " ", startingAt: 0))
            }

            let avgWER = results.reduce(0.0) { $0 + $1.wer } / Double(results.count)
            let avgCER = results.reduce(0.0) { $0 + $1.cer } / Double(results.count)
            let avgRTFx = results.reduce(0.0) { $0 + $1.rtfx } / Double(results.count)
            let totalDuration = results.reduce(0.0) { $0 + $1.totalDuration }
            let totalProcessed = results.reduce(0) { $0 + $1.samplesProcessed }
            let totalSkipped = results.reduce(0) { $0 + $1.samplesSkipped }

            print(String(repeating: "-", count: 89))
            let avgWerStr = String(format: "%.1f", avgWER * 100)
            let avgCerStr = String(format: "%.1f", avgCER * 100)
            let avgRtfxStr = String(format: "%.1f", avgRTFx)
            let totalDurationStr = String(format: "%.1fs", totalDuration)
            let totalProcessedStr = String(totalProcessed)
            let totalSkippedStr = totalSkipped > 0 ? String(totalSkipped) : "-"

            print(
                "AVERAGE".padding(toLength: 25, withPad: " ", startingAt: 0) + " | "
                    + avgWerStr.padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                    + avgCerStr.padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                    + avgRtfxStr.padding(toLength: 7, withPad: " ", startingAt: 0) + " | "
                    + totalDurationStr.padding(toLength: 8, withPad: " ", startingAt: 0) + " | "
                    + totalProcessedStr.padding(toLength: 9, withPad: " ", startingAt: 0) + " | "
                    + totalSkippedStr.padding(toLength: 7, withPad: " ", startingAt: 0))

            if totalSkipped > 0 {
                print("Note: \(totalSkipped) samples were skipped due to audio loading errors")
            }

        } catch {
            print("Benchmark failed: \(error)")
            exit(1)
        }
    }

    private static func runSingleFile(
        fileName: String,
        cacheDir: String,
        outputFile: String,
        debugMode: Bool,
        supportedLanguages: [String: String]
    ) async {
        let cliLogger = AppLogger(category: "FLEURSBenchmark")
        cliLogger.info("FLEURS Single File ASR Test")
        cliLogger.info(String(repeating: "=", count: 50))
        cliLogger.info("File: \(fileName)")

        // Find the file across all language directories
        guard
            let (filePath, language) = findFileInLanguageDirectories(
                fileName: fileName,
                cacheDir: cacheDir,
                supportedLanguages: supportedLanguages
            )
        else {
            cliLogger.error("File '\(fileName)' not found in any language directory")
            cliLogger.info("Searched in: \(cacheDir)")
            cliLogger.info("Supported languages: \(Array(supportedLanguages.keys).sorted().joined(separator: ", "))")
            exit(1)
        }

        let languageName = supportedLanguages[language] ?? language
        cliLogger.info("Language: \(languageName) (\(language))")
        cliLogger.info("Path: \(filePath)")

        // Create configuration for single language
        let config = FLEURSConfig(
            languages: [language],
            samplesPerLanguage: 1,
            outputFile: outputFile,
            cacheDir: cacheDir,
            debugMode: debugMode
        )

        let benchmark = FLEURSBenchmark(config: config)

        // Initialize ASR manager
        let asrConfig = ASRConfig(
            tdtConfig: TdtConfig()
        )

        let asrManager = AsrManager(config: asrConfig)

        do {
            cliLogger.info("Initializing ASR system...")
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)
            cliLogger.info("ASR system initialized")

            // Load the single sample directly
            let sample = try benchmark.loadSingleFLEURSSample(filePath: filePath, language: language)

            guard let sample = sample else {
                cliLogger.error("Could not load sample for file: \(fileName)")
                exit(1)
            }

            cliLogger.info("Processing single file...")
            cliLogger.info("Sample ID: \(sample.sampleId)")
            if !sample.transcription.isEmpty {
                cliLogger.info("Reference: \(sample.transcription)")
            } else {
                cliLogger.info("Reference: <no transcription available>")
            }

            // Process the single file directly
            let (result, highWERCase) = try await processSingleSample(
                sample: sample,
                language: language,
                asrManager: asrManager,
                debugMode: debugMode
            )

            // Save results
            try benchmark.saveResults([result], to: outputFile)

            // Display results
            cliLogger.info("Results:")
            let werPercent = result.wer * 100
            let cerPercent = result.cer * 100
            let rtfx = result.rtfx
            let duration = result.totalDuration
            let processingTime = result.processingTime

            cliLogger.info("  WER: \(String(format: "%.1f", werPercent))%")
            cliLogger.info("  CER: \(String(format: "%.1f", cerPercent))%")
            cliLogger.info("  RTFx: \(String(format: "%.1fx", rtfx))")
            cliLogger.info("  Duration: \(String(format: "%.2f", duration))s")
            cliLogger.info("  Processing time: \(String(format: "%.3f", processingTime))s")

            // Show high WER case if any
            if let highWERCase = highWERCase {
                cliLogger.warning("High WER detected:")
                benchmark.printAllHighWERCases([highWERCase])
            }

            cliLogger.info("Results saved to \(outputFile)")

        } catch {
            cliLogger.error("Single file test failed: \(error)")
            exit(1)
        }
    }

    private static func findFileInLanguageDirectories(
        fileName: String,
        cacheDir: String,
        supportedLanguages: [String: String]
    ) -> (filePath: String, language: String)? {
        let cacheDirURL = URL(fileURLWithPath: cacheDir)

        // First try to extract language from filename (e.g., "es_419_0001.wav" -> "es_419")
        let possibleLanguage = extractLanguageFromFileName(fileName)

        // If we can extract the language, check that directory first
        if let language = possibleLanguage, supportedLanguages.keys.contains(language) {
            let languageDir = cacheDirURL.appendingPathComponent(language)
            let targetFile = languageDir.appendingPathComponent(fileName)

            if FileManager.default.fileExists(atPath: targetFile.path) {
                return (targetFile.path, language)
            }
        }

        // Fallback: search all language directories
        for language in supportedLanguages.keys {
            let languageDir = cacheDirURL.appendingPathComponent(language)
            let targetFile = languageDir.appendingPathComponent(fileName)

            if FileManager.default.fileExists(atPath: targetFile.path) {
                return (targetFile.path, language)
            }
        }

        return nil
    }

    private static func processSingleSample(
        sample: FLEURSSample,
        language: String,
        asrManager: AsrManager,
        debugMode: Bool
    ) async throws -> (LanguageResults, HighWERCase?) {
        // Local logger for static context
        let logger = AppLogger(category: "FLEURSBenchmark")
        // Load audio
        let audioSamples: [Float]
        do {
            audioSamples = try AudioConverter().resampleAudioFile(path: sample.audioPath)
        } catch {
            throw NSError(
                domain: "FLEURSBenchmark",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load audio file: \(error.localizedDescription)"]
            )
        }

        let audioDuration = Double(audioSamples.count) / 16000.0
        logger.info("  Duration: \(String(format: "%.2f", audioDuration))s")

        // Measure only inference time for accurate RTFx calculation
        let url = URL(fileURLWithPath: sample.audioPath)
        let inferenceStartTime = Date()
        let result = try await asrManager.transcribe(url)
        let processingTime = Date().timeIntervalSince(inferenceStartTime)

        logger.info("  Hypothesis: \(result.text)")

        // Calculate metrics if reference transcription is available
        var wer = 0.0
        var cer = 0.0
        var highWERCase: HighWERCase? = nil

        if !sample.transcription.isEmpty {
            let normalizedHyp = TextNormalizer.normalize(result.text)
            let normalizedRef = TextNormalizer.normalize(sample.transcription)

            // Word-level
            let hypWords = normalizedHyp.split(separator: " ").map(String.init)
            let refWords = normalizedRef.split(separator: " ").map(String.init)
            let wordDistance = levenshteinDistance(hypWords, refWords)
            wer = refWords.isEmpty ? 0.0 : Double(wordDistance) / Double(refWords.count)

            // Character-level
            let hypChars = Array(normalizedHyp.replacingOccurrences(of: " ", with: ""))
            let refChars = Array(normalizedRef.replacingOccurrences(of: " ", with: ""))
            let charDistance = levenshteinDistance(hypChars.map(String.init), refChars.map(String.init))
            cer = refChars.isEmpty ? 0.0 : Double(charDistance) / Double(refChars.count)

            // Track high WER case for analysis
            if wer > ASRConstants.highWERThreshold {
                highWERCase = HighWERCase(
                    language: language,
                    sampleId: sample.sampleId,
                    reference: sample.transcription,
                    hypothesis: result.text,
                    normalizedRef: normalizedRef,
                    normalizedHyp: normalizedHyp,
                    wer: wer,
                    duration: audioDuration,
                    audioPath: sample.audioPath
                )
            }

            logger.info("Normalized Reference:\t\(normalizedRef)")
            logger.info("Normalized Hypothesis:\t\(normalizedHyp)")
        }

        let rtfx = processingTime > 0 ? audioDuration / processingTime : 0.0

        let languageResult = LanguageResults(
            language: language,
            wer: wer,
            cer: cer,
            rtfx: rtfx,
            samplesProcessed: 1,
            samplesSkipped: 0,
            totalDuration: audioDuration,
            processingTime: processingTime
        )

        return (languageResult, highWERCase)
    }

    private static func levenshteinDistance(_ s1: [String], _ s2: [String]) -> Int {
        let m = s1.count
        let n = s2.count

        if m == 0 { return n }
        if n == 0 { return m }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                let cost = s1[i - 1] == s2[j - 1] ? 0 : 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  // deletion
                    dp[i][j - 1] + 1,  // insertion
                    dp[i - 1][j - 1] + cost  // substitution
                )
            }
        }

        return dp[m][n]
    }

    private static func extractLanguageFromFileName(_ fileName: String) -> String? {
        // Remove file extension
        let baseName = URL(fileURLWithPath: fileName).deletingPathExtension().lastPathComponent

        // Look for patterns like "es_419_0001" -> "es_419"
        let parts = baseName.split(separator: "_")

        // Try different combinations for language extraction
        if parts.count >= 3 {
            // Pattern: lang_region_number (e.g., "es_419_0001")
            let possibleLang = "\(parts[0])_\(parts[1])"
            return String(possibleLang)
        } else if parts.count >= 2 {
            // Pattern: lang_region or lang_number (e.g., "en_us" or "en_001")
            let possibleLang = "\(parts[0])_\(parts[1])"
            return String(possibleLang)
        }

        return nil
    }

    private static func printUsage() {
        // Build available languages list dynamically to avoid drift
        let tmp = FLEURSBenchmark(
            config: FLEURSConfig(languages: [], samplesPerLanguage: 0, outputFile: "", cacheDir: "", debugMode: false)
        )
        let langs = Array(tmp.supportedLanguages.keys).sorted()
        let langsJoined = langs.joined(separator: ", ")
        let count = langs.count

        let logger = AppLogger(category: "FLEURSBenchmark")
        logger.info(
            """

            FLEURS Multilingual Benchmark Usage:
                fluidaudio fleurs-benchmark [options]

            Options:
                --languages <list>        Comma-separated list of language codes
                                         (default: all \(count) supported languages)
                                         Available: \(langsJoined)
                --samples <number|all>    Number of samples per language (default: all)
                --single-file <filename>  Test a single audio file (auto-detects language)
                --output <file>          Output JSON file path
                --cache-dir <path>       Directory for caching FLEURS data
                --debug                  Enable debug logging
                --help, -h              Show this help message

            Examples:
                # Test all \(count) languages with all available samples (~350 per language)
                fluidaudio fleurs-benchmark

                # Test specific languages only
                fluidaudio fleurs-benchmark --languages en_us,fr_fr,de_de,es_es

                # Quick test with only 10 samples per language
                fluidaudio fleurs-benchmark --samples 10

                # Test a single file (language auto-detected from filename)
                fluidaudio fleurs-benchmark --single-file es_419_0001.wav

                # Debug mode with custom output
                fluidaudio fleurs-benchmark --debug --output my_results.json

            Note:
                The FLEURS dataset will be downloaded automatically if not present.
                Audio files should be placed in the cache directory organized by language.

            """
        )
    }
}

// Helper extension for String repetition
extension String {
    static func * (lhs: String, rhs: Int) -> String {
        return String(repeating: lhs, count: rhs)
    }
}

#endif
