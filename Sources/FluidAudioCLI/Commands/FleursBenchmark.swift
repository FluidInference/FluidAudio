#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation
import OSLog

/// FLEURS multilingual dataset benchmark for ASR evaluation
@available(macOS 13.0, *)
public class FLEURSBenchmark {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "FLEURSBenchmark")

    // Language codes mapped to Parakeet TDT v3 supported languages
    // Based on the model's training data with reported WER performance
    let supportedLanguages: [String: String] = [
        // Best performing languages (WER < 5%) - Using FLEURS language codes
        "en_us": "English (US)",  // 4.85% WER
        "es_es": "Spanish (Spain)",  // 3.45% WER (FLEURS code)
        "it_it": "Italian (Italy)",  // 3.00% WER
        "fr_fr": "French (France)",  // 5.15% WER
        "de_de": "German (Germany)",  // 5.04% WER
        "pt_pt": "Portuguese (Portugal)",  // 4.76% WER (FLEURS code)

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
        let totalDuration: Double
        let processingTime: Double
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
                print("⚠️ Unsupported language: \(language)")
                continue
            }

            let languageDir = cacheDir.appendingPathComponent(language)

            // Check if already downloaded
            if FileManager.default.fileExists(atPath: languageDir.path) {
                do {
                    let contents = try FileManager.default.contentsOfDirectory(
                        at: languageDir, includingPropertiesForKeys: nil)
                    if contents.count > 10 {  // Assume downloaded if has files
                        print("✓ FLEURS \(language) already downloaded")
                        continue
                    }
                } catch {
                    // Directory exists but empty, re-download
                }
            }

            print("📥 Downloading FLEURS dataset for \(supportedLanguages[language]!)...")

            // Create language directory
            try FileManager.default.createDirectory(at: languageDir, withIntermediateDirectories: true)

            // Download sample metadata and audio files
            // Note: In a real implementation, you would fetch from Hugging Face API
            // For now, we'll create a structure for local testing
            try await downloadLanguageSamples(language: language, targetDir: languageDir)

            print("✓ Downloaded FLEURS \(language)")
        }
    }

    /// Download samples for a specific language
    private func downloadLanguageSamples(language: String, targetDir: URL) async throws {
        print("  📥 Downloading FLEURS test set for \(language)...")

        // Check if already downloaded (look for .trans.txt file)
        let transFile = targetDir.appendingPathComponent("\(language).trans.txt")
        if FileManager.default.fileExists(atPath: transFile.path) {
            do {
                let contents = try String(contentsOf: transFile)
                let lines = contents.components(separatedBy: .newlines).filter { !$0.isEmpty }
                if lines.count > 10 {
                    print("    ✓ Found existing data with \(lines.count) samples")
                    return
                }
            } catch {
                // File exists but can't read, re-download
            }
        }

        // Download from Hugging Face dataset: FluidInference/fleurs
        print("    📥 Downloading from HuggingFace: FluidInference/fleurs/\(language)...")

        // Use the existing HuggingFace download infrastructure
        let datasetRepo = "FluidInference/fleurs"
        // Use the official API with path query to list files in subfolder
        let apiBaseURL = "https://huggingface.co/api/datasets/\(datasetRepo)/tree/main?path=\(language)"

        do {
            // List files in the language directory using HuggingFace API
            guard let apiURL = URL(string: apiBaseURL) else {
                throw NSError(
                    domain: "FleursBenchmark", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Invalid API URL"])
            }

            let (data, response) = try await DownloadUtils.sharedSession.data(from: apiURL)

            guard let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 200
            else {
                throw URLError(.badServerResponse)
            }

            // Parse the JSON response to get file list
            struct HFFile: Codable {
                let type: String
                let path: String
                let size: Int?
            }

            let files = try JSONDecoder().decode([HFFile].self, from: data)

            // Find transcript file and audio files
            var audioFiles: [String] = []

            for file in files where file.type == "file" {
                let fileName = URL(fileURLWithPath: file.path).lastPathComponent
                if fileName == "\(language).trans.txt" {
                    // Download transcript file
                    let downloadURL = URL(
                        string: "https://huggingface.co/datasets/\(datasetRepo)/resolve/main/\(file.path)")!
                    let (transData, _) = try await DownloadUtils.sharedSession.data(from: downloadURL)
                    try transData.write(to: transFile)

                    let transcriptContent = String(data: transData, encoding: .utf8) ?? ""
                    let lines = transcriptContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
                    print("    ✓ Downloaded \(lines.count) transcriptions")
                } else if fileName.hasSuffix(".wav") {
                    audioFiles.append(file.path)
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
                    downloadedCount += 1
                    continue
                }

                // Download audio file using HuggingFace infrastructure
                let downloadURL = URL(
                    string: "https://huggingface.co/datasets/\(datasetRepo)/resolve/main/\(audioPath)")!

                do {
                    let (audioData, _) = try await DownloadUtils.sharedSession.data(from: downloadURL)
                    try audioData.write(to: audioFile)
                    downloadedCount += 1

                    if downloadedCount % 10 == 0 {
                        print("      Downloaded \(downloadedCount)/\(maxDownload) audio files...")
                    }
                } catch {
                    print("      ⚠️ Could not download \(fileName): \(error.localizedDescription)")
                }
            }

            print("    ✓ Downloaded \(downloadedCount) audio files")
            return

        } catch {
            print("    ⚠️ Could not download from HuggingFace: \(error)")

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

                        let transcriptContent = transcriptLines.joined(separator: "\n")
                        try transcriptContent.write(to: transFile, atomically: true, encoding: .utf8)

                        print("    ✓ Found \(audioFiles.count) audio files (no transcriptions)")
                        return
                    }
                } catch {
                    print("  ⚠️ Error reading audio directory: \(error)")
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
                print("⚠️ No data found for \(language). Please download first.")
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
                    print("⚠️ Could not read transcriptions for \(language): \(error)")
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
                print("  ✓ Loaded \(filteredAudioFiles.count) samples for \(language)")
            }
        }

        return allSamples
    }

    /// Run multilingual benchmark
    public func runMultilingualBenchmark(asrManager: AsrManager) async throws -> [LanguageResults] {
        print("\n🌍 Starting FLEURS Multilingual ASR Benchmark")
        print("=" * 50)

        var results: [LanguageResults] = []

        // Download datasets if needed
        try await downloadFLEURS(languages: config.languages)

        // Load samples
        let samples = try loadFLEURSSamples(languages: config.languages)

        if samples.isEmpty {
            print("⚠️ No samples found. Please ensure FLEURS data is available.")
            return []
        }

        print("\n📊 Processing \(samples.count) samples across \(config.languages.count) languages")

        // Group samples by language
        let languageGroups = Dictionary(grouping: samples, by: { $0.language })

        for (language, languageSamples) in languageGroups {
            print("\n🔤 Processing \(supportedLanguages[language] ?? language)...")

            let languageResult = try await processLanguageSamples(
                samples: languageSamples,
                language: language,
                asrManager: asrManager
            )

            results.append(languageResult)

            // Print language summary
            print(
                "  ✓ \(language): WER=\(String(format: "%.1f", languageResult.wer * 100))%, CER=\(String(format: "%.1f", languageResult.cer * 100))%, RTFx=\(String(format: "%.1f", languageResult.rtfx))x"
            )
        }

        return results
    }

    /// Process samples for a specific language
    private func processLanguageSamples(
        samples: [FLEURSSample],
        language: String,
        asrManager: AsrManager
    ) async throws -> LanguageResults {
        var totalWER = 0.0
        var totalCER = 0.0
        var totalDuration = 0.0
        var totalProcessingTime = 0.0
        var processedCount = 0

        for (index, sample) in samples.enumerated() {
            // Skip if audio file doesn't exist
            guard FileManager.default.fileExists(atPath: sample.audioPath) else {
                print("  ⚠️ Audio file not found: \(sample.audioPath)")
                continue
            }

            if config.debugMode {
                print("  Processing \(index + 1)/\(samples.count): \(sample.sampleId)")
            }

            do {
                // Load and process audio
                let startTime = Date()
                let audioSamples = try await AudioProcessor.loadAudioFile(path: sample.audioPath)
                let audioDuration = Double(audioSamples.count) / 16000.0

                // Transcribe
                let result = try await asrManager.transcribe(audioSamples)
                let processingTime = Date().timeIntervalSince(startTime)

                // Calculate metrics if reference transcription is available
                if !sample.transcription.isEmpty {
                    let metrics = calculateMetrics(
                        hypothesis: result.text,
                        reference: sample.transcription
                    )
                    totalWER += metrics.wer
                    totalCER += metrics.cer
                }

                totalDuration += audioDuration
                totalProcessingTime += processingTime
                processedCount += 1

                if config.debugMode {
                    print("    Hypothesis: \(result.text)")
                    if !sample.transcription.isEmpty {
                        print("    Reference:  \(sample.transcription)")
                    }
                }

            } catch {
                print("  ⚠️ Error processing \(sample.sampleId): \(error)")
            }
        }

        // Calculate averages
        let avgWER = processedCount > 0 ? totalWER / Double(processedCount) : 0.0
        let avgCER = processedCount > 0 ? totalCER / Double(processedCount) : 0.0
        let rtfx = totalProcessingTime > 0 ? totalDuration / totalProcessingTime : 0.0

        return LanguageResults(
            language: language,
            wer: avgWER,
            cer: avgCER,
            rtfx: rtfx,
            samplesProcessed: processedCount,
            totalDuration: totalDuration,
            processingTime: totalProcessingTime
        )
    }

    /// Calculate WER and CER metrics
    private func calculateMetrics(hypothesis: String, reference: String) -> (wer: Double, cer: Double) {
        let normalizedHyp = TextNormalizer.normalize(hypothesis)
        let normalizedRef = TextNormalizer.normalize(reference)

        // Word-level
        let hypWords = normalizedHyp.split(separator: " ").map(String.init)
        let refWords = normalizedRef.split(separator: " ").map(String.init)
        let wordDistance = levenshteinDistance(hypWords, refWords)
        let wer = refWords.isEmpty ? 0.0 : Double(wordDistance) / Double(refWords.count)

        // Character-level
        let hypChars = Array(normalizedHyp.replacingOccurrences(of: " ", with: ""))
        let refChars = Array(normalizedRef.replacingOccurrences(of: " ", with: ""))
        let charDistance = levenshteinDistance(hypChars.map(String.init), refChars.map(String.init))
        let cer = refChars.isEmpty ? 0.0 : Double(charDistance) / Double(refChars.count)

        return (wer, cer)
    }

    /// Simple Levenshtein distance calculation
    private func levenshteinDistance(_ s1: [String], _ s2: [String]) -> Int {
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
                    "totalDuration": result.totalDuration,
                    "processingTime": result.processingTime,
                ]
            },
            "summary": [
                "averageWER": results.reduce(0.0) { $0 + $1.wer } / Double(results.count),
                "averageCER": results.reduce(0.0) { $0 + $1.cer } / Double(results.count),
                "averageRTFx": results.reduce(0.0) { $0 + $1.rtfx } / Double(results.count),
                "totalSamples": results.reduce(0) { $0 + $1.samplesProcessed },
                "totalDuration": results.reduce(0.0) { $0 + $1.totalDuration },
                "totalProcessingTime": results.reduce(0.0) { $0 + $1.processingTime },
            ],
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: outputPath))
    }
}

/// CLI entry point for FLEURS benchmark
@available(macOS 13.0, *)
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
        var autoDownload = true  // Default to auto-download

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--languages":
                if i + 1 < arguments.count {
                    languages = arguments[i + 1].split(separator: ",").map(String.init)
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
            case "--no-download":
                autoDownload = false
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                print("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // If no languages specified, use all supported languages
        let finalLanguages = languages ?? Array(tempBenchmark.supportedLanguages.keys).sorted()

        print("\n🌍 FLEURS Multilingual ASR Benchmark")
        print("=" * 50)
        print(
            "Languages: \(finalLanguages.count == tempBenchmark.supportedLanguages.count ? "all (\(finalLanguages.count) languages)" : finalLanguages.joined(separator: ", "))"
        )
        print("Samples per language: \(samplesPerLanguage == Int.max ? "all" : String(samplesPerLanguage))")
        print("Output file: \(outputFile)")
        print("Cache directory: \(cacheDir)")

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
            enableDebug: debugMode,
            tdtConfig: TdtConfig()  // Uses default config
        )

        let asrManager = AsrManager(config: asrConfig)

        do {
            print("\nInitializing ASR system...")
            let models = try await AsrModels.downloadAndLoad()
            try await asrManager.initialize(models: models)
            print("✓ ASR system initialized")

            // Run benchmark
            let results = try await benchmark.runMultilingualBenchmark(asrManager: asrManager)

            // Save results
            try benchmark.saveResults(results, to: outputFile)

            // Print summary
            print("\n" + "=" * 50)
            print("📊 BENCHMARK SUMMARY")
            print("=" * 50)

            for result in results {
                let langName = benchmark.supportedLanguages[result.language] ?? result.language
                print("\n\(langName):")
                print("  WER: \(String(format: "%.1f", result.wer * 100))%")
                print("  CER: \(String(format: "%.1f", result.cer * 100))%")
                print("  RTFx: \(String(format: "%.1f", result.rtfx))x")
                print("  Samples: \(result.samplesProcessed)")
            }

            let avgWER = results.reduce(0.0) { $0 + $1.wer } / Double(results.count)
            let avgCER = results.reduce(0.0) { $0 + $1.cer } / Double(results.count)
            let avgRTFx = results.reduce(0.0) { $0 + $1.rtfx } / Double(results.count)

            print("\n📈 Overall Performance:")
            print("  Average WER: \(String(format: "%.1f", avgWER * 100))%")
            print("  Average CER: \(String(format: "%.1f", avgCER * 100))%")
            print("  Average RTFx: \(String(format: "%.1f", avgRTFx))x")

            print("\n✓ Results saved to: \(outputFile)")

        } catch {
            print("\n❌ Benchmark failed: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        // Build available languages list dynamically to avoid drift
        let tmp = FLEURSBenchmark(
            config: FLEURSConfig(languages: [], samplesPerLanguage: 0, outputFile: "", cacheDir: "", debugMode: false)
        )
        let langs = Array(tmp.supportedLanguages.keys).sorted()
        let langsJoined = langs.joined(separator: ", ")
        let count = langs.count

        print(
            """

            FLEURS Multilingual Benchmark Usage:
                fluidaudio fleurs-benchmark [options]

            Options:
                --languages <list>        Comma-separated list of language codes
                                         (default: all \(count) supported languages)
                                         Available: \(langsJoined)
                --samples <number|all>    Number of samples per language (default: all)
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
