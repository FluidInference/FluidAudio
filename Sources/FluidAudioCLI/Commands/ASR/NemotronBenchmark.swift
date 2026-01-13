#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Nemotron Speech Streaming 0.6B benchmark on LibriSpeech
public class NemotronBenchmark {
    private let logger = AppLogger(category: "NemotronBenchmark")

    public struct Config {
        var maxFiles: Int?
        var subset: String = "test-clean"
        var modelDir: URL?
        var encoderVariant: NemotronEncoderVariant = .int8

        public init() {}
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Run CLI benchmark
    public static func run(arguments: [String]) async {
        let logger = AppLogger(category: "NemotronBenchmark")

        var config = Config()

        // Parse arguments
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]

            switch arg {
            case "--max-files", "-n":
                i += 1
                if i < arguments.count, let n = Int(arguments[i]) {
                    config.maxFiles = n
                }
            case "--subset", "-s":
                i += 1
                if i < arguments.count {
                    config.subset = arguments[i]
                }
            case "--model-dir", "-m":
                i += 1
                if i < arguments.count {
                    config.modelDir = URL(fileURLWithPath: arguments[i])
                }
            case "--encoder", "-e":
                i += 1
                if i < arguments.count {
                    if arguments[i] == "float32" {
                        config.encoderVariant = .float32
                    } else {
                        config.encoderVariant = .int8
                    }
                }
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arg)")
            }
            i += 1
        }

        let benchmark = NemotronBenchmark(config: config)
        await benchmark.run()
    }

    private static func printUsage() {
        print(
            """
            Nemotron Speech Streaming 0.6B Benchmark

            Usage: fluidaudio nemotron-benchmark [options]

            Options:
                --max-files, -n <count>   Maximum files to process (default: all)
                --subset, -s <name>       LibriSpeech subset (default: test-clean)
                --model-dir, -m <path>    Path to Nemotron CoreML models
                --encoder, -e <variant>   Encoder variant: int8 or float32 (default: int8)
                --help, -h                Show this help

            Examples:
                fluidaudio nemotron-benchmark --max-files 100
                fluidaudio nemotron-benchmark --subset test-other --max-files 50
                fluidaudio nemotron-benchmark --encoder float32 --max-files 10
            """
        )
    }

    /// Run the benchmark
    public func run() async {
        logger.info(String(repeating: "=", count: 70))
        logger.info("NEMOTRON SPEECH STREAMING 0.6B BENCHMARK")
        logger.info(String(repeating: "=", count: 70))

        #if DEBUG
        logger.warning("WARNING: Running in DEBUG mode!")
        logger.warning("For accurate benchmarks, use: swift run -c release fluidaudio nemotron-benchmark")
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        #else
        logger.info("Running in RELEASE mode - optimal performance")
        #endif

        do {
            // 1. Download LibriSpeech if needed
            logger.info("Checking LibriSpeech \(config.subset)...")
            try await downloadLibriSpeech(subset: config.subset)

            // 2. Download Nemotron models if needed
            let modelDir = try await getOrDownloadModels()

            // 3. Load models
            logger.info("Loading Nemotron models with \(config.encoderVariant.rawValue) encoder...")
            let manager = NemotronStreamingAsrManager()
            try await manager.loadModels(modelDir: modelDir, encoderVariant: config.encoderVariant)
            logger.info("Models loaded successfully")

            // 4. Get audio files
            let datasetPath = getLibriSpeechDirectory().appendingPathComponent(config.subset)
            let audioFiles = try collectLibriSpeechFiles(from: datasetPath)

            let maxFiles = config.maxFiles ?? audioFiles.count
            let filesToProcess = Array(audioFiles.prefix(maxFiles))

            logger.info("Processing \(filesToProcess.count) files from LibriSpeech \(config.subset)")
            logger.info("")

            // 5. Run benchmark
            var totalErrors = 0
            var totalWords = 0
            var totalAudioDuration: Double = 0
            var totalProcessingTime: Double = 0

            for (index, file) in filesToProcess.enumerated() {
                let result = try await processFile(
                    manager: manager, file: file, index: index + 1, total: filesToProcess.count)

                totalErrors += result.errors
                totalWords += result.words
                totalAudioDuration += result.audioDuration
                totalProcessingTime += result.processingTime

                let runningWer = totalWords > 0 ? Double(totalErrors) / Double(totalWords) * 100.0 : 0.0

                if result.errors > 0 {
                    logger.info(
                        "  [\(index + 1)/\(filesToProcess.count)] \(file.fileName) -> \(result.errors) errs, WER: \(String(format: "%.2f", runningWer))%"
                    )
                    logger.info("       REF: \(file.transcript.prefix(70))...")
                    logger.info("       HYP: \(result.hypothesis.prefix(70))...")
                } else {
                    logger.info(
                        "  [\(index + 1)/\(filesToProcess.count)] \(file.fileName) -> 0 errs, WER: \(String(format: "%.2f", runningWer))%"
                    )
                }

                // Reset manager for next file
                await manager.reset()
            }

            // 6. Print summary
            let finalWer = totalWords > 0 ? Double(totalErrors) / Double(totalWords) * 100.0 : 0.0
            let rtf = totalAudioDuration > 0 ? totalProcessingTime / totalAudioDuration : 0.0
            let rtfx = rtf > 0 ? 1.0 / rtf : 0.0

            logger.info("")
            logger.info(String(repeating: "=", count: 70))
            logger.info("SUMMARY")
            logger.info(String(repeating: "=", count: 70))
            logger.info("Files processed:    \(filesToProcess.count)")
            logger.info("Total words:        \(totalWords)")
            logger.info("Total errors:       \(totalErrors)")
            logger.info("WER:                \(String(format: "%.2f", finalWer))%")
            logger.info("Audio duration:     \(String(format: "%.1f", totalAudioDuration))s")
            logger.info("Processing time:    \(String(format: "%.1f", totalProcessingTime))s")
            logger.info("RTFx:               \(String(format: "%.1f", rtfx))x")

        } catch {
            logger.error("Benchmark failed: \(error)")
        }
    }

    private struct FileResult {
        let hypothesis: String
        let errors: Int
        let words: Int
        let audioDuration: Double
        let processingTime: Double
    }

    private func processFile(
        manager: NemotronStreamingAsrManager, file: LibriSpeechFile, index: Int, total: Int
    ) async throws -> FileResult {
        // Load audio
        let audioFile = try AVAudioFile(forReading: file.audioPath)
        let buffer = AVAudioPCMBuffer(
            pcmFormat: audioFile.processingFormat,
            frameCapacity: AVAudioFrameCount(audioFile.length)
        )!
        try audioFile.read(into: buffer)

        let audioDuration = Double(audioFile.length) / audioFile.processingFormat.sampleRate

        // Process
        let startTime = Date()
        _ = try await manager.process(audioBuffer: buffer)
        let hypothesis = try await manager.finish()
        let processingTime = Date().timeIntervalSince(startTime)

        // Calculate WER
        let (errors, words) = calculateWER(reference: file.transcript, hypothesis: hypothesis)

        return FileResult(
            hypothesis: hypothesis,
            errors: errors,
            words: words,
            audioDuration: audioDuration,
            processingTime: processingTime
        )
    }

    private func calculateWER(reference: String, hypothesis: String) -> (errors: Int, words: Int) {
        let refWords = normalizeText(reference).split(separator: " ").map(String.init)
        let hypWords = normalizeText(hypothesis).split(separator: " ").map(String.init)

        let m = refWords.count
        let n = hypWords.count

        var d = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { d[i][0] = i }
        for j in 0...n { d[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if refWords[i - 1] == hypWords[j - 1] {
                    d[i][j] = d[i - 1][j - 1]
                } else {
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
                }
            }
        }

        return (d[m][n], m)
    }

    private func normalizeText(_ text: String) -> String {
        let cleaned = text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined(separator: " ")
        return cleaned.split(separator: " ").joined(separator: " ")
    }

    // MARK: - Dataset Management

    private func getLibriSpeechDirectory() -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/fluidaudio/datasets")
    }

    private func downloadLibriSpeech(subset: String) async throws {
        let datasetsDirectory = getLibriSpeechDirectory()
        let subsetDirectory = datasetsDirectory.appendingPathComponent(subset)

        // Check if already downloaded
        if FileManager.default.fileExists(atPath: subsetDirectory.path) {
            let enumerator = FileManager.default.enumerator(at: subsetDirectory, includingPropertiesForKeys: nil)
            var transcriptCount = 0

            while let url = enumerator?.nextObject() as? URL {
                if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                    transcriptCount += 1
                    if transcriptCount >= 5 { break }
                }
            }

            if transcriptCount >= 5 {
                logger.info("LibriSpeech \(subset) already downloaded")
                return
            }
        }

        logger.info("Downloading LibriSpeech \(subset)...")

        let downloadURL: String
        switch subset {
        case "test-clean":
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-clean.tar.gz")
                .absoluteString
        case "test-other":
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-other.tar.gz")
                .absoluteString
        default:
            throw ASRError.processingFailed("Unsupported LibriSpeech subset: \(subset)")
        }

        try await downloadAndExtractTarGz(
            url: downloadURL,
            extractTo: datasetsDirectory,
            expectedSubpath: subset
        )

        logger.info("LibriSpeech \(subset) downloaded successfully")
    }

    private func getOrDownloadModels() async throws -> URL {
        if let modelDir = config.modelDir {
            return modelDir
        }

        // Check default cache location
        // Note: downloadRepo appends repo.folderName internally, so we use the parent dir
        let modelsBaseDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/fluidaudio/models")
        let cacheDir = modelsBaseDir.appendingPathComponent(Repo.nemotronStreaming.folderName)

        // Check for encoder in subdirectory (new structure)
        let encoderPath = cacheDir.appendingPathComponent("encoder/encoder_int8.mlmodelc")

        if FileManager.default.fileExists(atPath: encoderPath.path) {
            logger.info("Using cached Nemotron models at \(cacheDir.path)")
            return cacheDir
        }

        // Download models (downloadRepo appends folderName internally)
        logger.info("Downloading Nemotron models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.nemotronStreaming, to: modelsBaseDir)

        return cacheDir
    }

    private func collectLibriSpeechFiles(from directory: URL) throws -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []
        var transcripts: [String: String] = [:]

        // First pass: collect all transcripts
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            if url.lastPathComponent.contains(".trans.txt") {
                let content = try String(contentsOf: url, encoding: .utf8)
                for line in content.components(separatedBy: .newlines) {
                    let parts = line.split(separator: " ", maxSplits: 1)
                    if parts.count == 2 {
                        transcripts[String(parts[0])] = String(parts[1]).lowercased()
                    }
                }
            }
        }

        // Second pass: match audio files with transcripts
        let enumerator2 = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        while let url = enumerator2?.nextObject() as? URL {
            if url.pathExtension == "flac" {
                let fileId = url.deletingPathExtension().lastPathComponent
                if let transcript = transcripts[fileId] {
                    files.append(
                        LibriSpeechFile(
                            fileName: url.lastPathComponent,
                            audioPath: url,
                            transcript: transcript
                        ))
                }
            }
        }

        return files.sorted { $0.fileName < $1.fileName }
    }

    private func downloadAndExtractTarGz(url: String, extractTo: URL, expectedSubpath: String) async throws {
        let tempFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".tar.gz")

        defer {
            try? FileManager.default.removeItem(at: tempFile)
        }

        // Download
        guard let downloadUrl = URL(string: url) else {
            throw ASRError.processingFailed("Invalid URL: \(url)")
        }

        let (data, _) = try await URLSession.shared.data(from: downloadUrl)
        try data.write(to: tempFile)

        // Extract
        try FileManager.default.createDirectory(at: extractTo, withIntermediateDirectories: true)

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        task.arguments = ["-xzf", tempFile.path, "-C", extractTo.path]
        try task.run()
        task.waitUntilExit()

        guard task.terminationStatus == 0 else {
            throw ASRError.processingFailed("Failed to extract tar.gz")
        }
    }
}
#endif
