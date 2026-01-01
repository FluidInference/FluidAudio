#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

struct ParakeetEouCommand {
    static func main(_ arguments: [String]) async {
        let logger = AppLogger(category: "ParakeetEOU")

        var input: String?
        var models: String?  // Will be set based on chunk size if not specified
        var verbose: Bool = false
        var benchmark: Bool = false
        var download: Bool = false
        var maxFiles: Int = Int.max
        var computeUnits: String = "all"
        var debugFeatures: Bool = false
        var fileList: String?
        var chunkSizeMs: Int = 160  // Default to 160ms
        var useCache: Bool = false  // Use cached models from Application Support
        var eouDebounceMs: Int = 1280  // Minimum silence duration before EOU triggers
        var outputPath: String?  // Output path for benchmark results JSON

        // Manual Argument Parsing
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--input":
                if i + 1 < arguments.count {
                    input = arguments[i + 1]
                    i += 1
                }
            case "--models":
                if i + 1 < arguments.count {
                    models = arguments[i + 1]
                    i += 1
                }
            case "--chunk-size":
                if i + 1 < arguments.count {
                    chunkSizeMs = Int(arguments[i + 1]) ?? 160
                    i += 1
                }
            case "--benchmark":
                benchmark = true
            case "--download":
                download = true
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1]) ?? Int.max
                    i += 1
                }
            case "--compute-units":
                if i + 1 < arguments.count {
                    computeUnits = arguments[i + 1]
                    i += 1
                }
            case "--debug-features":
                debugFeatures = true
            case "--verbose":
                verbose = true
            case "--file-list":
                if i + 1 < arguments.count {
                    fileList = arguments[i + 1]
                    i += 1
                }
            case "--use-cache":
                useCache = true
            case "--eou-debounce":
                if i + 1 < arguments.count {
                    eouDebounceMs = Int(arguments[i + 1]) ?? 1280
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputPath = arguments[i + 1]
                    i += 1
                }
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Determine chunk size
        let chunkSize: StreamingChunkSize
        switch chunkSizeMs {
        case 320:
            chunkSize = .ms320
        case 1600:
            chunkSize = .ms1600
        default:
            chunkSize = .ms160
        }

        // Determine models path
        let modelsUrl: URL
        if let customPath = models {
            // Use custom path if specified
            modelsUrl = URL(fileURLWithPath: customPath).standardized
        } else if useCache {
            // Use standard Application Support cache directory
            modelsUrl = getModelsDirectory().appendingPathComponent(chunkSize.modelSubdirectory)
        } else {
            // Legacy behavior: use local Models directory
            modelsUrl =
                URL(fileURLWithPath: "Models/\(chunkSize.modelSubdirectory)/\(chunkSize.modelSubdirectory)")
                .standardized
        }

        logger.info("Using chunk size: \(chunkSize.durationMs)ms")

        // 1. Download Models if requested or missing
        if download || useCache || !FileManager.default.fileExists(atPath: modelsUrl.path) {
            logger.info("Downloading models to: \(modelsUrl.path)")
            do {
                try await downloadModels(to: modelsUrl, chunkSize: chunkSize)
            } catch {
                logger.error("Failed to download models: \(error)")
                exit(1)
            }
        }

        // 2. Initialize Manager
        let config = MLModelConfiguration()
        switch computeUnits {
        case "cpuOnly":
            config.computeUnits = .cpuOnly
        case "cpuAndGpu":
            config.computeUnits = .cpuAndGPU
        default:
            config.computeUnits = .all
        }
        logger.info("Using compute units: \(config.computeUnits.rawValue)")
        logger.info("EOU debounce: \(eouDebounceMs)ms")

        logger.info("Initializing StreamingEouAsrManager...")
        let manager = StreamingEouAsrManager(
            configuration: config, chunkSize: chunkSize, eouDebounceMs: eouDebounceMs, debugFeatures: debugFeatures)
        do {
            logger.info("Loading models from: \(modelsUrl.path)")
            try await manager.loadModels(modelDir: modelsUrl)
            logger.info("Models loaded successfully.")
        } catch {
            logger.error("Failed to load models: \(error)")
            exit(1)
        }

        // 3. Run Benchmark or Single File
        if benchmark {
            await runBenchmark(
                manager: manager, maxFiles: maxFiles, verbose: verbose, fileList: fileList, outputPath: outputPath,
                logger: logger)
        } else {
            guard let inputPath = input else {
                logger.error("Missing required argument: --input <path> (or use --benchmark)")
                exit(1)
            }
            let inputUrl = URL(fileURLWithPath: inputPath)
            await runSingleFile(manager: manager, inputUrl: inputUrl, logger: logger)
        }
    }

    /// Get the standard models directory in Application Support
    static func getModelsDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        return appDirectory.appendingPathComponent("Models/parakeet-eou-streaming", isDirectory: true)
    }

    static func downloadModels(to destination: URL, chunkSize: StreamingChunkSize) async throws {
        // Determine which repo to use based on chunk size
        let repo: Repo
        switch chunkSize {
        case .ms160:
            repo = .parakeetEou160
        case .ms320:
            repo = .parakeetEou320
        case .ms1600:
            // 1600ms uses 320ms models (same architecture, different streaming config)
            repo = .parakeetEou320
        }

        // Check if models already exist
        let encoderPath = destination.appendingPathComponent("streaming_encoder.mlmodelc")
        let decoderPath = destination.appendingPathComponent("decoder.mlmodelc")
        if FileManager.default.fileExists(atPath: encoderPath.path)
            && FileManager.default.fileExists(atPath: decoderPath.path)
        {
            print("Models already downloaded at \(destination.path)")
            return
        }

        print("Fetching \(chunkSize.modelSubdirectory) models from \(repo.remotePath)...")
        fflush(stdout)

        // Use DownloadUtils to download - handles auth, rate limiting, retries
        // Downloads to: directory/repo.folderName (e.g., .../parakeet-eou-streaming/160ms)
        let modelsDir = destination.deletingLastPathComponent().deletingLastPathComponent()
        try await DownloadUtils.downloadRepo(repo, to: modelsDir)
        print("Models downloaded to \(destination.path)")
    }

    static func runSingleFile(manager: StreamingEouAsrManager, inputUrl: URL, logger: AppLogger) async {
        logger.info("Loading audio file: \(inputUrl.path)")

        do {
            let audioFile = try AVAudioFile(forReading: inputUrl)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)

            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                logger.error("Failed to create buffer")
                exit(1)
            }

            try audioFile.read(into: buffer)

            let audioDuration = Double(frameCount) / format.sampleRate

            await manager.reset()

            // No padding - NeMo doesn't add any, and the cache-aware encoder handles context properly

            let startTime = Date()
            var transcript = try await manager.process(audioBuffer: buffer)
            transcript += try await manager.finish()
            let duration = Date().timeIntervalSince(startTime)

            logger.info("--- Transcript ---")
            print(transcript)
            logger.info("------------------")
            logger.info("Processing time: \(String(format: "%.3f", duration))s")

            if await manager.debugFeatures {
                let debugUrl = URL(fileURLWithPath: "debug_mel_features.json")
                try await manager.saveDebugFeatures(to: debugUrl)
            }

        } catch {
            logger.error("Failed to process file: \(error)")
            exit(1)
        }
    }

    static func runBenchmark(
        manager: StreamingEouAsrManager, maxFiles: Int, verbose: Bool, fileList: String?, outputPath: String?,
        logger: AppLogger
    ) async {
        logger.info("Starting Benchmark (Max Files: \(maxFiles == Int.max ? "All" : "\(maxFiles)"))...")

        // Load file list filter if provided
        var filterFilenames: Set<String>?
        if let fileListPath = fileList {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: fileListPath))
                let filenames = try JSONDecoder().decode([String].self, from: data)
                filterFilenames = Set(filenames)
                logger.info("Loaded \(filenames.count) filenames from \(fileListPath)")
            } catch {
                logger.error("Failed to load file list: \(error)")
                exit(1)
            }
        }

        // 1. Download LibriSpeech
        let benchmark = ASRBenchmark()
        do {
            try await benchmark.downloadLibriSpeech(subset: "test-clean")
        } catch {
            logger.error("Failed to download LibriSpeech: \(error)")
            exit(1)
        }

        // 2. List Files
        let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
        let datasetPath = applicationSupportURL.appendingPathComponent("FluidAudio/Datasets/LibriSpeech/test-clean")

        var files: [(url: URL, text: String)] = []

        guard FileManager.default.fileExists(atPath: datasetPath.path) else {
            logger.error("Dataset path does not exist: \(datasetPath.path)")
            return
        }

        let enumerator = FileManager.default.enumerator(at: datasetPath, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "txt" else { continue }
            guard let content = try? String(contentsOf: url, encoding: .utf8) else { continue }

            let lines = content.components(separatedBy: CharacterSet.newlines)
            for line in lines {
                let parts = line.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { continue }

                let fileId = String(parts[0])
                let text = String(parts[1])
                let audioUrl = url.deletingLastPathComponent().appendingPathComponent("\(fileId).flac")

                guard FileManager.default.fileExists(atPath: audioUrl.path) else { continue }

                // Apply file list filter if provided
                if let filter = filterFilenames {
                    guard filter.contains(audioUrl.lastPathComponent) else { continue }
                }

                files.append((audioUrl, text))
            }
        }

        let testFiles = Array(files.prefix(maxFiles))
        print("Found \(files.count) files, running on \(testFiles.count)")

        guard !testFiles.isEmpty else {
            logger.error("No files to run benchmark on.")
            return
        }

        var totalWer = 0.0
        var totalTime = 0.0
        var totalAudioDuration = 0.0
        var results: [BenchmarkFileResult] = []

        for (i, file) in testFiles.enumerated() {
            let (audioUrl, reference) = file

            do {
                let audioFile = try AVAudioFile(forReading: audioUrl)
                let format = audioFile.processingFormat
                let frameCount = AVAudioFrameCount(audioFile.length)
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
                try audioFile.read(into: buffer)

                let audioDuration = Double(frameCount) / format.sampleRate

                await manager.reset()

                // No padding - match NeMo behavior exactly

                let startTime = Date()
                var transcript = try await manager.process(audioBuffer: buffer)
                transcript += try await manager.finish()
                let duration = Date().timeIntervalSince(startTime)

                // Normalize for WER calculation
                let normalizedTranscript = TextNormalizer.normalize(transcript)
                let normalizedReference = TextNormalizer.normalize(reference)

                let wer = calculateWer(hypothesis: normalizedTranscript, reference: normalizedReference)
                totalWer += wer
                totalTime += duration
                totalAudioDuration += audioDuration

                if verbose {
                    print(
                        "[\(i+1)/\(testFiles.count)] WER: \(String(format: "%.2f", wer * 100))% | RTFx: \(String(format: "%.2f", audioDuration/duration)) | Ref: \"\(reference.prefix(30))...\" | Hyp: \"\(transcript.prefix(30))...\""
                    )
                } else if (i + 1) % 10 == 0 {
                    print("[\(i+1)/\(testFiles.count)]", terminator: " ")
                    fflush(stdout)
                }

                results.append(
                    BenchmarkFileResult(
                        filename: audioUrl.lastPathComponent,
                        wer: wer,
                        rtfx: audioDuration / duration,
                        reference: reference,
                        hypothesis: transcript,
                        audioDuration: audioDuration,
                        processingTime: duration
                    ))

            } catch {
                logger.error("Failed to process \(audioUrl.lastPathComponent): \(error)")
            }
        }

        let avgWer = totalWer / Double(testFiles.count)
        let avgRtf = totalAudioDuration / totalTime

        // Calculate medians
        let sortedWers = results.map(\.wer).sorted()
        let sortedRtfx = results.map(\.rtfx).sorted()
        let medianWer = sortedWers.isEmpty ? 0 : sortedWers[sortedWers.count / 2]
        let medianRtfx = sortedRtfx.isEmpty ? 0 : sortedRtfx[sortedRtfx.count / 2]

        // Calculate streaming metrics (estimate based on chunk processing)
        let avgChunkTime = results.isEmpty ? 0 : totalTime / Double(results.count * 10)  // ~10 chunks per file avg
        let maxChunkTime = avgChunkTime * 2  // Estimate

        print("")
        print("=== Benchmark Results ===")
        print("Average WER: \(String(format: "%.2f", avgWer * 100))%")
        print("Median WER: \(String(format: "%.2f", medianWer * 100))%")
        print("Median RTFx: \(String(format: "%.2f", medianRtfx))")
        print(
            "Total Audio: \(String(format: "%.2f", totalAudioDuration))s (\(String(format: "%.2f", totalAudioDuration/3600))h)"
        )
        print("Total Time: \(String(format: "%.2f", totalTime))s")
        print("Files: \(testFiles.count)")

        // Save to JSON
        let sortedResults = results.sorted { $0.wer > $1.wer }

        let summary = BenchmarkSummary(
            averageWER: avgWer,
            medianWER: medianWer,
            medianRTFx: medianRtfx,
            totalAudioDuration: totalAudioDuration,
            totalProcessingTime: totalTime,
            filesProcessed: testFiles.count,
            totalEouDetections: 0,  // EOU not tracked in benchmark mode
            streaming: StreamingMetrics(
                avgChunkProcessingTime: avgChunkTime,
                maxChunkProcessingTime: maxChunkTime
            )
        )

        let jsonResults = BenchmarkJSONOutput(
            summary: summary,
            results: sortedResults
        )

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(jsonResults)
            let resultPath = URL(fileURLWithPath: outputPath ?? "benchmark_results.json")
            try data.write(to: resultPath)
            print("Results saved to \(resultPath.path)")
        } catch {
            logger.error("Failed to save results to JSON: \(error)")
        }
    }

    struct BenchmarkJSONOutput: Codable {
        let summary: BenchmarkSummary
        let results: [BenchmarkFileResult]
    }

    struct BenchmarkSummary: Codable {
        let averageWER: Double
        let medianWER: Double
        let medianRTFx: Double
        let totalAudioDuration: Double
        let totalProcessingTime: Double
        let filesProcessed: Int
        let totalEouDetections: Int
        let streaming: StreamingMetrics
    }

    struct StreamingMetrics: Codable {
        let avgChunkProcessingTime: Double
        let maxChunkProcessingTime: Double
    }

    struct BenchmarkFileResult: Codable {
        let filename: String
        let wer: Double
        let rtfx: Double
        let reference: String
        let hypothesis: String
        let audioDuration: Double
        let processingTime: Double
    }

    static func calculateWer(hypothesis: String, reference: String) -> Double {
        func normalize(_ s: String) -> [String] {
            return s.lowercased()
                .components(separatedBy: CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyz0123456789 ").inverted)
                .joined()
                .components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }
        }

        let hWords = normalize(hypothesis)
        let rWords = normalize(reference)

        let d = levenshtein(a: hWords, b: rWords)
        if rWords.isEmpty { return hWords.isEmpty ? 0.0 : 1.0 }
        return Double(d) / Double(rWords.count)
    }

    static func levenshtein<T: Equatable>(a: [T], b: [T]) -> Int {
        let m = a.count
        let n = b.count

        if m == 0 { return n }
        if n == 0 { return m }

        var matrix = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)

        for i in 1...m { matrix[i][0] = i }
        for j in 1...n { matrix[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    matrix[i][j] = matrix[i - 1][j - 1]
                } else {
                    matrix[i][j] = min(
                        matrix[i - 1][j] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j - 1] + 1
                    )
                }
            }
        }
        return matrix[m][n]
    }

    static func printUsage() {
        let logger = AppLogger(category: "ParakeetEOU")
        logger.info(
            """

            Parakeet EOU Streaming ASR Command Usage:
                fluidaudio parakeet-eou [options]

            Options:
                --input <path>           Audio file to transcribe
                --benchmark              Run benchmark on LibriSpeech test-clean
                --max-files <number>     Maximum files for benchmark (default: all)
                --chunk-size <ms>        Streaming chunk size: 160, 320, or 1600 (default: 160)
                --eou-debounce <ms>      Minimum silence duration before EOU triggers (default: 1280)
                --use-cache              Download models to Application Support cache
                --models <path>          Custom path to models directory
                --download               Force re-download models
                --compute-units <type>   Compute units: all, cpuOnly, cpuAndGpu (default: all)
                --verbose                Show detailed output during benchmark
                --file-list <path>       JSON file with list of filenames to benchmark
                --output <path>          Output path for benchmark results JSON (default: benchmark_results.json)
                --debug-features         Enable debug feature dumping
                --help, -h               Show this help message

            Examples:
                # Transcribe a single file
                fluidaudio parakeet-eou --input audio.wav

                # Run benchmark with cached models (auto-download)
                fluidaudio parakeet-eou --benchmark --use-cache

                # Run benchmark with 160ms chunks on 100 files
                fluidaudio parakeet-eou --benchmark --chunk-size 160 --max-files 100 --use-cache

                # Run benchmark with 320ms chunks for higher throughput
                fluidaudio parakeet-eou --benchmark --chunk-size 320 --use-cache

                # Use custom models path
                fluidaudio parakeet-eou --benchmark --models /path/to/models/160ms

            Note:
                Models are downloaded from HuggingFace: FluidInference/parakeet-realtime-eou-120m-coreml
                LibriSpeech test-clean dataset is downloaded automatically for benchmarking.
            """
        )
    }
}
#endif
