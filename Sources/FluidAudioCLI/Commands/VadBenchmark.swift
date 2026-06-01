#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// VAD benchmark implementation
struct VadBenchmark {
    private static let logger = AppLogger(category: "VAD")

    static func runVadBenchmark(arguments: [String]) async {
        do {
            try await runVadBenchmarkWithErrorHandling(arguments: arguments)
        } catch {
            logger.error("VAD Benchmark failed: \(error)")
        }
    }

    static func runVadBenchmarkWithErrorHandling(arguments: [String]) async throws {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        logger.info("Starting VAD Benchmark")
        var numFiles = -1
        var useAllFiles = true
        var vadThreshold: Float = 0.3
        var activityThreshold: Float = 0.1
        var outputFile: String?
        var dataset = "mini50"
        var debugMode = false
        var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
        var backend = "silero"  // silero | fsmn

        logger.info("Parsing arguments...")

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--num-files":
                if i + 1 < arguments.count {
                    numFiles = Int(arguments[i + 1]) ?? -1
                    useAllFiles = false
                    i += 1
                }
            case "--all-files":
                useAllFiles = true
                numFiles = -1
            case "--threshold":
                if i + 1 < arguments.count {
                    vadThreshold = Float(arguments[i + 1]) ?? 0.3
                    i += 1
                }
            case "--activity-threshold":
                if i + 1 < arguments.count {
                    activityThreshold = Float(arguments[i + 1]) ?? 0.3
                    i += 1
                }
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
            case "--backend":
                if i + 1 < arguments.count {
                    backend = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--compute-units":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "all": computeUnits = .cpuAndNeuralEngine
                    case "cpu-only": computeUnits = .cpuOnly
                    case "ane": computeUnits = .cpuAndNeuralEngine
                    default:
                        logger.warning("Invalid --compute-units value '\(arguments[i + 1])', using 'all'")
                    }
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        logger.info("Starting VAD Benchmark")
        logger.info("Test files: \(numFiles)")
        logger.info("VAD threshold: \(vadThreshold)")
        logger.info("Activity threshold: \(activityThreshold)")
        logger.info("Debug mode: \(debugMode)")

        if backend == "fsmn" {
            try await runFsmnVadBenchmark(
                dataset: dataset, count: useAllFiles ? -1 : numFiles, activityThreshold: activityThreshold)
            return
        }

        let vadManager = try await VadManager(
            config: VadConfig(
                defaultThreshold: vadThreshold,
                debugMode: debugMode,
                computeUnits: computeUnits
            ))

        logger.info("VAD system initialized")

        let testFiles = try await downloadVadTestFiles(
            count: useAllFiles ? -1 : numFiles, dataset: dataset)

        let result = try await runVadBenchmarkInternal(
            vadManager: vadManager, testFiles: testFiles, threshold: vadThreshold, activityThreshold: activityThreshold)

        let rtfx = try await calculateRTFx(result: result, testFiles: testFiles)

        logger.info("VAD Benchmark Results:")
        logger.info("Accuracy: \(String(format: "%.1f", result.accuracy))%")
        logger.info("Precision: \(String(format: "%.1f", result.precision))%")
        logger.info("Recall: \(String(format: "%.1f", result.recall))%")
        logger.info("F1-Score: \(String(format: "%.1f", result.f1Score))%")
        logger.info("Total Time: \(String(format: "%.2f", result.processingTime))s")
        if rtfx < 1.0 && rtfx > 0 {
            logger.info("RTFx: \(String(format: "%.1f", 1.0/rtfx))x faster than real-time")
        } else if rtfx >= 1.0 {
            logger.info("RTFx: \(String(format: "%.1f", rtfx))x slower than real-time")
        } else {
            logger.info("RTFx: N/A")
        }
        logger.info("Files Processed: \(result.totalFiles)")
        logger.info(
            "Avg Time per File: \(String(format: "%.3f", result.processingTime / Double(result.totalFiles)))s")

        if let outputFile = outputFile {
            try await saveVadBenchmarkResultsWithRTFx(
                result, testFiles: testFiles, to: outputFile)
            logger.info("Results saved to: \(outputFile)")
        } else {
            try await saveVadBenchmarkResultsWithRTFx(
                result, testFiles: testFiles, to: "vad_benchmark_results.json")
            logger.info("Results saved to: vad_benchmark_results.json")
        }

        if result.f1Score >= 70.0 {
            logger.info("EXCELLENT: F1-Score above 70%")
        } else if result.f1Score >= 60.0 {
            logger.warning("ACCEPTABLE: F1-Score above 60%")
        } else {
            logger.warning("NEEDS IMPROVEMENT: F1-Score below 60%")
        }
    }

    private static func printUsage() {
        let usage = """
            VAD Benchmark Command Usage:
                fluidaudio vad-benchmark [options]

            Options:
                --num-files <n>           Number of files to process (default: all)
                --all-files               Process all available files
                --threshold <float>        VAD probability threshold (default: 0.3)
                --activity-threshold <fl>  Activity threshold as fraction of chunks (default: 0.1)
                --dataset <name>           Dataset: mini50, mini100, musan-full, voices-subset (default: mini50)
                --output <file>            Save results to JSON file
                --debug                    Enable debug logging
                --compute-units <all|cpu-only|ane>
                                         ML compute units (default: all)

            Examples:
                fluidaudio vad-benchmark
                fluidaudio vad-benchmark --num-files 10 --compute-units cpu-only
            """
        fputs(usage, stderr)
        fflush(stderr)
    }

    /// FSMN-VAD backend: same dataset + per-file metric as silero, for apples-to-apples.
    /// Per-file prediction = (speech duration / file duration) >= activityThreshold.
    static func runFsmnVadBenchmark(dataset: String, count: Int, activityThreshold: Float) async throws {
        let vad = try await FsmnVadManager.load()
        let testFiles = try await downloadVadTestFiles(count: count, dataset: dataset)
        logger.info("Running FSMN-VAD benchmark on \(testFiles.count) files...")
        var predictions: [Int] = []
        var groundTruth: [Int] = []
        var audioSec = 0.0
        var procSec = 0.0
        for (idx, f) in testFiles.enumerated() {
            do {
                let af = try AVAudioFile(forReading: f.url)
                let dur = Double(af.length) / af.processingFormat.sampleRate
                audioSec += dur
                let t0 = Date()
                let segs = try await vad.detect(audioURL: f.url)
                procSec += Date().timeIntervalSince(t0)
                let speechMs = segs.reduce(0) { $0 + max(0, $1.endMs - $1.startMs) }
                let ratio = dur > 0 ? Float(speechMs) / Float(dur * 1000.0) : 0
                predictions.append(ratio >= activityThreshold ? 1 : 0)
            } catch {
                logger.warning("Error on \(f.name): \(error)")
                predictions.append(0)
            }
            groundTruth.append(f.expectedLabel)
            if (idx + 1) % 10 == 0 { logger.info("  \(idx + 1)/\(testFiles.count)") }
        }
        let m = calculateVadMetrics(predictions: predictions, groundTruth: groundTruth)
        let rtfx = procSec > 0 ? audioSec / procSec : 0
        logger.info("FSMN-VAD Benchmark Results (dataset=\(dataset), n=\(testFiles.count)):")
        logger.info("Accuracy: \(String(format: "%.1f", m.accuracy))%")
        logger.info("Precision: \(String(format: "%.1f", m.precision))%")
        logger.info("Recall: \(String(format: "%.1f", m.recall))%")
        logger.info("F1-Score: \(String(format: "%.1f", m.f1Score))%")
        logger.info("RTFx: \(String(format: "%.0f", rtfx))x")
    }

    static func downloadVadTestFiles(
        count: Int, dataset: String = "mini50"
    ) async throws
        -> [VadTestFile]
    {
        if count == -1 {
            logger.info("Loading all available test audio files...")
        } else {
            logger.info("Loading \(count) test audio files...")
        }

        if dataset == "musan-full" {
            if let musanFiles = try await loadFullMusanDataset(count: count) {
                return musanFiles
            }
        }

        if dataset == "voices-subset" {
            if let voicesFiles = try await loadVoicesSubset(count: count) {
                return voicesFiles
            }
        }

        if let localFiles = try await loadLocalDataset(count: count) {
            return localFiles
        }

        if let cachedFiles = try await loadHuggingFaceVadDataset(count: count, dataset: dataset) {
            return cachedFiles
        }

        logger.info("Downloading VAD dataset from Hugging Face...")
        if let hfFiles = try await downloadHuggingFaceVadDataset(count: count, dataset: dataset) {
            return hfFiles
        }

        logger.error(
            "Failed to load VAD dataset from all sources:\nLocal dataset not found\nHugging Face cache empty\nHugging Face download failed\nTry: swift run fluidaudiocli download --dataset vad"
        )
        throw NSError(
            domain: "VadError", code: 404,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "No VAD dataset available. Use 'download --dataset vad' to get real data."
            ])
    }

    static func loadLocalDataset(count: Int) async throws -> [VadTestFile]? {
        let possiblePaths = [
            "VADDataset/",
            "vad_test_data/",
            "datasets/vad/",
            "../datasets/vad/",
        ]

        for basePath in possiblePaths {
            let datasetDir = URL(fileURLWithPath: basePath)

            guard FileManager.default.fileExists(atPath: datasetDir.path) else {
                continue
            }

            logger.info("Found local dataset at: \(basePath)")

            var testFiles: [VadTestFile] = []

            let speechDir = datasetDir.appendingPathComponent("speech")
            let nonSpeechDir = datasetDir.appendingPathComponent("non_speech")

            if FileManager.default.fileExists(atPath: speechDir.path) {
                let maxSpeechFiles = count == -1 ? Int.max : count / 2
                let speechFiles = try loadAudioFiles(
                    from: speechDir, expectedLabel: 1, maxCount: maxSpeechFiles)
                testFiles.append(contentsOf: speechFiles)
                logger.info("Loaded \(speechFiles.count) speech files")
            }

            if FileManager.default.fileExists(atPath: nonSpeechDir.path) {
                let maxNoiseFiles = count == -1 ? Int.max : count - testFiles.count
                let nonSpeechFiles = try loadAudioFiles(
                    from: nonSpeechDir, expectedLabel: 0, maxCount: maxNoiseFiles)
                testFiles.append(contentsOf: nonSpeechFiles)
                logger.info("Loaded \(nonSpeechFiles.count) non-speech files")
            }

            if !testFiles.isEmpty {
                logger.info("Using local dataset: \(testFiles.count) files total")
                return testFiles
            }
        }

        return nil
    }

    static func loadAudioFiles(
        from directory: URL, expectedLabel: Int, maxCount: Int
    ) throws
        -> [VadTestFile]
    {
        let fileManager = FileManager.default
        let audioExtensions = ["wav", "mp3", "m4a", "aac", "aiff"]

        guard
            let enumerator = fileManager.enumerator(
                at: directory, includingPropertiesForKeys: nil)
        else {
            return []
        }

        var files: [VadTestFile] = []

        for case let fileURL as URL in enumerator {
            guard files.count < maxCount else { break }

            let fileExtension = fileURL.pathExtension.lowercased()
            guard audioExtensions.contains(fileExtension) else { continue }

            let fileName = fileURL.lastPathComponent
            files.append(
                VadTestFile(name: fileName, expectedLabel: expectedLabel, url: fileURL))
        }

        return files
    }

    /// Load VAD dataset from Hugging Face cache
    static func loadHuggingFaceVadDataset(
        count: Int, dataset: String = "mini50"
    ) async throws
        -> [VadTestFile]?
    {
        let cacheDir = getVadDatasetCacheDirectory()

        let speechDir = cacheDir.appendingPathComponent("speech")
        let noiseDir = cacheDir.appendingPathComponent("noise")

        guard
            FileManager.default.fileExists(atPath: speechDir.path)
                && FileManager.default.fileExists(atPath: noiseDir.path)
        else {
            return nil
        }

        var testFiles: [VadTestFile] = []

        let maxFilesForDataset = dataset == "mini100" ? 100 : 50

        if count == -1 {
            logger.info("Loading all available files from Hugging Face cache...")

            let speechFiles = try loadAudioFiles(
                from: speechDir, expectedLabel: 1, maxCount: maxFilesForDataset / 2)
            testFiles.append(contentsOf: speechFiles)

            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0, maxCount: maxFilesForDataset / 2)
            testFiles.append(contentsOf: noiseFiles)
        } else {
            let speechCount = count / 2
            let noiseCount = count - speechCount

            let speechFiles = try loadAudioFiles(
                from: speechDir, expectedLabel: 1, maxCount: speechCount)
            testFiles.append(contentsOf: speechFiles)

            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0, maxCount: noiseCount)
            testFiles.append(contentsOf: noiseFiles)
        }

        if testFiles.isEmpty {
            return nil
        }

        logger.info("Found cached Hugging Face dataset: \(testFiles.count) files total")
        return testFiles
    }

    /// Download VAD dataset from Hugging Face musan_mini50 or musan_mini100 repository
    static func downloadHuggingFaceVadDataset(
        count: Int, dataset: String = "mini50"
    )
        async throws -> [VadTestFile]?
    {
        let cacheDir = getVadDatasetCacheDirectory()

        let speechDir = cacheDir.appendingPathComponent("speech")
        let noiseDir = cacheDir.appendingPathComponent("noise")
        try FileManager.default.createDirectory(
            at: speechDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: noiseDir, withIntermediateDirectories: true)

        let repoName = dataset == "mini100" ? "musan_mini100" : "musan_mini50"
        let repoBase = ModelRegistry.resolveDatasetBase("alexwengg/\(repoName)")

        var testFiles: [VadTestFile] = []

        let maxFiles = dataset == "mini100" ? 100 : 50
        let speechCount = count == -1 ? maxFiles / 2 : count / 2
        let noiseCount = count == -1 ? maxFiles / 2 : count - speechCount

        do {
            logger.info("Downloading speech samples...")
            let speechFiles = try await DatasetDownloader.downloadVadFilesFromHF(
                baseUrl: "\(repoBase)/speech",
                targetDir: speechDir,
                expectedLabel: 1,
                count: speechCount,
                filePrefix: "speech",
                repoName: repoName
            )
            testFiles.append(contentsOf: speechFiles)

            logger.info("Downloading noise samples...")
            let noiseFiles = try await DatasetDownloader.downloadVadFilesFromHF(
                baseUrl: "\(repoBase)/noise",
                targetDir: noiseDir,
                expectedLabel: 0,
                count: noiseCount,
                filePrefix: "noise",
                repoName: repoName
            )
            testFiles.append(contentsOf: noiseFiles)

            if !testFiles.isEmpty {
                logger.info("Downloaded VAD dataset from Hugging Face: \(testFiles.count) files")
                return testFiles
            }

        } catch {
            logger.error("Failed to download from Hugging Face: \(error)")
            try? FileManager.default.removeItem(at: cacheDir)
        }

        return nil
    }

    /// Get VAD dataset cache directory
    static func getVadDatasetCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let cacheDir = appSupport.appendingPathComponent(
            "FluidAudio/vadDataset", isDirectory: true)

        try? FileManager.default.createDirectory(
            at: cacheDir, withIntermediateDirectories: true)
        return cacheDir
    }

    static func runVadBenchmarkInternal(
        vadManager: VadManager, testFiles: [VadTestFile], threshold: Float, activityThreshold: Float
    ) async throws -> VadBenchmarkResult {
        logger.info("Running VAD benchmark on \(testFiles.count) files...")

        let memoryOptimizer = ANEMemoryOptimizer()
        let startTime = Date()
        var predictions: [Int] = []
        var groundTruth: [Int] = []
        var fileDurations: [TimeInterval] = []
        var loadingTime: TimeInterval = 0
        var inferenceTime: TimeInterval = 0
        var totalAudioDuration: TimeInterval = 0

        for (index, testFile) in testFiles.enumerated() {
            let fileStartTime = Date()
            logger.info("Processing \(index + 1)/\(testFiles.count): \(testFile.name)")

            do {
                let loadStartTime = Date()
                let audioFile = try AVAudioFile(forReading: testFile.url)
                let audioDuration = Double(audioFile.length) / audioFile.processingFormat.sampleRate
                loadingTime += Date().timeIntervalSince(loadStartTime)
                totalAudioDuration += audioDuration

                let url = URL(fileURLWithPath: testFile.url.path)
                let inferenceStartTime = Date()
                let vadResults = try await vadManager.process(url)
                inferenceTime += Date().timeIntervalSince(inferenceStartTime)

                let activeChunks = vadResults.filter { $0.isVoiceActive }.count
                let activityRatio = Float(activeChunks) / Float(vadResults.count)
                let prediction = activityRatio >= activityThreshold ? 1 : 0
                let maxProbability = vadResults.map { $0.probability }.max() ?? 0.0

                predictions.append(prediction)
                groundTruth.append(testFile.expectedLabel)

                let fileProcessingTime = Date().timeIntervalSince(fileStartTime)
                fileDurations.append(fileProcessingTime)

                let fileRTFx = audioDuration > 0 ? fileProcessingTime / audioDuration : 0
                let rtfxDisplay =
                    if fileRTFx < 1.0 && fileRTFx > 0 {
                        String(format: "%.1fx", 1.0 / fileRTFx)
                    } else {
                        String(format: "%.1fx", fileRTFx)
                    }

                logger.info(
                    "Result: activity_ratio=\(String(format: "%.3f", activityRatio)), max_prob=\(String(format: "%.3f", maxProbability)), prediction=\(prediction), expected=\(testFile.expectedLabel), time=\(String(format: "%.3f", fileProcessingTime))s, RTFx=\(rtfxDisplay)"
                )

            } catch {
                logger.warning("Error: \(error)")
                predictions.append(0)
                groundTruth.append(testFile.expectedLabel)
                fileDurations.append(Date().timeIntervalSince(fileStartTime))
            }

            if (index + 1) % 10 == 0 {
                memoryOptimizer.clearBufferPool()
            }
        }

        let processingTime = Date().timeIntervalSince(startTime)

        let metrics = calculateVadMetrics(predictions: predictions, groundTruth: groundTruth)

        let avgProcessingTime =
            fileDurations.isEmpty ? 0 : fileDurations.reduce(0, +) / Double(fileDurations.count)
        let minProcessingTime = fileDurations.min() ?? 0
        let maxProcessingTime = fileDurations.max() ?? 0

        let rtfx = totalAudioDuration > 0 ? processingTime / totalAudioDuration : 0

        logger.info("Timing Statistics:")
        logger.info("Total processing time: \(String(format: "%.2f", processingTime))s")
        logger.info("Total audio duration: \(String(format: "%.2f", totalAudioDuration))s")
        if rtfx < 1.0 && rtfx > 0 {
            logger.info("RTFx: \(String(format: "%.1f", 1.0/rtfx))x faster than real-time")
        } else if rtfx >= 1.0 {
            logger.info("RTFx: \(String(format: "%.1f", rtfx))x slower than real-time")
        } else {
            logger.info("RTFx: N/A")
        }
        logger.info(
            "Audio loading time: \(String(format: "%.2f", loadingTime))s (\(String(format: "%.1f", loadingTime/processingTime*100))%)"
        )
        logger.info(
            "VAD inference time: \(String(format: "%.2f", inferenceTime))s (\(String(format: "%.1f", inferenceTime/processingTime*100))%)"
        )
        logger.info("Average per file: \(String(format: "%.3f", avgProcessingTime))s")
        logger.info("Min per file: \(String(format: "%.3f", minProcessingTime))s")
        logger.info("Max per file: \(String(format: "%.3f", maxProcessingTime))s")

        return VadBenchmarkResult(
            testName: "VAD_Benchmark_\(testFiles.count)_Files",
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            f1Score: metrics.f1Score,
            processingTime: processingTime,
            totalFiles: testFiles.count,
            correctPredictions: zip(predictions, groundTruth).filter { $0 == $1 }.count
        )
    }

    // VadManager.process(url:) which performs conversion internally.
    static func loadVadAudioData(_ audioFile: AVAudioFile) async throws -> [Float] {
        let converter: AudioConverter = AudioConverter()
        return try converter.resampleAudioFile(audioFile.url)
    }

    static func calculateVadMetrics(
        predictions: [Int], groundTruth: [Int]
    ) -> (
        accuracy: Float, precision: Float, recall: Float, f1Score: Float
    ) {
        guard predictions.count == groundTruth.count && !predictions.isEmpty else {
            return (0, 0, 0, 0)
        }

        var truePositives = 0
        var falsePositives = 0
        var trueNegatives = 0
        var falseNegatives = 0

        for (pred, truth) in zip(predictions, groundTruth) {
            switch (pred, truth) {
            case (1, 1): truePositives += 1
            case (1, 0): falsePositives += 1
            case (0, 0): trueNegatives += 1
            case (0, 1): falseNegatives += 1
            default: break
            }
        }

        let accuracy = Float(truePositives + trueNegatives) / Float(predictions.count) * 100
        let precision =
            truePositives + falsePositives > 0
            ? Float(truePositives) / Float(truePositives + falsePositives) * 100 : 0
        let recall =
            truePositives + falseNegatives > 0
            ? Float(truePositives) / Float(truePositives + falseNegatives) * 100 : 0
        let f1Score =
            precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0

        return (accuracy, precision, recall, f1Score)
    }

    static func calculateRTFx(
        result: VadBenchmarkResult, testFiles: [VadTestFile]
    ) async throws
        -> Double
    {
        var totalAudioDuration: TimeInterval = 0

        for testFile in testFiles {
            do {
                let audioFile = try AVAudioFile(forReading: testFile.url)
                let audioDuration =
                    Double(audioFile.length) / audioFile.processingFormat.sampleRate
                totalAudioDuration += audioDuration
            } catch {
                continue
            }
        }

        return totalAudioDuration > 0 ? result.processingTime / totalAudioDuration : 0
    }

    /// Load VOiCES subset dataset
    static func loadVoicesSubset(count: Int) async throws -> [VadTestFile]? {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let voicesDir = appSupport.appendingPathComponent("FluidAudio/voicesSubset")

        guard FileManager.default.fileExists(atPath: voicesDir.path) else {
            logger.warning(
                "VOiCES subset not found. Run: swift run fluidaudiocli download --dataset voices-subset"
            )
            return nil
        }

        logger.info("Loading VOiCES subset with mixed speech/non-speech samples...")

        var testFiles: [VadTestFile] = []

        let cleanDir = voicesDir.appendingPathComponent("clean")
        let noisyDir = voicesDir.appendingPathComponent("noisy")

        // count == -1 means "--all-files": load every VOiCES speech clip. Otherwise
        // split the budget evenly across clean + noisy speech (count / 4 each).
        let allFiles = count == -1
        let perSideSpeech = allFiles ? Int.max : max(1, count / 4)

        if FileManager.default.fileExists(atPath: cleanDir.path) {
            let cleanFiles = try loadAudioFiles(
                from: cleanDir, expectedLabel: 1, maxCount: perSideSpeech)
            testFiles.append(contentsOf: cleanFiles)
            logger.info("Loaded \(cleanFiles.count) clean speech files")
        }

        if FileManager.default.fileExists(atPath: noisyDir.path) {
            let noisyFiles = try loadAudioFiles(
                from: noisyDir, expectedLabel: 1, maxCount: perSideSpeech)
            testFiles.append(contentsOf: noisyFiles)
            logger.info("Loaded \(noisyFiles.count) noisy speech files")
        }

        let speechCount = testFiles.count
        logger.info("Loading non-speech samples from MUSAN...")
        let vadCacheDir = appSupport.appendingPathComponent("FluidAudio/vadDataset")
        let noiseDir = vadCacheDir.appendingPathComponent("noise")

        if FileManager.default.fileExists(atPath: noiseDir.path) {
            // Balance the negatives to the speech count when loading everything,
            // otherwise fill the remainder of the requested budget.
            let requestedNoiseCount = allFiles ? speechCount : max(0, count - speechCount)
            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0, maxCount: requestedNoiseCount)
            testFiles.append(contentsOf: noiseFiles)
            logger.info("Loaded \(noiseFiles.count) non-speech files from MUSAN")
        } else {
            logger.info("Downloading non-speech samples from MUSAN...")
            if let musanFiles = try await downloadHuggingFaceVadDataset(
                count: testFiles.count, dataset: "mini50")
            {
                let nonSpeechFiles = musanFiles.filter { $0.expectedLabel == 0 }
                testFiles.append(contentsOf: nonSpeechFiles)
                logger.info("Downloaded \(nonSpeechFiles.count) non-speech files")
            }
        }

        if testFiles.isEmpty {
            return nil
        }

        testFiles.shuffle()

        logger.info("Using VOiCES + MUSAN mixed dataset: \(testFiles.count) files total")
        logger.info("Speech samples: \(testFiles.filter { $0.expectedLabel == 1 }.count)")
        logger.info("Non-speech samples: \(testFiles.filter { $0.expectedLabel == 0 }.count)")
        logger.info("This tests VAD robustness in real-world acoustic conditions")
        return testFiles
    }

    /// Load full MUSAN dataset
    static func loadFullMusanDataset(count: Int) async throws -> [VadTestFile]? {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let musanDir = appSupport.appendingPathComponent("FluidAudio/musanFull/musan")

        guard FileManager.default.fileExists(atPath: musanDir.path) else {
            logger.warning(
                "Full MUSAN dataset not found. Run: swift run fluidaudiocli download --dataset musan-full"
            )
            return nil
        }

        logger.info("Loading full MUSAN dataset...")

        var testFiles: [VadTestFile] = []

        let speechDir = musanDir.appendingPathComponent("speech")
        if FileManager.default.fileExists(atPath: speechDir.path) {
            let speechFiles = try loadAudioFiles(
                from: speechDir, expectedLabel: 1, maxCount: count == -1 ? Int.max : count / 3)
            testFiles.append(contentsOf: speechFiles)
            logger.info("Loaded \(speechFiles.count) speech files")
        }

        let musicDir = musanDir.appendingPathComponent("music")
        if FileManager.default.fileExists(atPath: musicDir.path) {
            let musicFiles = try loadAudioFiles(
                from: musicDir, expectedLabel: 0, maxCount: count == -1 ? Int.max : count / 3)
            testFiles.append(contentsOf: musicFiles)
            logger.info("Loaded \(musicFiles.count) music files")
        }

        let noiseDir = musanDir.appendingPathComponent("noise")
        if FileManager.default.fileExists(atPath: noiseDir.path) {
            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0,
                maxCount: count == -1 ? Int.max : count - testFiles.count)
            testFiles.append(contentsOf: noiseFiles)
            logger.info("Loaded \(noiseFiles.count) noise files")
        }

        if testFiles.isEmpty {
            return nil
        }

        logger.info("Using full MUSAN dataset: \(testFiles.count) files total")
        return testFiles.shuffled()
    }

    static func saveVadBenchmarkResultsWithRTFx(
        _ result: VadBenchmarkResult, testFiles: [VadTestFile], to file: String
    ) async throws {
        var totalAudioDuration: TimeInterval = 0

        for testFile in testFiles {
            do {
                let audioFile = try AVAudioFile(forReading: testFile.url)
                let audioDuration =
                    Double(audioFile.length) / audioFile.processingFormat.sampleRate
                totalAudioDuration += audioDuration
            } catch {
                continue
            }
        }

        let rtfx = totalAudioDuration > 0 ? result.processingTime / totalAudioDuration : 0

        let resultsDict: [String: Any] = [
            "test_name": result.testName,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1Score,
            "processing_time_seconds": result.processingTime,
            "total_audio_duration_seconds": totalAudioDuration,
            "rtfx": rtfx,
            "avg_time_per_file": result.processingTime / Double(result.totalFiles),
            "total_files": result.totalFiles,
            "correct_predictions": result.correctPredictions,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "environment": "CLI",
        ]

        let jsonData = try JSONSerialization.data(
            withJSONObject: resultsDict, options: .prettyPrinted)
        try jsonData.write(to: URL(fileURLWithPath: file))
    }
}

#endif
