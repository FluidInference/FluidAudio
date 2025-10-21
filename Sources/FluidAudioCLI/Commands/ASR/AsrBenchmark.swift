#if os(macOS)
import AVFoundation
import FluidAudio
import OSLog

/// LibriSpeech dataset manager and ASR benchmarking
public class ASRBenchmark {

    private let logger = AppLogger(category: "Benchmark")
    private let config: ASRBenchmarkConfig

    public init(config: ASRBenchmarkConfig = ASRBenchmarkConfig()) {
        self.config = config
    }

    /// Download LibriSpeech test datasets
    public func downloadLibriSpeech(
        subset: String = "test-clean", forceDownload: Bool = false
    )
        async throws
    {
        let datasetsDirectory = getLibriSpeechDirectory()
        let subsetDirectory = datasetsDirectory.appendingPathComponent(subset)

        // Check if already downloaded by looking for transcript files (which indicate complete download)
        if !forceDownload && FileManager.default.fileExists(atPath: subsetDirectory.path) {
            let enumerator = FileManager.default.enumerator(
                at: subsetDirectory, includingPropertiesForKeys: nil)
            var transcriptCount = 0

            while let url = enumerator?.nextObject() as? URL {
                if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                    transcriptCount += 1
                    if transcriptCount >= 5 {  // Found enough transcript files, dataset exists
                        break
                    }
                }
            }

            if transcriptCount >= 5 {
                logger.info("LibriSpeech \(subset) already downloaded")
                logger.info("LibriSpeech \(subset) already available (dataset found)")
                return
            }
        }

        logger.info("Downloading LibriSpeech \(subset)...")

        let downloadURL: String
        switch subset {
        case "test-clean":
            downloadURL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
        case "test-other":
            downloadURL = "https://www.openslr.org/resources/12/test-other.tar.gz"
        case "dev-clean":
            downloadURL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
        case "dev-other":
            downloadURL = "https://www.openslr.org/resources/12/dev-other.tar.gz"
        default:
            throw ASRError.processingFailed("Unsupported LibriSpeech subset: \(subset)")
        }

        try await downloadAndExtractTarGz(
            url: downloadURL,
            extractTo: datasetsDirectory,
            expectedSubpath: "LibriSpeech/\(subset)"
        )

        logger.info("LibriSpeech \(subset) downloaded successfully")
    }

    /// Run ASR benchmark on LibriSpeech
    public func runLibriSpeechBenchmark(
        asrManager: AsrManager,
        models: AsrModels,
        subset: String = "test-clean",
        singleFile: String? = nil
    )
        async throws -> [ASRBenchmarkResult]
    {
        #if DEBUG
        logger.warning("WARNING: Running in DEBUG mode!")
        logger.warning("For accurate benchmarks, use: swift run -c release fluidaudio asr-benchmark")
        // Add a small delay so user sees the warning
        try? await Task.sleep(nanoseconds: 2_000_000_000)  // 2 seconds
        #else
        logger.info("Running in RELEASE mode - optimal performance")
        #endif

        // Ensure dataset is downloaded
        try await downloadLibriSpeech(subset: subset)

        let datasetPath = getLibriSpeechDirectory().appendingPathComponent(subset)
        let audioFiles = try collectLibriSpeechFiles(from: datasetPath)

        var filteredFiles = audioFiles

        // Handle single file processing
        if let singleFileName = singleFile {
            let targetFileName = singleFileName.hasSuffix(".flac") ? singleFileName : "\(singleFileName).flac"
            filteredFiles = audioFiles.filter { $0.fileName == targetFileName }
            if filteredFiles.isEmpty {
                throw ASRError.processingFailed("Single file '\(targetFileName)' not found in LibriSpeech \(subset)")
            }
            logger.info("🔍 Processing single file: \(targetFileName)")
        } else if config.longAudioOnly {
            filteredFiles = try await filterFilesByDuration(
                audioFiles, minDuration: 4.0, maxDuration: 20.0)
            logger.info(
                "Filtered to \(filteredFiles.count) files with duration 4-20 seconds (from \(audioFiles.count) total)"
            )
        }

        let maxFiles = singleFile != nil ? filteredFiles.count : (config.maxFiles ?? filteredFiles.count)
        let filesToProcess = Array(filteredFiles.prefix(maxFiles))

        logger.info(
            "📋 Processing \(filesToProcess.count) files (max files limit: \(config.maxFiles?.description ?? "unlimited"))"
        )

        logger.info(
            "Running ASR benchmark on \(filesToProcess.count) files from LibriSpeech \(subset)")

        var results: [ASRBenchmarkResult] = []

        for (index, audioFile) in filesToProcess.enumerated() {
            do {
                logger.info(
                    "Processing file \(index + 1)/\(filesToProcess.count): \(audioFile.fileName)")

                let result: ASRBenchmarkResult
                if config.testStreaming {
                    result = try await processLibriSpeechFileStreaming(
                        models: models, file: audioFile)
                } else {
                    result = try await processLibriSpeechFile(
                        asrManager: asrManager, file: audioFile)
                }
                results.append(result)

            } catch {
                logger.error("Failed to process \(audioFile.fileName): \(error)")
            }
        }

        return results
    }

    /// Process a single LibriSpeech file
    private func processLibriSpeechFile(
        asrManager: AsrManager, file: LibriSpeechFile
    ) async throws
        -> ASRBenchmarkResult
    {
        let audioSamples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Measure only inference time for accurate RTFx calculation
        let url = URL(fileURLWithPath: file.audioPath.path)
        let inferenceStartTime = Date()
        let asrResult = try await asrManager.transcribe(url)
        let processingTime = Date().timeIntervalSince(inferenceStartTime)

        let metrics = calculateASRMetrics(hypothesis: asrResult.text, reference: file.transcript)

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: asrResult.text,
            reference: file.transcript,
            metrics: metrics,
            processingTime: processingTime,
            audioLength: audioLength
        )
    }

    /// Process a single LibriSpeech file using the streaming pipeline with low-latency settings
    private func processLibriSpeechFileStreaming(
        models: AsrModels, file: LibriSpeechFile
    ) async throws -> ASRBenchmarkResult {
        let audioConverter = AudioConverter()
        let audioSamples = try audioConverter.resampleAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        logger.info("🔍 Starting streaming benchmark for \(file.fileName)")
        logger.info("🔍   Audio length: \(String(format: "%.2f", audioLength))s")
        logger.info(
            "🔍   Using streaming config (chunk=\(StreamingAsrConfig.streaming.chunkSeconds)s, left=\(StreamingAsrConfig.streaming.leftContextSeconds)s, right=\(StreamingAsrConfig.streaming.rightContextSeconds)s)"
        )

        let streamingConfig = StreamingAsrConfig.streaming.withVad(.disabled)
        let streamingAsr = StreamingAsrManager(config: streamingConfig)
        try await streamingAsr.start(models: models, source: .system)

        let chunkDuration: TimeInterval = 0.5  // Align with CLI streaming simulation cadence
        let samplesPerChunk = Int(chunkDuration * 16000.0)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        let streamingStart = Date()
        let latencyTask = Task { () -> (Double?, Double?) in
            var firstTokenLatency: Double?
            var firstConfirmedTokenLatency: Double?
            let updates = await streamingAsr.transcriptionUpdates
            for await update in updates {
                let normalized = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !normalized.isEmpty else { continue }

                let latency = update.timestamp.timeIntervalSince(streamingStart)
                if firstTokenLatency == nil, latency.isFinite {
                    firstTokenLatency = max(0, latency)
                }
                if update.isConfirmed, firstConfirmedTokenLatency == nil, latency.isFinite {
                    firstConfirmedTokenLatency = max(0, latency)
                }
            }
            return (firstTokenLatency, firstConfirmedTokenLatency)
        }

        var position = 0
        while position < audioSamples.count {
            let chunkSamples = min(samplesPerChunk, audioSamples.count - position)
            guard
                let chunkBuffer = AVAudioPCMBuffer(
                    pcmFormat: format,
                    frameCapacity: AVAudioFrameCount(chunkSamples)
                )
            else {
                throw ASRError.processingFailed("Failed to allocate streaming buffer for chunk")
            }
            chunkBuffer.frameLength = AVAudioFrameCount(chunkSamples)

            if let channelData = chunkBuffer.floatChannelData {
                audioSamples[position..<position + chunkSamples].withUnsafeBufferPointer { pointer in
                    channelData[0].update(from: pointer.baseAddress!, count: chunkSamples)
                }
            }

            await streamingAsr.streamAudio(chunkBuffer)
            position += chunkSamples
        }

        let finalText: String
        do {
            finalText = try await streamingAsr.finish()
        } catch {
            latencyTask.cancel()
            throw error
        }

        let (firstTokenLatency, firstConfirmedTokenLatency) = await latencyTask.value
        let totalProcessingTime = Date().timeIntervalSince(streamingStart)

        let metrics = calculateASRMetrics(hypothesis: finalText, reference: file.transcript)

        let streamingMetrics = StreamingMetrics(
            firstTokenLatency: firstTokenLatency,
            firstConfirmedTokenLatency: firstConfirmedTokenLatency,
            streamingRTFx: totalProcessingTime > 0 ? audioLength / totalProcessingTime : 0,
            chunkDuration: StreamingAsrConfig.streaming.chunkSeconds
        )

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: finalText,
            reference: file.transcript,
            metrics: metrics,
            processingTime: totalProcessingTime,
            audioLength: audioLength,
            streamingMetrics: streamingMetrics
        )
    }

    /// Calculate WER and CER metrics with HuggingFace-compatible normalization
    public func calculateASRMetrics(hypothesis: String, reference: String) -> ASRMetrics {
        let metrics = WERCalculator.calculateWERAndCER(hypothesis: hypothesis, reference: reference)
        return ASRMetrics(
            wer: metrics.wer,
            cer: metrics.cer,
            insertions: metrics.insertions,
            deletions: metrics.deletions,
            substitutions: metrics.substitutions,
            totalWords: metrics.totalWords,
            totalCharacters: metrics.totalCharacters
        )
    }

    // MARK: - Private Helper Methods

    /// Filter files by duration range
    private func filterFilesByDuration(
        _ files: [LibriSpeechFile], minDuration: Double, maxDuration: Double
    ) async throws -> [LibriSpeechFile] {
        var filteredFiles: [LibriSpeechFile] = []

        for file in files {
            do {
                let audioSamples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
                let duration = Double(audioSamples.count) / 16000.0

                if duration >= minDuration && duration <= maxDuration {
                    filteredFiles.append(file)
                }
            } catch {
                logger.warning(
                    "Could not load audio file \(file.fileName): \(error.localizedDescription)")
                continue
            }
        }

        return filteredFiles
    }

    private func getLibriSpeechDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDirectory = applicationSupportURL.appendingPathComponent(
            "FluidAudio", isDirectory: true)
        return appDirectory.appendingPathComponent("Datasets/LibriSpeech", isDirectory: true)
    }

    private func collectLibriSpeechFiles(from directory: URL) throws -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []

        let fileManager = FileManager.default
        let enumerator = fileManager.enumerator(at: directory, includingPropertiesForKeys: nil)

        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                let transcriptContent = try String(contentsOf: url)
                let lines = transcriptContent.components(separatedBy: .newlines).filter {
                    !$0.isEmpty
                }

                for line in lines {
                    let parts = line.components(separatedBy: " ")
                    guard parts.count >= 2 else { continue }

                    let audioId = parts[0]
                    let transcript = parts.dropFirst().joined(separator: " ")

                    let audioFileName = "\(audioId).flac"
                    let audioPath = url.deletingLastPathComponent().appendingPathComponent(
                        audioFileName)

                    if fileManager.fileExists(atPath: audioPath.path) {
                        files.append(
                            LibriSpeechFile(
                                fileName: audioFileName,
                                audioPath: audioPath,
                                transcript: transcript
                            ))
                    }
                }
            }
        }

        return files.sorted { $0.fileName < $1.fileName }
    }

    private func downloadAndExtractTarGz(
        url: String, extractTo: URL, expectedSubpath: String
    )
        async throws
    {
        let downloadURL = URL(string: url)!

        logger.info("Downloading \(url)...")
        let (tempFile, _) = try await DownloadUtils.sharedSession.download(from: downloadURL)

        try FileManager.default.createDirectory(at: extractTo, withIntermediateDirectories: true)

        logger.info("Extracting archive...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", tempFile.path, "-C", extractTo.path]

        // Capture stderr for better error reporting
        let errorPipe = Pipe()
        process.standardError = errorPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw ASRError.processingFailed("Failed to extract tar.gz file: \(errorMessage)")
        }

        let extractedPath = extractTo.appendingPathComponent(expectedSubpath)
        if FileManager.default.fileExists(atPath: extractedPath.path) {
            let targetPath = extractTo.appendingPathComponent(
                expectedSubpath.components(separatedBy: "/").last!)
            try? FileManager.default.removeItem(at: targetPath)
            try FileManager.default.moveItem(at: extractedPath, to: targetPath)

            try? FileManager.default.removeItem(at: extractTo.appendingPathComponent("LibriSpeech"))
        }

        logger.info("Dataset extracted successfully")
    }
}

// MARK: - Detailed WER Analysis

private struct WordDifference {
    let position: Int
    let reference: String?
    let hypothesis: String?
    let type: DifferenceType

    enum DifferenceType {
        case substitution
        case insertion
        case deletion
    }
}

extension ASRBenchmark {
    /// Print detailed analysis for files with WER > threshold
    private func printDetailedWERAnalysis(
        _ results: [ASRBenchmarkResult], threshold: Double = ASRConstants.highWERThreshold
    ) {
        let highWERResults = results.filter { $0.metrics.wer > threshold }

        guard !highWERResults.isEmpty else {
            return
        }

        logger.info("" + String(repeating: "=", count: 80))
        logger.info("📋 Detailed Analysis for Files with WER > \(Int(threshold * 100))%")
        logger.info(String(repeating: "=", count: 80))

        for result in highWERResults.sorted(by: { $0.metrics.wer > $1.metrics.wer }) {
            printSingleFileWERAnalysis(result)
        }
    }

    /// Print detailed analysis for a single file
    private func printSingleFileWERAnalysis(_ result: ASRBenchmarkResult) {
        let werPercent = result.metrics.wer * 100
        logger.info(
            "File: \(result.fileName) (WER: \(String(format: "%.1f", werPercent))%) (Duration: \(String(format: "%.2f", result.audioLength))s)"
        )
        logger.info(String(repeating: "-", count: 60))

        // Normalize the texts for comparison
        let normalizedReference = TextNormalizer.normalize(result.reference)
        let normalizedHypothesis = TextNormalizer.normalize(result.hypothesis)

        let refWords = normalizedReference.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let hypWords = normalizedHypothesis.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

        // Generate inline diff
        let (referenceDiff, hypothesisDiff) = generateInlineDiff(reference: refWords, hypothesis: hypWords)

        logger.info("Normalized Reference:\t\(referenceDiff)")
        logger.info("Normalized Hypothesis:\t\(hypothesisDiff)")
        logger.info("Original Hypothesis:\t\(result.hypothesis)")
    }

    /// Generate word-level differences between reference and hypothesis
    private func generateWordDifferences(reference: [String], hypothesis: [String]) -> [WordDifference] {
        let m = reference.count
        let n = hypothesis.count
        var differences: [WordDifference] = []

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

        // Backtrack to find actual differences
        var i = m
        var j = n
        var position = max(m, n) - 1

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] {
                // Match - no difference
                i -= 1
                j -= 1
                position -= 1
            } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
                // Substitution
                differences.append(
                    WordDifference(
                        position: position,
                        reference: reference[i - 1],
                        hypothesis: hypothesis[j - 1],
                        type: .substitution
                    ))
                i -= 1
                j -= 1
                position -= 1
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
                // Deletion
                differences.append(
                    WordDifference(
                        position: position,
                        reference: reference[i - 1],
                        hypothesis: nil,
                        type: .deletion
                    ))
                i -= 1
                position -= 1
            } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
                // Insertion
                differences.append(
                    WordDifference(
                        position: position,
                        reference: nil,
                        hypothesis: hypothesis[j - 1],
                        type: .insertion
                    ))
                j -= 1
                position -= 1
            } else {
                // Shouldn't happen, but break to avoid infinite loop
                break
            }
        }

        return differences.reversed()  // Reverse to get correct order
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
}

// IMPORTANT: RTFx Performance in CI Environments
// GitHub Actions and other CI environments use virtualized M1/M2 Macs where
// Neural Engine access is severely restricted. This results in significantly
// degraded performance compared to bare metal:
// - Physical M1/M2 Mac: ~21x real-time (RTFx)
// - GitHub Actions M1: ~3x real-time (7x slower due to virtualization)
//
// For accurate RTFx benchmarking, always test on physical Apple Silicon hardware.
// The WER (Word Error Rate) metrics remain accurate in CI environments.

/// Extension to provide CLI entry point
extension ASRBenchmark {
    public static func runASRBenchmark(arguments: [String]) async {
        // Create a local logger for the static CLI entrypoint
        let logger = AppLogger(category: "Benchmark")
        var subset = "test-clean"
        var maxFiles: Int?
        var singleFile: String?
        var outputFile = "asr_benchmark_results.json"
        var debugMode = false
        var autoDownload = true  // Default to true for automatic download
        var testStreaming = false
        var streamingChunkDuration = StreamingAsrConfig.streaming.chunkSeconds
        var modelVersion: AsrModelVersion = .v3  // Default to v3

        // Check for help flag first
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--subset":
                if i + 1 < arguments.count {
                    subset = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
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
            case "--debug":
                debugMode = true
            case "--auto-download":
                autoDownload = true
            case "--no-auto-download":
                autoDownload = false
            case "--test-streaming":
                testStreaming = true
            case "--chunk-duration":
                if i + 1 < arguments.count {
                    if let duration = Double(arguments[i + 1]), duration > 0 {
                        streamingChunkDuration = duration
                    } else {
                        logger.error("Invalid chunk duration: \(arguments[i + 1])")
                        exit(1)
                    }
                    i += 1
                }
            case "--model-version":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "v2", "2":
                        modelVersion = .v2
                    case "v3", "3":
                        modelVersion = .v3
                    default:
                        logger.error("Invalid model version: \(arguments[i + 1]). Use 'v2' or 'v3'")
                        exit(1)
                    }
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        logger.info("Starting ASR benchmark on LibriSpeech \(subset)")
        if singleFile != nil {
            logger.info("   Processing single file: \(singleFile!)")
        } else {
            logger.info("   Max files: \(maxFiles?.description ?? "all")")
        }
        logger.info("   Output file: \(outputFile)")
        logger.info("   Model version: \(modelVersion == .v2 ? "v2" : "v3")")
        logger.info("   Debug mode: \(debugMode ? "enabled" : "disabled")")
        logger.info("   Auto-download: \(autoDownload ? "enabled" : "disabled")")
        logger.info("   Test streaming: \(testStreaming ? "enabled" : "disabled")")
        if testStreaming {
            logger.info(
                "   Decoder chunk duration: \(StreamingAsrConfig.streaming.chunkSeconds)s (streaming preset)"
            )
        }

        let config = ASRBenchmarkConfig(
            dataset: "librispeech",
            subset: subset,
            maxFiles: maxFiles,
            debugMode: debugMode,
            longAudioOnly: false,
            testStreaming: testStreaming,
            streamingChunkDuration: streamingChunkDuration
        )

        let benchmark = ASRBenchmark(config: config)

        // Initialize ASR manager with fast benchmark preset
        let asrConfig = ASRConfig(
            tdtConfig: TdtConfig()
        )

        let asrManager = AsrManager(config: asrConfig)

        do {
            let startBenchmark = Date()

            logger.info("Initializing ASR system...")
            let models: AsrModels
            do {
                models = try await AsrModels.downloadAndLoad(version: modelVersion)
                try await asrManager.initialize(models: models)
                logger.info("ASR system initialized successfully")

            } catch {
                logger.error("Failed to initialize ASR system: \(error)")
                logger.error("   Error type: \(type(of: error))")
                logger.error("   Error details: \(error.localizedDescription)")

                if ProcessInfo.processInfo.environment["CI"] != nil {
                    logger.debug("🔍 CI Debug Information:")
                    let modelsDir = FileManager.default.homeDirectoryForCurrentUser
                        .appendingPathComponent(
                            "Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-\(modelVersion == .v2 ? "v2" : "v3")-coreml"
                        )
                    logger.debug("Models directory: \(modelsDir.path)")
                    logger.debug(
                        "   Directory exists: \(FileManager.default.fileExists(atPath: modelsDir.path))"
                    )

                    if FileManager.default.fileExists(atPath: modelsDir.path) {
                        do {
                            let contents = try FileManager.default.contentsOfDirectory(
                                at: modelsDir, includingPropertiesForKeys: nil)
                            logger.debug("   Directory contents: \(contents.map { $0.lastPathComponent })")
                        } catch {
                            logger.debug("   Failed to list directory contents: \(error)")
                        }
                    }
                }
                throw error
            }

            if autoDownload {
                try await benchmark.downloadLibriSpeech(subset: subset)
            }

            let results = try await benchmark.runLibriSpeechBenchmark(
                asrManager: asrManager,
                models: models,
                subset: subset,
                singleFile: singleFile)

            let totalWER = results.reduce(0.0) { $0 + $1.metrics.wer } / Double(results.count)
            let totalCER = results.reduce(0.0) { $0 + $1.metrics.cer } / Double(results.count)

            let rtfxValues = results.map { Float($0.rtfx) }
            let sortedRTFx = rtfxValues.sorted()
            let medianRTFx = sortedRTFx[sortedRTFx.count / 2]

            let totalAudioDuration = results.reduce(0.0) { $0 + $1.audioLength }
            let totalProcessingTime: Double = results.reduce(0.0) { $0 + $1.processingTime }

            let werValues = results.map { $0.metrics.wer }
            let sortedWER = werValues.sorted()
            let medianWER = sortedWER[sortedWER.count / 2]

            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "MM/dd/yyyy, h:mm a zzz"
            let dateString = dateFormatter.string(from: Date())

            let endTime = Date()
            let testRuntime = endTime.timeIntervalSince(startBenchmark)
            let minutes = Int(testRuntime) / 60
            let seconds = Int(testRuntime) % 60
            let runtimeString = "\(minutes)m \(seconds)s"

            var aggregateStreamingSummary:
                (
                    averageRTFx: Double,
                    averageFirstTokenLatency: Double?,
                    averageFirstConfirmedLatency: Double?
                )?

            if config.testStreaming {
                let streamingResults = results.compactMap { $0.streamingMetrics }
                if !streamingResults.isEmpty {
                    let avgRTFx =
                        streamingResults.map { $0.streamingRTFx }.reduce(0, +) / Double(streamingResults.count)
                    let firstTokenLatencies = streamingResults.compactMap { $0.firstTokenLatency }
                    let firstConfirmedLatencies = streamingResults.compactMap { $0.firstConfirmedTokenLatency }
                    let avgFirstTokenLatency =
                        firstTokenLatencies.isEmpty
                        ? nil
                        : firstTokenLatencies.reduce(0, +) / Double(firstTokenLatencies.count)
                    let avgFirstConfirmedLatency =
                        firstConfirmedLatencies.isEmpty
                        ? nil
                        : firstConfirmedLatencies.reduce(0, +) / Double(firstConfirmedLatencies.count)

                    aggregateStreamingSummary = (
                        averageRTFx: avgRTFx,
                        averageFirstTokenLatency: avgFirstTokenLatency,
                        averageFirstConfirmedLatency: avgFirstConfirmedLatency
                    )
                }
            }

            let overallRTFx: Double = totalProcessingTime > 0 ? (totalAudioDuration / totalProcessingTime) : 0.0

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

            var configDict: [String: Any] = [
                "dataset": config.dataset,
                "subset": config.subset,
                "maxFiles": config.maxFiles as Any,
                "debugMode": config.debugMode,
            ]

            if config.testStreaming {
                configDict["testStreaming"] = config.testStreaming
                configDict["streamingChunkDuration"] = StreamingAsrConfig.streaming.chunkSeconds
            }

            var summaryDict: [String: Any] = [
                "filesProcessed": results.count,
                "averageWER": totalWER,
                "medianWER": medianWER,
                "averageCER": totalCER,
                "medianRTFx": medianRTFx,
                "overallRTFx": overallRTFx,
                "totalAudioDuration": totalAudioDuration,
                "totalProcessingTime": totalProcessingTime,
            ]

            // Add streaming summary if available
            if let streamingSummary = aggregateStreamingSummary {
                var streamingSummaryDict: [String: Any] = [
                    "averageStreamingRTFx": streamingSummary.averageRTFx
                ]
                if let firstLatency = streamingSummary.averageFirstTokenLatency {
                    streamingSummaryDict["avgFirstTokenLatency"] = firstLatency
                }
                if let confirmedLatency = streamingSummary.averageFirstConfirmedLatency {
                    streamingSummaryDict["avgFirstConfirmedTokenLatency"] = confirmedLatency
                }
                summaryDict["streaming"] = streamingSummaryDict
            }

            let output =
                [
                    "config": configDict,
                    "summary": summaryDict,
                    "results": results.map { result in
                        var resultDict: [String: Any] = [
                            "fileName": result.fileName,
                            "hypothesis": result.hypothesis,
                            "reference": result.reference,
                            "wer": result.metrics.wer,
                            "cer": result.metrics.cer,
                            "rtfx": result.rtfx,
                            "audioLength": result.audioLength,
                            "processingTime": result.processingTime,
                        ]

                        // Add streaming metrics if available
                        if let streamingMetrics = result.streamingMetrics {
                            resultDict["streamingMetrics"] = [
                                "firstTokenLatency": streamingMetrics.firstTokenLatency as Any,
                                "firstConfirmedTokenLatency": streamingMetrics.firstConfirmedTokenLatency as Any,
                                "streamingRTFx": streamingMetrics.streamingRTFx,
                                "chunkDuration": streamingMetrics.chunkDuration,
                            ]
                        }

                        return resultDict
                    },
                ] as [String: Any]

            let jsonData = try JSONSerialization.data(
                withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: URL(fileURLWithPath: outputFile))

            // Print detailed analysis for files with high WER
            benchmark.printDetailedWERAnalysis(results)

            logger.info("\(results.count) files per dataset • Test runtime: \(runtimeString) • \(dateString)")

            logger.info("--- Benchmark Results ---")
            logger.info("   Dataset: \(config.dataset) \(config.subset)")
            logger.info("   Files processed: \(results.count)")

            logger.info("   Average WER: \(String(format: "%.1f", totalWER * 100))%")
            logger.info("   Median WER: \(String(format: "%.1f", medianWER * 100))%")
            logger.info("   Average CER: \(String(format: "%.1f", totalCER * 100))%")
            logger.info("   Median RTFx: \(String(format: "%.1f", medianRTFx))x")
            logger.info(
                "   Overall RTFx: \(String(format: "%.1f", overallRTFx))x (\(String(format: "%.1f", totalAudioDuration))s / \(String(format: "%.1f", totalProcessingTime))s)"
            )

            if let streamingSummary = aggregateStreamingSummary {
                logger.info("--- Streaming Metrics ---")
                logger.info(
                    "   Decoder chunk duration: \(StreamingAsrConfig.streaming.chunkSeconds)s"
                )
                logger.info(
                    "   Average streaming RTFx: \(String(format: "%.2f", streamingSummary.averageRTFx))x"
                )
                if let firstLatency = streamingSummary.averageFirstTokenLatency {
                    logger.info("   Avg first token latency: \(String(format: "%.3f", firstLatency))s")
                }
                if let confirmedLatency = streamingSummary.averageFirstConfirmedLatency {
                    logger.info(
                        "   Avg first confirmed token latency: \(String(format: "%.3f", confirmedLatency))s"
                    )
                }
            }

            logger.info("Results saved to: \(outputFile)")
            logger.info("ASR benchmark completed successfully")

        } catch {
            logger.error("ERROR: ASR benchmark failed: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        let logger = AppLogger(category: "Benchmark")
        logger.info(
            """
            ASR Benchmark Command Usage:
                fluidaudio asr-benchmark [options]

            Options:
                --subset <name>           LibriSpeech subset to use (default: test-clean)
                                         Available: test-clean, test-other, dev-clean, dev-other
                --max-files <number>      Maximum number of files to process (default: all)
                --single-file <id>        Process only a specific file (e.g., 1089-134686-0011)
                --output <file>           Output JSON file path (default: asr_benchmark_results.json)
                --model-version <version> ASR model version to use: v2 or v3 (default: v3)
                --debug                   Enable debug logging
                --auto-download           Automatically download LibriSpeech dataset (default)
                --no-auto-download        Disable automatic dataset download
                --test-streaming          Enable streaming simulation mode
                --chunk-duration <secs>   Chunk duration for streaming mode (default: 0.1s, min: 1.0s)
                --help, -h               Show this help message

            Description:
                The ASR benchmark command evaluates Automatic Speech Recognition performance
                on the LibriSpeech dataset, calculating WER (Word Error Rate) and CER
                (Character Error Rate) metrics, along with processing speed (RTFx).

            Streaming Mode:
                When --test-streaming is enabled, the benchmark simulates real-time streaming
                by processing audio in chunks. This measures:
                - First token latency
                - First confirmed token latency
                - Streaming real-time factor (RTFx)

            Examples:
                # Basic benchmark on test-clean subset
                fluidaudio asr-benchmark

                # Benchmark with 100 files from test-other subset
                fluidaudio asr-benchmark --subset test-other --max-files 100

                # Process a single specific file
                fluidaudio asr-benchmark --single-file 1089-134686-0011 --debug

                # Test streaming performance with 0.5s chunks
                fluidaudio asr-benchmark --test-streaming --chunk-duration 1-

                # Debug mode with custom output file
                fluidaudio asr-benchmark --debug --output my_results.json

            Expected Performance:
                - test-clean: 2-6% WER for good ASR systems
                - test-other: 5-15% WER for good ASR systems
                - RTFx: >1x indicates faster than real-time processing

            Note: First run will download LibriSpeech dataset (~1.1GB for test-clean).
                  ASR models will be downloaded automatically if not present.
            """
        )
    }
}
#endif
