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
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-clean.tar.gz")
                .absoluteString
        case "test-other":
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-other.tar.gz")
                .absoluteString
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
        subset: String = "test-clean",
        singleFile: String? = nil,
        customVocabulary: CustomVocabularyContext? = nil,
        compareCtc: Bool = false
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

        if let singleFileName = singleFile {
            // Check if it's an absolute path that exists
            let fileUrl = URL(fileURLWithPath: singleFileName)
            if FileManager.default.fileExists(atPath: fileUrl.path) {
                let file = LibriSpeechFile(
                    fileName: fileUrl.lastPathComponent,
                    audioPath: fileUrl,
                    transcript: "i'm going to tell you a story that could change your life"  // Known transcript
                )
                filteredFiles = [file]
                logger.info("🔍 Processing custom file: \(fileUrl.path)")
            } else {
                // Fallback to searching in dataset
                let targetFileName = singleFileName.hasSuffix(".flac") ? singleFileName : "\(singleFileName).flac"
                filteredFiles = audioFiles.filter { $0.fileName == targetFileName }
                if filteredFiles.isEmpty {
                    throw ASRError.processingFailed(
                        "Single file '\(targetFileName)' not found in LibriSpeech \(subset)")
                }
                logger.info("🔍 Processing single file from dataset: \(targetFileName)")
            }
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

        // Initialize Streaming EOU Manager if needed
        var streamingEouManager: StreamingEouAsrManager?
        if config.useStreamingEou {
            streamingEouManager = StreamingEouAsrManager()
            let modelDir = URL(fileURLWithPath: "/Users/kikow/brandon/FluidAudioSwift/Models/ParakeetEOU/Streaming")
            do {
                try await streamingEouManager?.loadModels(modelDir: modelDir)
                logger.info("Initialized Streaming EOU Manager")
            } catch {
                logger.error("Failed to initialize Streaming EOU Manager: \(error)")
                throw error
            }
        }

        for (index, audioFile) in filesToProcess.enumerated() {
            do {
                logger.info(
                    "Processing file \(index + 1)/\(filesToProcess.count): \(audioFile.fileName)")

                let result: ASRBenchmarkResult
                if config.useStreamingEou {
                    result = try await processLibriSpeechFilePureCoreML(
                        manager: streamingEouManager!, file: audioFile)
                } else if config.testStreaming {
                    result = try await processLibriSpeechFileStreaming(
                        asrManager: asrManager, file: audioFile, customVocabulary: customVocabulary)
                } else if compareCtc && customVocabulary != nil {
                    // CTC comparison mode: run both baseline and CTC-boosted
                    result = try await processLibriSpeechFileWithCtcComparison(
                        asrManager: asrManager, file: audioFile, customVocabulary: customVocabulary!)
                } else {
                    result = try await processLibriSpeechFile(
                        asrManager: asrManager, file: audioFile, customVocabulary: customVocabulary)
                }
                results.append(result)

            } catch {
                logger.error("Failed to process \(audioFile.fileName): \(error)")
            }
        }

        return results
    }

    /// Process a single LibriSpeech file using Pure CoreML pipeline
    private func processLibriSpeechFilePureCoreML(
        manager: StreamingEouAsrManager, file: LibriSpeechFile
    ) async throws
        -> ASRBenchmarkResult
    {
        let audioSamples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Read file into buffer
        let audioFile = try AVAudioFile(forReading: file.audioPath)
        let buffer = AVAudioPCMBuffer(
            pcmFormat: audioFile.processingFormat, frameCapacity: AVAudioFrameCount(audioFile.length))!
        try audioFile.read(into: buffer)

        let inferenceStartTime = Date()
        let transcript = try await manager.process(audioBuffer: buffer)
        let processingTime = Date().timeIntervalSince(inferenceStartTime)

        let metrics = calculateASRMetrics(hypothesis: transcript, reference: file.transcript)

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: transcript,
            reference: file.transcript,
            metrics: metrics,
            processingTime: processingTime,
            audioLength: audioLength
        )
    }

    /// Process a single LibriSpeech file
    private func processLibriSpeechFile(
        asrManager: AsrManager, file: LibriSpeechFile, customVocabulary: CustomVocabularyContext?
    ) async throws
        -> ASRBenchmarkResult
    {
        let audioSamples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Measure only inference time for accurate RTFx calculation
        let url = URL(fileURLWithPath: file.audioPath.path)
        let inferenceStartTime = Date()
        let asrResult = try await asrManager.transcribe(url, customVocabulary: customVocabulary)
        let processingTime = Date().timeIntervalSince(inferenceStartTime)

        let metrics = calculateASRMetrics(hypothesis: asrResult.text, reference: file.transcript)

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: asrResult.text,
            reference: file.transcript,
            metrics: metrics,
            processingTime: processingTime,
            audioLength: audioLength,
            ctcDetectedTerms: asrResult.ctcDetectedTerms,
            ctcAppliedTerms: asrResult.ctcAppliedTerms
        )
    }

    /// Process a single LibriSpeech file with CTC comparison (baseline vs CTC-boosted)
    private func processLibriSpeechFileWithCtcComparison(
        asrManager: AsrManager, file: LibriSpeechFile, customVocabulary: CustomVocabularyContext
    ) async throws
        -> ASRBenchmarkResult
    {
        let audioSamples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0
        let url = URL(fileURLWithPath: file.audioPath.path)

        // Run baseline (without custom vocabulary)
        logger.debug("   Running baseline transcription...")
        let baselineStartTime = Date()
        let baselineResult = try await asrManager.transcribe(url, customVocabulary: nil)
        let baselineTime = Date().timeIntervalSince(baselineStartTime)
        let baselineMetrics = calculateASRMetrics(hypothesis: baselineResult.text, reference: file.transcript)

        // Run CTC-boosted (with custom vocabulary)
        logger.debug("   Running CTC-boosted transcription...")
        let ctcStartTime = Date()
        let ctcResult = try await asrManager.transcribe(url, customVocabulary: customVocabulary)
        let ctcTime = Date().timeIntervalSince(ctcStartTime)
        let ctcMetrics = calculateASRMetrics(hypothesis: ctcResult.text, reference: file.transcript)

        let totalProcessingTime = baselineTime + ctcTime

        // Calculate improvement
        let werImprovement: Double?
        if baselineMetrics.wer > 0 {
            werImprovement = (baselineMetrics.wer - ctcMetrics.wer) / baselineMetrics.wer
        } else {
            werImprovement = nil
        }
        let baselinePct = String(format: "%.1f", baselineMetrics.wer * 100)
        let ctcPct = String(format: "%.1f", ctcMetrics.wer * 100)
        if let werImprovement {
            let improvementPct = String(format: "%.1f", werImprovement * 100)
            logger.info("   Baseline WER: \(baselinePct)% | CTC WER: \(ctcPct)% | Improvement: \(improvementPct)%")
        } else {
            logger.info("   Baseline WER: \(baselinePct)% | CTC WER: \(ctcPct)% | Improvement: N/A (baseline WER = 0)")
        }

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: ctcResult.text,  // Use CTC result as main hypothesis
            reference: file.transcript,
            metrics: ctcMetrics,  // Use CTC metrics as main metrics
            processingTime: totalProcessingTime,
            audioLength: audioLength,
            baselineHypothesis: baselineResult.text,
            baselineMetrics: baselineMetrics,
            ctcHypothesis: ctcResult.text,
            ctcMetrics: ctcMetrics,
            ctcDetectedTerms: ctcResult.ctcDetectedTerms,
            ctcAppliedTerms: ctcResult.ctcAppliedTerms
        )
    }

    /// Process a single LibriSpeech file with streaming simulation
    private func processLibriSpeechFileStreaming(
        asrManager: AsrManager, file: LibriSpeechFile, customVocabulary: CustomVocabularyContext?
    ) async throws
        -> ASRBenchmarkResult
    {
        let audioSamples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Streaming metrics tracking
        var chunkProcessingTimes: [TimeInterval] = []
        var firstTokenTime: Date?
        let overallStartTime = Date()

        // Calculate chunk size in samples (minimum 1 second to ensure reasonable context)
        let samplesPerChunk = max(Int(config.streamingChunkDuration * 16000.0), 16000)

        logger.info("🔍 Starting streaming simulation for \(file.fileName)")
        logger.info("🔍   Audio length: \(audioLength)s")
        logger.info("🔍   Total samples: \(audioSamples.count)")
        logger.info("🔍   Chunk duration: \(max(self.config.streamingChunkDuration, 1.0))s")
        logger.info("🔍   Samples per chunk: \(samplesPerChunk)")
        let totalChunks = (audioSamples.count + samplesPerChunk - 1) / samplesPerChunk
        logger.info("🔍   Expected total chunks: \(totalChunks)")

        // For streaming, we'll use the full file but measure chunk-by-chunk processing
        // This simulates how streaming would work with continuous audio
        var processedSamples = 0
        var accumulatedText = ""

        // Process the full audio file but track metrics as if streaming
        while processedSamples < audioSamples.count {
            let chunkNumber = chunkProcessingTimes.count + 1

            // Calculate how many samples we've "streamed" so far
            let nextChunkEnd = min(processedSamples + samplesPerChunk, audioSamples.count)
            let totalSamplesToProcess = nextChunkEnd
            let chunkSamples = nextChunkEnd - processedSamples
            let isLastChunk = nextChunkEnd >= audioSamples.count

            logger.debug(
                "🔍   Processing chunk \(chunkNumber): samples \(processedSamples) to \(nextChunkEnd) (chunkSize=\(chunkSamples), isLast=\(isLastChunk))"
            )

            // Process all audio up to this point (simulating accumulated streaming)
            let audioToProcess = Array(audioSamples[0..<totalSamplesToProcess])

            // Measure only inference time for this chunk
            let chunkInferenceStartTime = Date()
            let result = try await asrManager.transcribe(
                audioToProcess,
                source: .microphone,
                customVocabulary: customVocabulary
            )
            let chunkInferenceTime = Date().timeIntervalSince(chunkInferenceStartTime)

            // Track first token time
            if firstTokenTime == nil && !result.text.isEmpty {
                firstTokenTime = Date()
            }

            // Update accumulated text
            let previousText = accumulatedText
            accumulatedText = result.text

            // Use inference time for RTFx calculations, but keep total chunk time for debugging
            chunkProcessingTimes.append(chunkInferenceTime)

            let chunkDuration = Double(chunkSamples) / 16000.0
            logger.debug(
                "🔍   Chunk \(chunkNumber): processed \(String(format: "%.2f", chunkDuration))s in \(String(format: "%.3f", chunkInferenceTime))s (inference only)"
            )

            if isLastChunk {
                logger.debug(
                    "🔍   FINAL CHUNK \(chunkNumber): text change: '\(previousText)' -> '\(accumulatedText)'")
                logger.debug("🔍   FINAL CHUNK processing complete")
            }

            processedSamples = nextChunkEnd
        }

        // Use the final accumulated text
        let finalText = accumulatedText
        let metrics = calculateASRMetrics(hypothesis: finalText, reference: file.transcript)

        // Use sum of inference times for accurate RTFx calculation
        let totalInferenceTime = chunkProcessingTimes.reduce(0, +)
        let firstTokenLatency = firstTokenTime.map { $0.timeIntervalSince(overallStartTime) }

        // Calculate streaming metrics
        let avgChunkTime = chunkProcessingTimes.reduce(0, +) / Double(chunkProcessingTimes.count)
        let maxChunkTime = chunkProcessingTimes.max() ?? 0
        let minChunkTime = chunkProcessingTimes.min() ?? 0
        let streamingRTFx = audioLength / totalInferenceTime

        let streamingMetrics = StreamingMetrics(
            avgChunkProcessingTime: avgChunkTime,
            maxChunkProcessingTime: maxChunkTime,
            minChunkProcessingTime: minChunkTime,
            totalChunks: chunkProcessingTimes.count,
            firstTokenLatency: firstTokenLatency,
            streamingRTFx: streamingRTFx,
            chunkDuration: config.streamingChunkDuration
        )

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: finalText,
            reference: file.transcript,
            metrics: metrics,
            processingTime: totalInferenceTime,
            audioLength: audioLength,
            streamingMetrics: streamingMetrics,
            ctcDetectedTerms: nil
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

    public func getLibriSpeechDirectory() -> URL {
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
        case merge  // reference two words -> hypothesis one word
        case split  // reference one word -> hypothesis two words
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
        if m == 0 && n == 0 { return [] }

        // Helper to compare tokens after concatenation
        func concatEq(_ a: String, _ b: String) -> Bool {
            return a == b
        }

        // DP with compound transitions; store backpointers
        let INF = 1_000_000
        var dp = Array(repeating: Array(repeating: INF, count: n + 1), count: m + 1)
        enum BackType { case none, matchOrSub, ins, del, merge, split }
        var bt = Array(repeating: Array(repeating: BackType.none, count: n + 1), count: m + 1)
        // To know steps taken for merge/split
        struct Step {
            let di: Int
            let dj: Int
        }
        var step = Array(repeating: Array(repeating: Step(di: 0, dj: 0), count: n + 1), count: m + 1)

        dp[0][0] = 0
        bt[0][0] = .none
        for i in 1...m {
            dp[i][0] = i
            bt[i][0] = .del
            step[i][0] = Step(di: 1, dj: 0)
        }
        for j in 1...n {
            dp[0][j] = j
            bt[0][j] = .ins
            step[0][j] = Step(di: 0, dj: 1)
        }

        for i in 1...m {
            for j in 1...n {
                // match or substitution
                let costMS = dp[i - 1][j - 1] + (reference[i - 1] == hypothesis[j - 1] ? 0 : 1)
                if costMS < dp[i][j] {
                    dp[i][j] = costMS
                    bt[i][j] = .matchOrSub
                    step[i][j] = Step(di: 1, dj: 1)
                }
                // insertion
                let costI = dp[i][j - 1] + 1
                if costI < dp[i][j] {
                    dp[i][j] = costI
                    bt[i][j] = .ins
                    step[i][j] = Step(di: 0, dj: 1)
                }
                // deletion
                let costD = dp[i - 1][j] + 1
                if costD < dp[i][j] {
                    dp[i][j] = costD
                    bt[i][j] = .del
                    step[i][j] = Step(di: 1, dj: 0)
                }
                // merge: ref[i-2] + ref[i-1] == hyp[j-1]
                if i >= 2 {
                    let mergedRef = reference[i - 2] + reference[i - 1]
                    if concatEq(mergedRef, hypothesis[j - 1]) {
                        let costM = dp[i - 2][j - 1] + 1
                        if costM < dp[i][j] {
                            dp[i][j] = costM
                            bt[i][j] = .merge
                            step[i][j] = Step(di: 2, dj: 1)
                        }
                    }
                }
                // split: ref[i-1] == hyp[j-2] + hyp[j-1]
                if j >= 2 {
                    let mergedHyp = hypothesis[j - 2] + hypothesis[j - 1]
                    if concatEq(reference[i - 1], mergedHyp) {
                        let costS = dp[i - 1][j - 2] + 1
                        if costS < dp[i][j] {
                            dp[i][j] = costS
                            bt[i][j] = .split
                            step[i][j] = Step(di: 1, dj: 2)
                        }
                    }
                }
            }
        }

        // Backtrack
        var diffs: [WordDifference] = []
        var i = m
        var j = n
        var position = max(m, n) - 1
        while i > 0 || j > 0 {
            let b = bt[i][j]
            let s = step[i][j]
            switch b {
            case .matchOrSub:
                if s.di == 1 && s.dj == 1 {
                    if reference[i - 1] != hypothesis[j - 1] {
                        diffs.append(
                            WordDifference(
                                position: position, reference: reference[i - 1], hypothesis: hypothesis[j - 1],
                                type: .substitution))
                    }
                    i -= 1
                    j -= 1
                    position -= 1
                } else {
                    break
                }
            case .ins:
                diffs.append(
                    WordDifference(position: position, reference: nil, hypothesis: hypothesis[j - 1], type: .insertion))
                j -= 1
                position -= 1
            case .del:
                diffs.append(
                    WordDifference(position: position, reference: reference[i - 1], hypothesis: nil, type: .deletion))
                i -= 1
                position -= 1
            case .merge:
                // two ref words -> one hyp word
                let refCombined = reference[i - 2] + " " + reference[i - 1]
                diffs.append(
                    WordDifference(
                        position: position, reference: refCombined, hypothesis: hypothesis[j - 1], type: .merge))
                i -= 2
                j -= 1
                position -= 1
            case .split:
                // one ref word -> two hyp words
                let hypCombined = hypothesis[j - 2] + " " + hypothesis[j - 1]
                diffs.append(
                    WordDifference(
                        position: position, reference: reference[i - 1], hypothesis: hypCombined, type: .split))
                i -= 1
                j -= 2
                position -= 1
            case .none:
                // Fallback to avoid infinite loop
                if i > 0 { i -= 1 }
                if j > 0 { j -= 1 }
                position -= 1
            }
        }
        return diffs.reversed()
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

    /// Generate inline diff without ANSI colors, wrapping mismatches in square brackets.
    private func generateInlineDiffNoColor(reference: [String], hypothesis: [String]) -> (String, String) {
        let m = reference.count
        let n = hypothesis.count

        if n == 0 {
            let refString = reference.map { "[\($0)]" }.joined(separator: " ")
            return (refString, "")
        }
        if m == 0 {
            let hypString = hypothesis.map { "[\($0)]" }.joined(separator: " ")
            return ("", hypString)
        }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }
        for i in 1...m {
            for j in 1...n {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                }
            }
        }

        var i = m
        var j = n
        var refDiffWords: [(String, Bool)] = []
        var hypDiffWords: [(String, Bool)] = []
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] {
                refDiffWords.insert((reference[i - 1], false), at: 0)
                hypDiffWords.insert((hypothesis[j - 1], false), at: 0)
                i -= 1
                j -= 1
            } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
                refDiffWords.insert((reference[i - 1], true), at: 0)
                hypDiffWords.insert((hypothesis[j - 1], true), at: 0)
                i -= 1
                j -= 1
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
                refDiffWords.insert((reference[i - 1], true), at: 0)
                i -= 1
            } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
                hypDiffWords.insert((hypothesis[j - 1], true), at: 0)
                j -= 1
            } else {
                break
            }
        }

        let refString = refDiffWords.map { $0.1 ? "[\($0.0)]" : $0.0 }.joined(separator: " ")
        let hypString = hypDiffWords.map { $0.1 ? "[\($0.0)]" : $0.0 }.joined(separator: " ")
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
        var streamingChunkDuration = 10.0
        var useStreamingEou = false
        var modelVersion: AsrModelVersion = .v3  // Default to v3
        // Filtering/sorting options (defaults enabled)
        var minWER: Double? = 0.01  // Default: filter out WER < 1%
        var sortByWERDescending: Bool = false  // Default: sort results by WER high→low
        // Diff options
        var includeDiffs: Bool = false  // Include per-file word-level mismatches in JSON
        // CTC comparison mode
        var compareCtc: Bool = false

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
            case "--streaming-eou":
                useStreamingEou = true
            case "--dump-features":
                // Enable debug features if this flag is present
                debugMode = true
            case "--chunk-duration":
                if i + 1 < arguments.count {
                    if let duration = Double(arguments[i + 1]) {
                        streamingChunkDuration = duration
                    }
                    i += 1
                }
            case "--model-version":
                if i + 1 < arguments.count {
                    let versionString = arguments[i + 1].lowercased()
                    switch versionString {
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
            case "--min-wer":
                if i + 1 < arguments.count {
                    if let raw = Double(arguments[i + 1]) {
                        // Support either fraction (0.01) or percent (1.0 for 1%)
                        minWER = raw > 1.0 ? (raw / 100.0) : raw
                    } else {
                        logger.error("Invalid --min-wer value: \(arguments[i + 1])")
                        exit(1)
                    }
                    i += 1
                } else {
                    logger.error("--min-wer requires a value (e.g., 0.01 or 1 for 1%)")
                    exit(1)
                }
            case "--no-min-wer":
                // Disable WER filtering
                minWER = nil
            case "--sort-wer":
                // Enable descending sort by WER (highest to lowest)
                sortByWERDescending = true
            case "--no-sort-wer":
                // Disable WER sorting
                sortByWERDescending = false
            case "--include-diffs":
                includeDiffs = true
            case "--compare-ctc":
                compareCtc = true
            default:
                break
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
        logger.info("   Streaming EOU: \(useStreamingEou ? "enabled" : "disabled")")
        if testStreaming {
            logger.info("   Chunk duration: \(streamingChunkDuration)s")
        }
        if let minWER {
            logger.info("   Min WER filter: \(String(format: "%.2f", minWER * 100))%")
        } else {
            logger.info("   Min WER filter: disabled")
        }
        logger.info("   Sort by WER: \(sortByWERDescending ? "descending (high→low)" : "disabled")")
        logger.info("   Compare CTC: \(compareCtc ? "enabled" : "disabled")")

        let config = ASRBenchmarkConfig(
            dataset: "librispeech",
            subset: subset,
            maxFiles: maxFiles,
            debugMode: debugMode,
            longAudioOnly: false,
            testStreaming: testStreaming,
            streamingChunkDuration: streamingChunkDuration,
            useStreamingEou: useStreamingEou
        )

        let benchmark = ASRBenchmark(config: config)

        // Initialize ASR manager with fast benchmark preset
        let asrConfig = ASRConfig(
            tdtConfig: TdtConfig()
        )

        let asrManager = AsrManager(config: asrConfig)

        do {
            // If dumping features, we must be in streaming-eou mode and single file
            let dumpFeatures = arguments.contains("--dump-features")

            if dumpFeatures {
                guard useStreamingEou, let singleFile = singleFile else {
                    logger.error("Error: --dump-features requires --streaming-eou and --single-file")
                    exit(1)
                }

                logger.info("Running in Feature Dump Mode")

                let streamingEouManager = StreamingEouAsrManager(debugFeatures: true)
                let modelDir = URL(fileURLWithPath: "/Users/kikow/brandon/FluidAudioSwift/Models/ParakeetEOU/Streaming")
                try await streamingEouManager.loadModels(modelDir: modelDir)

                // Process single file
                let fileUrl = URL(fileURLWithPath: singleFile)

                let audioFile = try AVAudioFile(forReading: fileUrl)
                let buffer = AVAudioPCMBuffer(
                    pcmFormat: audioFile.processingFormat, frameCapacity: AVAudioFrameCount(audioFile.length))!
                try audioFile.read(into: buffer)

                _ = try await streamingEouManager.process(audioBuffer: buffer)
                _ = try await streamingEouManager.finish()

                let outputUrl = URL(fileURLWithPath: "coreml_mel_features.json")
                try await streamingEouManager.saveDebugFeatures(to: outputUrl)

                logger.info("Done. Features dumped to coreml_mel_features.json")
                exit(0)
            }

            let startBenchmark = Date()

            logger.info("Initializing ASR system...")
            do {
                let models = try await AsrModels.downloadAndLoad(version: modelVersion)
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

            // Load custom vocabulary if provided in arguments; support overrides
            // Supports both JSON format (.json) and simple text format (.txt)
            var customVocab: CustomVocabularyContext? = nil
            var customVocabPath: String? = nil
            if let idx = arguments.firstIndex(of: "--custom-vocab"), idx + 1 < arguments.count {
                let path = arguments[idx + 1]
                do {
                    let url = URL(fileURLWithPath: path)
                    let isJson = url.pathExtension.lowercased() == "json"
                    var ctx: CustomVocabularyContext
                    if isJson {
                        ctx = try CustomVocabularyContext.loadWithSentencePieceTokenization(from: url)
                    } else {
                        ctx = try CustomVocabularyContext.loadFromSimpleFormatWithTokenization(from: url)
                    }
                    customVocabPath = path
                    // Optional overrides
                    if let aIdx = arguments.firstIndex(of: "--alpha"), aIdx + 1 < arguments.count,
                        let val = Float(arguments[aIdx + 1])
                    {
                        ctx = CustomVocabularyContext(
                            terms: ctx.terms, alpha: val, contextScore: ctx.contextScore,
                            depthScaling: ctx.depthScaling, scorePerPhrase: ctx.scorePerPhrase)
                    }
                    if let cIdx = arguments.firstIndex(of: "--context-score"), cIdx + 1 < arguments.count,
                        let val = Float(arguments[cIdx + 1])
                    {
                        ctx = CustomVocabularyContext(
                            terms: ctx.terms, alpha: ctx.alpha, contextScore: val, depthScaling: ctx.depthScaling,
                            scorePerPhrase: ctx.scorePerPhrase)
                    }
                    if let dIdx = arguments.firstIndex(of: "--depth-scaling"), dIdx + 1 < arguments.count,
                        let val = Float(arguments[dIdx + 1])
                    {
                        ctx = CustomVocabularyContext(
                            terms: ctx.terms, alpha: ctx.alpha, contextScore: ctx.contextScore, depthScaling: val,
                            scorePerPhrase: ctx.scorePerPhrase)
                    }
                    customVocab = ctx
                    let termCount = ctx.terms.count
                    let alphaStr = String(format: "%.2f", ctx.alpha)
                    let cStr = String(format: "%.2f", ctx.contextScore)
                    logger.info("Loaded custom vocabulary: \(termCount) terms, alpha=\(alphaStr), contextScore=\(cStr)")
                } catch {
                    logger.error("Failed to load custom vocabulary at \(path): \(error.localizedDescription)")
                    exit(1)
                }
            }

            let results = try await benchmark.runLibriSpeechBenchmark(
                asrManager: asrManager, subset: subset, singleFile: singleFile, customVocabulary: customVocab,
                compareCtc: compareCtc)

            // Apply optional filtering/sorting before summarizing and exporting
            var filteredSortedResults = results
            if let minWER {
                filteredSortedResults = filteredSortedResults.filter { $0.metrics.wer >= minWER }
            }
            if sortByWERDescending {
                filteredSortedResults.sort { $0.metrics.wer > $1.metrics.wer }
            }

            let totalWER =
                filteredSortedResults.isEmpty
                ? 0.0
                : filteredSortedResults.reduce(0.0) { $0 + $1.metrics.wer } / Double(filteredSortedResults.count)
            let totalCER =
                filteredSortedResults.isEmpty
                ? 0.0
                : filteredSortedResults.reduce(0.0) { $0 + $1.metrics.cer } / Double(filteredSortedResults.count)

            let rtfxValues = filteredSortedResults.map { Float($0.rtfx) }
            let sortedRTFx = rtfxValues.sorted()
            let medianRTFx = sortedRTFx.isEmpty ? 0 : sortedRTFx[sortedRTFx.count / 2]

            let totalAudioDuration = filteredSortedResults.reduce(0.0) { $0 + $1.audioLength }
            let totalProcessingTime = filteredSortedResults.reduce(0.0) { $0 + $1.processingTime }

            let werValues = filteredSortedResults.map { $0.metrics.wer }
            let sortedWER = werValues.sorted()
            let medianWER = sortedWER.isEmpty ? 0 : sortedWER[sortedWER.count / 2]

            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "MM/dd/yyyy, h:mm a zzz"
            let dateString = dateFormatter.string(from: Date())

            let endTime = Date()
            let testRuntime = endTime.timeIntervalSince(startBenchmark)
            let minutes = Int(testRuntime) / 60
            let seconds = Int(testRuntime) % 60
            let runtimeString = "\(minutes)m \(seconds)s"

            // Print streaming metrics if available
            if config.testStreaming {
                logger.info("--- Streaming Metrics ---")

                // Calculate aggregate streaming metrics
                let streamingResults = filteredSortedResults.compactMap { $0.streamingMetrics }
                if !streamingResults.isEmpty {
                    let avgChunkTime =
                        streamingResults.map { $0.avgChunkProcessingTime }.reduce(0, +) / Double(streamingResults.count)
                    let maxChunkTime = streamingResults.map { $0.maxChunkProcessingTime }.max() ?? 0
                    let totalChunks = streamingResults.map { $0.totalChunks }.reduce(0, +)
                    let avgFirstTokenLatency =
                        streamingResults.compactMap { $0.firstTokenLatency }.reduce(0, +)
                        / Double(streamingResults.compactMap { $0.firstTokenLatency }.count)

                    logger.info("   Chunk duration: \(config.streamingChunkDuration)s")
                    logger.info("   Total chunks processed: \(totalChunks)")
                    logger.info("   Avg chunk processing time: \(String(format: "%.3f", avgChunkTime))s")
                    logger.info("   Max chunk processing time: \(String(format: "%.3f", maxChunkTime))s")
                    if streamingResults.compactMap({ $0.firstTokenLatency }).count > 0 {
                        logger.info("   Avg first token latency: \(String(format: "%.3f", avgFirstTokenLatency))s")
                    }
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

            if let customVocabPath {
                configDict["customVocabularyPath"] = customVocabPath
                configDict["ctcTokenization"] = "sentencepiece"
            }

            if config.testStreaming {
                configDict["testStreaming"] = config.testStreaming
                configDict["streamingChunkDuration"] = config.streamingChunkDuration
            }

            var summaryDict: [String: Any] = [
                "filesProcessed": filteredSortedResults.count,
                "averageWER": totalWER,
                "medianWER": medianWER,
                "averageCER": totalCER,
                "medianRTFx": medianRTFx,
                "overallRTFx": overallRTFx,
                "totalAudioDuration": totalAudioDuration,
                "totalProcessingTime": totalProcessingTime,
            ]

            // Add CTC comparison summary if available
            if compareCtc {
                let ctcResults = filteredSortedResults.filter { $0.baselineMetrics != nil && $0.ctcMetrics != nil }
                if !ctcResults.isEmpty {
                    let avgBaselineWER =
                        ctcResults.reduce(0.0) { $0 + ($1.baselineMetrics?.wer ?? 0.0) } / Double(ctcResults.count)
                    let avgCtcWER =
                        ctcResults.reduce(0.0) { $0 + ($1.ctcMetrics?.wer ?? 0.0) } / Double(ctcResults.count)
                    let avgImprovement =
                        ctcResults.reduce(0.0) { $0 + ($1.werImprovement ?? 0.0) } / Double(ctcResults.count)

                    summaryDict["ctcComparison"] = [
                        "averageBaselineWER": avgBaselineWER,
                        "averageCtcWER": avgCtcWER,
                        "averageWerImprovement": avgImprovement,
                        "filesCompared": ctcResults.count,
                    ]
                }
            }

            // Add streaming summary if available
            if config.testStreaming {
                let streamingResults = results.compactMap { $0.streamingMetrics }
                if !streamingResults.isEmpty {
                    let avgChunkTime =
                        streamingResults.map { $0.avgChunkProcessingTime }.reduce(0, +) / Double(streamingResults.count)
                    let maxChunkTime = streamingResults.map { $0.maxChunkProcessingTime }.max() ?? 0
                    let totalChunks = streamingResults.map { $0.totalChunks }.reduce(0, +)
                    let firstTokenLatencies = streamingResults.compactMap { $0.firstTokenLatency }

                    var streamingSummary: [String: Any] = [
                        "avgChunkProcessingTime": avgChunkTime,
                        "maxChunkProcessingTime": maxChunkTime,
                        "totalChunksProcessed": totalChunks,
                    ]

                    if !firstTokenLatencies.isEmpty {
                        streamingSummary["avgFirstTokenLatency"] =
                            firstTokenLatencies.reduce(0, +) / Double(firstTokenLatencies.count)
                    }

                    summaryDict["streaming"] = streamingSummary
                }
            }

            let output =
                [
                    "config": configDict,
                    "summary": summaryDict,
                    "results": filteredSortedResults.map { result in
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

                        // Add CTC comparison data if available
                        if let baselineMetrics = result.baselineMetrics, let ctcMetrics = result.ctcMetrics {
                            resultDict["baselineWER"] = baselineMetrics.wer
                            resultDict["ctcWER"] = ctcMetrics.wer
                            resultDict["werImprovement"] = result.werImprovement ?? 0.0
                            resultDict["baselineHypothesis"] = result.baselineHypothesis ?? ""
                            resultDict["ctcHypothesis"] = result.ctcHypothesis ?? ""
                        }

                        if includeDiffs {
                            // Build word-level differences using normalized texts
                            let normalizedReference = TextNormalizer.normalize(result.reference)
                            let normalizedHypothesis = TextNormalizer.normalize(result.hypothesis)
                            let refWords = normalizedReference.components(separatedBy: .whitespacesAndNewlines)
                                .filter { !$0.isEmpty }
                            let hypWords = normalizedHypothesis.components(separatedBy: .whitespacesAndNewlines)
                                .filter { !$0.isEmpty }
                            let diffs = benchmark.generateWordDifferences(reference: refWords, hypothesis: hypWords)
                            let diffsArray: [[String: Any]] = diffs.map { d in
                                var entry: [String: Any] = [
                                    "position": d.position
                                ]
                                switch d.type {
                                case .substitution: entry["type"] = "substitution"
                                case .insertion: entry["type"] = "insertion"
                                case .deletion: entry["type"] = "deletion"
                                case .merge: entry["type"] = "merge"
                                case .split: entry["type"] = "split"
                                }
                                entry["reference"] = d.reference ?? NSNull()
                                entry["hypothesis"] = d.hypothesis ?? NSNull()
                                return entry
                            }
                            resultDict["differences"] = diffsArray

                            // Also provide inline marked strings (no ANSI) for quick viewing
                            let (refInline, hypInline) = benchmark.generateInlineDiffNoColor(
                                reference: refWords, hypothesis: hypWords)
                            resultDict["referenceDiffInline"] = refInline
                            resultDict["hypothesisDiffInline"] = hypInline
                        }

                        // Add streaming metrics if available
                        if let streamingMetrics = result.streamingMetrics {
                            resultDict["streamingMetrics"] = [
                                "avgChunkProcessingTime": streamingMetrics.avgChunkProcessingTime,
                                "maxChunkProcessingTime": streamingMetrics.maxChunkProcessingTime,
                                "minChunkProcessingTime": streamingMetrics.minChunkProcessingTime,
                                "totalChunks": streamingMetrics.totalChunks,
                                "firstTokenLatency": streamingMetrics.firstTokenLatency as Any,
                                "streamingRTFx": streamingMetrics.streamingRTFx,
                                "chunkDuration": streamingMetrics.chunkDuration,
                            ]
                        }

                        if let detected = result.ctcDetectedTerms, !detected.isEmpty {
                            resultDict["ctcDetectedTerms"] = detected
                        }
                        if let applied = result.ctcAppliedTerms, !applied.isEmpty {
                            resultDict["ctcAppliedTerms"] = applied
                        }

                        return resultDict
                    },
                ] as [String: Any]

            let jsonData = try JSONSerialization.data(
                withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: URL(fileURLWithPath: outputFile))

            // Print detailed analysis for files with high WER (use filtered set)
            benchmark.printDetailedWERAnalysis(filteredSortedResults)

            logger.info(
                "\(filteredSortedResults.count) files per dataset • Test runtime: \(runtimeString) • \(dateString)")

            print("--- Benchmark Results ---")
            print("   Dataset: \(config.dataset) \(config.subset)")
            print("   Files processed: \(filteredSortedResults.count)")

            print("   Average WER: \(String(format: "%.1f", totalWER * 100))%")
            print("   Median WER: \(String(format: "%.1f", medianWER * 100))%")
            print("   Average CER: \(String(format: "%.1f", totalCER * 100))%")
            print("   Median RTFx: \(String(format: "%.1f", medianRTFx))x")
            print(
                "   Overall RTFx: \(String(format: "%.1f", overallRTFx))x (\(String(format: "%.1f", totalAudioDuration))s / \(String(format: "%.1f", totalProcessingTime))s)"
            )

            // Print CTC comparison results if available
            if compareCtc {
                let ctcResults = filteredSortedResults.filter { $0.baselineMetrics != nil && $0.ctcMetrics != nil }
                if !ctcResults.isEmpty {
                    let avgBaselineWER =
                        ctcResults.reduce(0.0) { $0 + ($1.baselineMetrics?.wer ?? 0.0) } / Double(ctcResults.count)
                    let avgCtcWER =
                        ctcResults.reduce(0.0) { $0 + ($1.ctcMetrics?.wer ?? 0.0) } / Double(ctcResults.count)
                    let avgImprovement = (avgBaselineWER - avgCtcWER) / avgBaselineWER

                    print("\n--- CTC Comparison Results ---")
                    print("   Files compared: \(ctcResults.count)")
                    print("   Average baseline WER: \(String(format: "%.1f", avgBaselineWER * 100))%")
                    print("   Average CTC WER: \(String(format: "%.1f", avgCtcWER * 100))%")
                    print("   Average WER improvement: \(String(format: "%.1f", avgImprovement * 100))%")
                }
            }
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
                --custom-vocab <path>     Load custom vocabulary JSON for context boosting (batch mode)
                --compare-ctc             Run both baseline and CTC-boosted transcriptions for comparison (requires --custom-vocab)
                --min-wer <value>        Filter results by minimum WER (default: 1%). Accepts fraction (0.01) or percent (1 = 1%).
                --no-min-wer             Disable minimum WER filtering
                --sort-wer               Sort output results by WER descending (default: enabled)
                --no-sort-wer            Disable sorting by WER
                --include-diffs           Include word-level differences per file in JSON output
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
                - Per-chunk processing latency
                - First token latency
                - Streaming real-time factor (RTFx)
                - Min/max/average chunk processing times

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

                # Only include files with WER >= 1% and sort by WER (desc)
                fluidaudio asr-benchmark --subset test-clean --min-wer 1 --sort-wer

                # Compare baseline vs CTC-boosted transcriptions
                fluidaudio asr-benchmark --subset test-clean --max-files 100 \\
                    --custom-vocab custom_vocabulary.json --compare-ctc

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
