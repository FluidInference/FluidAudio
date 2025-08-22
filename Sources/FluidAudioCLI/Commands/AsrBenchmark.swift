#if os(macOS)
import AVFoundation
import FluidAudio
import OSLog
import Foundation

/// LibriSpeech dataset manager and ASR benchmarking
@available(macOS 13.0, *)
public class ASRBenchmark {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "Benchmark")
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
                print("LibriSpeech \(subset) already available (dataset found)")
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
        asrManager: AsrManager, subset: String = "test-clean"
    )
        async throws -> [ASRBenchmarkResult]
    {
        #if DEBUG
        print("")
        print("WARNING: Running in DEBUG mode!")
        print("Performance will be significantly slower (~2x).")
        print("For accurate benchmarks, use: swift run -c release fluidaudio asr-benchmark")
        print("")
        // Add a small delay so user sees the warning
        try? await Task.sleep(nanoseconds: 2_000_000_000)  // 2 seconds
        #else
        print("Running in RELEASE mode - optimal performance")
        #endif

        // Ensure dataset is downloaded
        try await downloadLibriSpeech(subset: subset)

        let datasetPath = getLibriSpeechDirectory().appendingPathComponent(subset)
        let audioFiles = try collectLibriSpeechFiles(from: datasetPath)

        var filteredFiles = audioFiles
        if config.longAudioOnly {
            filteredFiles = try await filterFilesByDuration(
                audioFiles, minDuration: 4.0, maxDuration: 20.0)
            print(
                "Filtered to \(filteredFiles.count) files with duration 4-20 seconds (from \(audioFiles.count) total)"
            )
        }

        let maxFiles = config.maxFiles ?? filteredFiles.count
        let filesToProcess = Array(filteredFiles.prefix(maxFiles))

        print(
            "📋 Processing \(filesToProcess.count) files (max files limit: \(config.maxFiles?.description ?? "unlimited"))"
        )

        logger.info(
            "Running ASR benchmark on \(filesToProcess.count) files from LibriSpeech \(subset)")

        var results: [ASRBenchmarkResult] = []

        for (index, audioFile) in filesToProcess.enumerated() {
            do {
                if config.debugMode {
                    logger.info(
                        "Processing file \(index + 1)/\(filesToProcess.count): \(audioFile.fileName)"
                    )
                }

                if config.debugMode && index > 0 {
                    logger.info("   🔍 Processing file \(index + 1)")
                }

                // Reset decoder state for each new file
                try await asrManager.resetDecoderState(for: .microphone)

                let result: ASRBenchmarkResult
                if config.testStreaming {
                    result = try await processLibriSpeechFileStreaming(
                        asrManager: asrManager, file: audioFile)
                } else {
                    result = try await processLibriSpeechFile(
                        asrManager: asrManager, file: audioFile)
                }

                // Log file processing details
                print("   =========================================")
                print("   File: \(audioFile.fileName)")
                print("   Duration: \(String(format: "%.2f", result.audioLength))s")

                // Show actual chunk information
                if let chunkOutputs = result.chunkOutputs {
                    print("   Chunks processed: \(chunkOutputs.count)")
                    print("   Transcription calls: \(chunkOutputs.count)")

                    // Display chunk outputs
                    print("   Chunk outputs:")
                    for chunk in chunkOutputs {
                        let timeRange =
                            "[\(String(format: "%.1f", chunk.startTime))-\(String(format: "%.1f", chunk.endTime))s]"
                        print("     \(timeRange): \"\(chunk.text)\"")
                    }
                } else if let streamingMetrics = result.streamingMetrics {
                    print("   Chunks processed: \(streamingMetrics.totalChunks)")
                    print("   Transcription calls: \(streamingMetrics.totalChunks)")
                } else {
                    print("   Single chunk processed (no chunking data available)")
                }

                // Only show error details for files with WER > 5%
                if result.metrics.wer > 0.05 {
                    print(
                        "   ⚠️ WER: \(String(format: "%.1f%%", result.metrics.wer * 100))")
                    print("   Hypothesis: \(result.hypothesis)")

                    // Show normalized versions with highlighted differences
                    let normalizedRef = TextNormalizer.normalize(result.reference)
                    let normalizedHyp = TextNormalizer.normalize(result.hypothesis)
                    let highlightedComparison = highlightDifferences(
                        reference: normalizedRef,
                        hypothesis: normalizedHyp
                    )
                    print("   Normalized Reference: \(highlightedComparison.reference)")
                    print("   Normalized Hypothesis: \(highlightedComparison.hypothesis)")
                } else {
                    print("   ✅ WER: \(String(format: "%.1f%%", result.metrics.wer * 100))")
                }

                results.append(result)

            } catch {
                logger.error("Failed to process \(audioFile.fileName): \(error)")
                print("ERROR: Failed to process \(audioFile.fileName): \(error)")
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
        let startTime = Date()

        let audioSamples = try await AudioProcessor.loadAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        let (asrResult, chunkOutputs) = try await transcribeAudioWithChunks(
            asrManager: asrManager, audioSamples: audioSamples)

        let metrics = calculateASRMetrics(hypothesis: asrResult.text, reference: file.transcript)

        let processingTime = Date().timeIntervalSince(startTime)

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: asrResult.text,
            reference: file.transcript,
            metrics: metrics,
            processingTime: processingTime,
            audioLength: audioLength,
            chunkOutputs: chunkOutputs
        )
    }

    /// Process a single LibriSpeech file with streaming simulation
    private func processLibriSpeechFileStreaming(
        asrManager: AsrManager, file: LibriSpeechFile
    ) async throws
        -> ASRBenchmarkResult
    {
        let startTime = Date()
        let audioSamples = try await AudioProcessor.loadAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Streaming metrics tracking
        var chunkProcessingTimes: [TimeInterval] = []
        var firstTokenTime: Date?

        // Calculate chunk size in samples (minimum 1 second to ensure reasonable context)
        let samplesPerChunk = max(Int(config.streamingChunkDuration * 16000.0), 16000)

        if config.debugMode {
            logger.info("Starting streaming simulation for \(file.fileName)")
            logger.info("  Audio length: \(audioLength)s")
            logger.info("  Chunk duration: \(max(self.config.streamingChunkDuration, 1.0))s")
            logger.info("  Samples per chunk: \(samplesPerChunk)")
        }

        // For streaming, we'll use the full file but measure chunk-by-chunk processing
        // This simulates how streaming would work with continuous audio
        var processedSamples = 0
        var accumulatedText = ""

        // Process the full audio file but track metrics as if streaming
        while processedSamples < audioSamples.count {
            let chunkStartTime = Date()

            // Calculate how many samples we've "streamed" so far
            let nextChunkEnd = min(processedSamples + samplesPerChunk, audioSamples.count)
            let totalSamplesToProcess = nextChunkEnd

            // Process all audio up to this point (simulating accumulated streaming)
            let audioToProcess = Array(audioSamples[0..<totalSamplesToProcess])
            let result = try await asrManager.transcribe(audioToProcess, source: .microphone)

            // Track first token time
            if firstTokenTime == nil && !result.text.isEmpty {
                firstTokenTime = Date()
            }

            // Update accumulated text
            accumulatedText = result.text

            let chunkProcessingTime = Date().timeIntervalSince(chunkStartTime)
            chunkProcessingTimes.append(chunkProcessingTime)

            if config.debugMode {
                let chunkDuration = Double(nextChunkEnd - processedSamples) / 16000.0
                logger.info(
                    "  Chunk \(chunkProcessingTimes.count): processed \(String(format: "%.2f", chunkDuration))s in \(String(format: "%.3f", chunkProcessingTime))s"
                )
            }

            processedSamples = nextChunkEnd
        }

        // Use the final accumulated text
        let finalText = accumulatedText
        let metrics = calculateASRMetrics(hypothesis: finalText, reference: file.transcript)

        let totalProcessingTime = Date().timeIntervalSince(startTime)
        let firstTokenLatency = firstTokenTime.map { $0.timeIntervalSince(startTime) }

        // Calculate streaming metrics
        let avgChunkTime = chunkProcessingTimes.reduce(0, +) / Double(chunkProcessingTimes.count)
        let maxChunkTime = chunkProcessingTimes.max() ?? 0
        let minChunkTime = chunkProcessingTimes.min() ?? 0
        let streamingRTFx = audioLength / totalProcessingTime

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
            processingTime: totalProcessingTime,
            audioLength: audioLength,
            streamingMetrics: streamingMetrics
        )
    }

    /// Transcribe audio with chunk tracking - captures detailed chunk outputs
    internal func transcribeAudioWithChunks(
        asrManager: AsrManager, audioSamples: [Float]
    ) async throws -> (result: ASRResult, chunkOutputs: [ChunkOutput]) {

        let audioLength = Double(audioSamples.count) / 16000.0

        // For short audio (<=10s), use single chunk processing
        if audioLength <= 10.0 {
            let chunkStartTime = Date()
            let result = try await asrManager.transcribe(audioSamples)
            let processingTime = Date().timeIntervalSince(chunkStartTime)

            let chunkOutput = ChunkOutput(
                chunkIndex: 0,
                startTime: 0.0,
                endTime: audioLength,
                text: result.text,
                tokenCount: result.text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.count,
                processingTime: processingTime
            )

            return (result: result, chunkOutputs: [chunkOutput])
        }

        // For longer audio, use sliding window with chunk tracking
        let chunkSizeSeconds = 10.0  // 10 second chunks
        let overlapSeconds = 1.0  // 1 second overlap
        let stepSizeSeconds = chunkSizeSeconds - overlapSeconds  // 9 seconds
        let chunkSizeSamples = Int(chunkSizeSeconds * 16000.0)
        let stepSizeSamples = Int(stepSizeSeconds * 16000.0)

        var chunkOutputs: [ChunkOutput] = []
        var position = 0
        var chunkIndex = 0
        var previousTokens: [String] = []
        var allText = ""

        while position < audioSamples.count {
            let chunkStartTime = Date()

            let endPosition = min(position + chunkSizeSamples, audioSamples.count)
            let chunkSamples = Array(audioSamples[position..<endPosition])

            // Pad audio if needed (simplified local implementation)
            let paddedChunk: [Float]
            if chunkSamples.count < chunkSizeSamples {
                paddedChunk = chunkSamples + Array(repeating: 0, count: chunkSizeSamples - chunkSamples.count)
            } else {
                paddedChunk = chunkSamples
            }

            // Calculate actual time bounds for this chunk
            let startTimeSeconds = Double(position) / 16000.0
            let endTimeSeconds = Double(endPosition) / 16000.0

            // Get transcription for this chunk
            let chunkResult = try await asrManager.transcribe(paddedChunk)
            let processingTime = Date().timeIntervalSince(chunkStartTime)

            // Extract words from the result
            let chunkWords = chunkResult.text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

            // For overlapping chunks, try to identify new content
            var newWords: [String] = []
            if chunkIndex > 0 && !previousTokens.isEmpty && !chunkWords.isEmpty {
                // Find where the new content starts by looking for overlap
                var skipCount = 0
                let searchWindow = min(previousTokens.count, chunkWords.count, 10)

                for windowSize in (1...searchWindow).reversed() {
                    let suffix = Array(previousTokens.suffix(windowSize))
                    if chunkWords.starts(with: suffix) {
                        skipCount = windowSize
                        break
                    }
                }

                newWords = Array(chunkWords.dropFirst(skipCount))

                if config.debugMode {
                    print(
                        "DEBUG: Chunk \(chunkIndex) overlap - skipping \(skipCount) words, adding \(newWords.count) new words"
                    )
                }
            } else {
                newWords = chunkWords
            }

            let chunkText = newWords.joined(separator: " ")

            // Create chunk output
            let chunkOutput = ChunkOutput(
                chunkIndex: chunkIndex,
                startTime: startTimeSeconds,
                endTime: endTimeSeconds,
                text: chunkText,
                tokenCount: newWords.count,
                processingTime: processingTime
            )
            chunkOutputs.append(chunkOutput)

            // Accumulate text
            if !chunkText.isEmpty {
                if !allText.isEmpty && !allText.hasSuffix(" ") && !chunkText.hasPrefix(" ") {
                    allText += " "
                }
                allText += chunkText
            }

            previousTokens = chunkWords
            position += stepSizeSamples
            chunkIndex += 1
        }

        // Create final result
        let finalResult = ASRResult(
            text: allText,
            confidence: 1.0,
            duration: audioLength,
            processingTime: chunkOutputs.reduce(0.0) { $0 + $1.processingTime },
            tokenTimings: nil
        )

        return (result: finalResult, chunkOutputs: chunkOutputs)
    }

    /// Transcribe audio - now supports long files through AsrManager chunking
    internal func transcribeAudio(
        asrManager: AsrManager, audioSamples: [Float]
    ) async throws
        -> ASRResult
    {
        let (result, _) = try await transcribeAudioWithChunks(asrManager: asrManager, audioSamples: audioSamples)

        if ProcessInfo.processInfo.environment["CI"] != nil && result.text.isEmpty {
            print("⚠️ CI: Transcription returned empty text")
            print("   Audio samples: \(audioSamples.count)")
            print("   Audio duration: \(Float(audioSamples.count) / 16000.0)s")
            print("   Result confidence: \(result.confidence)")
        }

        return result
    }

    /// Calculate WER and CER metrics using standard ASR evaluation normalization
    public func calculateASRMetrics(hypothesis: String, reference: String) -> ASRMetrics {
        // Use standard HuggingFace-aligned normalization
        let normalizedHypothesis = TextNormalizer.normalize(hypothesis)
        let normalizedReference = TextNormalizer.normalize(reference)

        let hypWords = normalizedHypothesis.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let refWords = normalizedReference.components(separatedBy: .whitespacesAndNewlines).filter {
            !$0.isEmpty
        }

        let wordEditDistance = editDistance(hypWords, refWords)
        let wer = refWords.isEmpty ? 0.0 : Double(wordEditDistance.total) / Double(refWords.count)

        let hypChars = Array(normalizedHypothesis.replacingOccurrences(of: " ", with: ""))
        let refChars = Array(normalizedReference.replacingOccurrences(of: " ", with: ""))
        let charEditDistance = editDistance(hypChars.map(String.init), refChars.map(String.init))
        let cer = refChars.isEmpty ? 0.0 : Double(charEditDistance.total) / Double(refChars.count)

        return ASRMetrics(
            wer: wer,
            cer: cer,
            insertions: wordEditDistance.insertions,
            deletions: wordEditDistance.deletions,
            substitutions: wordEditDistance.substitutions,
            totalWords: refWords.count,
            totalCharacters: refChars.count
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
                let audioSamples = try await AudioProcessor.loadAudioFile(path: file.audioPath.path)
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

        print("Downloading \(url)...")
        let (tempFile, _) = try await DownloadUtils.sharedSession.download(from: downloadURL)

        try FileManager.default.createDirectory(at: extractTo, withIntermediateDirectories: true)

        print("Extracting archive...")

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

        print("Dataset extracted successfully")
    }
}

// MARK: - Text Highlighting

private struct HighlightedComparison {
    let reference: String
    let hypothesis: String
}

/// Highlight differences between reference and hypothesis text using ANSI escape codes
private func highlightDifferences(reference: String, hypothesis: String) -> HighlightedComparison {
    let refWords = reference.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
    let hypWords = hypothesis.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

    // Use edit distance to align the sequences and identify operations
    let alignment = getAlignment(refWords, hypWords)

    var highlightedRef = ""
    var highlightedHyp = ""

    for operation in alignment {
        switch operation {
        case .match(let word):
            highlightedRef += word + " "
            highlightedHyp += word + " "
        case .substitute(let refWord, let hypWord):
            highlightedRef += "\u{001B}[41m\(refWord)\u{001B}[0m "  // Red background for reference
            highlightedHyp += "\u{001B}[43m\(hypWord)\u{001B}[0m "  // Yellow background for hypothesis
        case .delete(let word):
            highlightedRef += "\u{001B}[41m\(word)\u{001B}[0m "  // Red background for deleted
            highlightedHyp += "\u{001B}[90m[DELETED]\u{001B}[0m "  // Gray for placeholder
        case .insert(let word):
            highlightedRef += "\u{001B}[90m[INSERTED]\u{001B}[0m "  // Gray for placeholder
            highlightedHyp += "\u{001B}[42m\(word)\u{001B}[0m "  // Green background for inserted
        }
    }

    return HighlightedComparison(
        reference: highlightedRef.trimmingCharacters(in: .whitespaces),
        hypothesis: highlightedHyp.trimmingCharacters(in: .whitespaces)
    )
}

/// Alignment operations for highlighting differences
private enum AlignmentOperation {
    case match(String)
    case substitute(String, String)
    case delete(String)
    case insert(String)
}

/// Get alignment between two word sequences to identify operations
private func getAlignment(_ seq1: [String], _ seq2: [String]) -> [AlignmentOperation] {
    let m = seq1.count
    let n = seq2.count

    if m == 0 {
        return seq2.map { .insert($0) }
    }
    if n == 0 {
        return seq1.map { .delete($0) }
    }

    // Dynamic programming table for edit distance
    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

    for i in 0...m {
        dp[i][0] = i
    }
    for j in 0...n {
        dp[0][j] = j
    }

    for i in 1...m {
        for j in 1...n {
            if seq1[i - 1] == seq2[j - 1] {
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

    // Backtrack to get the alignment
    var operations: [AlignmentOperation] = []
    var i = m
    var j = n

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && seq1[i - 1] == seq2[j - 1] {
            operations.append(.match(seq1[i - 1]))
            i -= 1
            j -= 1
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            operations.append(.substitute(seq1[i - 1], seq2[j - 1]))
            i -= 1
            j -= 1
        } else if i > 0 && (j == 0 || dp[i][j] == dp[i - 1][j] + 1) {
            operations.append(.delete(seq1[i - 1]))
            i -= 1
        } else if j > 0 {
            operations.append(.insert(seq2[j - 1]))
            j -= 1
        }
    }

    return operations.reversed()
}

// MARK: - Edit Distance Algorithm

private struct EditDistanceResult {
    let total: Int
    let insertions: Int
    let deletions: Int
    let substitutions: Int
}

private func editDistance<T: Equatable>(_ seq1: [T], _ seq2: [T]) -> EditDistanceResult {
    let m = seq1.count
    let n = seq2.count

    if m == 0 {
        return EditDistanceResult(total: n, insertions: n, deletions: 0, substitutions: 0)
    }
    if n == 0 {
        return EditDistanceResult(total: m, insertions: 0, deletions: m, substitutions: 0)
    }

    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

    for i in 0...m {
        dp[i][0] = i
    }
    for j in 0...n {
        dp[0][j] = j
    }

    for i in 1...m {
        for j in 1...n {
            if seq1[i - 1] == seq2[j - 1] {
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

    var i = m
    var j = n
    var insertions = 0
    var deletions = 0
    var substitutions = 0

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && seq1[i - 1] == seq2[j - 1] {
            i -= 1
            j -= 1
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            substitutions += 1
            i -= 1
            j -= 1
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            deletions += 1
            i -= 1
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            insertions += 1
            j -= 1
        } else {
            break
        }
    }

    return EditDistanceResult(
        total: dp[m][n],
        insertions: insertions,
        deletions: deletions,
        substitutions: substitutions
    )
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
@available(macOS 13.0, iOS 16.0, *)
extension ASRBenchmark {
    public static func runASRBenchmark(arguments: [String]) async {
        var subset = "test-clean"
        var maxFiles: Int? = 100  // Default to 100 files for standard evaluation
        var outputFile = "asr_benchmark_results.json"
        var debugMode = false
        var autoDownload = true  // Default to true for automatic download
        var testStreaming = false
        var streamingChunkDuration = 0.1  // Default 100ms chunks

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
                        print("Invalid chunk duration: \(arguments[i + 1])")
                        exit(1)
                    }
                    i += 1
                }
            default:
                print("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("\nStarting ASR benchmark on LibriSpeech \(subset)")
        print("   Max files: \(maxFiles?.description ?? "all") (default: 100)")
        print("   Output file: \(outputFile)")
        print("   Debug mode: \(debugMode ? "enabled" : "disabled")")
        print("   Auto-download: \(autoDownload ? "enabled" : "disabled")")
        print("   Test streaming: \(testStreaming ? "enabled" : "disabled")")
        if testStreaming {
            print("   Chunk duration: \(streamingChunkDuration)s")
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
            maxSymbolsPerFrame: 3,
            enableDebug: debugMode,
            realtimeMode: false,
            chunkSizeMs: 2000,
            tdtConfig: TdtConfig(
                durations: [0, 1, 2, 3, 4],
                includeTokenDuration: true,
                maxSymbolsPerStep: 3
            )
        )

        let asrManager = AsrManager(config: asrConfig)

        do {
            let startBenchmark = Date()

            print("Initializing ASR system...")
            do {
                let models = try await AsrModels.downloadAndLoad()
                try await asrManager.initialize(models: models)
                print("ASR system initialized successfully")

                // Profile Neural Engine optimizations
                asrManager.profilePerformance()

                if ProcessInfo.processInfo.environment["CI"] != nil {
                    print("🔍 CI: Verifying ASR models with test audio...")
                    let testSamples = Array(repeating: Float(0.0), count: 16000)  // 1 second of silence
                    let testResult = try await asrManager.transcribe(testSamples)
                    print("   Test transcription result: '\(testResult.text)'")
                    print("   Models appear to be working: \(asrManager.isAvailable)")
                }
            } catch {
                print("Failed to initialize ASR system: \(error)")
                print("   Error type: \(type(of: error))")
                print("   Error details: \(error.localizedDescription)")

                if ProcessInfo.processInfo.environment["CI"] != nil {
                    print("🔍 CI Debug Information:")
                    let modelsDir = FileManager.default.homeDirectoryForCurrentUser
                        .appendingPathComponent(
                            "Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3-coreml"
                        )
                    print("   Models directory: \(modelsDir.path)")
                    print(
                        "   Directory exists: \(FileManager.default.fileExists(atPath: modelsDir.path))"
                    )

                    if FileManager.default.fileExists(atPath: modelsDir.path) {
                        do {
                            let contents = try FileManager.default.contentsOfDirectory(
                                at: modelsDir, includingPropertiesForKeys: nil)
                            print("   Directory contents: \(contents.map { $0.lastPathComponent })")
                        } catch {
                            print("   Failed to list directory contents: \(error)")
                        }
                    }
                }
                throw error
            }

            if autoDownload {
                try await benchmark.downloadLibriSpeech(subset: subset)
            }

            let results = try await benchmark.runLibriSpeechBenchmark(
                asrManager: asrManager, subset: subset)

            let totalWER = results.reduce(0.0) { $0 + $1.metrics.wer } / Double(results.count)
            let totalCER = results.reduce(0.0) { $0 + $1.metrics.cer } / Double(results.count)

            let rtfxValues = results.map { Float($0.rtfx) }
            let sortedRTFx = rtfxValues.sorted()
            let medianRTFx = sortedRTFx[sortedRTFx.count / 2]

            let totalAudioDuration = results.reduce(0.0) { $0 + $1.audioLength }
            let totalProcessingTime = results.reduce(0.0) { $0 + $1.processingTime }

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

            print(
                "\n\(results.count) files per dataset • Test runtime: \(runtimeString) • \(dateString)"
            )

            print("--- Benchmark Results ---")
            #if DEBUG
            print("   Mode: DEBUG (slow performance)")
            #else
            print("   Mode: RELEASE (optimal performance)")
            #endif
            print("   Dataset: \(config.dataset) \(config.subset)")
            print("   Files processed: \(results.count)")
            let overallRTFx = totalAudioDuration / totalProcessingTime

            print("   Average WER: \(String(format: "%.1f", totalWER * 100))%")
            print("   Median WER: \(String(format: "%.1f", medianWER * 100))%")
            print("   Average CER: \(String(format: "%.1f", totalCER * 100))%")
            print("   Median RTFx: \(String(format: "%.1f", medianRTFx))x")
            print(
                "   Overall RTFx: \(String(format: "%.1f", overallRTFx))x (\(String(format: "%.1f", totalAudioDuration))s / \(String(format: "%.1f", totalProcessingTime))s)"
            )

            // Print streaming metrics if available
            if config.testStreaming {
                print("\n--- Streaming Metrics ---")

                // Calculate aggregate streaming metrics
                let streamingResults = results.compactMap { $0.streamingMetrics }
                if !streamingResults.isEmpty {
                    let avgChunkTime =
                        streamingResults.map { $0.avgChunkProcessingTime }.reduce(0, +) / Double(streamingResults.count)
                    let maxChunkTime = streamingResults.map { $0.maxChunkProcessingTime }.max() ?? 0
                    let totalChunks = streamingResults.map { $0.totalChunks }.reduce(0, +)
                    let avgFirstTokenLatency =
                        streamingResults.compactMap { $0.firstTokenLatency }.reduce(0, +)
                        / Double(streamingResults.compactMap { $0.firstTokenLatency }.count)

                    print("   Chunk duration: \(config.streamingChunkDuration)s")
                    print("   Total chunks processed: \(totalChunks)")
                    print("   Avg chunk processing time: \(String(format: "%.3f", avgChunkTime))s")
                    print("   Max chunk processing time: \(String(format: "%.3f", maxChunkTime))s")
                    if streamingResults.compactMap({ $0.firstTokenLatency }).count > 0 {
                        print("   Avg first token latency: \(String(format: "%.3f", avgFirstTokenLatency))s")
                    }
                }
            }

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
                configDict["streamingChunkDuration"] = config.streamingChunkDuration
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
                                "avgChunkProcessingTime": streamingMetrics.avgChunkProcessingTime,
                                "maxChunkProcessingTime": streamingMetrics.maxChunkProcessingTime,
                                "minChunkProcessingTime": streamingMetrics.minChunkProcessingTime,
                                "totalChunks": streamingMetrics.totalChunks,
                                "firstTokenLatency": streamingMetrics.firstTokenLatency as Any,
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

            print("\nResults saved to: \(outputFile)")
            print("ASR benchmark completed successfully")

        } catch {
            print("\nERROR: ASR benchmark failed: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """
            ASR Benchmark Command Usage:
                fluidaudio asr-benchmark [options]

            Options:
                --subset <name>           LibriSpeech subset to use (default: test-clean)
                                         Available: test-clean, test-other, dev-clean, dev-other
                --max-files <number>      Maximum number of files to process (default: all)
                --output <file>           Output JSON file path (default: asr_benchmark_results.json)
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
                fluidaudio asr-benchmark --subset test-other --max-files 100r

                # Test streaming performance with 0.5s chunks
                fluidaudio asr-benchmark --test-streaming --chunk-duration 0.5

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
