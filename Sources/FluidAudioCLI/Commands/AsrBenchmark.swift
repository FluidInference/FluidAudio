import Foundation
import AVFoundation
import OSLog
import FluidAudio

/// ASR evaluation metrics
public struct ASRMetrics: Sendable {
    public let wer: Double           // Word Error Rate
    public let cer: Double           // Character Error Rate
    public let insertions: Int
    public let deletions: Int
    public let substitutions: Int
    public let totalWords: Int
    public let totalCharacters: Int

    public init(wer: Double, cer: Double, insertions: Int, deletions: Int, substitutions: Int, totalWords: Int, totalCharacters: Int) {
        self.wer = wer
        self.cer = cer
        self.insertions = insertions
        self.deletions = deletions
        self.substitutions = substitutions
        self.totalWords = totalWords
        self.totalCharacters = totalCharacters
    }
}

/// Single ASR benchmark result
public struct ASRBenchmarkResult: Sendable {
    public let fileName: String
    public let hypothesis: String
    public let reference: String
    public let metrics: ASRMetrics
    public let processingTime: TimeInterval
    public let audioLength: TimeInterval
    public let rtf: Double             // Real-Time Factor

    public init(fileName: String, hypothesis: String, reference: String, metrics: ASRMetrics, processingTime: TimeInterval, audioLength: TimeInterval) {
        self.fileName = fileName
        self.hypothesis = hypothesis
        self.reference = reference
        self.metrics = metrics
        self.processingTime = processingTime
        self.audioLength = audioLength
        self.rtf = processingTime / audioLength
    }
}

/// ASR benchmark configuration
///
/// ## LibriSpeech Dataset Subsets
/// - **test-clean**: Clean, studio-quality recordings with clear speech from native speakers
///   - Easier benchmark subset with minimal noise/accents
///   - Expected WER: 2-6% for good ASR systems
///   - Use for baseline performance evaluation
///
/// - **test-other**: More challenging recordings with various acoustic conditions
///   - Includes accented speech, background noise, and non-native speakers
///   - Expected WER: 5-15% for good ASR systems
///   - Use for robustness testing
///
/// Both subsets contain ~5.4 hours of audio from different speakers reading books.
public struct ASRBenchmarkConfig: Sendable {
    public let dataset: String
    public let subset: String
    public let maxFiles: Int?
    public let debugMode: Bool
    public let longAudioOnly: Bool

    public init(dataset: String = "librispeech", subset: String = "test-clean", maxFiles: Int? = nil, debugMode: Bool = false, longAudioOnly: Bool = false) {
        self.dataset = dataset
        self.subset = subset
        self.maxFiles = maxFiles
        self.debugMode = debugMode
        self.longAudioOnly = longAudioOnly
    }
}

/// LibriSpeech dataset manager and ASR benchmarking
@available(macOS 13.0, *)
public class ASRBenchmark {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "Benchmark")
    private let config: ASRBenchmarkConfig

    public init(config: ASRBenchmarkConfig = ASRBenchmarkConfig()) {
        self.config = config
    }

    /// Download LibriSpeech test datasets
    public func downloadLibriSpeech(subset: String = "test-clean", forceDownload: Bool = false) async throws {
        let datasetsDirectory = getLibriSpeechDirectory()
        let subsetDirectory = datasetsDirectory.appendingPathComponent(subset)

        // Check if already downloaded by looking for transcript files (which indicate complete download)
        if !forceDownload && FileManager.default.fileExists(atPath: subsetDirectory.path) {
            // Look for transcript files recursively to verify complete dataset
            let enumerator = FileManager.default.enumerator(at: subsetDirectory, includingPropertiesForKeys: nil)
            var transcriptCount = 0

            while let url = enumerator?.nextObject() as? URL {
                if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                    transcriptCount += 1
                    if transcriptCount >= 5 { // Found enough transcript files, dataset exists
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

        // LibriSpeech dataset URLs
        // test-clean: High-quality recordings, easier to transcribe
        // test-other: Challenging recordings with accents/noise
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
    public func runLibriSpeechBenchmark(asrManager: AsrManager, subset: String = "test-clean") async throws -> [ASRBenchmarkResult] {
        // Check if running in release mode and warn if not
        #if DEBUG
        print("")
        print("WARNING: Running in DEBUG mode!")
        print("Performance will be significantly slower (~2x).")
        print("For accurate benchmarks, use: swift run -c release fluidaudio asr-benchmark")
        print("")
        // Add a small delay so user sees the warning
        try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
        #else
        print("Running in RELEASE mode - optimal performance")
        #endif

        // Ensure dataset is downloaded
        try await downloadLibriSpeech(subset: subset)

        let datasetPath = getLibriSpeechDirectory().appendingPathComponent(subset)
        let audioFiles = try collectLibriSpeechFiles(from: datasetPath)

        // Filter by duration if requested
        var filteredFiles = audioFiles
        if config.longAudioOnly {
            filteredFiles = try filterFilesByDuration(audioFiles, minDuration: 4.0, maxDuration: 20.0)
            print("Filtered to \(filteredFiles.count) files with duration 4-10 seconds (from \(audioFiles.count) total)")
        }

        let maxFiles = config.maxFiles ?? filteredFiles.count // Process all files if not specified
        let filesToProcess = Array(filteredFiles.prefix(maxFiles))

        print("📋 Processing \(filesToProcess.count) files (max files limit: \(config.maxFiles?.description ?? "unlimited"))")

        logger.info("Running ASR benchmark on \(filesToProcess.count) files from LibriSpeech \(subset)")

        var results: [ASRBenchmarkResult] = []

        // var previousStateFingerprint: String? = nil // Removed unused variable

        for (index, audioFile) in filesToProcess.enumerated() {
            do {
                if config.debugMode {
                    logger.info("Processing file \(index + 1)/\(filesToProcess.count): \(audioFile.fileName)")
                }
                // print("🎵 Processing (\(index + 1)/\(filesToProcess.count)): \(audioFile.fileName)")

                // State verification: Check for state persistence between files
                if config.debugMode && index > 0 {
                    // Note: We can't directly access decoderState from AsrManager in this context
                    // State verification is handled internally by AsrManager
                    logger.info("   🔍 Processing file \(index + 1)")
                }

                // Note: ASR state is managed internally by AsrManager
                // Each transcription handles its own state initialization

                let result = try await processLibriSpeechFile(asrManager: asrManager, file: audioFile)
                results.append(result)

                // print("   WER: \(String(format: "%.1f", result.metrics.wer * 100))%, RTF: \(String(format: "%.3f", result.rtf))x, RTFx: \(String(format: "%.1f", 1.0/result.rtf))x, Duration: \(String(format: "%.1f", result.audioLength))s")

                // Show text comparison for all files (always visible for better analysis)
                // printTextComparison(result: result, maxLength: 150, showFileNumber: index + 1)

            } catch {
                logger.error("Failed to process \(audioFile.fileName): \(error)")
                print("ERROR: Failed to process \(audioFile.fileName): \(error)")
            }
        }

        return results
    }

    /// Process a single LibriSpeech file
    private func processLibriSpeechFile(asrManager: AsrManager, file: LibriSpeechFile) async throws -> ASRBenchmarkResult {
        let startTime = Date()

        // Load and convert audio to the required format
        let audioSamples = try loadAudioFile(url: file.audioPath)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Run ASR transcription in chunks if needed
        let asrResult = try await transcribeAudio(asrManager: asrManager, audioSamples: audioSamples)

        // Calculate metrics
        let metrics = calculateASRMetrics(hypothesis: asrResult.text, reference: file.transcript)

        let processingTime = Date().timeIntervalSince(startTime)

        return ASRBenchmarkResult(
            fileName: file.fileName,
            hypothesis: asrResult.text,
            reference: file.transcript,
            metrics: metrics,
            processingTime: processingTime,
            audioLength: audioLength
        )
    }

    /// Transcribe audio - now supports long files through AsrManager chunking
    internal func transcribeAudio(asrManager: AsrManager, audioSamples: [Float]) async throws -> ASRResult {
        // AsrManager now handles chunking internally for audio > 10 seconds
        let result = try await asrManager.transcribe(audioSamples)

        // CI debugging
        if ProcessInfo.processInfo.environment["CI"] != nil && result.text.isEmpty {
            print("⚠️ CI: Transcription returned empty text")
            print("   Audio samples: \(audioSamples.count)")
            print("   Audio duration: \(Float(audioSamples.count) / 16000.0)s")
            print("   Result confidence: \(result.confidence)")
        }

        return result
    }

    /// Calculate WER and CER metrics with HuggingFace-compatible normalization
    public func calculateASRMetrics(hypothesis: String, reference: String) -> ASRMetrics {
        // Apply HuggingFace-compatible text normalization
        let normalizedHypothesis = TextNormalizer.normalize(hypothesis)
        let normalizedReference = TextNormalizer.normalize(reference)

        let hypWords = normalizedHypothesis.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let refWords = normalizedReference.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

        // Calculate word-level edit distance
        let wordEditDistance = editDistance(hypWords, refWords)
        let wer = refWords.isEmpty ? 0.0 : Double(wordEditDistance.total) / Double(refWords.count)

        // Calculate character-level edit distance using normalized text
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

    /// Generate benchmark summary statistics

    /// Print text comparison between ground truth and model output
    private func printTextComparison(result: ASRBenchmarkResult, maxLength: Int = 200, showFileNumber: Int? = nil) {
        let groundTruth = result.reference
        let modelOutput = result.hypothesis

        // Truncate for display if too long
        let gtDisplay = groundTruth.count > maxLength ? String(groundTruth.prefix(maxLength)) + "..." : groundTruth
        let moDisplay = modelOutput.count > maxLength ? String(modelOutput.prefix(maxLength)) + "..." : modelOutput

        let filePrefix = showFileNumber != nil ? "[\(showFileNumber!)] " : ""

        print("   \(filePrefix)Text Comparison (Duration: \(String(format: "%.1f", result.audioLength))s):")
        print("   Expected: \"\(gtDisplay)\"")
        print("   Got:      \"\(moDisplay)\"")

        // Quick analysis with more detailed feedback
        var issues: [String] = []

        if modelOutput.isEmpty {
            issues.append("No output")
        } else {
            if modelOutput.count < groundTruth.count / 2 {
                issues.append("Too short")
            } else if modelOutput.count > groundTruth.count * 2 {
                issues.append("Too long")
            }

            if hasRepetitivePatterns(modelOutput) {
                issues.append("Repetition")
            }

            // Check for case issues
            if modelOutput.lowercased() == groundTruth.lowercased() {
                issues.append("Case mismatch")
            }

            // Check for partial match
            let gtWords = groundTruth.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            let moWords = modelOutput.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            let commonWords = Set(gtWords).intersection(Set(moWords)).count
            let matchPercent = gtWords.isEmpty ? 0 : Double(commonWords) / Double(gtWords.count) * 100

            if matchPercent > 50 && matchPercent < 90 {
                issues.append("Partial match (\(String(format: "%.0f", matchPercent))%)")
            } else if matchPercent >= 90 {
                issues.append("Good match (\(String(format: "%.0f", matchPercent))%)")
            }
        }

        if !issues.isEmpty {
            print("   Issues: \(issues.joined(separator: ", "))")
        }

        print("   " + String(repeating: "─", count: 80))
    }

    /// Detect repetitive patterns in text that suggest token loops
    private func hasRepetitivePatterns(_ text: String) -> Bool {
        let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        guard words.count > 5 else { return false }

        // Check for immediate word repetition (same word repeated 3+ times)
        for i in 0..<(words.count - 2) {
            if words[i] == words[i + 1] && words[i] == words[i + 2] {
                return true
            }
        }

        // Check for phrase repetition (3+ word phrases repeated)
        for phraseLen in 2...min(5, words.count / 3) {
            for i in 0..<(words.count - phraseLen * 2) {
                let phrase1 = words[i..<(i + phraseLen)]
                let phrase2 = words[(i + phraseLen)..<(i + phraseLen * 2)]
                if Array(phrase1) == Array(phrase2) {
                    return true
                }
            }
        }

        return false
    }

    // MARK: - Private Helper Methods

    /// Filter files by duration range
    private func filterFilesByDuration(_ files: [LibriSpeechFile], minDuration: Double, maxDuration: Double) throws -> [LibriSpeechFile] {
        var filteredFiles: [LibriSpeechFile] = []

        for file in files {
            do {
                let audioSamples = try loadAudioFile(url: file.audioPath)
                let duration = Double(audioSamples.count) / 16000.0

                if duration >= minDuration && duration <= maxDuration {
                    filteredFiles.append(file)
                }
            } catch {
                // Skip files that can't be loaded
                logger.warning("Could not load audio file \(file.fileName): \(error.localizedDescription)")
                continue
            }
        }

        return filteredFiles
    }

    private func getLibriSpeechDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        return appDirectory.appendingPathComponent("Datasets/LibriSpeech", isDirectory: true)
    }

    private func collectLibriSpeechFiles(from directory: URL) throws -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []

        let fileManager = FileManager.default
        let enumerator = fileManager.enumerator(at: directory, includingPropertiesForKeys: nil)

        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                // Found transcript file, look for corresponding audio
                let transcriptContent = try String(contentsOf: url)
                let lines = transcriptContent.components(separatedBy: .newlines).filter { !$0.isEmpty }

                for line in lines {
                    let parts = line.components(separatedBy: " ")
                    guard parts.count >= 2 else { continue }

                    let audioId = parts[0]
                    let transcript = parts.dropFirst().joined(separator: " ")

                    // Construct audio file path
                    let audioFileName = "\(audioId).flac"
                    let audioPath = url.deletingLastPathComponent().appendingPathComponent(audioFileName)

                    if fileManager.fileExists(atPath: audioPath.path) {
                        files.append(LibriSpeechFile(
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

    internal func loadAudioFile(url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = UInt32(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ASRError.processingFailed("Failed to create audio buffer")
        }

        try audioFile.read(into: buffer)

        // Convert to mono 16kHz float array
        let channelCount = Int(format.channelCount)
        let sampleRate = format.sampleRate

        var samples: [Float] = []

        if channelCount == 1 {
            samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData?[0], count: Int(frameCount)))
        } else {
            // Mix down to mono
            for i in 0..<Int(frameCount) {
                var sample: Float = 0
                for channel in 0..<channelCount {
                    sample += buffer.floatChannelData?[channel][i] ?? 0
                }
                samples.append(sample / Float(channelCount))
            }
        }

        // Resample to 16kHz if needed
        if sampleRate != 16000 {
            samples = resampleAudio(samples, fromRate: sampleRate, toRate: 16000)
        }

        return samples
    }

    private func resampleAudio(_ samples: [Float], fromRate: Double, toRate: Double) -> [Float] {
        if fromRate == toRate {
            return samples
        }

        let ratio = toRate / fromRate
        let outputLength = Int(Double(samples.count) * ratio)
        var resampled = Array<Float>(repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let sourceIndex = Double(i) / ratio
            let leftIndex = Int(floor(sourceIndex))
            let rightIndex = min(leftIndex + 1, samples.count - 1)
            let fraction = Float(sourceIndex - Double(leftIndex))

            if leftIndex < samples.count {
                resampled[i] = samples[leftIndex] * (1 - fraction) + samples[rightIndex] * fraction
            }
        }

        return resampled
    }

    private func downloadAndExtractTarGz(url: String, extractTo: URL, expectedSubpath: String) async throws {
        let downloadURL = URL(string: url)!

        print("Downloading \(url)...")
        let (tempFile, _) = try await URLSession.shared.download(from: downloadURL)

        try FileManager.default.createDirectory(at: extractTo, withIntermediateDirectories: true)

        print("Extracting archive...")

        // Extract tar.gz using system tar command
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", tempFile.path, "-C", extractTo.path]

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw ASRError.processingFailed("Failed to extract tar.gz file")
        }

        // Move files from LibriSpeech subfolder if needed
        let extractedPath = extractTo.appendingPathComponent(expectedSubpath)
        if FileManager.default.fileExists(atPath: extractedPath.path) {
            let targetPath = extractTo.appendingPathComponent(expectedSubpath.components(separatedBy: "/").last!)
            try? FileManager.default.removeItem(at: targetPath)
            try FileManager.default.moveItem(at: extractedPath, to: targetPath)

            // Clean up LibriSpeech parent directory
            try? FileManager.default.removeItem(at: extractTo.appendingPathComponent("LibriSpeech"))
        }

        print("Dataset extracted successfully")
    }
}

// MARK: - Supporting Types

public struct LibriSpeechFile {
    public let fileName: String
    public let audioPath: URL
    public let transcript: String
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

    // Handle empty sequences
    if m == 0 {
        return EditDistanceResult(total: n, insertions: n, deletions: 0, substitutions: 0)
    }
    if n == 0 {
        return EditDistanceResult(total: m, insertions: 0, deletions: m, substitutions: 0)
    }

    // Dynamic programming matrix
    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

    // Initialize base cases
    for i in 0...m {
        dp[i][0] = i
    }
    for j in 0...n {
        dp[0][j] = j
    }

    // Fill the matrix
    for i in 1...m {
        for j in 1...n {
            if seq1[i-1] == seq2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = 1 + min(
                    dp[i-1][j],     // deletion
                    dp[i][j-1],     // insertion
                    dp[i-1][j-1]    // substitution
                )
            }
        }
    }

    // Backtrack to count operation types
    var i = m, j = n
    var insertions = 0, deletions = 0, substitutions = 0

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && seq1[i-1] == seq2[j-1] {
            i -= 1
            j -= 1
        } else if i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] + 1 {
            substitutions += 1
            i -= 1
            j -= 1
        } else if i > 0 && dp[i][j] == dp[i-1][j] + 1 {
            deletions += 1
            i -= 1
        } else if j > 0 && dp[i][j] == dp[i][j-1] + 1 {
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

/// Extension to provide CLI entry point
@available(macOS 13.0, iOS 16.0, *)
extension ASRBenchmark {
    public static func runASRBenchmark(arguments: [String]) async {
        print("DEBUG: Starting runASRBenchmark")
        var subset = "test-clean"
        var maxFiles: Int?
        var outputFile = "asr_benchmark_results.json"
        var debugMode = false
        var autoDownload = true  // Default to true for automatic download

        // Parse arguments
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
            default:
                print("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("\nStarting ASR benchmark on LibriSpeech \(subset)")
        print("   Max files: \(maxFiles?.description ?? "all")")
        print("   Output file: \(outputFile)")
        print("   Debug mode: \(debugMode ? "enabled" : "disabled")")
        print("   Auto-download: \(autoDownload ? "enabled" : "disabled")")

        let config = ASRBenchmarkConfig(
            dataset: "librispeech",
            subset: subset,
            maxFiles: maxFiles,
            debugMode: debugMode,
            longAudioOnly: false
        )

        let benchmark = ASRBenchmark(config: config)

        // Initialize ASR manager with fast benchmark preset
        let asrConfig = ASRConfig(
            maxSymbolsPerFrame: 3,
            enableDebug: debugMode,
            realtimeMode: false,
            chunkSizeMs: 2000,
            tdtConfig: TDTConfig(
                durations: [0, 1, 2, 3, 4],
                includeTokenDuration: true,
                includeDurationConfidence: false,
                maxSymbolsPerStep: 3
            )
        )

        let asrManager = AsrManager(config: asrConfig)

        do {
            // Track benchmark start time
            let startBenchmark = Date()
            
            // Initialize ASR system
            print("Initializing ASR system...")
            do {
                try await asrManager.initialize()
                print("ASR system initialized successfully")

                // Verify models are actually working
                if ProcessInfo.processInfo.environment["CI"] != nil {
                    print("🔍 CI: Verifying ASR models with test audio...")
                    let testSamples = Array(repeating: Float(0.0), count: 16000) // 1 second of silence
                    let testResult = try await asrManager.transcribe(testSamples)
                    print("   Test transcription result: '\(testResult.text)'")
                    print("   Models appear to be working: \(asrManager.isAvailable)")
                }
            } catch {
                print("❌ Failed to initialize ASR system: \(error)")
                print("   Error type: \(type(of: error))")
                print("   Error details: \(error.localizedDescription)")

                // Additional debugging in CI
                if ProcessInfo.processInfo.environment["CI"] != nil {
                    print("🔍 CI Debug Information:")
                    let modelsDir = FileManager.default.homeDirectoryForCurrentUser
                        .appendingPathComponent("Library/Application Support/FluidAudio/Models/Parakeet")
                    print("   Models directory: \(modelsDir.path)")
                    print("   Directory exists: \(FileManager.default.fileExists(atPath: modelsDir.path))")

                    if FileManager.default.fileExists(atPath: modelsDir.path) {
                        do {
                            let contents = try FileManager.default.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil)
                            print("   Directory contents: \(contents.map { $0.lastPathComponent })")
                        } catch {
                            print("   Failed to list directory contents: \(error)")
                        }
                    }
                }
                throw error
            }

            // Download dataset if requested
            if autoDownload {
                try await benchmark.downloadLibriSpeech(subset: subset)
            }

            // Run benchmark
            let results = try await benchmark.runLibriSpeechBenchmark(asrManager: asrManager, subset: subset)

            // Calculate overall metrics
            let totalWER = results.reduce(0.0) { $0 + $1.metrics.wer } / Double(results.count)
            let totalCER = results.reduce(0.0) { $0 + $1.metrics.cer } / Double(results.count)
            let totalRTF = results.reduce(0.0) { $0 + $1.rtf } / Double(results.count)
            let rtfxValues = results.map { Float(1.0 / $0.rtf) }
            let meanRTFx = rtfxValues.reduce(0, +) / Float(rtfxValues.count)
            let medianRTFx = rtfxValues.sorted()[rtfxValues.count / 2]
            let sumRTFx = rtfxValues.reduce(0, +)
            
            // Calculate standard deviation for RTFx
            let deviations = rtfxValues.map { ($0 - meanRTFx) * ($0 - meanRTFx) }
            let variance = deviations.reduce(0, +) / Float(rtfxValues.count)
            let stdRTFx = sqrt(variance)
            
            // Calculate total durations
            let totalAudioDuration = results.reduce(0.0) { $0 + $1.audioLength }
            let totalProcessingTime = results.reduce(0.0) { $0 + $1.processingTime }

            // Calculate median WER
            let werValues = results.map { $0.metrics.wer }
            let sortedWER = werValues.sorted()
            let medianWER = sortedWER[sortedWER.count / 2]

            // Get current date/time
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "MM/dd/yyyy, h:mm a zzz"
            let dateString = dateFormatter.string(from: Date())
            
            // Calculate test runtime
            let endTime = Date()
            let testRuntime = endTime.timeIntervalSince(startBenchmark)
            let minutes = Int(testRuntime) / 60
            let seconds = Int(testRuntime) % 60
            let runtimeString = "\(minutes)m \(seconds)s"
            
            // Print summary with improved formatting
            print("\n\(results.count) files per dataset • Test runtime: \(runtimeString) • \(dateString)")
            print("")
            print("Inverse Real Time Factor (RTFx)")
            print("RTFx measures the latency of speech recognition systems - how long it takes to process a given amount of speech.")
            print("")
            print("RTFx = (number of seconds of audio inferred) / (compute time in seconds)")
            print("")
            print("• RTFx of 1.0 = system processes speech as fast as it's spoken")
            print("• RTFx of 2.0 = system takes half the time (2x faster than real-time)")
            print("• Higher RTFx = lower latency = better performance")
            print("")
            print("Processing time includes: Model inference on Apple Neural Engine, audio preprocessing, state resets between files,")
            print("token-to-text conversion, and file I/O overhead.")
            print("")
            print("--- Benchmark Results ---")
            #if DEBUG
            print("   Mode: DEBUG (slow performance)")
            #else
            print("   Mode: RELEASE (optimal performance)")
            #endif
            print("   Dataset: \(config.dataset) \(config.subset)")
            print("   Files processed: \(results.count)")
            print("   Average WER: \(String(format: "%.1f", totalWER * 100))%")
            print("   Median WER: \(String(format: "%.1f", medianWER * 100))%")
            print("   Average CER: \(String(format: "%.1f", totalCER * 100))%")
            print("   Average RTF: \(String(format: "%.3f", totalRTF))x")
            print("   Mean RTFx: \(String(format: "%.1f", meanRTFx))x")
            print("   Median RTFx: \(String(format: "%.1f", medianRTFx))x")
            print("   Std RTFx: \(String(format: "%.1f", stdRTFx))x")
            print("   Total audio duration: \(String(format: "%.1f", totalAudioDuration))s")
            print("   Total processing time: \(String(format: "%.1f", totalProcessingTime))s")
            let overallRTFx = totalAudioDuration / totalProcessingTime
            print("   Overall RTFx: \(String(format: "%.1f", overallRTFx))x (\(String(format: "%.1f", totalAudioDuration))s / \(String(format: "%.1f", totalProcessingTime))s)")

            // Save results
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

            let output = [
                "config": [
                    "dataset": config.dataset,
                    "subset": config.subset,
                    "maxFiles": config.maxFiles as Any,
                    "debugMode": config.debugMode
                ],
                "summary": [
                    "filesProcessed": results.count,
                    "averageWER": totalWER,
                    "medianWER": medianWER,
                    "averageCER": totalCER,
                    "averageRTF": totalRTF,
                    "meanRTFx": meanRTFx,
                    "medianRTFx": medianRTFx,
                    "stdRTFx": stdRTFx,
                    "sumRTFx": sumRTFx,
                    "totalAudioDuration": totalAudioDuration,
                    "totalProcessingTime": totalProcessingTime
                ],
                "results": results.map { result in
                    [
                        "fileName": result.fileName,
                        "hypothesis": result.hypothesis,
                        "reference": result.reference,
                        "wer": result.metrics.wer,
                        "cer": result.metrics.cer,
                        "rtf": result.rtf,
                        "audioLength": result.audioLength,
                        "processingTime": result.processingTime
                    ]
                }
            ] as [String: Any]

            let jsonData = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: URL(fileURLWithPath: outputFile))

            print("\nResults saved to: \(outputFile)")
            print("ASR benchmark completed successfully")

        } catch {
            print("\nERROR: ASR benchmark failed: \(error)")
            exit(1)
        }
    }
}
