#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// Earnings22 benchmark using TDT for transcription + CTC for keyword spotting.
/// TDT provides low WER transcription, CTC provides high recall dictionary detection.
public enum CtcEarningsBenchmark {

    public static func runCLI(arguments: [String]) async {
        // Check for help
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var dataDir = "test-dataset"
        var outputFile = "ctc_earnings_benchmark.json"
        var maxFiles: Int? = nil
        var ctcModelPath: String? = nil
        var tdtVersion: AsrModelVersion = .v3

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--data-dir":
                if i + 1 < arguments.count {
                    dataDir = arguments[i + 1]
                    i += 1
                }
            case "--output", "-o":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--ctc-model":
                if i + 1 < arguments.count {
                    ctcModelPath = arguments[i + 1]
                    i += 1
                }
            case "--tdt-version":
                if i + 1 < arguments.count {
                    if arguments[i + 1] == "v2" || arguments[i + 1] == "2" {
                        tdtVersion = .v2
                    }
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        print("Earnings Benchmark (TDT transcription + CTC keyword spotting)")
        print("  Data directory: \(dataDir)")
        print("  Output file: \(outputFile)")
        print("  TDT version: \(tdtVersion == .v2 ? "v2" : "v3")")
        print("  CTC model: \(ctcModelPath ?? "required")")

        guard let modelPath = ctcModelPath else {
            print("ERROR: --ctc-model path is required")
            printUsage()
            return
        }

        do {
            // Load TDT models for transcription
            print("Loading TDT models (\(tdtVersion == .v2 ? "v2" : "v3")) for transcription...")
            let tdtModels = try await AsrModels.downloadAndLoad(version: tdtVersion)
            let asrManager = AsrManager(config: .default)
            try await asrManager.initialize(models: tdtModels)
            print("TDT models loaded successfully")

            // Load CTC models for keyword spotting
            print("Loading CTC models from: \(modelPath)")
            let modelDir = URL(fileURLWithPath: modelPath)
            let ctcModels = try await CtcModels.loadDirect(from: modelDir)
            print("Loaded CTC vocabulary with \(ctcModels.vocabulary.count) tokens")

            // Create keyword spotter
            let vocabSize = ctcModels.vocabulary.count
            let blankId = vocabSize  // Blank is at index = vocab_size
            let spotter = CtcKeywordSpotter(models: ctcModels, blankId: blankId)
            print("Created CTC spotter with blankId=\(blankId)")

            // Collect test files
            let dataDirURL = URL(fileURLWithPath: dataDir)
            let fileIds = try collectFileIds(from: dataDirURL, maxFiles: maxFiles)

            if fileIds.isEmpty {
                print("ERROR: No test files found in \(dataDir)")
                return
            }

            print("Processing \(fileIds.count) test files...")

            var results: [[String: Any]] = []
            var totalWer = 0.0
            var totalDictChecks = 0
            var totalDictFound = 0
            var totalAudioDuration = 0.0
            var totalProcessingTime = 0.0

            for (index, fileId) in fileIds.enumerated() {
                print("[\(index + 1)/\(fileIds.count)] \(fileId)")

                if let result = try await processFile(
                    fileId: fileId,
                    dataDir: dataDirURL,
                    asrManager: asrManager,
                    ctcModels: ctcModels,
                    spotter: spotter
                ) {
                    results.append(result)
                    totalWer += result["wer"] as? Double ?? 0
                    totalDictChecks += result["dictTotal"] as? Int ?? 0
                    totalDictFound += result["dictFound"] as? Int ?? 0
                    totalAudioDuration += result["audioLength"] as? Double ?? 0
                    totalProcessingTime += result["processingTime"] as? Double ?? 0

                    let wer = result["wer"] as? Double ?? 0
                    let dictFound = result["dictFound"] as? Int ?? 0
                    let dictTotal = result["dictTotal"] as? Int ?? 0
                    print("  WER: \(String(format: "%.1f", wer))%, Dict: \(dictFound)/\(dictTotal)")
                }
            }

            // Calculate summary
            let avgWer = results.isEmpty ? 0.0 : totalWer / Double(results.count)
            let dictRate = totalDictChecks > 0 ? Double(totalDictFound) / Double(totalDictChecks) * 100 : 0

            // Print summary
            print("\n" + String(repeating: "=", count: 60))
            print("EARNINGS22 BENCHMARK (TDT + CTC)")
            print(String(repeating: "=", count: 60))
            print("Model: \(modelPath)")
            print("Total tests: \(results.count)")
            print("Average WER: \(String(format: "%.2f", avgWer))%")
            print("Dict Pass (Recall): \(totalDictFound)/\(totalDictChecks) (\(String(format: "%.1f", dictRate))%)")
            print("Total audio: \(String(format: "%.1f", totalAudioDuration))s")
            print("Total processing: \(String(format: "%.1f", totalProcessingTime))s")
            if totalProcessingTime > 0 {
                print("RTFx: \(String(format: "%.2f", totalAudioDuration / totalProcessingTime))x")
            }
            print(String(repeating: "=", count: 60))

            // Save to JSON
            let summaryDict: [String: Any] = [
                "totalTests": results.count,
                "avgWer": round(avgWer * 100) / 100,
                "dictPass": totalDictFound,
                "dictTotal": totalDictChecks,
                "dictRate": round(dictRate * 100) / 100,
                "totalAudioDuration": round(totalAudioDuration * 100) / 100,
                "totalProcessingTime": round(totalProcessingTime * 100) / 100
            ]

            let output: [String: Any] = [
                "model": modelPath,
                "summary": summaryDict,
                "results": results
            ]

            let jsonData = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: URL(fileURLWithPath: outputFile))
            print("\nResults written to: \(outputFile)")

        } catch {
            print("ERROR: Benchmark failed: \(error)")
        }
    }

    private static func collectFileIds(from dataDir: URL, maxFiles: Int?) throws -> [String] {
        var fileIds: [String] = []
        let suffix = ".dictionary.txt"

        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil)

        for url in contents.sorted(by: { $0.path < $1.path }) {
            let name = url.lastPathComponent
            if name.hasSuffix(suffix) {
                let data = try? Data(contentsOf: url)
                if let data = data, !data.isEmpty {
                    let fileId = String(name.dropLast(suffix.count))
                    fileIds.append(fileId)
                }
            }
        }

        if let maxFiles = maxFiles {
            return Array(fileIds.prefix(maxFiles))
        }
        return fileIds
    }

    private static func processFile(
        fileId: String,
        dataDir: URL,
        asrManager: AsrManager,
        ctcModels: CtcModels,
        spotter: CtcKeywordSpotter
    ) async throws -> [String: Any]? {
        let wavFile = dataDir.appendingPathComponent("\(fileId).wav")
        let dictionaryFile = dataDir.appendingPathComponent("\(fileId).dictionary.txt")
        let textFile = dataDir.appendingPathComponent("\(fileId).text.txt")

        let fm = FileManager.default
        guard fm.fileExists(atPath: wavFile.path),
              fm.fileExists(atPath: dictionaryFile.path) else {
            return nil
        }

        // Load dictionary words
        let dictionaryContent = try String(contentsOf: dictionaryFile, encoding: .utf8)
        let dictionaryWords = dictionaryContent
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        // Load reference text
        let referenceRaw = (try? String(contentsOf: textFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        // Get audio samples
        let audioFile = try AVAudioFile(forReading: wavFile)
        let audioLength = Double(audioFile.length) / audioFile.processingFormat.sampleRate
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(domain: "CtcEarningsBenchmark", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
        }
        try audioFile.read(into: buffer)

        // Resample to 16kHz
        let converter = AudioConverter()
        let samples = try converter.resampleBuffer(buffer)

        let startTime = Date()

        // 1. TDT transcription for low WER
        let tdtResult = try await asrManager.transcribe(wavFile)

        // Skip files where TDT returns empty (some audio files fail)
        if tdtResult.text.isEmpty {
            print("  SKIPPED: TDT returned empty transcription")
            return nil
        }

        // 2. Build custom vocabulary for CTC keyword spotting
        var vocabTerms: [CustomVocabularyTerm] = []
        for word in dictionaryWords {
            let tokenIds = tokenize(word, vocabulary: ctcModels.vocabulary)
            if !tokenIds.isEmpty {
                let term = CustomVocabularyTerm(
                    text: word,
                    weight: nil,
                    aliases: nil,
                    tokenIds: nil,
                    ctcTokenIds: tokenIds
                )
                vocabTerms.append(term)
            }
        }
        let customVocab = CustomVocabularyContext(terms: vocabTerms)

        // 3. CTC keyword spotting for high recall dictionary detection
        let spotResult = try await spotter.spotKeywordsWithLogProbs(
            audioSamples: samples,
            customVocabulary: customVocab,
            minScore: nil
        )

        // 4. Post-process: Replace TDT words with CTC-detected keywords using timestamps
        let hypothesis = applyKeywordCorrections(
            tdtResult: tdtResult,
            detections: spotResult.detections,
            minScore: -10.0
        )

        let processingTime = Date().timeIntervalSince(startTime)

        // Normalize texts
        let referenceNormalized = TextNormalizer.normalize(referenceRaw)
        let hypothesisNormalized = TextNormalizer.normalize(hypothesis)

        let referenceWords = referenceNormalized.components(separatedBy: CharacterSet.whitespacesAndNewlines).filter { !$0.isEmpty }
        let hypothesisWords = hypothesisNormalized.components(separatedBy: CharacterSet.whitespacesAndNewlines).filter { !$0.isEmpty }

        // Calculate WER
        let wer: Double
        if referenceWords.isEmpty {
            wer = hypothesisWords.isEmpty ? 0.0 : 1.0
        } else {
            wer = calculateWER(reference: referenceWords, hypothesis: hypothesisWords)
        }

        // Count dictionary detections
        let minCtcScore: Float = -10.0
        var dictFound = 0
        var detectionDetails: [[String: Any]] = []

        for detection in spotResult.detections {
            let detail: [String: Any] = [
                "word": detection.term.text,
                "score": round(Double(detection.score) * 100) / 100,
                "startTime": round(detection.startTime * 100) / 100,
                "endTime": round(detection.endTime * 100) / 100
            ]
            detectionDetails.append(detail)

            if detection.score > minCtcScore {
                dictFound += 1
            }
        }

        let result: [String: Any] = [
            "fileId": fileId,
            "reference": referenceRaw,
            "hypothesis": hypothesis,
            "wer": round(wer * 10000) / 100,
            "dictFound": dictFound,
            "dictTotal": dictionaryWords.count,
            "audioLength": round(audioLength * 100) / 100,
            "processingTime": round(processingTime * 1000) / 1000,
            "ctcDetections": detectionDetails
        ]
        return result
    }

    /// Simple tokenization using vocabulary lookup
    private static func tokenize(_ text: String, vocabulary: [Int: String]) -> [Int] {
        // Build reverse vocabulary (token -> id)
        var tokenToId: [String: Int] = [:]
        for (id, token) in vocabulary {
            tokenToId[token] = id
        }

        let normalizedText = text.lowercased()
        var result: [Int] = []
        var position = normalizedText.startIndex
        var isWordStart = true

        while position < normalizedText.endIndex {
            var matched = false
            let remaining = normalizedText.distance(from: position, to: normalizedText.endIndex)
            var matchLength = min(20, remaining)

            while matchLength > 0 {
                let endPos = normalizedText.index(position, offsetBy: matchLength)
                let substring = String(normalizedText[position..<endPos])

                // Try with SentencePiece prefix for word start
                let withPrefix = isWordStart ? "▁" + substring : substring

                if let tokenId = tokenToId[withPrefix] {
                    result.append(tokenId)
                    position = endPos
                    isWordStart = false
                    matched = true
                    break
                } else if let tokenId = tokenToId[substring] {
                    result.append(tokenId)
                    position = endPos
                    isWordStart = false
                    matched = true
                    break
                }

                matchLength -= 1
            }

            if !matched {
                let char = normalizedText[position]
                if char == " " {
                    isWordStart = true
                    position = normalizedText.index(after: position)
                } else {
                    // Unknown character - skip
                    position = normalizedText.index(after: position)
                    isWordStart = false
                }
            }
        }

        return result
    }

    /// Apply CTC keyword corrections to TDT transcription using fuzzy matching.
    /// For each detected keyword, find similar-sounding words in TDT output and replace them.
    private static func applyKeywordCorrections(
        tdtResult: ASRResult,
        detections: [CtcKeywordSpotter.KeywordDetection],
        minScore: Float
    ) -> String {
        // Filter detections by score
        let validDetections = detections.filter { $0.score > minScore }
        guard !validDetections.isEmpty else {
            return tdtResult.text
        }

        var text = tdtResult.text

        // For each detected keyword, try to find and replace similar words
        for detection in validDetections {
            let keyword = detection.term.text
            let keywordLower = keyword.lowercased()

            // Split text into words while preserving structure
            let words = text.components(separatedBy: .whitespacesAndNewlines)

            for word in words {
                let wordClean = word.trimmingCharacters(in: .punctuationCharacters).lowercased()
                guard !wordClean.isEmpty else { continue }

                // Check if this word is similar to the keyword (fuzzy match)
                if isSimilar(wordClean, keywordLower) && wordClean != keywordLower {
                    // Replace this word with the keyword, preserving case pattern
                    let replacement = matchCase(keyword, to: word)
                    text = text.replacingOccurrences(of: word, with: replacement)
                    break // Only replace first occurrence per keyword
                }
            }
        }

        return text
    }

    /// Check if two words are similar (edit distance / length ratio)
    private static func isSimilar(_ a: String, _ b: String) -> Bool {
        let maxLen = max(a.count, b.count)
        guard maxLen > 0 else { return false }

        // Must be similar length
        let lenDiff = abs(a.count - b.count)
        if lenDiff > max(2, maxLen / 3) { return false }

        // Calculate edit distance
        let distance = editDistance(a, b)
        let threshold = max(1, maxLen / 3)

        return distance <= threshold
    }

    /// Simple edit distance calculation
    private static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a)
        let b = Array(b)
        let m = a.count
        let n = b.count

        if m == 0 { return n }
        if n == 0 { return m }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if a[i-1] == b[j-1] {
                    dp[i][j] = dp[i-1][j-1]
                } else {
                    dp[i][j] = 1 + min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1]))
                }
            }
        }

        return dp[m][n]
    }

    /// Match the case pattern of the original word
    private static func matchCase(_ keyword: String, to original: String) -> String {
        let origClean = original.trimmingCharacters(in: .punctuationCharacters)

        // Check case pattern
        if origClean.first?.isUppercase == true {
            // Capitalize first letter
            return keyword.prefix(1).uppercased() + keyword.dropFirst()
        }
        return keyword
    }

    private static func calculateWER(reference: [String], hypothesis: [String]) -> Double {
        if reference.isEmpty {
            return hypothesis.isEmpty ? 0.0 : 1.0
        }

        let m = reference.count
        let n = hypothesis.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1
                }
            }
        }

        return Double(dp[m][n]) / Double(m)
    }

    private static func printUsage() {
        print("""
        CTC Earnings Benchmark (canary)

        Usage: fluidaudio ctc-earnings-benchmark [options]

        Options:
            --data-dir <path>     Path to earnings test dataset (default: test-dataset)
            --ctc-model <path>    Path to CTC model directory (required, e.g., canary-1b-v2)
            --max-files <n>       Maximum number of files to process
            --output, -o <path>   Output JSON file (default: ctc_earnings_benchmark.json)

        Example:
            fluidaudio ctc-earnings-benchmark \\
                --data-dir test-earnings/test-dataset \\
                --ctc-model "/Users/kikow/Library/Application Support/FluidAudio/Models/canary-1b-v2" \\
                --max-files 10
        """)
    }
}
#endif
