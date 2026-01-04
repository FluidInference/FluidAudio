#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Earnings22 benchmark using ONLY the Hybrid 110M model (single encoder).
/// CTC head provides both transcription AND keyword spotting from the same encoder.
public enum HybridEarningsBenchmark {

    public static func runCLI(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var outputFile = "hybrid_earnings_benchmark.json"
        var maxFiles: Int? = nil

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
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
            default:
                break
            }
            i += 1
        }

        let dataDir = DatasetDownloader.getEarnings22Directory().appendingPathComponent("test-dataset")
        guard FileManager.default.fileExists(atPath: dataDir.path) else {
            print("ERROR: Earnings dataset not found at \(dataDir.path)")
            print("Download with: fluidaudio download --dataset earnings22-kws")
            return
        }

        print("Hybrid 110M Earnings Benchmark (Single Encoder: CTC for both transcription & keywords)")
        print("  Output file: \(outputFile)")

        do {
            // Load Hybrid 110M model (single encoder with CTC head)
            print("Loading Hybrid 110M model...")
            let hybridModels = try await HybridAsrModels.downloadAndLoad()
            let hybridManager = HybridAsrManager(models: hybridModels)
            let spotter = HybridKeywordSpotter(vocabulary: hybridModels.vocabulary, blankId: hybridModels.blankId)
            print("  Vocab size: \(hybridModels.vocabSize)")

            // Collect test files
            let fileIds = try collectFileIds(from: dataDir, maxFiles: maxFiles)

            if fileIds.isEmpty {
                print("ERROR: No test files found")
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
                    dataDir: dataDir,
                    hybridManager: hybridManager,
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
            print("HYBRID 110M BENCHMARK (Single Encoder)")
            print(String(repeating: "=", count: 60))
            print("Model: parakeet-ctc-110m-coreml (Hybrid CTC transcription + keywords)")
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
                "totalProcessingTime": round(totalProcessingTime * 100) / 100,
            ]

            let output: [String: Any] = [
                //"model": "parakeet-tdt-ctc-110m-hybrid",
                "model": "parakeet-ctc-110m-coreml",
                "approach": "single-encoder",
                "summary": summaryDict,
                "results": results,
            ]

            let jsonData = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: URL(fileURLWithPath: outputFile))
            print("\nResults written to: \(outputFile)")

        } catch {
            print("ERROR: \(error)")
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
        hybridManager: HybridAsrManager,
        spotter: HybridKeywordSpotter
    ) async throws -> [String: Any]? {
        let wavFile = dataDir.appendingPathComponent("\(fileId).wav")
        let dictionaryFile = dataDir.appendingPathComponent("\(fileId).dictionary.txt")
        let textFile = dataDir.appendingPathComponent("\(fileId).text.txt")

        let fm = FileManager.default
        guard fm.fileExists(atPath: wavFile.path),
            fm.fileExists(atPath: dictionaryFile.path)
        else {
            return nil
        }

        // Load dictionary words
        let dictionaryContent = try String(contentsOf: dictionaryFile, encoding: .utf8)
        let dictionaryWords =
            dictionaryContent
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        // Load reference text
        let referenceRaw =
            (try? String(contentsOf: textFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        // Get audio samples
        let audioFile = try AVAudioFile(forReading: wavFile)
        let audioLength = Double(audioFile.length) / audioFile.processingFormat.sampleRate
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        try audioFile.read(into: buffer)

        // Resample to 16kHz
        let converter = AudioConverter()
        let samples = try converter.resampleBuffer(buffer)

        let startTime = Date()

        // Build custom vocabulary for keyword spotting
        var vocabTerms: [CustomVocabularyTerm] = []
        for word in dictionaryWords {
            let term = CustomVocabularyTerm(
                text: word,
                weight: nil,
                aliases: nil,
                tokenIds: nil,
                ctcTokenIds: nil
            )
            vocabTerms.append(term)
        }
        let customVocab = CustomVocabularyContext(terms: vocabTerms)

        // Run Hybrid 110M (single encoder for BOTH transcription and keywords)
        let hybridResult = try await hybridManager.transcribe(
            audioSamples: samples,
            customVocabulary: customVocab
        )

        // Skip if empty transcription
        if hybridResult.text.isEmpty {
            print("  SKIPPED: Empty transcription")
            return nil
        }

        // Spot keywords using CTC log-probs from same encoder
        let detections = spotter.spotKeywords(
            ctcLogProbs: hybridResult.ctcLogProbs,
            frameDuration: hybridResult.frameDuration,
            customVocabulary: customVocab
        )

        let processingTime = Date().timeIntervalSince(startTime)

        // Use hybrid transcription as hypothesis
        let hypothesis = hybridResult.text

        // Normalize texts
        let referenceNormalized = TextNormalizer.normalize(referenceRaw)
        let hypothesisNormalized = TextNormalizer.normalize(hypothesis)

        let referenceWords = referenceNormalized.components(separatedBy: CharacterSet.whitespacesAndNewlines).filter {
            !$0.isEmpty
        }
        let hypothesisWords = hypothesisNormalized.components(separatedBy: CharacterSet.whitespacesAndNewlines).filter {
            !$0.isEmpty
        }

        // Calculate WER
        let wer: Double
        if referenceWords.isEmpty {
            wer = hypothesisWords.isEmpty ? 0.0 : 1.0
        } else {
            wer = calculateWER(reference: referenceWords, hypothesis: hypothesisWords)
        }

        // Count dictionary detections
        let minCtcScore: Float = -15.0
        var dictFound = 0
        var detectionDetails: [[String: Any]] = []
        var foundWords: Set<String> = []

        // CTC detections
        for detection in detections {
            let detail: [String: Any] = [
                "word": detection.term.text,
                "score": round(Double(detection.score) * 100) / 100,
                "startTime": round(detection.startTime * 100) / 100,
                "endTime": round(detection.endTime * 100) / 100,
                "source": "ctc",
            ]
            detectionDetails.append(detail)

            if detection.score >= minCtcScore {
                dictFound += 1
                foundWords.insert(detection.term.text.lowercased())
            }
        }

        // Fallback: check hypothesis for dictionary words not found by CTC
        let hypothesisLower = hypothesis.lowercased()
        for word in dictionaryWords {
            let wordLower = word.lowercased()
            if !foundWords.contains(wordLower) {
                let pattern = "\\b\(NSRegularExpression.escapedPattern(for: wordLower))\\b"
                if let regex = try? NSRegularExpression(pattern: pattern, options: []),
                    regex.firstMatch(
                        in: hypothesisLower, options: [],
                        range: NSRange(hypothesisLower.startIndex..., in: hypothesisLower)) != nil
                {
                    dictFound += 1
                    foundWords.insert(wordLower)
                    let detail: [String: Any] = [
                        "word": word,
                        "score": 0.0,
                        "startTime": 0.0,
                        "endTime": 0.0,
                        "source": "hypothesis",
                    ]
                    detectionDetails.append(detail)
                }
            }
        }

        let result: [String: Any] = [
            "fileId": fileId,
            "reference": referenceRaw,
            "hypothesis": hypothesis,
            "referenceNormalized": referenceNormalized,
            "hypothesisNormalized": hypothesisNormalized,
            "wer": round(wer * 10000) / 100,
            "dictFound": dictFound,
            "dictTotal": dictionaryWords.count,
            "audioLength": round(audioLength * 100) / 100,
            "processingTime": round(processingTime * 1000) / 1000,
            "ctcDetections": detectionDetails,
        ]
        return result
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
        print(
            """
            Hybrid 110M Earnings Benchmark (Single Encoder)

            Usage: fluidaudio hybrid-earnings-benchmark [options]

            This benchmark uses ONLY the Hybrid 110M model:
            - Single encoder provides CTC log-probs
            - CTC greedy decode for transcription
            - CTC keyword spotting from same encoder output

            Options:
                --max-files <n>       Maximum number of files to process
                --output, -o <path>   Output JSON file (default: hybrid_earnings_benchmark.json)

            Compare with:
                fluidaudio ctc-earnings-benchmark  (Canary-CTC + TDT 0.6B, two encoders)
            """)
    }
}
#endif
