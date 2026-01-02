#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Earnings22 benchmark using ONLY the Hybrid 110M model (single encoder).
/// CTC head provides both transcription AND keyword spotting from the same encoder.
public enum HybridEarningsBenchmark {

    private enum KeywordMode: String {
        case chunk
        case file
    }

    public static func runCLI(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var outputFile = "hybrid_earnings_benchmark.json"
        var maxFiles: Int? = nil
        var decodingMode: HybridDecodingMode = .tdt
        var useRescoring = false
        var keywordMode: KeywordMode = .chunk

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
            case "--ctc":
                decodingMode = .ctc
            case "--tdt":
                decodingMode = .tdt
            case "--rescore":
                useRescoring = true
            case "--keyword-mode":
                if i + 1 < arguments.count, let mode = parseKeywordMode(arguments[i + 1]) {
                    keywordMode = mode
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        let dataDir = DatasetDownloader.getEarnings22Directory().appendingPathComponent("test-dataset")
        if !FileManager.default.fileExists(atPath: dataDir.path) {
            print("📥 Earnings dataset not found, downloading earnings22-kws...")
            await DatasetDownloader.downloadEarnings22KWS(force: false)
        }

        guard FileManager.default.fileExists(atPath: dataDir.path) else {
            print("ERROR: Earnings dataset not found at \(dataDir.path)")
            print("Download with: fluidaudio download --dataset earnings22-kws")
            return
        }

        let modeStr = decodingMode == .ctc ? "CTC" : "TDT"
        let rescoringStr = useRescoring ? " + Rescoring" : ""
        print("Hybrid 110M Earnings Benchmark (Decoding: \(modeStr)\(rescoringStr))")
        print("  Output file: \(outputFile)")
        print("  Decoding mode: \(modeStr)")
        print("  Rescoring: \(useRescoring ? "enabled" : "disabled")")
        print("  Keyword mode: \(keywordMode.rawValue)")

        do {
            // Load Hybrid 110M model (single encoder with CTC head)
            print("Loading Hybrid 110M model...")
            let hybridModels = try await HybridAsrModels.downloadAndLoad()
            let hybridManager = HybridAsrManager(models: hybridModels, decodingMode: decodingMode)
            let spotter = HybridKeywordSpotter(vocabulary: hybridModels.vocabulary, blankId: hybridModels.blankId)
            print("  Vocab size: \(hybridModels.vocabSize)")

            // Collect test files
            let fileIds = try collectFileIds(from: dataDir, maxFiles: maxFiles)
            let keywordIndex = try buildKeywordIndex(dataDir: dataDir, keywordMode: keywordMode)

            if fileIds.isEmpty {
                print("ERROR: No test files found")
                return
            }

            print("Processing \(fileIds.count) test files...")

            var results: [[String: Any]] = []
            var totalWer = 0.0
            var totalKeywordReference = 0
            var totalKeywordPredicted = 0
            var totalKeywordTruePositives = 0
            var totalKeywordFalsePositives = 0
            var totalKeywordFalseNegatives = 0
            var totalAudioDuration = 0.0
            var totalProcessingTime = 0.0

            for (index, fileId) in fileIds.enumerated() {
                print("[\(index + 1)/\(fileIds.count)] \(fileId)")

                if let result = try await processFile(
                    fileId: fileId,
                    dataDir: dataDir,
                    hybridManager: hybridManager,
                    spotter: spotter,
                    useRescoring: useRescoring,
                    keywordMode: keywordMode,
                    keywordIndex: keywordIndex
                ) {
                    results.append(result)
                    totalWer += result["wer"] as? Double ?? 0
                    totalKeywordReference += result["keywordReference"] as? Int ?? 0
                    totalKeywordPredicted += result["keywordPredicted"] as? Int ?? 0
                    totalKeywordTruePositives += result["keywordTruePositives"] as? Int ?? 0
                    totalKeywordFalsePositives += result["keywordFalsePositives"] as? Int ?? 0
                    totalKeywordFalseNegatives += result["keywordFalseNegatives"] as? Int ?? 0
                    totalAudioDuration += result["audioLength"] as? Double ?? 0
                    totalProcessingTime += result["processingTime"] as? Double ?? 0

                    let wer = result["wer"] as? Double ?? 0
                    let precision = result["keywordPrecision"] as? Double ?? 0
                    let recall = result["keywordRecall"] as? Double ?? 0
                    let fscore = result["keywordFscore"] as? Double ?? 0
                    print(
                        "  WER: \(String(format: "%.1f", wer))%, " +
                            "KW P/R/F: \(String(format: "%.2f", precision))/" +
                            "\(String(format: "%.2f", recall))/" +
                            "\(String(format: "%.2f", fscore))"
                    )
                }
            }

            // Calculate summary
            let avgWer = results.isEmpty ? 0.0 : totalWer / Double(results.count)
            let keywordPrecision =
                totalKeywordPredicted > 0
                ? Double(totalKeywordTruePositives) / Double(totalKeywordPredicted)
                : 0
            let keywordRecall =
                totalKeywordReference > 0
                ? Double(totalKeywordTruePositives) / Double(totalKeywordReference)
                : 0
            let keywordFscore =
                (keywordPrecision + keywordRecall) > 0
                ? 2 * keywordPrecision * keywordRecall / (keywordPrecision + keywordRecall)
                : 0

            // Print summary
            print("\n" + String(repeating: "=", count: 60))
            print("HYBRID 110M BENCHMARK (\(modeStr)\(rescoringStr))")
            print(String(repeating: "=", count: 60))
            print("Model: parakeet-tdt-ctc-110m-hybrid")
            print("Decoding: \(modeStr), Rescoring: \(useRescoring ? "yes" : "no")")
            print("Total tests: \(results.count)")
            print("Average WER: \(String(format: "%.2f", avgWer))%")
            print(
                "Keyword Precision/Recall/F1: " +
                    "\(String(format: "%.2f", keywordPrecision))/" +
                    "\(String(format: "%.2f", keywordRecall))/" +
                    "\(String(format: "%.2f", keywordFscore))"
            )
            print("Total audio: \(String(format: "%.1f", totalAudioDuration))s")
            print("Total processing: \(String(format: "%.1f", totalProcessingTime))s")
            if totalProcessingTime > 0 {
                print("RTFx: \(String(format: "%.2f", totalAudioDuration / totalProcessingTime))x")
            }
            print(String(repeating: "=", count: 60))

            // Sort results by WER descending (worst first)
            let sortedResults = results.sorted { r1, r2 in
                let wer1 = r1["wer"] as? Double ?? 0
                let wer2 = r2["wer"] as? Double ?? 0
                return wer1 > wer2
            }

            // Save to JSON
            let summaryDict: [String: Any] = [
                "totalTests": results.count,
                "avgWer": round(avgWer * 100) / 100,
                "keywordTruePositives": totalKeywordTruePositives,
                "keywordFalsePositives": totalKeywordFalsePositives,
                "keywordFalseNegatives": totalKeywordFalseNegatives,
                "keywordPredicted": totalKeywordPredicted,
                "keywordReference": totalKeywordReference,
                "keywordPrecision": round(keywordPrecision * 1000) / 1000,
                "keywordRecall": round(keywordRecall * 1000) / 1000,
                "keywordFscore": round(keywordFscore * 1000) / 1000,
                "totalAudioDuration": round(totalAudioDuration * 100) / 100,
                "totalProcessingTime": round(totalProcessingTime * 100) / 100,
            ]

            let output: [String: Any] = [
                "model": "parakeet-tdt-ctc-110m-hybrid",
                "approach": "single-encoder",
                "decodingMode": modeStr,
                "rescoring": useRescoring,
                "keywordMode": keywordMode.rawValue,
                "summary": summaryDict,
                "results": sortedResults,
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
        spotter: HybridKeywordSpotter,
        useRescoring: Bool,
        keywordMode: KeywordMode,
        keywordIndex: [String: [String]]
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

        // Load dictionary words (chunk or file keywords)
        let dictionaryWords = try loadDictionaryWords(
            fileId: fileId,
            dictionaryFile: dictionaryFile,
            keywordMode: keywordMode,
            keywordIndex: keywordIndex
        )

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

        // Run Hybrid 110M using new API (TDT transcription + CTC keyword detection)
        let rescorerConfig: HybridTextRescorer.Config? = useRescoring ? .default : nil
        let hybridResult = try await hybridManager.transcribeHybrid(
            audioSamples: samples,
            customVocabulary: customVocab,
            rescorerConfig: rescorerConfig
        )

        // Skip if empty transcription
        if hybridResult.text.isEmpty {
            print("  SKIPPED: Empty transcription")
            return nil
        }

        let detections = hybridResult.keywordDetections
        let processingTime = hybridResult.processingTime

        // Use hybrid transcription as hypothesis (may be rescored if enabled)
        let hypothesis = hybridResult.text

        // Normalize texts
        let referenceNormalized = TextNormalizer.normalize(referenceRaw)
        let hypothesisNormalized = TextNormalizer.normalize(hypothesis)

        // Keyword metrics with alignment-based matching (OpenBench-style).
        let keywordStats = computeKeywordStats(
            referenceText: referenceNormalized,
            hypothesisText: hypothesisNormalized,
            dictionaryWords: dictionaryWords
        )
        let keywordPrecision = keywordStats.precision
        let keywordRecall = keywordStats.recall
        let keywordFscore = keywordStats.fscore

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

        // Count dictionary detections for debugging
        let minCtcScore: Float = -15.0
        var detectionDetails: [[String: Any]] = []
        var foundWords: Set<String> = []

        // CTC detections
        for detection in detections {
            let inRef = keywordStats.referenceKeywords.contains(detection.term.text.lowercased())
            let detail: [String: Any] = [
                "word": detection.term.text,
                "score": round(Double(detection.score) * 100) / 100,
                "startTime": round(detection.startTime * 100) / 100,
                "endTime": round(detection.endTime * 100) / 100,
                "source": "ctc",
                "inReference": inRef,
            ]
            detectionDetails.append(detail)

            if detection.score >= minCtcScore {
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
                    foundWords.insert(wordLower)
                    let inRef = keywordStats.referenceKeywords.contains(wordLower)
                    let detail: [String: Any] = [
                        "word": word,
                        "score": 0.0,
                        "startTime": 0.0,
                        "endTime": 0.0,
                        "source": "hypothesis",
                        "inReference": inRef,
                    ]
                    detectionDetails.append(detail)
                }
            }
        }

        let result: [String: Any] = [
            "fileId": fileId,
            "reference": referenceNormalized,
            "hypothesis": hypothesisNormalized,
            "wer": round(wer * 10000) / 100,
            "dictFound": keywordStats.predictedCount,
            "dictTotal": keywordStats.groundTruthCount,
            "keywordPredicted": keywordStats.predictedCount,
            "keywordReference": keywordStats.groundTruthCount,
            "keywordTruePositives": keywordStats.truePositives,
            "keywordFalsePositives": keywordStats.falsePositives,
            "keywordFalseNegatives": keywordStats.falseNegatives,
            "keywordPrecision": round(keywordPrecision * 1000) / 1000,
            "keywordRecall": round(keywordRecall * 1000) / 1000,
            "keywordFscore": round(keywordFscore * 1000) / 1000,
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
                --keyword-mode <mode> Keyword mode: chunk or file (default: chunk)

            Compare with:
                fluidaudio ctc-earnings-benchmark  (Canary-CTC + TDT 0.6B, two encoders)
            """)
    }

    private static func parseKeywordMode(_ value: String) -> KeywordMode? {
        switch value.lowercased() {
        case "chunk", "chunk-keywords":
            return .chunk
        case "file", "file-keywords":
            return .file
        default:
            return nil
        }
    }

    private static func parentId(from fileId: String) -> String {
        guard let range = fileId.range(of: "_chunk") else {
            return fileId
        }
        return String(fileId[..<range.lowerBound])
    }

    private static func buildKeywordIndex(dataDir: URL, keywordMode: KeywordMode) throws -> [String: [String]] {
        guard keywordMode == .file else {
            return [:]
        }

        var index: [String: Set<String>] = [:]
        let suffix = ".dictionary.txt"
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil)

        for url in contents {
            let name = url.lastPathComponent
            guard name.hasSuffix(suffix) else { continue }
            let fileId = String(name.dropLast(suffix.count))
            let parent = parentId(from: fileId)
            let words = try loadDictionaryWords(from: url)
            var set = index[parent] ?? Set<String>()
            set.formUnion(words)
            index[parent] = set
        }

        return index.mapValues { Array($0).sorted() }
    }

    private static func loadDictionaryWords(
        fileId: String,
        dictionaryFile: URL,
        keywordMode: KeywordMode,
        keywordIndex: [String: [String]]
    ) throws -> [String] {
        switch keywordMode {
        case .chunk:
            return try loadDictionaryWords(from: dictionaryFile)
        case .file:
            let parent = parentId(from: fileId)
            if let words = keywordIndex[parent] {
                return words
            }
            return try loadDictionaryWords(from: dictionaryFile)
        }
    }

    private static func loadDictionaryWords(from url: URL) throws -> [String] {
        let dictionaryContent = try String(contentsOf: url, encoding: .utf8)
        return dictionaryContent
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private struct KeywordStats {
        let truePositives: Int
        let falsePositives: Int
        let falseNegatives: Int
        let groundTruthCount: Int
        let predictedCount: Int
        let precision: Double
        let recall: Double
        let fscore: Double
        let referenceKeywords: Set<String>
    }

    private static func computeKeywordStats(
        referenceText: String,
        hypothesisText: String,
        dictionaryWords: [String]
    ) -> KeywordStats {
        let normalizedRef = referenceText
        let normalizedHyp = hypothesisText
        let normalizedKeywords = dictionaryWords.map { TextNormalizer.normalize($0) }.filter { !$0.isEmpty }

        let refWords = normalizedRef.split(separator: " ").map(String.init)
        let hypWords = normalizedHyp.split(separator: " ").map(String.init)

        let alignment = alignWords(reference: refWords, hypothesis: hypWords)
        let keywordStats = computeAlignedKeywordStats(alignment: alignment, keywords: normalizedKeywords)

        let tp = keywordStats.truePositives
        let fp = keywordStats.falsePositives
        let gt = keywordStats.groundTruth
        let precision = (tp + fp) > 0 ? Double(tp) / Double(tp + fp) : 0
        let recall = gt > 0 ? Double(tp) / Double(gt) : 0
        let fscore = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0

        return KeywordStats(
            truePositives: tp,
            falsePositives: fp,
            falseNegatives: max(0, gt - tp),
            groundTruthCount: gt,
            predictedCount: tp + fp,
            precision: precision,
            recall: recall,
            fscore: fscore,
            referenceKeywords: keywordStats.referenceKeywords
        )
    }

    private static func alignWords(
        reference: [String],
        hypothesis: [String]
    ) -> [(ref: String, hyp: String)] {
        let m = reference.count
        let n = hypothesis.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        var back = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }
        for i in 1...m { back[i][0] = 1 }  // delete
        for j in 1...n { back[0][j] = 2 }  // insert

        for i in 1...m {
            for j in 1...n {
                let cost = reference[i - 1] == hypothesis[j - 1] ? 0 : 1
                let subCost = dp[i - 1][j - 1] + cost
                let delCost = dp[i - 1][j] + 1
                let insCost = dp[i][j - 1] + 1

                let minCost = min(subCost, min(delCost, insCost))
                dp[i][j] = minCost

                if minCost == subCost {
                    back[i][j] = 0
                } else if minCost == delCost {
                    back[i][j] = 1
                } else {
                    back[i][j] = 2
                }
            }
        }

        var aligned: [(ref: String, hyp: String)] = []
        var i = m
        var j = n
        let eps = "<eps>"

        while i > 0 || j > 0 {
            let op = back[i][j]
            if op == 0, i > 0, j > 0 {
                aligned.append((reference[i - 1], hypothesis[j - 1]))
                i -= 1
                j -= 1
                continue
            }
            if op == 1, i > 0 {
                aligned.append((reference[i - 1], eps))
                i -= 1
                continue
            }
            if j > 0 {
                aligned.append((eps, hypothesis[j - 1]))
                j -= 1
                continue
            }
            if i > 0 {
                aligned.append((reference[i - 1], eps))
                i -= 1
            }
        }

        return aligned.reversed()
    }

    private struct AlignedKeywordStats {
        let truePositives: Int
        let groundTruth: Int
        let falsePositives: Int
        let referenceKeywords: Set<String>
    }

    private static func computeAlignedKeywordStats(
        alignment: [(ref: String, hyp: String)],
        keywords: [String]
    ) -> AlignedKeywordStats {
        guard !keywords.isEmpty else {
            return AlignedKeywordStats(truePositives: 0, groundTruth: 0, falsePositives: 0, referenceKeywords: [])
        }

        var stats: [String: (tp: Int, gt: Int, fp: Int)] = [:]
        let keywordSet = Set(keywords)
        for keyword in keywords {
            stats[keyword] = (0, 0, 0)
        }

        let eps = "<eps>"
        var maxOrder = 1
        for keyword in keywords {
            let count = keyword.split(separator: " ").count
            if count > maxOrder {
                maxOrder = count
            }
        }

        for pair in alignment {
            let ref = pair.ref
            let hyp = pair.hyp
            if let existing = stats[ref] {
                let updated = (tp: existing.tp + (ref == hyp ? 1 : 0), gt: existing.gt + 1, fp: existing.fp)
                stats[ref] = updated
                continue
            }
            if let existing = stats[hyp] {
                let updated = (tp: existing.tp, gt: existing.gt, fp: existing.fp + 1)
                stats[hyp] = updated
            }
        }

        if maxOrder > 1 {
            for ngramOrder in 2...maxOrder {
                var idx = 0
                var itemRef: [(word: String, index: Int)] = []
                while idx < alignment.count {
                    if !itemRef.isEmpty {
                        let nextIndex = itemRef[0].index + 1
                        if itemRef.count > 1 {
                            itemRef = [itemRef[1]]
                            idx = nextIndex
                            continue
                        }
                        itemRef = []
                        idx = nextIndex
                    }
                    while itemRef.count != ngramOrder, idx < alignment.count {
                        let word = alignment[idx].ref
                        idx += 1
                        if word == eps {
                            continue
                        }
                        itemRef.append((word, idx - 1))
                    }
                    if itemRef.count == ngramOrder {
                        let phraseRef = itemRef.map { $0.word }.joined(separator: " ")
                        let phraseHyp = itemRef.map { alignment[$0.index].hyp }.joined(separator: " ")
                        if let existing = stats[phraseRef] {
                            let updated = (
                                tp: existing.tp + (phraseRef == phraseHyp ? 1 : 0),
                                gt: existing.gt + 1,
                                fp: existing.fp
                            )
                            stats[phraseRef] = updated
                        }
                    }
                }

                idx = 0
                var itemHyp: [(word: String, index: Int)] = []
                while idx < alignment.count {
                    if !itemHyp.isEmpty {
                        let nextIndex = itemHyp[0].index + 1
                        if itemHyp.count > 1 {
                            itemHyp = [itemHyp[1]]
                            idx = nextIndex
                            continue
                        }
                        itemHyp = []
                        idx = nextIndex
                    }
                    while itemHyp.count != ngramOrder, idx < alignment.count {
                        let word = alignment[idx].hyp
                        idx += 1
                        if word == eps {
                            continue
                        }
                        itemHyp.append((word, idx - 1))
                    }
                    if itemHyp.count == ngramOrder {
                        let phraseHyp = itemHyp.map { $0.word }.joined(separator: " ")
                        let phraseRef = itemHyp.map { alignment[$0.index].ref }.joined(separator: " ")
                        if phraseHyp != phraseRef, let existing = stats[phraseHyp] {
                            let updated = (tp: existing.tp, gt: existing.gt, fp: existing.fp + 1)
                            stats[phraseHyp] = updated
                        }
                    }
                }
            }
        }

        var tp = 0
        var gt = 0
        var fp = 0
        for (_, values) in stats {
            tp += values.tp
            gt += values.gt
            fp += values.fp
        }

        return AlignedKeywordStats(
            truePositives: tp,
            groundTruth: gt,
            falsePositives: fp,
            referenceKeywords: keywordSet
        )
    }
}
#endif
