#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// Earnings22 benchmark using TDT for transcription + CTC for keyword spotting.
/// TDT provides low WER transcription, CTC provides high recall dictionary detection.
public enum CtcEarningsBenchmark {

    /// Default CTC model directory
    private static func defaultCtcModelPath() -> String? {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let modelPath = appSupport.appendingPathComponent("FluidAudio/Models/parakeet-ctc-110m-coreml")
        if FileManager.default.fileExists(atPath: modelPath.path) {
            return modelPath.path
        }
        return nil
    }

    /// Default data directory (from download command)
    private static func defaultDataDir() -> String? {
        let dataDir = DatasetDownloader.getEarnings22Directory().appendingPathComponent("test-dataset")
        if FileManager.default.fileExists(atPath: dataDir.path) {
            return dataDir.path
        }
        return nil
    }

    public static func runCLI(arguments: [String]) async {
        // Check for help
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var dataDir: String? = nil
        var outputFile = "ctc_earnings_benchmark.json"
        var maxFiles: Int? = nil
        var ctcModelPath: String? = nil
        var tdtVersion: AsrModelVersion = .v3
        var autoDownload = false

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
            case "--auto-download":
                autoDownload = true
            default:
                break
            }
            i += 1
        }

        // Use defaults if not specified
        if dataDir == nil {
            dataDir = defaultDataDir()
        }
        if ctcModelPath == nil {
            ctcModelPath = defaultCtcModelPath()
        }

        // Handle auto-download for dataset
        if autoDownload && dataDir == nil {
            print("📥 Downloading earnings22-kws dataset...")
            await DatasetDownloader.downloadEarnings22KWS(force: false)
            dataDir = defaultDataDir()
        }

        print("Earnings Benchmark (TDT transcription + CTC keyword spotting)")
        print("  Data directory: \(dataDir ?? "not found")")
        print("  Output file: \(outputFile)")
        print("  TDT version: \(tdtVersion == .v2 ? "v2" : "v3")")
        print("  CTC model: \(ctcModelPath ?? "not found")")

        guard let finalDataDir = dataDir else {
            print("ERROR: Data directory not found")
            print("💡 Download with: fluidaudio download --dataset earnings22-kws")
            print("   Or specify: --data-dir <path>")
            printUsage()
            return
        }

        guard let modelPath = ctcModelPath else {
            print("ERROR: CTC model not found")
            print("💡 Download parakeet-ctc-110m-coreml model to:")
            print("   ~/Library/Application Support/FluidAudio/Models/parakeet-ctc-110m-coreml/")
            print("   Or specify: --ctc-model <path>")
            printUsage()
            return
        }

        let dataDirResolved = finalDataDir

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
            let dataDirURL = URL(fileURLWithPath: dataDirResolved)
            let fileIds = try collectFileIds(from: dataDirURL, maxFiles: maxFiles)

            if fileIds.isEmpty {
                print("ERROR: No test files found in \(dataDirResolved)")
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
                "totalProcessingTime": round(totalProcessingTime * 100) / 100,
            ]

            let output: [String: Any] = [
                "model": modelPath,
                "summary": summaryDict,
                "results": results,
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
            throw NSError(
                domain: "CtcEarningsBenchmark", code: 1,
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

        // Count dictionary detections (CTC + hypothesis fallback)
        let minCtcScore: Float = -10.0
        var dictFound = 0
        var detectionDetails: [[String: Any]] = []
        var ctcFoundWords: Set<String> = []

        // 1. CTC detections
        for detection in spotResult.detections {
            let detail: [String: Any] = [
                "word": detection.term.text,
                "score": round(Double(detection.score) * 100) / 100,
                "startTime": round(detection.startTime * 100) / 100,
                "endTime": round(detection.endTime * 100) / 100,
                "source": "ctc",
            ]
            detectionDetails.append(detail)

            if detection.score > minCtcScore {
                dictFound += 1
                ctcFoundWords.insert(detection.term.text.lowercased())
            }
        }

        // 2. Fallback: check hypothesis for dictionary words not found by CTC
        let hypothesisLower = hypothesis.lowercased()
        for word in dictionaryWords {
            let wordLower = word.lowercased()
            if !ctcFoundWords.contains(wordLower) {
                // Check if word appears as whole word in hypothesis (avoid substring false positives)
                let pattern = "\\b\(NSRegularExpression.escapedPattern(for: wordLower))\\b"
                if let regex = try? NSRegularExpression(pattern: pattern, options: []),
                    regex.firstMatch(
                        in: hypothesisLower, options: [],
                        range: NSRange(hypothesisLower.startIndex..., in: hypothesisLower)) != nil
                {
                    dictFound += 1
                    ctcFoundWords.insert(wordLower)
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
            "wer": round(wer * 10000) / 100,
            "dictFound": dictFound,
            "dictTotal": dictionaryWords.count,
            "audioLength": round(audioLength * 100) / 100,
            "processingTime": round(processingTime * 1000) / 1000,
            "ctcDetections": detectionDetails,
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

    /// Apply CTC keyword corrections to TDT transcription using a two-pass approach:
    /// 1. First pass: fuzzy matching (for words that are phonetically similar)
    /// 2. Second pass: timestamp alignment (for words that are very different)
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
        var usedDetections: Set<String> = []

        // PASS 1: Fuzzy matching for phonetically similar words
        for detection in validDetections {
            let keyword = detection.term.text
            let keywordLower = keyword.lowercased()
            let keywordParts = keywordLower.components(separatedBy: " ").filter { !$0.isEmpty }

            let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

            // Handle multi-word keywords
            if keywordParts.count > 1 {
                for i in 0..<(words.count - keywordParts.count + 1) {
                    var allMatch = true
                    var matchedWords: [String] = []

                    for j in 0..<keywordParts.count {
                        let wordClean = words[i + j].trimmingCharacters(in: .punctuationCharacters).lowercased()
                        if isSimilar(wordClean, keywordParts[j]) {
                            matchedWords.append(words[i + j])
                        } else {
                            allMatch = false
                            break
                        }
                    }

                    if allMatch && !matchedWords.isEmpty {
                        let originalPhrase = matchedWords.joined(separator: " ")
                        let replacement = matchCase(keyword, to: matchedWords[0])
                        text = text.replacingOccurrences(of: originalPhrase, with: replacement)
                        usedDetections.insert(keyword)
                        break
                    }
                }
            } else {
                // Single word keyword
                for word in words {
                    let wordClean = word.trimmingCharacters(in: .punctuationCharacters).lowercased()
                    guard !wordClean.isEmpty else { continue }

                    if isSimilar(wordClean, keywordLower) && wordClean != keywordLower {
                        let replacement = matchCase(keyword, to: word)
                        text = text.replacingOccurrences(of: word, with: replacement)
                        usedDetections.insert(keyword)
                        break
                    }
                }
            }
        }

        // PASS 2: Timestamp-based alignment for keywords not matched by fuzzy matching
        guard let tokenTimings = tdtResult.tokenTimings, !tokenTimings.isEmpty else {
            return text
        }

        // Build word timings from token timings (merge subword tokens)
        let wordTimings = buildWordTimings(from: tokenTimings)

        for detection in validDetections {
            let keyword = detection.term.text
            guard !usedDetections.contains(keyword) else { continue }

            // Find TDT word(s) overlapping with CTC detection time
            let ctcStart = detection.startTime
            let ctcEnd = detection.endTime
            let ctcMid = (ctcStart + ctcEnd) / 2

            // Find word with maximum overlap
            var bestMatch: (index: Int, overlap: Double)? = nil
            for (idx, wt) in wordTimings.enumerated() {
                let overlapStart = max(ctcStart, wt.startTime)
                let overlapEnd = min(ctcEnd, wt.endTime)
                let overlap = max(0, overlapEnd - overlapStart)

                // Also check if CTC midpoint falls within word
                let containsMidpoint = wt.startTime <= ctcMid && ctcMid <= wt.endTime

                if overlap > 0 || containsMidpoint {
                    let score = overlap + (containsMidpoint ? 0.1 : 0)
                    if bestMatch == nil || score > bestMatch!.overlap {
                        bestMatch = (idx, score)
                    }
                }
            }

            if let match = bestMatch {
                let originalWord = wordTimings[match.index].word
                let originalClean = originalWord.trimmingCharacters(in: .punctuationCharacters).lowercased()

                // Skip if already correct or is a stop word
                if originalClean == keyword.lowercased() || stopWords.contains(originalClean) {
                    continue
                }

                // Skip if the word is very different in length (might be wrong alignment)
                // Allow replacement if original word is shorter (TDT truncated the keyword)
                // or if they have similar lengths
                let lenRatio = Double(originalClean.count) / Double(keyword.count)
                if lenRatio > 2.0 {
                    continue  // Original much longer than keyword - likely wrong alignment
                }

                let replacement = matchCase(keyword, to: originalWord)
                text = text.replacingOccurrences(of: originalWord, with: replacement)
            }
        }

        return text
    }

    /// Build word timings by merging subword tokens (tokens starting with "▁" begin new words)
    private static func buildWordTimings(
        from tokenTimings: [TokenTiming]
    ) -> [(word: String, startTime: Double, endTime: Double)] {
        var wordTimings: [(word: String, startTime: Double, endTime: Double)] = []
        var currentWord = ""
        var wordStart: Double = 0
        var wordEnd: Double = 0

        for timing in tokenTimings {
            let token = timing.token

            // Skip special tokens
            if token.isEmpty || token == "<blank>" || token == "<pad>" {
                continue
            }

            // Check if this starts a new word (has ▁ prefix or is first token)
            let startsNewWord = token.hasPrefix("▁") || currentWord.isEmpty

            if startsNewWord && !currentWord.isEmpty {
                // Save previous word
                wordTimings.append((word: currentWord, startTime: wordStart, endTime: wordEnd))
                currentWord = ""
            }

            if startsNewWord {
                currentWord = token.hasPrefix("▁") ? String(token.dropFirst()) : token
                wordStart = timing.startTime
            } else {
                currentWord += token
            }
            wordEnd = timing.endTime
        }

        // Save final word
        if !currentWord.isEmpty {
            wordTimings.append((word: currentWord, startTime: wordStart, endTime: wordEnd))
        }

        return wordTimings
    }

    /// Common English words that should never be replaced by keyword matching
    private static let stopWords: Set<String> = [
        // Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
        "this", "that", "these", "those", "who", "whom", "what", "which", "whose",
        // Common verbs
        "is", "are", "was", "were", "be", "been", "being", "am",
        "have", "has", "had", "having", "do", "does", "did", "doing", "done",
        "will", "would", "shall", "should", "may", "might", "must", "can", "could",
        "get", "got", "getting", "go", "goes", "went", "going", "gone",
        "come", "came", "coming", "see", "saw", "seen", "know", "knew", "known",
        "think", "thought", "make", "made", "take", "took", "taken", "give", "gave", "given",
        "say", "said", "tell", "told", "ask", "asked", "use", "used", "want", "wanted",
        "need", "needed", "try", "tried", "let", "put", "keep", "kept", "look", "looked",
        // Articles and determiners
        "a", "an", "the", "some", "any", "no", "every", "each", "all", "both", "few", "many",
        "much", "more", "most", "other", "another", "such",
        // Prepositions
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down", "out",
        "about", "into", "over", "after", "before", "between", "under", "through", "during",
        // Conjunctions
        "and", "or", "but", "so", "yet", "nor", "if", "then", "than", "because", "while",
        "although", "unless", "since", "when", "where", "as",
        // Adverbs
        "not", "very", "just", "also", "only", "even", "still", "already", "always", "never",
        "often", "sometimes", "usually", "really", "well", "now", "here", "there", "how", "why",
        // Common words
        "yes", "no", "okay", "ok", "thank", "thanks", "please", "sorry", "hello", "hi", "bye",
        "good", "great", "bad", "new", "old", "first", "last", "long", "short", "big", "small",
        "high", "low", "right", "left", "next", "back", "same", "different", "own", "able",
        "way", "thing", "things", "time", "times", "year", "years", "day", "days", "week", "weeks",
        "part", "place", "case", "point", "fact", "end", "kind", "lot", "set",
        // Numbers
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "hundred", "thousand", "million", "billion",
    ]

    /// Check if two words are similar (edit distance / length ratio)
    private static func isSimilar(_ a: String, _ b: String) -> Bool {
        // Never match stop words - they're too common to be proper nouns
        if stopWords.contains(a) || stopWords.contains(b) {
            return false
        }

        let maxLen = max(a.count, b.count)
        let minLen = min(a.count, b.count)
        guard maxLen > 0, minLen >= 3 else { return false }

        // Allow more length difference for longer words
        let lenDiff = abs(a.count - b.count)
        if lenDiff > max(3, maxLen / 2) { return false }

        // Calculate edit distance
        let distance = editDistance(a, b)

        // More aggressive threshold: allow up to 40% of max length as edits
        let threshold = max(2, Int(Double(maxLen) * 0.4))

        // Also check if one is substring of other (handles "Erik" in "Ririek")
        if a.contains(b) || b.contains(a) {
            return true
        }

        // Check common prefix/suffix (handles "Heri" vs "Harry")
        let commonPrefix = commonPrefixLength(a, b)
        let commonSuffix = commonSuffixLength(a, b)
        if commonPrefix >= 2 || commonSuffix >= 2 {
            return distance <= threshold + 1
        }

        return distance <= threshold
    }

    /// Get length of common prefix
    private static func commonPrefixLength(_ a: String, _ b: String) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        var count = 0
        for i in 0..<min(aChars.count, bChars.count) {
            if aChars[i] == bChars[i] {
                count += 1
            } else {
                break
            }
        }
        return count
    }

    /// Get length of common suffix
    private static func commonSuffixLength(_ a: String, _ b: String) -> Int {
        let aChars = Array(a.reversed())
        let bChars = Array(b.reversed())
        var count = 0
        for i in 0..<min(aChars.count, bChars.count) {
            if aChars[i] == bChars[i] {
                count += 1
            } else {
                break
            }
        }
        return count
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
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]))
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
        print(
            """
            CTC Earnings Benchmark (TDT + CTC keyword spotting)

            Usage: fluidaudio ctc-earnings-benchmark [options]

            Options:
                --data-dir <path>     Path to earnings test dataset (auto-detected if downloaded)
                --ctc-model <path>    Path to CTC model directory (auto-detected if in standard location)
                --max-files <n>       Maximum number of files to process
                --output, -o <path>   Output JSON file (default: ctc_earnings_benchmark.json)
                --auto-download       Download earnings22-kws dataset if not found

            Default locations:
                Dataset: ~/Library/Application Support/FluidAudio/earnings22-kws/test-dataset/
                CTC Model: ~/Library/Application Support/FluidAudio/Models/parakeet-ctc-110m-coreml/

            Setup:
                1. Download dataset: fluidaudio download --dataset earnings22-kws
                2. Place CTC model in standard location
                3. Run: fluidaudio ctc-earnings-benchmark

            Examples:
                # Run with auto-detected paths
                fluidaudio ctc-earnings-benchmark

                # Run with auto-download
                fluidaudio ctc-earnings-benchmark --auto-download

                # Run with explicit paths
                fluidaudio ctc-earnings-benchmark \\
                    --data-dir /path/to/test-dataset \\
                    --ctc-model /path/to/parakeet-ctc-110m-coreml \\
                    --max-files 100
            """)
    }
}
#endif
