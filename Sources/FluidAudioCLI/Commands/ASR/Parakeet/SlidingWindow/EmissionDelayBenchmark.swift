#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// Measures the systematic offset between TDT-emitted token timestamps and the
/// frames where the same token actually achieves its CTC log-prob peak.
///
/// Premise: the rescorer's window is built from TDT timestamps and expanded by
/// `marginSeconds` (default 0.5s). RNN-T-style decoders (TDT included) routinely
/// emit a token 1–3 encoder frames *after* the acoustic event. If the offset is
/// real and consistent, subtracting it from TDT timestamps should let
/// `marginSeconds` shrink without losing recall.
///
/// For each TDT-recognized word with a clean CTC acoustic match, we record the
/// offset in encoder frames (1 frame = 80 ms at 12.5 fps).
public enum EmissionDelayBenchmark {

    /// One offset sample for a single (file, word) pair.
    private struct OffsetSample {
        let fileId: String
        let word: String
        let tokenCount: Int
        let tdtStartFrame: Int
        let tdtEndFrame: Int
        let ctcStartFrame: Int
        let ctcEndFrame: Int
        let ctcScore: Float
        let frameDuration: Double
        /// Average TDT decoder confidence across the tokens in this word.
        let tdtAvgConfidence: Float
        /// Min TDT decoder confidence across the tokens in this word.
        let tdtMinConfidence: Float
        /// TDT word duration in encoder frames.
        let tdtDurationFrames: Int

        var offsetStart: Int { ctcStartFrame - tdtStartFrame }
        var offsetEnd: Int { ctcEndFrame - tdtEndFrame }
        var offsetCenter: Double {
            let tdtCenter = Double(tdtStartFrame + tdtEndFrame) / 2.0
            let ctcCenter = Double(ctcStartFrame + ctcEndFrame) / 2.0
            return ctcCenter - tdtCenter
        }
    }

    public static func runCLI(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        var dataDir: String? = nil
        var outputFile = "emission_delay_benchmark.json"
        var maxFiles: Int? = nil
        var ctcModelPath: String? = nil
        // Production CTC variant for the rescorer.
        let ctcVariant: CtcModelVariant = .ctc110m
        // v2 mirrors the production ctc-earnings-benchmark default.
        var tdtVersion: AsrModelVersion = .v2
        // Outlier guards on the recorded offsets.
        // Drop samples whose CTC peak landed > maxOffsetSeconds from the TDT center.
        // The search itself is constrained to ±searchHalfWidthSeconds.
        var maxOffsetSeconds: Double = 1.5
        var searchHalfWidthSeconds: Double = 2.0
        var minCtcScore: Float = -15.0
        var minWordChars: Int = 5

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
                    let v = arguments[i + 1].lowercased()
                    switch v {
                    case "v2": tdtVersion = .v2
                    case "v3": tdtVersion = .v3
                    case "110m", "tdt-ctc-110m": tdtVersion = .tdtCtc110m
                    default:
                        print("Unknown TDT version '\(v)', keeping v2")
                        break
                    }
                    i += 1
                }
            case "--max-offset-seconds":
                if i + 1 < arguments.count, let v = Double(arguments[i + 1]) {
                    maxOffsetSeconds = v
                    i += 1
                }
            case "--search-half-width":
                if i + 1 < arguments.count, let v = Double(arguments[i + 1]) {
                    searchHalfWidthSeconds = v
                    i += 1
                }
            case "--min-ctc-score":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    minCtcScore = v
                    i += 1
                }
            case "--min-word-chars":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    minWordChars = v
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        if dataDir == nil { dataDir = defaultDataDir() }
        if ctcModelPath == nil { ctcModelPath = defaultCtcModelPath(for: ctcVariant) }

        guard let finalDataDir = dataDir else {
            print("ERROR: Data directory not found")
            print("💡 Download with: fluidaudio download --dataset earnings22-kws")
            print("   Or specify: --data-dir <path>")
            return
        }
        guard let modelPath = ctcModelPath else {
            print("ERROR: CTC model not found at default path")
            print("   Specify: --ctc-model <path>")
            return
        }

        print("Emission Delay Benchmark (TDT timestamps vs CTC argmax peaks)")
        print("  Data directory: \(finalDataDir)")
        print("  CTC model: \(modelPath)")
        print("  TDT version: \(tdtVersionLabel(tdtVersion))")
        print(
            "  Outlier guard: |offset| <= \(maxOffsetSeconds)s, ctcScore >= \(minCtcScore), word len >= \(minWordChars)"
        )

        do {
            print("Loading TDT (\(tdtVersionLabel(tdtVersion)))...")
            let tdtModels = try await AsrModels.downloadAndLoad(version: tdtVersion)
            let asrManager = AsrManager(config: .default)
            try await asrManager.loadModels(tdtModels)

            print("Loading CTC models...")
            let ctcModelDir = URL(fileURLWithPath: modelPath)
            let ctcModels = try await CtcModels.loadDirect(from: ctcModelDir, variant: ctcVariant)
            let blankId = ctcModels.vocabulary.count
            let spotter = CtcKeywordSpotter(models: ctcModels, blankId: blankId)
            let tokenizer = try await CtcTokenizer.load(from: ctcModelDir)

            let dataDirURL = URL(fileURLWithPath: finalDataDir)
            let fileIds = try collectFileIds(from: dataDirURL, maxFiles: maxFiles)
            guard !fileIds.isEmpty else {
                print("ERROR: No test files found in \(finalDataDir)")
                return
            }

            print("Processing \(fileIds.count) file\(fileIds.count == 1 ? "" : "s")...")

            var samples: [OffsetSample] = []
            var skippedTooShort = 0
            var skippedNoTokens = 0
            var skippedLowScore = 0
            var skippedFarPeak = 0
            var totalWords = 0

            for (idx, fileId) in fileIds.enumerated() {
                let wavFile = dataDirURL.appendingPathComponent("\(fileId).wav")
                guard FileManager.default.fileExists(atPath: wavFile.path) else { continue }

                let audioFile = try AVAudioFile(forReading: wavFile)
                let frameCount = AVAudioFrameCount(audioFile.length)
                guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount)
                else { continue }
                try audioFile.read(into: buffer)
                let converter = AudioConverter()
                let audioSamples = try converter.resampleBuffer(buffer)

                // 1. TDT: get token timings.
                var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)
                let tdtResult = try await asrManager.transcribe(wavFile, decoderState: &decoderState)
                guard !tdtResult.text.isEmpty,
                    let tokenTimings = tdtResult.tokenTimings,
                    !tokenTimings.isEmpty
                else {
                    print("  [\(idx + 1)/\(fileIds.count)] \(fileId): empty TDT, skipping")
                    continue
                }

                // 2. CTC: run once, reuse log-probs across all words.
                let emptyVocab = CustomVocabularyContext(terms: [])
                let probsResult = try await spotter.spotKeywordsWithLogProbs(
                    audioSamples: audioSamples,
                    customVocabulary: emptyVocab,
                    minScore: nil
                )
                let logProbs = probsResult.logProbs
                let frameDuration = probsResult.frameDuration
                guard !logProbs.isEmpty, frameDuration > 0 else {
                    print("  [\(idx + 1)/\(fileIds.count)] \(fileId): empty CTC log-probs, skipping")
                    continue
                }

                // 3. Build TDT word timings.
                let wordTimings = buildWordTimings(from: tokenTimings)

                var fileSamples = 0
                for wt in wordTimings {
                    totalWords += 1

                    // Trim and lowercase.
                    let raw = wt.word.trimmingCharacters(in: .whitespacesAndNewlines)
                    let cleaned = raw.lowercased().filter { $0.isLetter || $0.isNumber || $0 == "'" }
                    guard cleaned.count >= minWordChars else {
                        skippedTooShort += 1
                        continue
                    }

                    let ctcTokens = tokenizer.encode(cleaned)
                    guard !ctcTokens.isEmpty else {
                        skippedNoTokens += 1
                        continue
                    }

                    let term = CustomVocabularyTerm(
                        text: cleaned,
                        weight: nil,
                        aliases: nil,
                        tokenIds: nil,
                        ctcTokenIds: ctcTokens
                    )
                    // minTermLength is enforced inside the spotter; lower it here so
                    // 4-letter words still go through.
                    let singleton = CustomVocabularyContext(
                        terms: [term],
                        minTermLength: max(1, minWordChars)
                    )

                    // Constrain the search to a window around the TDT center.
                    // This avoids the spotter latching onto a different occurrence
                    // of the same word elsewhere in the audio.
                    let tdtCenter = (wt.startTime + wt.endTime) / 2.0
                    let halfFrames = Int((searchHalfWidthSeconds / frameDuration).rounded())
                    let centerFrame = Int((tdtCenter / frameDuration).rounded())
                    let windowStart = max(0, centerFrame - halfFrames)
                    let windowEnd = min(logProbs.count, centerFrame + halfFrames)
                    guard windowEnd > windowStart else {
                        skippedFarPeak += 1
                        continue
                    }
                    let windowed = Array(logProbs[windowStart..<windowEnd])

                    let spotResult = spotter.spotKeywordsFromLogProbs(
                        logProbs: windowed,
                        frameDuration: frameDuration,
                        customVocabulary: singleton,
                        minScore: -50.0
                    )
                    guard !spotResult.detections.isEmpty else {
                        skippedNoTokens += 1
                        continue
                    }

                    // Pick the highest-scoring detection inside the window.
                    let best = spotResult.detections.max { $0.score < $1.score }!

                    if best.score < minCtcScore {
                        skippedLowScore += 1
                        continue
                    }

                    // Translate window-local frames back to global.
                    let ctcStartFrame = best.startFrame + windowStart
                    let ctcEndFrame = best.endFrame + windowStart

                    let tdtStartFrame = Int((wt.startTime / frameDuration).rounded())
                    let tdtEndFrame = Int((wt.endTime / frameDuration).rounded())

                    let tdtCenterFrame = Double(tdtStartFrame + tdtEndFrame) / 2.0
                    let ctcCenterFrame = Double(ctcStartFrame + ctcEndFrame) / 2.0
                    if abs(ctcCenterFrame - tdtCenterFrame) * frameDuration > maxOffsetSeconds {
                        skippedFarPeak += 1
                        continue
                    }

                    samples.append(
                        OffsetSample(
                            fileId: fileId,
                            word: cleaned,
                            tokenCount: ctcTokens.count,
                            tdtStartFrame: tdtStartFrame,
                            tdtEndFrame: tdtEndFrame,
                            ctcStartFrame: ctcStartFrame,
                            ctcEndFrame: ctcEndFrame,
                            ctcScore: best.score,
                            frameDuration: frameDuration,
                            tdtAvgConfidence: wt.avgConfidence,
                            tdtMinConfidence: wt.minConfidence,
                            tdtDurationFrames: max(0, tdtEndFrame - tdtStartFrame)
                        ))
                    fileSamples += 1
                }

                let pad = fileId.padding(toLength: 25, withPad: " ", startingAt: 0)
                print("  [\(idx + 1)/\(fileIds.count)] \(pad) words: \(wordTimings.count), kept: \(fileSamples)")
            }

            // Summary.
            print()
            print(String(repeating: "=", count: 70))
            print("EMISSION DELAY MEASUREMENT")
            print(String(repeating: "=", count: 70))
            print("Total TDT words seen: \(totalWords)")
            print("Skipped (too short):  \(skippedTooShort)")
            print("Skipped (no tokens):  \(skippedNoTokens)")
            print("Skipped (low score):  \(skippedLowScore)")
            print("Skipped (far peak):   \(skippedFarPeak)")
            print("Kept samples:         \(samples.count)")
            print()

            guard !samples.isEmpty else {
                print("No samples retained — try loosening guards.")
                return
            }

            let frameDuration = samples[0].frameDuration
            let frameMs = frameDuration * 1000.0
            print("Frame duration: \(String(format: "%.4f", frameDuration))s (\(String(format: "%.2f", frameMs))ms)")
            print()

            printOffsetStats(
                label: "offsetStart  (frames)", values: samples.map { Double($0.offsetStart) }, frameMs: frameMs)
            printOffsetStats(
                label: "offsetEnd    (frames)", values: samples.map { Double($0.offsetEnd) }, frameMs: frameMs)
            printOffsetStats(label: "offsetCenter (frames)", values: samples.map { $0.offsetCenter }, frameMs: frameMs)
            print()

            // Recommendation: snap mean offsetStart to the nearest integer frame.
            let meanStart = mean(samples.map { Double($0.offsetStart) })
            let medianStart = percentile(samples.map { Double($0.offsetStart) }, p: 0.5)
            let snapped = Int(medianStart.rounded())
            let snappedMs = Double(snapped) * frameMs
            print(
                "Suggested correction: subtract \(snapped) frame\(snapped == 1 ? "" : "s") "
                    + "(~\(String(format: "%.0f", snappedMs))ms) from TDT word startTime")
            print(
                "  rationale: median offsetStart = \(String(format: "%.2f", medianStart)) frames, "
                    + "mean = \(String(format: "%.2f", meanStart)) frames")
            print(String(repeating: "=", count: 70))

            // Persist.
            let summaryDict: [String: Any] = [
                "totalWords": totalWords,
                "keptSamples": samples.count,
                "skippedTooShort": skippedTooShort,
                "skippedNoTokens": skippedNoTokens,
                "skippedLowScore": skippedLowScore,
                "skippedFarPeak": skippedFarPeak,
                "frameDurationSeconds": frameDuration,
                "offsetStartFrames": statsDict(samples.map { Double($0.offsetStart) }),
                "offsetEndFrames": statsDict(samples.map { Double($0.offsetEnd) }),
                "offsetCenterFrames": statsDict(samples.map { $0.offsetCenter }),
                "suggestedFrameCorrection": snapped,
                "suggestedMsCorrection": snappedMs,
            ]

            let perSampleJson: [[String: Any]] = samples.map { s in
                [
                    "file": s.fileId,
                    "word": s.word,
                    "tokens": s.tokenCount,
                    "tdtStartFrame": s.tdtStartFrame,
                    "tdtEndFrame": s.tdtEndFrame,
                    "ctcStartFrame": s.ctcStartFrame,
                    "ctcEndFrame": s.ctcEndFrame,
                    "ctcScore": Double(s.ctcScore),
                    "offsetStart": s.offsetStart,
                    "offsetEnd": s.offsetEnd,
                    "offsetCenter": s.offsetCenter,
                    "tdtAvgConfidence": Double(s.tdtAvgConfidence),
                    "tdtMinConfidence": Double(s.tdtMinConfidence),
                    "tdtDurationFrames": s.tdtDurationFrames,
                ]
            }

            let output: [String: Any] = [
                "config": [
                    "tdtVersion": tdtVersionLabel(tdtVersion),
                    "ctcVariant": ctcVariant.displayName,
                    "maxOffsetSeconds": maxOffsetSeconds,
                    "minCtcScore": Double(minCtcScore),
                    "minWordChars": minWordChars,
                ],
                "summary": summaryDict,
                "samples": perSampleJson,
            ]

            let jsonData = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: URL(fileURLWithPath: outputFile))
            print("Results written to: \(outputFile)")
        } catch {
            print("ERROR: \(error)")
        }
    }

    // MARK: - Helpers

    private static func tdtVersionLabel(_ v: AsrModelVersion) -> String {
        switch v {
        case .v2: return "v2"
        case .v3: return "v3"
        case .tdtCtc110m: return "tdt-ctc-110m"
        default: return "\(v)"
        }
    }

    private static func defaultCtcModelPath(for variant: CtcModelVariant) -> String? {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let modelPath = appSupport.appendingPathComponent("FluidAudio/Models/\(variant.repo.folderName)")
        return FileManager.default.fileExists(atPath: modelPath.path) ? modelPath.path : nil
    }

    private static func defaultDataDir() -> String? {
        let dataDir = DatasetDownloader.getEarnings22Directory().appendingPathComponent("test-dataset")
        return FileManager.default.fileExists(atPath: dataDir.path) ? dataDir.path : nil
    }

    private static func collectFileIds(from dataDir: URL, maxFiles: Int?) throws -> [String] {
        var fileIds: [String] = []
        let suffix = ".dictionary.txt"
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil)
        for url in contents.sorted(by: { $0.path < $1.path }) {
            let name = url.lastPathComponent
            if name.hasSuffix(suffix) {
                let data = try? Data(contentsOf: url)
                if let data = data, !data.isEmpty {
                    fileIds.append(String(name.dropLast(suffix.count)))
                }
            }
        }
        if let m = maxFiles { return Array(fileIds.prefix(m)) }
        return fileIds
    }

    private struct WordTiming {
        let word: String
        let startTime: Double
        let endTime: Double
        let avgConfidence: Float
        let minConfidence: Float
    }

    /// Mirrors `VocabularyRescorer.buildWordTimings` (internal to the
    /// FluidAudio module). Tokens prefixed with " " or "▁" begin new words.
    /// Also accumulates per-word confidence summaries.
    private static func buildWordTimings(from tokenTimings: [TokenTiming]) -> [WordTiming] {
        var out: [WordTiming] = []
        var currentWord = ""
        var wordStart: Double = 0
        var wordEnd: Double = 0
        var wordConfSum: Float = 0
        var wordConfMin: Float = .infinity
        var wordTokenCount = 0

        func flush() {
            let trimmed = currentWord.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, wordTokenCount > 0 else { return }
            let avg = wordConfSum / Float(wordTokenCount)
            out.append(
                WordTiming(
                    word: trimmed,
                    startTime: wordStart,
                    endTime: wordEnd,
                    avgConfidence: avg,
                    minConfidence: wordConfMin
                ))
        }

        for timing in tokenTimings {
            let token = timing.token
            if token.isEmpty || token == "<blank>" || token == "<pad>" { continue }

            let startsNewWord = isWordBoundary(token) || currentWord.isEmpty
            if startsNewWord && !currentWord.isEmpty {
                flush()
                currentWord = ""
                wordConfSum = 0
                wordConfMin = .infinity
                wordTokenCount = 0
            }
            if startsNewWord {
                currentWord = stripWordBoundaryPrefix(token)
                wordStart = timing.startTime
            } else {
                currentWord += token
            }
            wordEnd = timing.endTime
            wordConfSum += timing.confidence
            wordConfMin = min(wordConfMin, timing.confidence)
            wordTokenCount += 1
        }
        flush()
        return out
    }

    // MARK: - Statistics

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func stdev(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let m = mean(values)
        let sq = values.reduce(0.0) { $0 + ($1 - m) * ($1 - m) }
        return (sq / Double(values.count - 1)).squareRoot()
    }

    private static func percentile(_ values: [Double], p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let idx = max(0, min(sorted.count - 1, Int((Double(sorted.count - 1) * p).rounded())))
        return sorted[idx]
    }

    private static func printOffsetStats(label: String, values: [Double], frameMs: Double) {
        let m = mean(values)
        let sd = stdev(values)
        let p10 = percentile(values, p: 0.10)
        let p25 = percentile(values, p: 0.25)
        let p50 = percentile(values, p: 0.50)
        let p75 = percentile(values, p: 0.75)
        let p90 = percentile(values, p: 0.90)
        print("\(label):")
        print(
            "  mean=\(String(format: "%.2f", m)) (~\(String(format: "%.0f", m * frameMs))ms), "
                + "stdev=\(String(format: "%.2f", sd))")
        print(
            "  p10=\(String(format: "%.1f", p10)), p25=\(String(format: "%.1f", p25)), "
                + "p50=\(String(format: "%.1f", p50)), p75=\(String(format: "%.1f", p75)), "
                + "p90=\(String(format: "%.1f", p90))")
    }

    private static func statsDict(_ values: [Double]) -> [String: Double] {
        return [
            "mean": mean(values),
            "stdev": stdev(values),
            "p10": percentile(values, p: 0.10),
            "p25": percentile(values, p: 0.25),
            "p50": percentile(values, p: 0.50),
            "p75": percentile(values, p: 0.75),
            "p90": percentile(values, p: 0.90),
        ]
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio emission-delay-benchmark [options]

            Measures the systematic offset between TDT-emitted token timestamps and
            CTC argmax peaks for the same words. Used to calibrate a TDT timestamp
            correction so the rescorer's marginSeconds can be tightened.

            Options:
              --data-dir <path>          Earnings22 test-dataset directory
                                         (default: ~/Library/Application Support/.../earnings22-kws/test-dataset)
              --output, -o <file>        Output JSON file (default: emission_delay_benchmark.json)
              --max-files <N>            Process at most N files
              --ctc-model <path>         CTC model directory (default: parakeet-ctc-110m-coreml)
              --tdt-version <v>          v2 (default), v3, or 110m
              --max-offset-seconds <s>   Drop samples with |center delta| > s (default 1.5)
              --min-ctc-score <f>        Drop samples with CTC normalized score < f (default -3.0)
              --min-word-chars <N>       Skip TDT words shorter than N chars (default 4)
              --help, -h                 Show this help

            Example:
              fluidaudio emission-delay-benchmark --max-files 50
            """)
    }
}
#endif
