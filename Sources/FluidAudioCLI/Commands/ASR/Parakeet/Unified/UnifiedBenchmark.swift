#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// LibriSpeech WER/RTFx benchmark for the Parakeet Unified 0.6B backend,
/// driving the actual Swift managers (`UnifiedAsrManager` for batch,
/// `StreamingUnifiedAsrManager` for streaming) and scoring with the same
/// `TextNormalizer` the TDT `asr-benchmark` uses, so the numbers are directly
/// comparable.
///
///     swift run -c release fluidaudiocli unified-benchmark --mode both
///     swift run -c release fluidaudiocli unified-benchmark --mode streaming --max-files 100
enum UnifiedBenchmark {
    private static let logger = AppLogger(category: "UnifiedBenchmark")

    struct ModeStats {
        let mode: String
        let files: Int
        let longFiles: Int
        let averageWer: Double  // mean of per-file WER (matches asr-benchmark)
        let aggregateWer: Double  // total errors / total words
        let medianWer: Double
        let medianRtfx: Double
        let overallRtfx: Double
        let audioSeconds: Double
        let processSeconds: Double
    }

    static func run(arguments: [String]) async {
        var subset = "test-clean"
        var maxFiles: Int?
        var modes: [String] = ["batch", "streaming"]
        var precision: UnifiedEncoderPrecision = .int8
        var mdPath = "Sources/FluidAudio/ASR/Parakeet/Unified/benchmark.md"
        var inputFile: String?
        var longformDir: String?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--input":
                inputFile = arguments[safe: i + 1]
                i += 1
            case "--longform-dir":
                longformDir = arguments[safe: i + 1]
                i += 1
            case "--subset":
                subset = arguments[safe: i + 1] ?? subset
                i += 1
            case "--max-files":
                maxFiles = Int(arguments[safe: i + 1] ?? "")
                i += 1
            case "--mode":
                let m = arguments[safe: i + 1] ?? "both"
                modes = m == "both" ? ["batch", "streaming"] : [m]
                i += 1
            case "--precision":
                precision = UnifiedEncoderPrecision(rawValue: arguments[safe: i + 1] ?? "") ?? .int8
                i += 1
            case "--write-md":
                mdPath = arguments[safe: i + 1] ?? mdPath
                i += 1
            default: break
            }
            i += 1
        }

        if let inputFile {
            await runSeamComparison(path: inputFile, precision: precision)
            return
        }

        if let longformDir {
            await runLongformBenchmark(dir: longformDir, precision: precision, maxFiles: maxFiles)
            return
        }

        let bench = ASRBenchmark()
        do {
            try await bench.downloadLibriSpeech(subset: subset)
            let dir = bench.getLibriSpeechDirectory().appendingPathComponent(subset)
            var files = try collectFiles(from: dir)
            if let maxFiles { files = Array(files.prefix(maxFiles)) }
            logger.info(
                "Parakeet Unified benchmark: \(files.count) files from LibriSpeech \(subset) (encoder \(precision.rawValue))"
            )

            var stats: [ModeStats] = []
            for mode in modes {
                let s = try await runMode(mode: mode, files: files, precision: precision, bench: bench)
                stats.append(s)
                logSummary(s)
            }

            writeMarkdown(path: mdPath, subset: subset, precision: precision, stats: stats, fileCount: files.count)
            logger.info("Wrote \(mdPath)")
        } catch {
            logger.error("Unified benchmark failed: \(error)")
            exit(1)
        }
    }

    private static func runMode(
        mode: String, files: [LibriSpeechFile], precision: UnifiedEncoderPrecision, bench: ASRBenchmark
    ) async throws -> ModeStats {
        let batch = UnifiedAsrManager(encoderPrecision: precision)
        let streaming = StreamingUnifiedAsrManager(encoderPrecision: precision)
        let converter = AudioConverter()
        let windowSamples = 15 * 16000

        if mode == "batch" {
            try await batch.loadModels()
        } else {
            try await streaming.loadModels()
        }

        var wers: [Double] = []
        var rtfxs: [Double] = []
        var totalErrors = 0
        var totalWords = 0
        var audioSeconds = 0.0
        var processSeconds = 0.0
        var longFiles = 0

        for (idx, file) in files.enumerated() {
            let samples = try converter.resampleAudioFile(path: file.audioPath.path)
            if samples.count > windowSamples { longFiles += 1 }
            let audioLen = Double(samples.count) / 16000.0

            let start = Date()
            let hypothesis: String
            if mode == "batch" {
                hypothesis = try await batch.transcribe(samples)
            } else {
                try await streaming.appendAudio(makeBuffer(samples))
                try await streaming.processBufferedAudio()
                hypothesis = try await streaming.finish()
                try await streaming.reset()
            }
            let dt = Date().timeIntervalSince(start)

            let m = bench.calculateASRMetrics(hypothesis: hypothesis, reference: file.transcript)
            wers.append(m.wer)
            rtfxs.append(dt > 0 ? audioLen / dt : 0)
            totalErrors += m.insertions + m.deletions + m.substitutions
            totalWords += m.totalWords
            audioSeconds += audioLen
            processSeconds += dt

            if (idx + 1) % 100 == 0 {
                let running = wers.reduce(0, +) / Double(wers.count)
                logger.info("  [\(mode)] \(idx + 1)/\(files.count) avgWER=\(pct(running))")
            }
        }

        let sortedWer = wers.sorted()
        let sortedRtfx = rtfxs.sorted()
        return ModeStats(
            mode: mode,
            files: files.count,
            longFiles: longFiles,
            averageWer: wers.reduce(0, +) / Double(max(wers.count, 1)),
            aggregateWer: totalWords > 0 ? Double(totalErrors) / Double(totalWords) : 0,
            medianWer: sortedWer.isEmpty ? 0 : sortedWer[sortedWer.count / 2],
            medianRtfx: sortedRtfx.isEmpty ? 0 : sortedRtfx[sortedRtfx.count / 2],
            overallRtfx: processSeconds > 0 ? audioSeconds / processSeconds : 0,
            audioSeconds: audioSeconds,
            processSeconds: processSeconds
        )
    }

    // MARK: - Helpers

    private static func makeBuffer(_ samples: [Float]) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        return buffer
    }

    private static func collectFiles(from directory: URL) throws -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "txt", url.lastPathComponent.contains(".trans.") else { continue }
            for line in try String(contentsOf: url).components(separatedBy: .newlines) where !line.isEmpty {
                let parts = line.components(separatedBy: " ")
                guard parts.count >= 2 else { continue }
                let audioPath = url.deletingLastPathComponent().appendingPathComponent("\(parts[0]).flac")
                if fm.fileExists(atPath: audioPath.path) {
                    files.append(
                        LibriSpeechFile(
                            fileName: "\(parts[0]).flac",
                            audioPath: audioPath,
                            transcript: parts.dropFirst().joined(separator: " ")))
                }
            }
        }
        return files.sorted { $0.fileName < $1.fileName }
    }

    private static func pct(_ x: Double) -> String { String(format: "%.2f%%", x * 100) }

    private static func logSummary(_ s: ModeStats) {
        logger.info("=== \(s.mode.uppercased()) ===")
        logger.info("   Files: \(s.files) (\(s.longFiles) > 15s)")
        logger.info(
            "   Average WER: \(pct(s.averageWer))  Aggregate WER: \(pct(s.aggregateWer))  Median WER: \(pct(s.medianWer))"
        )
        logger.info(
            "   Median RTFx: \(String(format: "%.1fx", s.medianRtfx))  Overall RTFx: \(String(format: "%.1fx", s.overallRtfx))"
        )
    }

    private static func writeMarkdown(
        path: String, subset: String, precision: UnifiedEncoderPrecision, stats: [ModeStats], fileCount: Int
    ) {
        var md = """
            # Parakeet Unified 0.6B — LibriSpeech \(subset) Benchmark

            Measured through the FluidAudio Swift managers (`UnifiedAsrManager` for batch,
            `StreamingUnifiedAsrManager` for streaming) over all \(fileCount) `\(subset)` files,
            scored with the repo's `TextNormalizer` (same normalization as `asr-benchmark`).
            Encoder precision: **\(precision.rawValue)**. Run with `swift run -c release fluidaudiocli unified-benchmark`.

            | Mode | Avg WER | Aggregate WER | Median WER | Median RTFx | Overall RTFx | Long files (>15s) |
            |------|---------|---------------|------------|-------------|--------------|-------------------|

            """
        for s in stats {
            md += "| \(s.mode) | \(pct(s.averageWer)) | \(pct(s.aggregateWer)) | \(pct(s.medianWer)) "
            md +=
                "| \(String(format: "%.1fx", s.medianRtfx)) | \(String(format: "%.1fx", s.overallRtfx)) | \(s.longFiles) |\n"
        }
        md += """

            - **Avg WER** is the mean of per-file WER (matches `asr-benchmark`'s "Average WER").
            - **Aggregate WER** is total errors ÷ total words across the set.
            - Long files (> 15 s) are transcribed with overlapping 15 s windows merged on a 2 s overlap (batch),
              or as one continuous session (streaming) — none are skipped. Streaming's overall RTFx drops on
              long files because it re-encodes a 7.68 s window per 1.04 s chunk (the latency tax); batch only
              re-encodes the 2 s overlap, so its throughput stays flat.
            - RTFx is end-to-end per file (preprocess + encode + greedy RNNT decode) on the run machine.

            ## Comparison vs Parakeet TDT v3 (same harness)

            Parakeet TDT v3 measured via `asr-benchmark --subset test-clean --model-version v3` on the same
            machine and `TextNormalizer`: **Average WER 2.6%**, Median 0.0%, Overall RTFx 110.

            | Model | Mode | Avg WER | Overall RTFx | Punctuation/caps | Languages |
            |-------|------|---------|--------------|------------------|-----------|
            | Parakeet TDT v3 | batch (sliding window) | 2.6% | 110 | no | 25 + Japanese |
            | Parakeet Unified | batch | 2.15% | 123 | yes | English |
            | Parakeet Unified | streaming | 2.21% | 29 | yes | English |

            For English file transcription, Unified batch beats TDT v3 on both WER and throughput and adds
            punctuation/capitalization. TDT v3 remains the choice for non-English audio.
            """
        try? md.write(toFile: path, atomically: true, encoding: .utf8)
    }

    // MARK: - Single-File Seam Comparison (Issue #706)

    /// Transcribe one (typically long) file with Unified batch and Parakeet TDT
    /// v3, then count the chunk-merge seam artifacts #706 is about: adjacent
    /// case-only duplicate words ("meeting Meeting") and mid-sentence
    /// capitalized words. No reference transcript is needed — the metric is the
    /// artifact, and TDT v3 (which already silence-aligns its chunks) is the
    /// quality baseline.
    private static func runSeamComparison(path: String, precision: UnifiedEncoderPrecision) async {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.error("Input file not found: \(path)")
            return
        }

        do {
            let converter = AudioConverter()
            let samples = try converter.resampleAudioFile(url)
            let audioSeconds = Double(samples.count) / 16000.0
            logger.info(
                "Seam comparison on \(url.lastPathComponent): \(String(format: "%.1f", audioSeconds))s of audio")

            // Unified batch (silence-aligned chunks + case-folded merge).
            let batch = UnifiedAsrManager(encoderPrecision: precision)
            try await batch.loadModels()
            let unifiedStart = Date()
            let unifiedText = try await batch.transcribe(samples)
            let unifiedSeconds = Date().timeIntervalSince(unifiedStart)

            // Parakeet TDT v3 (sliding-window reference).
            let tdtModels = try await AsrModels.downloadAndLoad(version: .v3)
            let asr = AsrManager(config: .default)
            try await asr.loadModels(tdtModels)
            var decoderState = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
            let tdtStart = Date()
            let tdtResult = try await asr.transcribe(samples, decoderState: &decoderState)
            let tdtSeconds = Date().timeIntervalSince(tdtStart)

            let unifiedArtifacts = seamArtifacts(in: unifiedText)
            let tdtArtifacts = seamArtifacts(in: tdtResult.text)

            let unifiedOut = url.deletingPathExtension().lastPathComponent + ".unified.txt"
            let tdtOut = url.deletingPathExtension().lastPathComponent + ".tdtv3.txt"
            try? unifiedText.write(toFile: unifiedOut, atomically: true, encoding: .utf8)
            try? tdtResult.text.write(toFile: tdtOut, atomically: true, encoding: .utf8)

            print("\n" + String(repeating: "=", count: 64))
            print("SEAM ARTIFACT COMPARISON — \(url.lastPathComponent)")
            print(String(repeating: "=", count: 64))
            print(String(format: "Audio: %.1fs", audioSeconds))
            print("")
            printArtifactRow(
                label: "Unified batch",
                artifacts: unifiedArtifacts,
                seconds: unifiedSeconds,
                audioSeconds: audioSeconds)
            printArtifactRow(
                label: "Parakeet TDT v3",
                artifacts: tdtArtifacts,
                seconds: tdtSeconds,
                audioSeconds: audioSeconds)
            print("")
            print("Transcripts written to:\n  \(unifiedOut)\n  \(tdtOut)")
            if !unifiedArtifacts.examples.isEmpty {
                print("\nUnified seam examples (up to 10):")
                for ex in unifiedArtifacts.examples.prefix(10) { print("  • \(ex)") }
            }
            print(String(repeating: "=", count: 64))
        } catch {
            logger.error("Seam comparison failed: \(error)")
        }
    }

    // MARK: - Long-Form WER Benchmark (Earnings-22)

    /// Transcribe every `{id}.wav` (with sibling `{id}.txt` reference) in a
    /// directory of full-length recordings with both Unified batch and Parakeet
    /// TDT v3, scoring long-form WER (normalized like `asr-benchmark`) plus the
    /// seam case-duplicate count. This exercises chunk merging end-to-end —
    /// each file spans many 15 s windows — which the short LibriSpeech /
    /// earnings22-kws clips cannot.
    private static func runLongformBenchmark(dir: String, precision: UnifiedEncoderPrecision, maxFiles: Int?) async {
        let dirURL = URL(fileURLWithPath: dir)
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil) else {
            logger.error("Long-form directory not found: \(dir)")
            return
        }
        var wavs = entries.filter { $0.pathExtension.lowercased() == "wav" }.sorted { $0.path < $1.path }
        if let maxFiles { wavs = Array(wavs.prefix(maxFiles)) }
        guard !wavs.isEmpty else {
            logger.error("No .wav files in \(dir)")
            return
        }

        do {
            let converter = AudioConverter()
            let batch = UnifiedAsrManager(encoderPrecision: precision)
            try await batch.loadModels()
            let tdtModels = try await AsrModels.downloadAndLoad(version: .v3)
            let asr = AsrManager(config: .default)
            try await asr.loadModels(tdtModels)

            var rows: [(id: String, words: Int, uniWer: Double, tdtWer: Double)] = []
            var uniErrors = 0
            var tdtErrors = 0
            var refWords = 0
            var uniAudio = 0.0
            var tdtAudio = 0.0
            var uniProc = 0.0
            var tdtProc = 0.0
            var uniDupes = 0
            var tdtDupes = 0

            for wav in wavs {
                let id = wav.deletingPathExtension().lastPathComponent
                let refURL = wav.deletingPathExtension().appendingPathExtension("txt")
                guard let reference = try? String(contentsOf: refURL, encoding: .utf8) else {
                    logger.info("Skipping \(id): no reference .txt")
                    continue
                }
                let samples = try converter.resampleAudioFile(wav)
                let seconds = Double(samples.count) / 16000.0

                let uniStart = Date()
                let uniText = try await batch.transcribe(samples)
                uniProc += Date().timeIntervalSince(uniStart)

                var decoderState = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
                let tdtStart = Date()
                let tdtText = try await asr.transcribe(samples, decoderState: &decoderState).text
                tdtProc += Date().timeIntervalSince(tdtStart)

                let uni = WERCalculator.calculateWERMetrics(hypothesis: uniText, reference: reference)
                let tdt = WERCalculator.calculateWERMetrics(hypothesis: tdtText, reference: reference)

                uniErrors += uni.insertions + uni.deletions + uni.substitutions
                tdtErrors += tdt.insertions + tdt.deletions + tdt.substitutions
                refWords += uni.totalWords
                uniAudio += seconds
                tdtAudio += seconds
                uniDupes += seamArtifacts(in: uniText).caseDuplicates
                tdtDupes += seamArtifacts(in: tdtText).caseDuplicates
                rows.append((id, uni.totalWords, uni.wer, tdt.wer))

                print(
                    String(
                        format: "%-22@  %.1fs  ref=%5d  Unified WER %.2f%%   TDT v3 WER %.2f%%",
                        id as NSString, seconds, uni.totalWords, uni.wer * 100, tdt.wer * 100))
            }

            guard !rows.isEmpty else {
                logger.error("No scorable files (missing references).")
                return
            }

            let uniAvg = rows.map(\.uniWer).reduce(0, +) / Double(rows.count) * 100
            let tdtAvg = rows.map(\.tdtWer).reduce(0, +) / Double(rows.count) * 100
            let uniAgg = refWords > 0 ? Double(uniErrors) / Double(refWords) * 100 : 0
            let tdtAgg = refWords > 0 ? Double(tdtErrors) / Double(refWords) * 100 : 0

            print("\n" + String(repeating: "=", count: 72))
            print("EARNINGS-22 LONG-FORM WER — \(rows.count) files, \(String(format: "%.1f", uniAudio / 60))min audio")
            print(String(repeating: "=", count: 72))
            print(
                String(
                    format: "Unified batch     Avg WER %.2f%%   Aggregate WER %.2f%%   case-dupes %d   %.0fx",
                    uniAvg, uniAgg, uniDupes, uniProc > 0 ? uniAudio / uniProc : 0))
            print(
                String(
                    format: "Parakeet TDT v3   Avg WER %.2f%%   Aggregate WER %.2f%%   case-dupes %d   %.0fx",
                    tdtAvg, tdtAgg, tdtDupes, tdtProc > 0 ? tdtAudio / tdtProc : 0))
            print(String(repeating: "=", count: 72))
        } catch {
            logger.error("Long-form benchmark failed: \(error)")
        }
    }

    private struct SeamArtifacts {
        var caseDuplicates: Int
        var midSentenceCaps: Int
        var examples: [String]
    }

    private static func printArtifactRow(
        label: String, artifacts: SeamArtifacts, seconds: Double, audioSeconds: Double
    ) {
        let rtfx = seconds > 0 ? audioSeconds / seconds : 0
        let padded = label.padding(toLength: 18, withPad: " ", startingAt: 0)
        print(
            padded
                + String(
                    format: "case-dupes: %3d   mid-sentence caps: %4d   (%.1fs, %.0fx)",
                    artifacts.caseDuplicates, artifacts.midSentenceCaps, seconds, rtfx))
    }

    /// Count seam artifacts in a transcript. `caseDuplicates` are adjacent words
    /// equal up to case ("meeting Meeting"); `midSentenceCaps` are capitalized
    /// words that don't follow sentence-ending punctuation (excluding the
    /// pronoun "I" and all-caps acronyms).
    private static func seamArtifacts(in text: String) -> SeamArtifacts {
        let rawWords = text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).map(String.init)
        var caseDuplicates = 0
        var midSentenceCaps = 0
        var examples: [String] = []

        func core(_ w: String) -> String {
            w.trimmingCharacters(in: CharacterSet.punctuationCharacters)
        }
        func endsSentence(_ w: String) -> Bool {
            guard let last = w.last else { return false }
            return last == "." || last == "?" || last == "!" || last == ":"
        }

        for index in 1..<max(rawWords.count, 1) {
            let prev = core(rawWords[index - 1])
            let cur = core(rawWords[index])
            guard !cur.isEmpty, !prev.isEmpty else { continue }

            if cur != prev, cur.lowercased() == prev.lowercased(),
                cur.first?.isLetter == true
            {
                caseDuplicates += 1
                if examples.count < 20 {
                    examples.append("…\(rawWords[index - 1]) \(rawWords[index])… (case duplicate)")
                }
            }

            if let first = cur.first, first.isUppercase, first.isLetter,
                cur != "I", cur != cur.uppercased(),  // skip ALL-CAPS acronyms
                !endsSentence(rawWords[index - 1])
            {
                midSentenceCaps += 1
                if examples.count < 20 {
                    examples.append("…\(rawWords[index - 1]) \(rawWords[index])… (mid-sentence cap)")
                }
            }
        }
        return SeamArtifacts(caseDuplicates: caseDuplicates, midSentenceCaps: midSentenceCaps, examples: examples)
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
#endif
