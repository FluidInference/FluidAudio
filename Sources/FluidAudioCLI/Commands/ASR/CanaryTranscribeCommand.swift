#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `canary-transcribe <audio> [--fp16|--int8] [--reference "..."] [--verbose]`
/// `canary-transcribe --benchmark <librispeech-dir> [--max-files N] [--fp16|--int8]`
///
/// Canary-1B-v2 attention encoder-decoder ASR. Default precision int4 (ANE).
enum CanaryTranscribeCommand {
    private static let logger = AppLogger(category: "CanaryTranscribe")

    static func run(arguments: [String]) async {
        var audioPath: String?
        var benchmarkDir: String?
        var reference: String?
        var precision: CanaryPrecision = .int4
        var maxFiles = Int.max
        var maxDuration = Double.greatestFiniteMagnitude
        var verbose = false

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--fp16": precision = .fp16
            case "--int8": precision = .int8
            case "--int4": precision = .int4
            case "--reference":
                i += 1
                if i < arguments.count { reference = arguments[i] }
            case "--benchmark":
                i += 1
                if i < arguments.count { benchmarkDir = arguments[i] }
            case "--max-files":
                i += 1
                if i < arguments.count { maxFiles = Int(arguments[i]) ?? .max }
            case "--max-duration":
                i += 1
                if i < arguments.count { maxDuration = Double(arguments[i]) ?? .greatestFiniteMagnitude }
            case "--verbose", "-v": verbose = true
            case "--help", "-h":
                printUsage()
                return
            default: if audioPath == nil { audioPath = arguments[i] }
            }
            i += 1
        }

        do {
            logger.info("Loading Canary models (\(precision.rawValue))...")
            let loadStart = Date()
            let manager = try await CanaryManager.load(precision: precision)
            if verbose { logger.info("Loaded in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s") }

            if let dir = benchmarkDir {
                await runBenchmark(
                    manager: manager, dir: dir, maxFiles: maxFiles, maxDuration: maxDuration, precision: precision)
                return
            }

            guard let audioPath else {
                logger.error("Error: No audio file specified")
                printUsage()
                return
            }
            let audioURL = URL(fileURLWithPath: audioPath)
            guard FileManager.default.fileExists(atPath: audioURL.path) else {
                logger.error("Error: Audio file not found: \(audioPath)")
                return
            }

            let duration = audioDuration(audioURL)
            let start = Date()
            let text = try await manager.transcribe(audioURL: audioURL)
            let elapsed = Date().timeIntervalSince(start)
            let rtfx = duration > 0 ? duration / elapsed : 0

            print(text)
            logger.info(
                "Time \(String(format: "%.2f", elapsed))s | audio \(String(format: "%.2f", duration))s | RTFx \(String(format: "%.2f", rtfx))x"
            )
            if let reference {
                let m = WERCalculator.calculateWERMetrics(hypothesis: text, reference: reference)
                logger.info(
                    "WER \(String(format: "%.2f", m.wer * 100))% (\(m.substitutions)S \(m.deletions)D \(m.insertions)I / \(m.totalWords)w)"
                )
            }
        } catch {
            logger.error("Transcription failed: \(error)")
        }
    }

    // MARK: - Benchmark (LibriSpeech-style directory)

    private static func runBenchmark(
        manager: CanaryManager, dir: String, maxFiles: Int, maxDuration: Double, precision: CanaryPrecision
    ) async {
        let root = URL(fileURLWithPath: dir)
        let refs = loadLibriSpeechReferences(root: root)
        var audios = findAudio(root: root).sorted { $0.lastPathComponent < $1.lastPathComponent }
        var skippedLong = 0
        if maxDuration < .greatestFiniteMagnitude {
            let before = audios.count
            audios = audios.filter { audioDuration($0) <= maxDuration }
            skippedLong = before - audios.count
        }
        guard !audios.isEmpty else {
            logger.error("No .wav/.flac files found under \(dir)")
            return
        }
        let files = Array(audios.prefix(maxFiles))
        logger.info(
            "Benchmarking Canary (\(precision.rawValue)) on \(files.count) files (refs: \(refs.count), skipped \(skippedLong) > \(maxDuration)s)..."
        )

        var totalAudio = 0.0
        var totalCompute = 0.0
        var totalWords = 0
        var totalErrors = 0
        var scored = 0

        for (idx, url) in files.enumerated() {
            do {
                let duration = audioDuration(url)
                let start = Date()
                let hyp = try await manager.transcribe(audioURL: url)
                let elapsed = Date().timeIntervalSince(start)
                totalAudio += duration
                totalCompute += elapsed

                let key = url.deletingPathExtension().lastPathComponent
                if let ref = refs[key] {
                    let m = WERCalculator.calculateWERMetrics(hypothesis: hyp, reference: ref)
                    totalErrors += Int((m.wer * Double(m.totalWords)).rounded())
                    totalWords += m.totalWords
                    scored += 1
                }
                if (idx + 1) % 10 == 0 || idx + 1 == files.count {
                    logger.info("  \(idx + 1)/\(files.count) done")
                }
            } catch {
                logger.error("  failed \(url.lastPathComponent): \(error)")
            }
        }

        let rtfx = totalCompute > 0 ? totalAudio / totalCompute : 0
        let wer = totalWords > 0 ? Double(totalErrors) / Double(totalWords) * 100 : -1
        print("")
        print("===== Canary \(precision.rawValue) benchmark =====")
        print("Files          : \(files.count) (\(scored) scored)")
        print("Audio total    : \(String(format: "%.1f", totalAudio))s")
        print("Compute total  : \(String(format: "%.1f", totalCompute))s")
        print("RTFx           : \(String(format: "%.2f", rtfx))x")
        if wer >= 0 { print("WER            : \(String(format: "%.2f", wer))% (\(totalWords) words)") }
        print("================================")
    }

    private static func findAudio(root: URL) -> [URL] {
        guard let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else { return [] }
        var out: [URL] = []
        for case let u as URL in en where ["wav", "flac"].contains(u.pathExtension.lowercased()) {
            out.append(u)
        }
        return out
    }

    /// LibriSpeech `*.trans.txt`: each line is `<utt-id> TRANSCRIPT`.
    private static func loadLibriSpeechReferences(root: URL) -> [String: String] {
        var map: [String: String] = [:]
        guard let en = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else { return map }
        for case let u as URL in en where u.lastPathComponent.hasSuffix(".trans.txt") {
            guard let content = try? String(contentsOf: u, encoding: .utf8) else { continue }
            for line in content.split(separator: "\n") {
                let parts = line.split(separator: " ", maxSplits: 1)
                if parts.count == 2 { map[String(parts[0])] = String(parts[1]) }
            }
        }
        return map
    }

    private static func audioDuration(_ url: URL) -> Double {
        let asset = AVURLAsset(url: url)
        return CMTimeGetSeconds(asset.duration)
    }

    private static func printUsage() {
        print(
            """
            Usage:
              fluidaudio canary-transcribe <audio-file> [options]
              fluidaudio canary-transcribe --benchmark <librispeech-dir> [options]

            Canary-1B-v2 attention encoder-decoder ASR (25 European languages, 15 s window).

            Options:
              --int4         int4 encoder/decoder (ANE, ~573 MB, iOS18) — default
              --fp16         fp16 encoder/decoder (ANE, exact parity, iOS17)
              --int8         int8 encoder/decoder (CPU only)
              --reference T  reference transcript for single-file WER
              --benchmark D  run over a LibriSpeech-style dir (uses *.trans.txt refs)
              --max-files N  limit benchmark file count
              --verbose,-v   print load + per-file timing
              --help,-h      show this help
            """
        )
    }
}
#endif
