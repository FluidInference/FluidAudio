#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// `nemotron35-transcribe <audio> [--no-speakers] [--out t.txt] [--json t.json] [--model-dir dir]
///  [--compute-units gpu|ane|all] [--diarizer-units ane|gpu|all]`
///
/// Nemotron 3.5 Transcribe Lite: words with 40 ms start times from the ASR encoder + CTC head, and
/// (by default) speakers from Nemotron-3-Diarization running concurrently on the Neural Engine.
enum Nemotron35TranscribeCommand {
    private static let logger = AppLogger(category: "Nemotron35Transcribe")

    static func run(arguments: [String]) async {
        var audioPath: String?
        var speakers = true
        var outPath: String?
        var jsonPath: String?
        var modelDir: String?
        var computeUnits: MLComputeUnits = .cpuAndGPU
        var diarizerUnits: MLComputeUnits = .cpuAndNeuralEngine

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            func next() -> String? {
                i += 1
                return i < arguments.count ? arguments[i] : nil
            }
            switch arg {
            case "--no-speakers": speakers = false
            case "--out": outPath = next()
            case "--json": jsonPath = next()
            case "--model-dir": modelDir = next()
            case "--compute-units": computeUnits = parseUnits(next(), default: .cpuAndGPU)
            case "--diarizer-units": diarizerUnits = parseUnits(next(), default: .cpuAndNeuralEngine)
            case "--help", "-h":
                printUsage()
                return
            default:
                if audioPath == nil { audioPath = arg }
            }
            i += 1
        }

        guard let audioPath else {
            printUsage()
            return
        }
        guard FileManager.default.fileExists(atPath: audioPath) else {
            logger.error("Audio file not found: \(audioPath)")
            exit(1)
        }

        do {
            let audio = try AudioConverter().resampleAudioFile(path: audioPath)
            let duration = Double(audio.count) / Double(NemotronTranscribeLiteConfig.sampleRate)

            print("Loading Nemotron 3.5 Transcribe Lite (not part of the timed run) ...")
            let models: NemotronTranscribeLiteModels
            if let modelDir {
                models = try NemotronTranscribeLiteModels.load(
                    from: URL(fileURLWithPath: modelDir), computeUnits: computeUnits)
            } else {
                models = try await NemotronTranscribeLiteModels.downloadAndLoad(computeUnits: computeUnits)
            }
            let manager = NemotronTranscribeLiteManager(models: models)
            try await manager.warmup()

            let result: NemotronTranscribeLiteResult
            if speakers {
                // Load (and on first use compile) the diarizer on a 10 s clip before timing.
                let warm = Array(audio.prefix(10 * NemotronTranscribeLiteConfig.sampleRate))
                _ = try await manager.transcribeWithSpeakers(audio: warm, diarizerComputeUnits: diarizerUnits)
                result = try await manager.transcribeWithSpeakers(audio: audio, diarizerComputeUnits: diarizerUnits)
            } else {
                result = try await manager.transcribe(audio: audio)
            }

            printTranscript(result)
            let t = result.timings
            let nSpeakers = Set(result.words.compactMap(\.speaker)).count
            print(
                "\n\(String(format: "%.1f", duration / 60)) min of audio -> \(result.words.count) words"
                    + (speakers ? ", \(nSpeakers) speakers" : "")
                    + " in \(String(format: "%.2f", t.totalSeconds))s (\(String(format: "%.0f", result.rtfx))x realtime)"
            )
            var detail =
                "mel + encoder \(String(format: "%.2f", t.encoderSeconds))s, CTC head \(String(format: "%.2f", t.ctcSeconds))s"
            if let d = t.diarizerSeconds { detail += ", diarizer \(String(format: "%.2f", d))s (in parallel)" }
            print(detail)

            if let outPath { try taggedText(result).write(toFile: outPath, atomically: true, encoding: .utf8) }
            if let jsonPath { try json(result, audioPath: audioPath).write(to: URL(fileURLWithPath: jsonPath)) }
        } catch {
            logger.error("Transcription failed: \(error)")
            exit(1)
        }
    }

    /// `[mm:ss] Speaker N: words`, speakers numbered by first appearance.
    private static func printTranscript(_ result: NemotronTranscribeLiteResult) {
        var display: [Int: Int] = [:]
        for turn in result.turns {
            let stamp = String(format: "[%02d:%02d]", Int(turn.start) / 60, Int(turn.start) % 60)
            if let s = turn.speaker {
                let n = display[s] ?? display.count + 1
                display[s] = n
                print("\(stamp) Speaker \(n): \(turn.text)")
            } else {
                print("\(stamp) \(turn.text)")
            }
        }
    }

    /// Plain transcript with a `<spk:N>` tag (raw diarizer slot) at every speaker change.
    private static func taggedText(_ result: NemotronTranscribeLiteResult) -> String {
        result.turns.map { turn in
            (turn.speaker.map { "<spk:\($0)> " } ?? "") + turn.text
        }.joined(separator: " ")
    }

    private static func json(_ result: NemotronTranscribeLiteResult, audioPath: String) throws -> Data {
        let t = result.timings
        var timings: [String: Double] = [
            "encoder": t.encoderSeconds, "ctc": t.ctcSeconds, "total": t.totalSeconds,
        ]
        if let d = t.diarizerSeconds { timings["diarizer"] = d }
        let words: [[String: Any]] = result.words.map { w in
            var d: [String: Any] = ["t": (w.start * 100).rounded() / 100, "w": w.text]
            if let s = w.speaker { d["spk"] = s }
            return d
        }
        let object: [String: Any] = [
            "audio": audioPath, "duration": result.durationSeconds, "timings": timings, "words": words,
        ]
        return try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    }

    private static func parseUnits(_ s: String?, default fallback: MLComputeUnits) -> MLComputeUnits {
        switch s {
        case "gpu": return .cpuAndGPU
        case "ane": return .cpuAndNeuralEngine
        case "cpu": return .cpuOnly
        case "all": return .all
        default: return fallback
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudiocli nemotron35-transcribe <audio> [options]

            Nemotron 3.5 Transcribe Lite: NVIDIA Nemotron 3.5 Transcribe's ASR encoder + CTC head (no LLM),
            words with 40 ms start times, plus speakers from Nemotron-3-Diarization run in parallel.

            Options:
              --no-speakers           Words only (skip the diarizer)
              --out <file>            Write the transcript with <spk:N> tags at speaker changes
              --json <file>           Write words ({t, spk, w}) and timings as JSON
              --model-dir <dir>       Load from a local copy of the HF repo instead of downloading
              --compute-units <u>     Encoder + CTC head: gpu (default), ane, cpu, all
              --diarizer-units <u>    Diarizer: ane (default), gpu, cpu, all
            """)
    }
}
#endif
