import FluidAudio
import Foundation

@available(macOS 13.0, *)
public struct TTS {

    public static func run(arguments: [String]) async {
        // Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart] [--metrics metrics.json] [--chunk-dir dir]
        guard !arguments.isEmpty else {
            printUsage()
            return
        }

        let text = arguments[0]
        var output = "output.wav"
        var voice = "af_heart"
        var metricsPath: String? = nil
        var chunkDirectory: String? = nil
        // Always ensure required files in CLI

        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                return
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--voice", "-v":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            case "--metrics":
                if i + 1 < arguments.count {
                    metricsPath = arguments[i + 1]
                    i += 1
                }
            case "--chunk-dir":
                if i + 1 < arguments.count {
                    chunkDirectory = arguments[i + 1]
                    i += 1
                }
            case "--auto-download":
                // No-op: downloads are always ensured by the CLI
                ()
            default:
                // Unknown flag; ignore for forward-compat
                ()
            }
            i += 1
        }

        do {
            // Timing buckets
            let tStart = Date()

            // 1) Ensure required resources (dictionary)
            try await KokoroModel.ensureRequiredFiles()

            // 2) Ensure voice embedding in cache
            try? await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: voice)

            // 3) Load/compile model if needed
            let tLoad0 = Date()
            try await KokoroModel.loadModel()
            let tLoad1 = Date()

            // 4) Synthesize (includes chunking + inference + stitching + WAV pack)
            let tSynth0 = Date()
            let detailed = try await KokoroModel.synthesizeDetailed(text: text, voice: voice)
            let wav = detailed.audio
            let tSynth1 = Date()

            // Write WAV
            let outURL = URL(fileURLWithPath: output)
            try wav.write(to: outURL)
            print("Saved: \(outURL.path)")

            var chunkFileMap: [Int: String] = [:]
            if let chunkDirectory = chunkDirectory {
                let dirURL = URL(fileURLWithPath: chunkDirectory, isDirectory: true)
                try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
                for chunk in detailed.chunks {
                    let fileName = String(format: "chunk_%03d.wav", chunk.index)
                    let fileURL = dirURL.appendingPathComponent(fileName)
                    let chunkData = try AudioWAV.data(from: chunk.samples, sampleRate: 24_000)
                    try chunkData.write(to: fileURL)
                    chunkFileMap[chunk.index] = fileURL.path
                }
                print("Saved \(chunkFileMap.count) chunk WAV files to \(dirURL.path)")
            }

            // Metrics
            if let metricsPath = metricsPath {
                let loadS = tLoad1.timeIntervalSince(tLoad0)
                let synthS = tSynth1.timeIntervalSince(tSynth0)
                let totalS = tSynth1.timeIntervalSince(tStart)

                // Approx audio seconds from WAV header (24 kHz mono)
                let audioSecs: Double = {
                    // 44-byte header typical, but use Data length minus header if possible.
                    // We store raw bytes; Safe estimate: payload / (24000 * 2)
                    let bytes = wav.count
                    let payload = max(0, bytes - 44)
                    return Double(payload) / Double(24000 * 2)
                }()
                let rtf = audioSecs > 0 ? (synthS / audioSecs) : 0
                let realtimeSpeed = rtf > 0 ? (1.0 / rtf) : 0

                // Run ASR on the generated audio for comparison
                var asrHypothesis: String? = nil
                var werValue: Double? = nil

                print("\n--- Running ASR for TTS evaluation ---")
                do {
                    // Load ASR models and initialize
                    let models = try await AsrModels.downloadAndLoad()
                    let asr = AsrManager()
                    try await asr.initialize(models: models)

                    // Transcribe the generated audio file
                    let transcription = try await asr.transcribe(outURL)
                    asrHypothesis = transcription.text

                    // Calculate WER
                    werValue = calculateWER(reference: text, hypothesis: transcription.text)

                    print("Reference: \(text)")
                    print("ASR Output: \(transcription.text)")
                    print(String(format: "WER: %.1f%%", werValue! * 100))

                    // Clean up ASR resources
                    asr.cleanup()
                } catch {
                    print("ASR evaluation failed: \(error.localizedDescription)")
                }

                var metricsDict: [String: Any] = [
                    "inference_time_s": synthS,
                    "realtime_speed": realtimeSpeed,
                    "audio_duration_s": audioSecs,
                    "model_load_time_s": loadS,
                    "total_time_s": totalS,
                ]

                // Add ASR comparison if available
                if let asrHypothesis = asrHypothesis {
                    metricsDict["asr_hypothesis"] = asrHypothesis
                    if let werValue = werValue {
                        metricsDict["wer"] = werValue
                    }
                }

                if !detailed.chunks.isEmpty {
                    let chunkMetrics = detailed.chunks.map { chunk -> [String: Any] in
                        var entry: [String: Any] = [
                            "index": chunk.index,
                            "text": chunk.text,
                            "pause_after_ms": chunk.pauseAfterMs,
                            "tokens": chunk.tokenCount,
                        ]
                        entry["word_count"] = chunk.wordCount
                        if !chunk.words.isEmpty {
                            entry["normalized_words"] = chunk.words
                        }
                        if let path = chunkFileMap[chunk.index] {
                            entry["audio_file"] = path
                        }
                        return entry
                    }
                    metricsDict["chunks"] = chunkMetrics
                }

                let dict: [String: Any] = [
                    "text": text,
                    "voice": voice,
                    "output": outURL.path,
                    "metrics": metricsDict,
                ]

                // Write JSON
                let json = try JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted])
                let mURL = URL(fileURLWithPath: metricsPath)
                try json.write(to: mURL)
                print("\nMetrics saved: \(mURL.path)")
            }
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart] [--metrics metrics.json]

            Options:
              --output, -o         Output WAV path (default: output.wav)
              --voice, -v          Voice name (default: af_heart)
              --metrics            Write timing metrics to a JSON file (also runs ASR for evaluation)
              --chunk-dir          Directory where individual chunk WAVs will be written
              (models/dictionary auto-download is always on in CLI)
              --help, -h           Show this help
            """
        )
    }

    private static func calculateWER(reference: String, hypothesis: String) -> Double {
        // Normalize text for comparison
        func normalize(_ text: String) -> String {
            return
                text
                .lowercased()
                .replacingOccurrences(of: "\u{2019}", with: "'")  // smart quotes
                .replacingOccurrences(of: "\u{2018}", with: "'")
                .replacingOccurrences(of: "\u{201c}", with: "\"")
                .replacingOccurrences(of: "\u{201d}", with: "\"")
                .replacingOccurrences(of: "\u{2014}", with: "-")  // em dash
                .replacingOccurrences(of: "\u{2026}", with: "...")  // ellipsis
        }

        // Tokenize into words (alphanumeric + apostrophe only)
        func tokenize(_ text: String) -> [String] {
            var tokens: [String] = []
            var currentToken = ""

            for char in normalize(text) {
                if char.isLetter || char.isNumber || char == "'" {
                    currentToken.append(char)
                } else {
                    if !currentToken.isEmpty {
                        tokens.append(currentToken)
                        currentToken = ""
                    }
                }
            }
            if !currentToken.isEmpty {
                tokens.append(currentToken)
            }
            return tokens
        }

        let refTokens = tokenize(reference)
        let hypTokens = tokenize(hypothesis)

        // Handle empty cases
        if refTokens.isEmpty {
            return hypTokens.isEmpty ? 0.0 : 1.0
        }

        // Calculate Levenshtein distance using dynamic programming
        let n = refTokens.count
        let m = hypTokens.count
        var dp = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1)

        // Initialize base cases
        for i in 0...n {
            dp[i][0] = i
        }
        for j in 0...m {
            dp[0][j] = j
        }

        // Fill the DP table
        for i in 1...n {
            for j in 1...m {
                let cost = refTokens[i - 1] == hypTokens[j - 1] ? 0 : 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  // deletion
                    dp[i][j - 1] + 1,  // insertion
                    dp[i - 1][j - 1] + cost  // substitution
                )
            }
        }

        // WER = edits / reference_length
        return Double(dp[n][m]) / Double(n)
    }
}
