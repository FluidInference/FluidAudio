import FluidAudio
import FluidAudioTTS
import Foundation

public struct ChineseTTS {

    private static let logger = AppLogger(category: "ChineseTTSCommand")

    public static func run(arguments: [String]) async {
        var output = "output_zh.wav"
        var voice = "zf_001"  // Default Chinese female voice
        var text: String? = nil
        var modelPath: String? = nil
        var dataPath: String? = nil
        var rawPhonemes: String? = nil  // Direct phonemes, bypassing G2P

        var i = 0
        while i < arguments.count {
            let argument = arguments[i]
            switch argument {
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
            case "--model", "-m":
                if i + 1 < arguments.count {
                    modelPath = arguments[i + 1]
                    i += 1
                }
            case "--data", "-d":
                if i + 1 < arguments.count {
                    dataPath = arguments[i + 1]
                    i += 1
                }
            case "--phonemes", "-p":
                if i + 1 < arguments.count {
                    rawPhonemes = arguments[i + 1]
                    i += 1
                }
            default:
                if text == nil {
                    text = argument
                } else {
                    logger.warning("Ignoring unexpected argument '\(argument)'")
                }
            }
            i += 1
        }

        // Need either text or phonemes
        guard text != nil || rawPhonemes != nil else {
            printUsage()
            return
        }

        do {
            let tStart = Date()

            // Resolve paths
            let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)

            // Find model directory
            let modelDir: URL
            if let modelPath = modelPath {
                let expanded = (modelPath as NSString).expandingTildeInPath
                modelDir =
                    expanded.hasPrefix("/")
                    ? URL(fileURLWithPath: expanded, isDirectory: true)
                    : cwd.appendingPathComponent(expanded, isDirectory: true)
            } else {
                // Default: look for kokoro-82m-v1.1-zh-mlx in convert directory
                modelDir = cwd.appendingPathComponent(
                    "convert/kokoro-82m-v1.1-zh/kokoro-82m-v1.1-zh-mlx", isDirectory: true)
            }

            logger.info("Model directory: \(modelDir.path)")

            // Check model path exists
            guard FileManager.default.fileExists(atPath: modelDir.path) else {
                logger.error("Model directory not found: \(modelDir.path)")
                print("Model directory not found. Use --model to specify path.")
                exit(1)
            }

            // Initialize synthesizer
            logger.info("[1/3] Initializing Chinese TTS synthesizer...")
            let synthesizer = KokoroChineseSynthesizer()

            // Load G2P dictionaries only if using text input (not raw phonemes)
            if rawPhonemes == nil {
                // Find data directory (for G2P dictionaries)
                let dataDir: URL
                if let dataPath = dataPath {
                    let expanded = (dataPath as NSString).expandingTildeInPath
                    dataDir =
                        expanded.hasPrefix("/")
                        ? URL(fileURLWithPath: expanded, isDirectory: true)
                        : cwd.appendingPathComponent(expanded, isDirectory: true)
                } else {
                    // Default: look for swift_data in convert directory
                    dataDir = cwd.appendingPathComponent("convert/kokoro-82m-v1.1-zh/swift_data", isDirectory: true)
                }

                logger.info("Data directory: \(dataDir.path)")

                guard FileManager.default.fileExists(atPath: dataDir.path) else {
                    logger.error("Data directory not found: \(dataDir.path)")
                    print("Data directory not found. Use --data to specify path.")
                    exit(1)
                }

                logger.info("[2/3] Loading G2P dictionaries...")
                let jiebaURL = dataDir.appendingPathComponent("jieba.bin.gz")
                let pinyinSingleURL = dataDir.appendingPathComponent("pinyin_single.bin.gz")
                let pinyinPhrasesURL = dataDir.appendingPathComponent("pinyin_phrases.bin.gz")

                try synthesizer.loadG2P(
                    jiebaURL: jiebaURL,
                    pinyinSingleURL: pinyinSingleURL,
                    pinyinPhrasesURL: pinyinPhrasesURL
                )
                logger.info("  G2P loaded successfully")
            } else {
                logger.info("[2/3] Skipping G2P (using raw phonemes)")
            }

            // Load model
            logger.info("[3/3] Loading Kokoro model...")
            let weightsURL = modelDir.appendingPathComponent("model.safetensors")
            let configURL = modelDir.appendingPathComponent("config.json")

            try synthesizer.loadModel(weightsURL: weightsURL, configURL: configURL)
            logger.info("  Model loaded successfully")

            // Load voice
            let voiceURL = modelDir.appendingPathComponent("voices/\(voice).npy")
            guard FileManager.default.fileExists(atPath: voiceURL.path) else {
                logger.error("Voice not found: \(voiceURL.path)")
                print("Voice '\(voice)' not found. Check available voices in: \(modelDir.path)/voices/")
                exit(1)
            }

            try synthesizer.loadVoice(name: voice, from: voiceURL)
            logger.info("  Voice '\(voice)' loaded")

            let tLoad = Date()

            // Synthesize
            logger.info("[4/4] Synthesizing audio...")

            let result: SynthesisResult
            if let phonemes = rawPhonemes {
                logger.info("  Phonemes: \(phonemes)")
                result = try synthesizer.synthesizeFromPhonemes(phonemes, voice: voice, speed: 1.0)
            } else {
                logger.info("  Input: \(text!)")
                result = try synthesizer.synthesize(text!, voice: voice, speed: 1.0)
            }

            let tSynth = Date()

            logger.info("  Phonemes: \(result.phonemes)")
            logger.info("  Duration: \(String(format: "%.2f", result.duration))s")

            // Write WAV file
            let outURL: URL = {
                let expanded = (output as NSString).expandingTildeInPath
                if expanded.hasPrefix("/") {
                    return URL(fileURLWithPath: expanded)
                }
                return cwd.appendingPathComponent(expanded)
            }()

            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let wavData = try AudioWAV.data(
                from: result.audio,
                sampleRate: Double(result.sampleRate)
            )
            try wavData.write(to: outURL)

            let tEnd = Date()

            // Print summary
            let separator = String(repeating: "=", count: 60)
            let dashLine = String(repeating: "-", count: 60)
            print("")
            print(separator)
            print("Chinese TTS Synthesis Complete")
            print(separator)
            print("Input:      \(text ?? "(raw phonemes)")")
            print("Phonemes:   \(result.phonemes)")
            print("Voice:      \(voice)")
            print("Duration:   \(String(format: "%.2f", result.duration))s")
            print("Output:     \(outURL.path)")
            print(dashLine)
            print("Load time:  \(String(format: "%.2f", tLoad.timeIntervalSince(tStart)))s")
            print("Synth time: \(String(format: "%.2f", tSynth.timeIntervalSince(tLoad)))s")
            print("RTFx:       \(String(format: "%.1f", result.duration / tSynth.timeIntervalSince(tLoad)))x")
            print(separator)

        } catch {
            logger.error("Chinese TTS Error: \(error)")
            print("Chinese TTS failed: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts-zh "Chinese text" [options]

            Synthesize Chinese text to speech using Kokoro TTS with Bopomofo G2P.

            Options:
              --output, -o    Output WAV path (default: output_zh.wav)
              --voice, -v     Voice name (default: zf_001 - Chinese female)
              --model, -m     Path to model directory (default: convert/kokoro-82m-v1.1-zh/kokoro-82m-v1.1-zh-mlx)
              --data, -d      Path to G2P data directory (default: convert/kokoro-82m-v1.1-zh/swift_data)
              --phonemes, -p  Raw Bopomofo phonemes (bypasses G2P, no text required)
              --help, -h      Show this help

            Available voices (Chinese):
              zf_001 - zf_103   Chinese female voices
              zm_010 - zm_020   Chinese male voices

            Examples:
              fluidaudio tts-zh "你好世界"
              fluidaudio tts-zh "今天天气很好" --voice zm_010 --output weather.wav
              fluidaudio tts-zh --phonemes "ㄋㄧ3ㄏㄠ3" --output hello.wav
            """
        )
    }
}
