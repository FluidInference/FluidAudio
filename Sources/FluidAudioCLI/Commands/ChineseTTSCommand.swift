import FluidAudio
import FluidAudioTTS
import Foundation

public struct ChineseTTS {

    private static let logger = AppLogger(category: "ChineseTTSCommand")

    /// HuggingFace repo for Chinese TTS model
    private static let hfRepo = "FluidInference/kokoro-82m-v1.1-zh-mlx"

    /// Required files for Chinese TTS
    private static let requiredFiles = [
        "model.safetensors",
        "config.json",
        "g2p/jieba.bin.gz",
        "g2p/pinyin_single.bin.gz",
        "g2p/pinyin_phrases.bin.gz",
    ]

    public static func run(arguments: [String]) async {
        var output = "output_zh.wav"
        var voice = "zf_001"  // Default Chinese female voice
        var text: String? = nil
        var modelPath: String? = nil
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

            // Resolve model directory
            let modelDir: URL
            if let modelPath = modelPath {
                let expanded = (modelPath as NSString).expandingTildeInPath
                modelDir =
                    expanded.hasPrefix("/")
                    ? URL(fileURLWithPath: expanded, isDirectory: true)
                    : URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                        .appendingPathComponent(expanded, isDirectory: true)
            } else {
                // Default: use cache directory
                let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                    .appendingPathComponent("FluidAudio/kokoro-zh", isDirectory: true)
                modelDir = cacheDir
            }

            // Check if model needs downloading
            let needsDownload = !FileManager.default.fileExists(
                atPath: modelDir.appendingPathComponent("model.safetensors").path)

            if needsDownload {
                print("Downloading Chinese TTS model from HuggingFace...")
                print("Repository: \(hfRepo)")
                try await downloadModel(to: modelDir, voice: voice)
                print("Download complete!")
            }

            logger.info("Model directory: \(modelDir.path)")

            // Initialize synthesizer
            logger.info("[1/3] Initializing Chinese TTS synthesizer...")
            let synthesizer = KokoroChineseSynthesizer()

            // Load G2P dictionaries only if using text input (not raw phonemes)
            if rawPhonemes == nil {
                let g2pDir = modelDir.appendingPathComponent("g2p", isDirectory: true)

                logger.info("[2/3] Loading G2P dictionaries...")
                let jiebaURL = g2pDir.appendingPathComponent("jieba.bin.gz")
                let pinyinSingleURL = g2pDir.appendingPathComponent("pinyin_single.bin.gz")
                let pinyinPhrasesURL = g2pDir.appendingPathComponent("pinyin_phrases.bin.gz")

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
                return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                    .appendingPathComponent(expanded)
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

    /// Download model files from HuggingFace
    private static func downloadModel(to directory: URL, voice: String) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let baseURL = "https://huggingface.co/\(hfRepo)/resolve/main"

        // Files to download
        var filesToDownload = requiredFiles
        filesToDownload.append("voices/\(voice).npy")  // Add requested voice

        // Create subdirectories
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("g2p"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("voices"), withIntermediateDirectories: true)

        for file in filesToDownload {
            let remoteURL = URL(string: "\(baseURL)/\(file)")!
            let localURL = directory.appendingPathComponent(file)

            // Skip if already exists
            if FileManager.default.fileExists(atPath: localURL.path) {
                print("  ✓ \(file) (cached)")
                continue
            }

            print("  ↓ \(file)...")
            let (data, response) = try await URLSession.shared.data(from: remoteURL)

            guard let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 200
            else {
                throw NSError(
                    domain: "ChineseTTS", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to download \(file)"])
            }

            try data.write(to: localURL)

            let sizeMB = Double(data.count) / 1_000_000
            print("  ✓ \(file) (\(String(format: "%.1f", sizeMB)) MB)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts-zh "Chinese text" [options]

            Synthesize Chinese text to speech using Kokoro TTS with Bopomofo G2P.
            Model is auto-downloaded from HuggingFace on first run (~370MB).

            Options:
              --output, -o    Output WAV path (default: output_zh.wav)
              --voice, -v     Voice name (default: zf_001 - Chinese female)
              --model, -m     Path to model directory (default: ~/Library/Caches/FluidAudio/kokoro-zh)
              --phonemes, -p  Raw Bopomofo phonemes (bypasses G2P, no text required)
              --help, -h      Show this help

            Available voices (Chinese):
              zf_001 - zf_099   Chinese female voices
              zm_009 - zm_100   Chinese male voices

            Examples:
              fluidaudio tts-zh "你好世界"
              fluidaudio tts-zh "今天天气很好" --voice zm_010 --output weather.wav
              fluidaudio tts-zh --phonemes "ㄋㄧ3ㄏㄠ3" --output hello.wav
            """
        )
    }
}
