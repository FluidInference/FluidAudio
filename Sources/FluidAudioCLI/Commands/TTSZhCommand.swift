import FluidAudio
import Foundation

/// Convenience wrapper for Mandarin: enforces zh-lexicon G2P and sensible defaults.
enum TTSZh {
    private static let logger = AppLogger(category: "TTSZhCommand")

    public static func run(arguments: [String]) async {
        var output = "out.wav"
        var voice = "zf_003"
        var modelPath: String? = nil
        var voicesDir: String? = nil
        var speed: Float = 1.0
        var text: String? = nil

        // Parse a minimal set of flags
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--help", "-h":
                print("Usage: fluidaudio tts-zh \"text\" [--output out.wav] [--voice zf_003] [--model-path path] [--voices-dir dir] [--speed 1.0]")
                return
            case "--output", "-o":
                if i+1 < arguments.count { output = arguments[i+1]; i += 1 }
            case "--voice", "-v":
                if i+1 < arguments.count { voice = arguments[i+1]; i += 1 }
            case "--model", "--model-path":
                if i+1 < arguments.count { modelPath = arguments[i+1]; i += 1 }
            case "--voices-dir":
                if i+1 < arguments.count { voicesDir = arguments[i+1]; i += 1 }
            case "--speed":
                if i+1 < arguments.count, let val = Float(arguments[i+1]) { speed = val; i += 1 }
            default:
                if text == nil { text = arg } else { logger.warning("Ignoring unexpected argument '\(arg)'") }
            }
            i += 1
        }

        guard let text = text else {
            print("Usage: fluidaudio tts-zh \"text\" [--output out.wav] [--voice zf_003] [--model-path path] [--voices-dir dir] [--speed 1.0]")
            return
        }

        do {
            // Prefer a local zh vocabulary next to CWD if present, before any model detection.
            do {
                let fm = FileManager.default
                let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
                let cwdZh = cwd.appendingPathComponent("zh_vocab_index.json")
                if fm.fileExists(atPath: cwdZh.path) {
                    await KokoroVocabulary.shared.setOverrideURL(cwdZh)
                }
            }

            // Auto-detect model if not provided
            if modelPath == nil {
                let fm = FileManager.default
                let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
                let candidates: [URL] = [
                    cwd.appendingPathComponent("kokoro_v21_zh.mlmodelc"),
                    cwd.appendingPathComponent("tts-zh/kokoro_v21_zh.mlmodelc"),
                    cwd.appendingPathComponent("build/kokoro_v21_zh_compiled/kokoro_v21_zh.mlmodelc"),
                    cwd.appendingPathComponent("kokoro_v21_zh.mlpackage"),
                    cwd.appendingPathComponent("tts-zh/kokoro_v21_zh.mlpackage")
                ]
                if let found = candidates.first(where: { fm.fileExists(atPath: $0.path) }) {
                    modelPath = found.path
                }
            }

            // If model path found, auto-detect vocab next to it. Otherwise, try CWD fallbacks.
            let fm = FileManager.default
            if let modelPath, !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let modelURL = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath)
                let root = modelURL.deletingLastPathComponent()
                let zhVocab = root.appendingPathComponent("zh_vocab_index.json")
                let genericVocab = root.appendingPathComponent("vocab_index.json")
                if fm.fileExists(atPath: zhVocab.path) {
                    await KokoroVocabulary.shared.setOverrideURL(zhVocab)
                } else if fm.fileExists(atPath: genericVocab.path) {
                    await KokoroVocabulary.shared.setOverrideURL(genericVocab)
                }
            } else {
                let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
                let cwdZh = cwd.appendingPathComponent("zh_vocab_index.json")
                let cwdGeneric = cwd.appendingPathComponent("vocab_index.json")
                if fm.fileExists(atPath: cwdZh.path) {
                    await KokoroVocabulary.shared.setOverrideURL(cwdZh)
                } else if fm.fileExists(atPath: cwdGeneric.path) {
                    await KokoroVocabulary.shared.setOverrideURL(cwdGeneric)
                }
            }

            // Provide extra voices dirs: explicit + sibling voices
            var extraDirs: [URL] = []
            if let dir = voicesDir, !dir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                extraDirs.append(URL(fileURLWithPath: (dir as NSString).expandingTildeInPath, isDirectory: true))
            }
            if let modelPath {
                let root = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath).deletingLastPathComponent()
                let sibling = root.appendingPathComponent("voices", isDirectory: true)
                if FileManager.default.fileExists(atPath: sibling.path) { extraDirs.append(sibling) }
            }
            if !extraDirs.isEmpty { KokoroSynthesizer.setAdditionalVoiceDirectories(extraDirs) }

            let manager = TtSManager()
            let requestedVoice = voice.trimmingCharacters(in: .whitespacesAndNewlines)
            let voiceOverride = requestedVoice.isEmpty ? nil : requestedVoice
            let preloadVoices = voiceOverride.map { Set([$0]) }

            // Initialize models: use local model if provided; otherwise download defaults
            if let localPath = modelPath, !localPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let models = try await TtsModels.loadLocal(at: localPath)
                try await manager.initialize(models: models, preloadVoices: preloadVoices)
            } else {
                try await manager.initialize(preloadVoices: preloadVoices)
            }

            // Lexicon-based Mandarin encoding
            let vocab = try await KokoroVocabulary.shared.getVocabulary()
            let allowed = Set(vocab.keys)
            let phonemeString = try await ZhCharLexicon.shared.encode(text: text, allowedTokens: allowed)

            let detailed = try await manager.synthesizePhonemesDetailed(
                phonemes: phonemeString,
                voice: voiceOverride,
                voiceSpeed: speed,
                variantPreference: .fifteenSecond
            )

            // Write WAV
            let outURL = {
                let expanded = (output as NSString).expandingTildeInPath
                if expanded.hasPrefix("/") { return URL(fileURLWithPath: expanded) }
                let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                return cwd.appendingPathComponent(expanded)
            }()
            try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try detailed.audio.write(to: outURL)
            logger.info("Saved output WAV: \(outURL.path)")
        } catch {
            logger.error("TTS zh failed: \(error)")
            print("❌ TTS zh failed: \(error)")
        }
    }
}
