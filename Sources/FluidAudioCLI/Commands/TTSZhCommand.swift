import FluidAudio
import Foundation

/// Convenience wrapper for Mandarin: enforces zh-lexicon G2P and sensible defaults.
enum TTSZh {
    private static let logger = AppLogger(category: "TTSZhCommand")

    public static func run(arguments: [String]) async {
        // If the user already supplied their own phonemes or g2p, pass-through.
        var args = arguments

        func hasFlag(_ flag: String) -> Bool { args.contains(where: { $0 == flag }) }
        func hasAny(_ flags: [String]) -> Bool { flags.contains(where: { hasFlag($0) }) }
        func optionPresent(_ name: String) -> Bool {
            if let i = args.firstIndex(of: name) { return (i + 1) < args.count }
            return false
        }

        // Ensure a default Mandarin voice if none supplied
        if !hasAny(["--voice", "-v"]) {
            args.append(contentsOf: ["--voice", "zf_003"]) // common zh female
        }

        // Enforce zh-lexicon unless user passed phonemes or g2p explicitly
        let userProvidedPhonemes = hasAny(["--phonemes", "--phonemes-file", "--phoneme-json"]) || optionPresent("--phonemes") || optionPresent("--phonemes-file") || optionPresent("--phoneme-json")
        let userProvidedG2P = optionPresent("--g2p")
        if !userProvidedPhonemes && !userProvidedG2P {
            args.append(contentsOf: ["--g2p", "zh-lexicon"])
        }

        // Auto-detect model path if not supplied: prefer common local paths
        let userProvidedModel = optionPresent("--model") || optionPresent("--model-path")
        if !userProvidedModel {
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
                args.append(contentsOf: ["--model-path", found.path])
            }
        }

        // Provide zh vocab and lexicon defaults if files exist locally
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let localVocab = cwd.appendingPathComponent("zh_vocab_index.json").path
        if !optionPresent("--vocab-path") && fm.fileExists(atPath: localVocab) {
            args.append(contentsOf: ["--vocab-path", localVocab])
        }

        let localLex = cwd.appendingPathComponent("zh_char_phonemes.json").path
        if !optionPresent("--zh-lexicon") && fm.fileExists(atPath: localLex) {
            args.append(contentsOf: ["--zh-lexicon", localLex])
        }

        // Call the standard TTS command with augmented arguments
        logger.info("tts-zh forwarding to TTS with args: \(args)")
        await TTS.run(arguments: args)
    }
}
