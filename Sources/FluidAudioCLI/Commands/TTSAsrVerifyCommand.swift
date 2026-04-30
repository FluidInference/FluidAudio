#if os(macOS)
import FluidAudio
import Foundation

/// Batch TTS→ASR roundtrip verification.
///
/// Reads a list of phrases from a file, synthesizes each one with the requested
/// TTS backend, transcribes the resulting WAV with Parakeet, computes per-phrase
/// and aggregate WER, and writes a JSON report.
///
/// Usage:
///   fluidaudio tts-asr-verify --backend kokoro-ane \
///       --texts-file phrases.txt \
///       --voice af_heart \
///       --output-json verify-results.json \
///       [--audio-dir /tmp/lai-wavs]
public enum TTSAsrVerifyCommand {

    private static let logger = AppLogger(category: "TTSAsrVerifyCommand")

    public static func run(arguments: [String]) async {
        var backendName = "kokoro-ane"
        var textsFile: String?
        var voice: String = TtsConstants.recommendedVoice
        var outputJson: String?
        var audioDir: String?

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--backend":
                if i + 1 < arguments.count {
                    backendName = arguments[i + 1]
                    i += 1
                }
            case "--texts-file":
                if i + 1 < arguments.count {
                    textsFile = arguments[i + 1]
                    i += 1
                }
            case "--voice":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            case "--output-json":
                if i + 1 < arguments.count {
                    outputJson = arguments[i + 1]
                    i += 1
                }
            case "--audio-dir":
                if i + 1 < arguments.count {
                    audioDir = arguments[i + 1]
                    i += 1
                }
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arg)")
            }
            i += 1
        }

        guard let textsFile else {
            logger.error("--texts-file is required")
            printUsage()
            exit(1)
        }

        let phrases: [String]
        do {
            phrases = try readPhrases(from: textsFile)
        } catch {
            logger.error("Failed to read texts file: \(error.localizedDescription)")
            exit(1)
        }
        guard !phrases.isEmpty else {
            logger.error("No phrases found in \(textsFile)")
            exit(1)
        }
        logger.info("Loaded \(phrases.count) phrase(s) from \(textsFile)")

        let backend = parseBackend(backendName)
        guard backend == .kokoroAne else {
            logger.error(
                "tts-asr-verify currently supports --backend kokoro-ane only (got '\(backendName)')")
            exit(1)
        }

        let resolvedVoice =
            voice == TtsConstants.recommendedVoice
            ? KokoroAneConstants.defaultVoice : voice

        do {
            // Set up TTS once.
            let manager = KokoroAneManager(defaultVoice: resolvedVoice)
            try await manager.initialize()
            logger.info("KokoroAne initialized (voice=\(resolvedVoice))")

            // Set up ASR once.
            let asrModels = try await AsrModels.downloadAndLoad()
            let asr = AsrManager()
            try await asr.loadModels(asrModels)
            let decoderLayers = await asr.decoderLayerCount

            // Optional audio output directory.
            var audioDirURL: URL? = nil
            if let audioDir {
                let url = resolveURL(audioDir, isDirectory: true)
                try FileManager.default.createDirectory(
                    at: url, withIntermediateDirectories: true)
                audioDirURL = url
            }

            // Iterate phrases.
            var perPhrase: [[String: Any]] = []
            var totalAudioS = 0.0
            var totalSynthS = 0.0
            var totalAsrS = 0.0
            var werValues: [Double] = []
            var totalRefWords = 0
            var totalEditDistance = 0

            for (idx, phrase) in phrases.enumerated() {
                let label = String(format: "[%02d/%02d]", idx + 1, phrases.count)
                logger.info("\(label) Synthesizing: \(phrase)")

                let synth0 = Date()
                let detailed = try await manager.synthesizeDetailed(
                    text: phrase, voice: resolvedVoice, speed: 1.0)
                let wav = try AudioWAV.data(
                    from: detailed.samples, sampleRate: Double(detailed.sampleRate))
                let synthS = Date().timeIntervalSince(synth0)

                // Persist WAV (audioDir if set, else temp file).
                let wavURL: URL
                if let audioDirURL {
                    wavURL = audioDirURL.appendingPathComponent(
                        String(format: "phrase_%03d.wav", idx + 1))
                } else {
                    wavURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent("tts-asr-verify-\(UUID().uuidString).wav")
                }
                try wav.write(to: wavURL)

                let audioS = Double(detailed.samples.count) / Double(detailed.sampleRate)

                // Transcribe.
                let asr0 = Date()
                var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                let transcription = try await asr.transcribe(
                    wavURL, decoderState: &decoderState)
                let asrS = Date().timeIntervalSince(asr0)

                // WER.
                let m = WERCalculator.calculateWERMetrics(
                    hypothesis: transcription.text, reference: phrase)
                werValues.append(m.wer)
                totalRefWords += m.totalWords
                totalEditDistance += m.insertions + m.deletions + m.substitutions
                totalAudioS += audioS
                totalSynthS += synthS
                totalAsrS += asrS

                logger.info("  ref: \(phrase)")
                logger.info("  hyp: \(transcription.text)")
                logger.info(
                    String(
                        format: "  wer=%.1f%%  audio=%.2fs  synth=%.2fs  asr=%.2fs",
                        m.wer * 100, audioS, synthS, asrS))

                if audioDirURL == nil {
                    try? FileManager.default.removeItem(at: wavURL)
                }

                perPhrase.append([
                    "index": idx + 1,
                    "reference": phrase,
                    "hypothesis": transcription.text,
                    "wer": m.wer,
                    "insertions": m.insertions,
                    "deletions": m.deletions,
                    "substitutions": m.substitutions,
                    "ref_word_count": m.totalWords,
                    "audio_s": audioS,
                    "synth_s": synthS,
                    "asr_s": asrS,
                    "encoder_tokens": detailed.encoderTokens,
                    "acoustic_frames": detailed.acousticFrames,
                    "wav_path": audioDirURL == nil ? "" : wavURL.path,
                    "stage_timings_ms": [
                        "albert": detailed.timings.albert,
                        "post_albert": detailed.timings.postAlbert,
                        "alignment": detailed.timings.alignment,
                        "prosody": detailed.timings.prosody,
                        "noise": detailed.timings.noise,
                        "vocoder": detailed.timings.vocoder,
                        "tail": detailed.timings.tail,
                        "total": detailed.timings.totalMs,
                    ],
                ])
            }

            await asr.cleanup()

            // Aggregate.
            let macroWer =
                werValues.isEmpty
                ? 0.0 : werValues.reduce(0, +) / Double(werValues.count)
            let microWer =
                totalRefWords == 0
                ? 0.0 : Double(totalEditDistance) / Double(totalRefWords)
            let rtfx = totalSynthS > 0 ? totalAudioS / totalSynthS : 0

            logger.info("--- Summary ---")
            logger.info("  phrases: \(phrases.count)")
            logger.info(String(format: "  macro WER: %.2f%%", macroWer * 100))
            logger.info(String(format: "  micro WER: %.2f%%", microWer * 100))
            logger.info(String(format: "  total audio: %.2fs", totalAudioS))
            logger.info(String(format: "  total synth: %.2fs (RTFx %.2fx)", totalSynthS, rtfx))
            logger.info(String(format: "  total asr:   %.2fs", totalAsrS))

            // Write JSON.
            if let outputJson {
                let summary: [String: Any] = [
                    "backend": backendName,
                    "voice": resolvedVoice,
                    "phrase_count": phrases.count,
                    "macro_wer": macroWer,
                    "micro_wer": microWer,
                    "total_audio_s": totalAudioS,
                    "total_synth_s": totalSynthS,
                    "total_asr_s": totalAsrS,
                    "realtime_speed": rtfx,
                ]
                let report: [String: Any] = [
                    "summary": summary,
                    "phrases": perPhrase,
                ]
                let url = resolveURL(outputJson, isDirectory: false)
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                let data = try JSONSerialization.data(
                    withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: url)
                logger.info("Report written: \(url.path)")
            }
        } catch {
            logger.error("tts-asr-verify failed: \(error)")
            exit(1)
        }
    }

    // MARK: - Helpers

    private static func parseBackend(_ name: String) -> TtsBackend {
        switch name.lowercased() {
        case "kokoro": return .kokoro
        case "pocket", "pockettts", "pocket-tts": return .pocketTts
        case "kokoro-ane", "kokoroane", "lai": return .kokoroAne
        default: return .kokoroAne
        }
    }

    private static func readPhrases(from path: String) throws -> [String] {
        let url = resolveURL(path, isDirectory: false)
        let raw = try String(contentsOf: url, encoding: .utf8)
        return raw.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    private static func resolveURL(_ path: String, isDirectory: Bool) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded, isDirectory: isDirectory)
        }
        let cwd = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return cwd.appendingPathComponent(expanded, isDirectory: isDirectory)
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts-asr-verify --texts-file phrases.txt [options]

            Reads phrases (one per line, '#' comments ignored), synthesizes each
            with the chosen TTS backend, transcribes with Parakeet, computes
            per-phrase + aggregate WER, and writes a JSON report.

            Options:
              --backend <name>      TTS backend: kokoro-ane (default)
              --texts-file <path>   Phrases file (required)
              --voice <name>        Voice name (default: af_heart)
              --output-json <path>  Output JSON report path
              --audio-dir <path>    Optional dir to keep generated WAVs
              --help, -h            Show this help

            Example:
              fluidaudio tts-asr-verify \\
                  --backend kokoro-ane \\
                  --texts-file phrases.txt \\
                  --voice af_heart \\
                  --output-json verify-results.json
            """
        )
    }
}
#endif
