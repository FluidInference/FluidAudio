#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// `fluidaudio tts-benchmark` — quantitative TTS benchmark harness.
///
/// Reports **TTFT / cold-start / warm-start latency, per-stage timings,
/// peak RSS, WER + CER per category** — i.e. the things conversational
/// TTS users actually feel — instead of just RTFx.
///
/// Backends:
///   kokoro-ane    — 7-stage ANE pipeline (per-stage timings, per-stage CU)
///   pocket-tts    — streaming flow-matching (no per-stage timings)
///   styletts2     — LibriTTS iteration_3, zero-shot w/ reference audio
///   supertonic3   — 4-stage multilingual flow-matching diffusion (31 langs)
///
/// Usage:
///   fluidaudio tts-benchmark --backend kokoro-ane \
///       --corpus minimax-english \
///       --voice af_heart \
///       --compute-units default \
///       --output-json bench.json
///
/// Corpora land in `Benchmarks/tts/corpus/minimax/<lang>.txt` —
/// the MiniMax Multilingual TTS Test Set (CC-BY-SA-4.0,
/// 24 languages × 100 phrases). The `.txt` files are gitignored;
/// populate them with `swift run fluidaudio minimax-corpus`. See
/// `Documentation/TTS/MinimaxCorpus.md` for attribution + reproduction
/// notes and `Documentation/TTS/Benchmarks.md` for the per-backend ↔
/// language coverage matrix. Reference with `--corpus minimax-<lang>`
/// (e.g. `minimax-english`, `minimax-chinese`, `minimax-vietnamese`, …).
public enum TtsBenchmarkCommand {

    private static let logger = AppLogger(category: "TtsBenchmarkCommand")

    // MARK: - Per-phrase sample emitted by every backend driver.
    private struct BackendPhraseSample {
        let synthMs: Double
        let ttftMs: Double  // For one-shot backends, == synthMs.
        let samples: [Float]
        let sampleRate: Int
        let stageMs: [String: Double]  // Empty if backend has no per-stage timings.
        let extraFields: [String: Any]  // encoder_tokens, finished_on_eos, etc.
    }

    // MARK: - ASR backend selection
    //
    // The harness supports two ASR backends for the TTS→ASR roundtrip:
    //   .parakeet — Parakeet TDT (English-only, auto-downloaded).
    //   .cohere   — Cohere Transcribe cache-external (14 languages incl. zh).
    // Non-English TTS output (e.g. KokoroAne `mandarin` variant) needs
    // `.cohere` for a meaningful CER — Parakeet's English-only output
    // collapses to ~100% WER on zh.
    fileprivate enum AsrChoice {
        case skip
        case parakeet
        case cohere(modelDir: URL, language: CohereAsrConfig.Language, computeUnits: MLComputeUnits)

        var label: String {
            switch self {
            case .skip: return "skip"
            case .parakeet: return "parakeet-tdt"
            case .cohere(_, let lang, let cu):
                return "cohere-transcribe-\(lang.rawValue)/\(Self.computeLabel(cu))"
            }
        }

        private static func computeLabel(_ cu: MLComputeUnits) -> String {
            switch cu {
            case .all: return "all"
            case .cpuAndNeuralEngine: return "cpu+ane"
            case .cpuAndGPU: return "cpu+gpu"
            case .cpuOnly: return "cpu"
            @unknown default: return "unknown"
            }
        }

        var skipped: Bool {
            if case .skip = self { return true } else { return false }
        }
    }

    /// Closure-based ASR adapter so `runPhraseLoop` doesn't have to know
    /// which backend it's driving. Built once before the per-phrase loop,
    /// torn down after.
    fileprivate struct AsrLoop {
        let label: String
        let transcribeOne: (URL) async throws -> String
        let cleanup: () async -> Void
    }

    public static func run(arguments: [String]) async {
        var backendName = "kokoro-ane"
        var corpusName: String?
        var corpusPath: String?
        var voice: String?
        var languageName: String?
        var computeUnitsName = "default"
        var outputJson: String?
        var audioDir: String?
        var skipAsr = false
        var asrBackendName: String?
        var cohereModelDirArg: String?
        var asrLanguageArg: String?
        var cohereComputeUnitsArg: String?
        var referencePath: String?
        var variantArg: String?
        var voiceStylePath: String?
        var totalStepsArg: Int?
        var speedArg: Float?

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--backend":
                if i + 1 < arguments.count {
                    backendName = arguments[i + 1]
                    i += 1
                }
            case "--corpus":
                if i + 1 < arguments.count {
                    corpusName = arguments[i + 1]
                    i += 1
                }
            case "--corpus-path":
                if i + 1 < arguments.count {
                    corpusPath = arguments[i + 1]
                    i += 1
                }
            case "--voice":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            case "--language":
                if i + 1 < arguments.count {
                    languageName = arguments[i + 1]
                    i += 1
                }
            case "--compute-units":
                if i + 1 < arguments.count {
                    computeUnitsName = arguments[i + 1]
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
            case "--skip-asr":
                skipAsr = true
            case "--asr-backend":
                if i + 1 < arguments.count {
                    asrBackendName = arguments[i + 1]
                    i += 1
                }
            case "--cohere-model-dir":
                if i + 1 < arguments.count {
                    cohereModelDirArg = arguments[i + 1]
                    i += 1
                }
            case "--asr-language":
                if i + 1 < arguments.count {
                    asrLanguageArg = arguments[i + 1]
                    i += 1
                }
            case "--cohere-compute-units":
                if i + 1 < arguments.count {
                    cohereComputeUnitsArg = arguments[i + 1]
                    i += 1
                }
            case "--reference":
                if i + 1 < arguments.count {
                    referencePath = arguments[i + 1]
                    i += 1
                }
            case "--variant":
                if i + 1 < arguments.count {
                    variantArg = arguments[i + 1]
                    i += 1
                }
            case "--voice-style":
                if i + 1 < arguments.count {
                    voiceStylePath = arguments[i + 1]
                    i += 1
                }
            case "--total-steps":
                if i + 1 < arguments.count {
                    totalStepsArg = Int(arguments[i + 1])
                    i += 1
                }
            case "--speed":
                if i + 1 < arguments.count {
                    speedArg = Float(arguments[i + 1])
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

        let backend = parseBackend(backendName)

        // Resolve corpus.
        let phrases: [(category: String, text: String)]
        let corpusLabel: String
        do {
            if let corpusPath {
                let url = resolveURL(corpusPath, isDirectory: false)
                let raw = try String(contentsOf: url, encoding: .utf8)
                phrases = parseCorpus(raw, category: url.deletingPathExtension().lastPathComponent)
                corpusLabel = url.lastPathComponent
            } else {
                let resolved = corpusName ?? backend.defaultCorpus
                phrases = try loadShippedCorpus(resolved)
                corpusLabel = resolved
            }
        } catch {
            logger.error("Failed to load corpus: \(error.localizedDescription)")
            exit(1)
        }
        guard !phrases.isEmpty else {
            logger.error("Corpus is empty after parsing")
            exit(1)
        }
        logger.info("Loaded \(phrases.count) phrase(s) from corpus '\(corpusLabel)'")

        guard let preset = TtsComputeUnitPreset(cliValue: computeUnitsName) else {
            logger.error(
                "Unknown --compute-units value: \(computeUnitsName). Expected default | all-ane | cpu-and-gpu | cpu-only | ane-tail-gpu."
            )
            exit(1)
        }

        // Resolve ASR backend choice. Precedence:
        //   --skip-asr or --asr-backend none → .skip
        //   --asr-backend cohere             → .cohere(modelDir, language)
        //   --asr-backend parakeet           → .parakeet
        //   no flag, otherwise               → .parakeet
        let asrChoice: AsrChoice
        do {
            asrChoice = try await resolveAsrChoice(
                skipAsrFlag: skipAsr,
                backendName: asrBackendName,
                cohereModelDir: cohereModelDirArg,
                asrLanguage: asrLanguageArg,
                cohereComputeUnits: cohereComputeUnitsArg,
                corpusLabel: corpusLabel,
                ttsBackend: backend)
        } catch {
            logger.error("Failed to resolve ASR backend: \(error.localizedDescription)")
            exit(1)
        }
        logger.info("ASR backend: \(asrChoice.label)")

        do {
            switch backend {
            case .kokoroAne:
                let kaVariant = parseKokoroAneVariant(variantArg)
                try await runKokoroAne(
                    phrases: phrases, corpusLabel: corpusLabel,
                    variant: kaVariant,
                    voice: voice ?? kaVariant.defaultVoice,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    asrChoice: asrChoice)
            case .pocketTts:
                try await runPocketTts(
                    phrases: phrases, corpusLabel: corpusLabel,
                    voice: voice ?? PocketTtsConstants.defaultVoice,
                    languageName: languageName,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    asrChoice: asrChoice)
            case .styleTts2:
                try await runStyleTTS2(
                    phrases: phrases, corpusLabel: corpusLabel,
                    referencePath: referencePath,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    asrChoice: asrChoice)
            case .supertonic3:
                try await runSupertonic3(
                    phrases: phrases, corpusLabel: corpusLabel,
                    voiceStylePath: voiceStylePath,
                    languageName: languageName,
                    totalSteps: totalStepsArg
                        ?? Supertonic3Constants.defaultTotalSteps,
                    speed: speedArg ?? Supertonic3Constants.defaultSpeed,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    asrChoice: asrChoice)
            }
        } catch {
            logger.error("tts-benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - Kokoro ANE driver

    private static func runKokoroAne(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        variant: KokoroAneVariant,
        voice: String,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        asrChoice: AsrChoice
    ) async throws {
        let units = KokoroAneComputeUnits(preset: preset)
        let manager = KokoroAneManager(variant: variant, defaultVoice: voice, computeUnits: units)

        let coldStart = Date()
        try await manager.initialize()
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (initialize): %.2fs", coldStartS))

        let firstStart = Date()
        _ = try await manager.synthesizeDetailed(
            text: "Initialization warm-up.", voice: voice, speed: 1.0)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(String(format: "First synth: %.0f ms", firstSynthMs))

        try await runPhraseLoop(
            backendId: "kokoro-ane",
            voiceLabel: voice,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            asrChoice: asrChoice,
            extraSummary: ["voice": voice]
        ) { text in
            let t0 = Date()
            let result = try await manager.synthesizeDetailed(
                text: text, voice: voice, speed: 1.0)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: result.samples,
                sampleRate: result.sampleRate,
                stageMs: [
                    "albert": result.timings.albert,
                    "post_albert": result.timings.postAlbert,
                    "alignment": result.timings.alignment,
                    "prosody": result.timings.prosody,
                    "noise": result.timings.noise,
                    "vocoder": result.timings.vocoder,
                    "tail": result.timings.tail,
                    "total": result.timings.totalMs,
                ],
                extraFields: [
                    "encoder_tokens": result.encoderTokens,
                    "acoustic_frames": result.acousticFrames,
                ]
            )
        }
    }

    // MARK: - PocketTTS driver

    private static func runPocketTts(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        voice: String,
        languageName: String?,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        asrChoice: AsrChoice
    ) async throws {
        if preset != .default {
            logger.warning(
                "PocketTTS does not expose per-call compute-unit overrides; --compute-units \(preset.cliValue) ignored."
            )
        }
        let language = parsePocketLanguage(languageName)
        logger.info("PocketTTS language: \(language.rawValue)")

        let manager = PocketTtsManager(defaultVoice: voice, language: language)

        let coldStart = Date()
        try await manager.initialize()
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (initialize): %.2fs", coldStartS))

        let firstStart = Date()
        var firstFrameMs: Double = 0
        var firstFrameCount = 0
        let warmupStream = try await manager.synthesizeStreaming(
            text: "Initialization warm-up.", voice: voice)
        for try await frame in warmupStream {
            if firstFrameCount == 0 {
                firstFrameMs = Date().timeIntervalSince(firstStart) * 1000
            }
            firstFrameCount += 1
            _ = frame.samples
        }
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(
            String(
                format: "First synth: %.0f ms total, %.0f ms TTFT (frames=%d)",
                firstSynthMs, firstFrameMs, firstFrameCount))

        try await runPhraseLoop(
            backendId: "pocket-tts",
            voiceLabel: voice,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            asrChoice: asrChoice,
            extraSummary: ["voice": voice, "language": language.rawValue]
        ) { text in
            // PocketTTS is streaming-first: we measure TTFT (time to first
            // audio frame) separately from total synth time so the benchmark
            // numbers reflect what a streaming consumer actually experiences.
            let t0 = Date()
            let stream = try await manager.synthesizeStreaming(text: text, voice: voice)
            var aggregated: [Float] = []
            var ttftMs: Double = 0
            var frameCount = 0
            var lastChunkCount = 0
            for try await frame in stream {
                if frameCount == 0 {
                    ttftMs = Date().timeIntervalSince(t0) * 1000
                }
                aggregated.append(contentsOf: frame.samples)
                frameCount += 1
                lastChunkCount = frame.chunkCount
            }
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: ttftMs,
                samples: aggregated,
                sampleRate: PocketTtsConstants.audioSampleRate,
                stageMs: [:],
                extraFields: [
                    "frame_count": frameCount,
                    "chunk_count": lastChunkCount,
                ]
            )
        }
    }

    // MARK: - StyleTTS2 driver

    private static func runStyleTTS2(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        referencePath: String?,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        asrChoice: AsrChoice
    ) async throws {
        guard let referencePath, !referencePath.isEmpty else {
            logger.error(
                "styletts2 backend requires --reference <speaker-audio-file> "
                    + "(any sample rate / channel layout — resampled to 24 kHz mono).")
            exit(1)
        }
        let referenceURL = resolveURL(referencePath, isDirectory: false)
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            logger.error("Reference audio not found: \(referenceURL.path)")
            exit(1)
        }
        logger.info("StyleTTS2 reference audio: \(referenceURL.path)")

        let units = preset.uniformUnits ?? .cpuAndNeuralEngine
        let manager = StyleTTS2Manager(computeUnits: units)

        let coldStart = Date()
        try await manager.initialize()
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (initialize): %.2fs", coldStartS))

        let firstStart = Date()
        _ = try await manager.synthesize(
            text: "Initialization warm-up.",
            referenceAudioURL: referenceURL)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(String(format: "First synth: %.0f ms", firstSynthMs))

        try await runPhraseLoop(
            backendId: "styletts2",
            voiceLabel: referenceURL.lastPathComponent,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            asrChoice: asrChoice,
            extraSummary: [
                "reference": referenceURL.path,
                "alpha": Double(StyleTTS2Constants.defaultAlpha),
                "beta": Double(StyleTTS2Constants.defaultBeta),
            ]
        ) { text in
            // StyleTTS2 is a one-shot diffusion-based synthesizer — no
            // streaming yield, so TTFT == synthMs. The per-phrase mel
            // recompute is tiny vs. the 5-step ADPM2 + decoder cost.
            let t0 = Date()
            let samples = try await manager.synthesize(
                text: text, referenceAudioURL: referenceURL)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: samples,
                sampleRate: StyleTTS2Constants.sampleRate,
                stageMs: [:],
                extraFields: [:]
            )
        }
    }

    // MARK: - Supertonic-3 driver

    private static func runSupertonic3(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        voiceStylePath: String?,
        languageName: String?,
        totalSteps: Int,
        speed: Float,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        asrChoice: AsrChoice
    ) async throws {
        guard let voiceStylePath, !voiceStylePath.isEmpty else {
            logger.error(
                "supertonic3 backend requires --voice-style <path-to-voice.json> "
                    + "(e.g. M1.json from FluidInference/supertonic-3-coreml).")
            exit(1)
        }
        let voiceURL = resolveURL(voiceStylePath, isDirectory: false)
        guard FileManager.default.fileExists(atPath: voiceURL.path) else {
            logger.error("Voice style JSON not found: \(voiceURL.path)")
            exit(1)
        }
        let style: Supertonic3VoiceStyle
        do {
            style = try Supertonic3VoiceStyle.load(from: voiceURL)
        } catch {
            logger.error("Failed to load voice style: \(error.localizedDescription)")
            exit(1)
        }

        let language = resolveSupertonic3Language(
            explicit: languageName, corpus: corpusLabel)
        logger.info(
            "Supertonic-3 voice=\(style.name) lang=\(language) "
                + "steps=\(totalSteps) speed=\(speed)")

        let units = preset.uniformUnits ?? .cpuAndNeuralEngine
        let manager = Supertonic3Manager(computeUnits: units)

        let coldStart = Date()
        try await manager.initialize()
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (initialize): %.2fs", coldStartS))

        let firstStart = Date()
        _ = try await manager.synthesize(
            text: "Initialization warm-up.",
            language: language,
            style: style,
            totalSteps: totalSteps,
            speed: speed)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(String(format: "First synth: %.0f ms", firstSynthMs))

        try await runPhraseLoop(
            backendId: "supertonic3",
            voiceLabel: style.name,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            asrChoice: asrChoice,
            extraSummary: [
                "voice_style": style.name,
                "language": language,
                "total_steps": totalSteps,
                "speed": Double(speed),
            ]
        ) { text in
            // Supertonic-3 is a one-shot diffusion synthesizer — `synthesize`
            // returns the full waveform after the 8-step vector_estimator
            // loop completes, so TTFT == synthMs (no incremental yield).
            let t0 = Date()
            let result = try await manager.synthesize(
                text: text, language: language, style: style,
                totalSteps: totalSteps, speed: speed)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: result.samples,
                sampleRate: Supertonic3Constants.sampleRate,
                stageMs: [:],
                extraFields: [
                    "total_steps": totalSteps,
                    "speed": Double(speed),
                ]
            )
        }
    }

    // MARK: - Shared per-phrase loop + summary

    private static func runPhraseLoop(
        backendId: String,
        voiceLabel: String,
        corpusLabel: String,
        phrases: [(category: String, text: String)],
        preset: TtsComputeUnitPreset,
        coldStartS: Double,
        firstSynthMs: Double,
        outputJson: String?,
        audioDir: String?,
        asrChoice: AsrChoice,
        extraSummary: [String: Any],
        synthOne: (String) async throws -> BackendPhraseSample
    ) async throws {
        // Optional output dir for WAVs.
        var audioDirURL: URL? = nil
        if let audioDir {
            let url = resolveURL(audioDir, isDirectory: true)
            try FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: true)
            audioDirURL = url
        }

        // Build optional ASR backend (Parakeet, Cohere, or none).
        let asrLoop = try await buildAsrLoop(asrChoice)

        var perPhrase: [[String: Any]] = []
        var byCategory: [String: [Int]] = [:]

        for (idx, item) in phrases.enumerated() {
            let label = String(format: "[%02d/%02d]", idx + 1, phrases.count)
            logger.info("\(label) [\(item.category)] \(item.text)")

            let sample = try await synthOne(item.text)
            let audioMs =
                Double(sample.samples.count) / Double(sample.sampleRate) * 1000
            let rtfx = sample.synthMs > 0 ? audioMs / sample.synthMs : 0

            // Persist WAV (audioDir if set, else temp file for ASR).
            let wavURL: URL
            if let audioDirURL {
                wavURL = audioDirURL.appendingPathComponent(
                    String(format: "phrase_%03d.wav", idx + 1))
            } else {
                wavURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("tts-benchmark-\(UUID().uuidString).wav")
            }
            let wavData = try AudioWAV.data(
                from: sample.samples, sampleRate: Double(sample.sampleRate))
            try wavData.write(to: wavURL)

            var werValue = Double.nan
            var cerValue = Double.nan
            var hypothesis = ""
            var asrMs = 0.0
            if let asrLoop {
                let asr0 = Date()
                hypothesis = try await asrLoop.transcribeOne(wavURL)
                asrMs = Date().timeIntervalSince(asr0) * 1000
                let m = WERCalculator.calculateWERAndCER(
                    hypothesis: hypothesis, reference: item.text)
                werValue = m.wer
                cerValue = m.cer
            }

            if audioDirURL == nil {
                try? FileManager.default.removeItem(at: wavURL)
            }

            logger.info(
                String(
                    format:
                        "  ttft=%.0f ms  synth=%.0f ms  audio=%.0f ms  rtfx=%.2fx  wer=%.1f%%  cer=%.1f%%",
                    sample.ttftMs, sample.synthMs, audioMs, rtfx,
                    werValue.isNaN ? 0 : werValue * 100,
                    cerValue.isNaN ? 0 : cerValue * 100))

            byCategory[item.category, default: []].append(perPhrase.count)
            var phraseDict: [String: Any] = [
                "index": idx + 1,
                "category": item.category,
                "reference": item.text,
                "hypothesis": hypothesis,
                "ttft_ms": sample.ttftMs,
                "synth_ms": sample.synthMs,
                "audio_ms": audioMs,
                "rtfx": rtfx,
                "wer": werValue.isNaN ? NSNull() : werValue as Any,
                "cer": cerValue.isNaN ? NSNull() : cerValue as Any,
                "asr_ms": asrMs,
                "stage_ms": sample.stageMs,
                "wav_path": audioDirURL == nil ? "" : wavURL.path,
            ]
            for (k, v) in sample.extraFields {
                phraseDict[k] = v
            }
            perPhrase.append(phraseDict)
        }

        if let asrLoop {
            await asrLoop.cleanup()
        }

        // Aggregate.
        let totalSynthMs = perPhrase.reduce(0.0) { $0 + ($1["synth_ms"] as? Double ?? 0) }
        let totalAudioMs = perPhrase.reduce(0.0) { $0 + ($1["audio_ms"] as? Double ?? 0) }
        let aggRtfx = totalSynthMs > 0 ? totalAudioMs / totalSynthMs : 0

        let synthMsValues = perPhrase.compactMap { $0["synth_ms"] as? Double }.sorted()
        let p50 = percentile(synthMsValues, 0.5)
        let p95 = percentile(synthMsValues, 0.95)
        let ttftValues = perPhrase.compactMap { $0["ttft_ms"] as? Double }.sorted()
        let ttftP50 = percentile(ttftValues, 0.5)
        let ttftP95 = percentile(ttftValues, 0.95)

        var categories: [[String: Any]] = []
        for (cat, indexes) in byCategory.sorted(by: { $0.key < $1.key }) {
            let werVals = indexes.compactMap { perPhrase[$0]["wer"] as? Double }
            let cerVals = indexes.compactMap { perPhrase[$0]["cer"] as? Double }
            let synthVals = indexes.compactMap { perPhrase[$0]["synth_ms"] as? Double }
            let audioVals = indexes.compactMap { perPhrase[$0]["audio_ms"] as? Double }
            let synthSum = synthVals.reduce(0, +)
            let audioSum = audioVals.reduce(0, +)
            let macroWer =
                werVals.isEmpty ? Double.nan : werVals.reduce(0, +) / Double(werVals.count)
            let macroCer =
                cerVals.isEmpty ? Double.nan : cerVals.reduce(0, +) / Double(cerVals.count)
            categories.append([
                "category": cat,
                "phrase_count": indexes.count,
                "macro_wer": macroWer.isNaN ? NSNull() : macroWer as Any,
                "macro_cer": macroCer.isNaN ? NSNull() : macroCer as Any,
                "synth_ms_p50": percentile(synthVals.sorted(), 0.5),
                "synth_ms_p95": percentile(synthVals.sorted(), 0.95),
                "rtfx": synthSum > 0 ? audioSum / synthSum : 0,
            ])
        }

        let peakRssMb =
            Double(FluidAudioCLI.fetchPeakMemoryUsageBytes() ?? 0) / 1024 / 1024

        // Banner.
        logger.info("--- Summary ---")
        logger.info("  backend:        \(backendId)")
        logger.info("  voice/speaker:  \(voiceLabel)")
        logger.info("  corpus:         \(corpusLabel) (n=\(phrases.count))")
        logger.info("  compute units:  \(preset.cliValue)")
        logger.info(String(format: "  cold start:     %.2fs", coldStartS))
        logger.info(String(format: "  first synth:    %.0f ms", firstSynthMs))
        logger.info(String(format: "  TTFT p50/p95:   %.0f / %.0f ms", ttftP50, ttftP95))
        logger.info(String(format: "  warm synth p50: %.0f ms", p50))
        logger.info(String(format: "  warm synth p95: %.0f ms", p95))
        logger.info(String(format: "  agg RTFx:       %.2fx", aggRtfx))
        logger.info(String(format: "  peak RSS:       %.0f MB", peakRssMb))
        if !asrChoice.skipped {
            let werVals = perPhrase.compactMap { $0["wer"] as? Double }
            let cerVals = perPhrase.compactMap { $0["cer"] as? Double }
            let macroWer =
                werVals.isEmpty ? 0 : werVals.reduce(0, +) / Double(werVals.count)
            let macroCer =
                cerVals.isEmpty ? 0 : cerVals.reduce(0, +) / Double(cerVals.count)
            logger.info("  ASR backend:    \(asrChoice.label)")
            logger.info(String(format: "  macro WER:      %.2f%%", macroWer * 100))
            logger.info(String(format: "  macro CER:      %.2f%%", macroCer * 100))
            // Word-level WER is meaningless on whitespace-free scripts (zh, ja).
            // Surface that explicitly so readers don't trust ~100% WER for zh.
            if case .cohere(_, let lang, _) = asrChoice,
                lang == .chinese || lang == .japanese
            {
                logger.info(
                    "  note:           WER is whitespace-tokenized; trust CER for \(lang.rawValue).")
            }
        } else {
            logger.info("  WER/CER:        skipped")
        }

        if let outputJson {
            var summary: [String: Any] = [
                "backend": backendId,
                "corpus": corpusLabel,
                "phrase_count": phrases.count,
                "compute_units": preset.cliValue,
                "cold_start_s": coldStartS,
                "first_synth_ms": firstSynthMs,
                "ttft_ms_p50": ttftP50,
                "ttft_ms_p95": ttftP95,
                "warm_synth_ms_p50": p50,
                "warm_synth_ms_p95": p95,
                "agg_rtfx": aggRtfx,
                "peak_rss_mb": peakRssMb,
                "asr_skipped": asrChoice.skipped,
                "asr_backend": asrChoice.label,
            ]
            for (k, v) in extraSummary {
                summary[k] = v
            }
            let report: [String: Any] = [
                "summary": summary,
                "categories": categories,
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
    }

    // MARK: - Corpus loading

    private static func loadShippedCorpus(
        _ name: String
    ) throws -> [(category: String, text: String)] {
        let cwd = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        let relativePath = corpusRelativePath(for: name)
        let url = cwd.appendingPathComponent(relativePath, isDirectory: false)
        let raw = try String(contentsOf: url, encoding: .utf8)
        return parseCorpus(raw, category: name)
    }

    /// Map a `--corpus` name to its on-disk relative path.
    ///
    /// All shipped corpora are MiniMax Multilingual TTS Test Set
    /// languages — `minimax-<lang>` resolves to
    /// `Benchmarks/tts/corpus/minimax/<lang>.txt`. The CC-BY-SA-4.0
    /// attribution lives next to the data in `minimax/README.md`.
    /// Pass `--corpus-path` for ad-hoc files outside the shipped set.
    private static func corpusRelativePath(for name: String) -> String {
        let prefix = "minimax-"
        if name.hasPrefix(prefix) {
            let lang = String(name.dropFirst(prefix.count))
            return "Benchmarks/tts/corpus/minimax/\(lang).txt"
        }
        // Back-compat shim — anything else is assumed to live next to
        // the minimax subdirectory. Prefer `--corpus-path` for non-shipped
        // corpora.
        return "Benchmarks/tts/corpus/\(name).txt"
    }

    private static func parseCorpus(
        _ raw: String, category: String
    ) -> [(category: String, text: String)] {
        return
            raw
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            .map { (category: category, text: $0) }
    }

    // MARK: - Backend dispatch

    private enum Backend: String {
        case kokoroAne
        case pocketTts
        case styleTts2
        case supertonic3

        var defaultCorpus: String {
            return "minimax-english"
        }
    }

    private static func parseBackend(_ name: String) -> Backend {
        switch name.lowercased() {
        case "kokoro-ane", "kokoroane", "kokoro_ane", "kokoro", "lai":
            return .kokoroAne
        case "pocket-tts", "pockettts", "pocket":
            return .pocketTts
        case "styletts2", "style-tts2", "styletts", "style-tts":
            return .styleTts2
        case "supertonic3", "supertonic-3", "sup3", "supertonic":
            return .supertonic3
        default:
            logger.warning("Unknown backend '\(name)' — defaulting to kokoro-ane")
            return .kokoroAne
        }
    }

    private static func parsePocketLanguage(_ name: String?) -> PocketTtsLanguage {
        guard let name, let l = PocketTtsLanguage(rawValue: name.lowercased()) else {
            return .english
        }
        return l
    }

    /// Map an explicit `--language` flag or a `minimax-<lang>` corpus name
    /// onto one of the 31 ISO codes accepted by Supertonic-3
    /// (`Supertonic3Constants.availableLanguages`). Falls back to English.
    private static func resolveSupertonic3Language(
        explicit: String?, corpus: String
    ) -> String {
        if let explicit, !explicit.isEmpty {
            let lower = explicit.lowercased()
            if Supertonic3Constants.availableLanguages.contains(lower) {
                return lower
            }
            if let mapped = supertonic3LanguageAliases[lower] {
                return mapped
            }
        }
        let lower = corpus.lowercased()
        for (needle, code) in supertonic3CorpusToLanguage where lower.contains(needle) {
            return code
        }
        return "en"
    }

    /// Long-name → ISO code aliases for the `--language` flag.
    private static let supertonic3LanguageAliases: [String: String] = [
        "english": "en", "korean": "ko", "japanese": "ja", "arabic": "ar",
        "bulgarian": "bg", "czech": "cs", "danish": "da", "german": "de",
        "greek": "el", "spanish": "es", "estonian": "et", "finnish": "fi",
        "french": "fr", "hindi": "hi", "croatian": "hr", "hungarian": "hu",
        "indonesian": "id", "italian": "it", "lithuanian": "lt",
        "latvian": "lv", "dutch": "nl", "polish": "pl", "portuguese": "pt",
        "romanian": "ro", "russian": "ru", "slovak": "sk", "slovenian": "sl",
        "swedish": "sv", "turkish": "tr", "ukrainian": "uk",
        "vietnamese": "vi",
    ]

    /// Ordered scan list for `minimax-<lang>` corpus labels — longest /
    /// least-ambiguous matches first.
    private static let supertonic3CorpusToLanguage: [(String, String)] = [
        ("korean", "ko"), ("japanese", "ja"), ("arabic", "ar"),
        ("bulgarian", "bg"), ("czech", "cs"), ("danish", "da"),
        ("german", "de"), ("greek", "el"), ("spanish", "es"),
        ("estonian", "et"), ("finnish", "fi"), ("french", "fr"),
        ("hindi", "hi"), ("croatian", "hr"), ("hungarian", "hu"),
        ("indonesian", "id"), ("italian", "it"), ("lithuanian", "lt"),
        ("latvian", "lv"), ("dutch", "nl"), ("polish", "pl"),
        ("portuguese", "pt"), ("romanian", "ro"), ("russian", "ru"),
        ("slovak", "sk"), ("slovenian", "sl"), ("swedish", "sv"),
        ("turkish", "tr"), ("ukrainian", "uk"), ("vietnamese", "vi"),
        ("english", "en"),
    ]

    private static func parseKokoroAneVariant(_ name: String?) -> KokoroAneVariant {
        switch name?.lowercased() {
        case "mandarin", "zh", "chinese", "zh-cn":
            return .mandarin
        case "english", "en", "en-us", nil, "":
            return .english
        default:
            return .english
        }
    }

    // MARK: - Helpers

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let idx = Int((Double(sorted.count - 1) * p).rounded())
        return sorted[max(0, min(sorted.count - 1, idx))]
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

    // MARK: - ASR backend resolution & adapter construction

    /// Map CLI flags + TTS backend defaults to a concrete `AsrChoice`.
    ///
    /// Precedence: `--skip-asr` and `--asr-backend none` always win. With
    /// no flag, defaults to Parakeet TDT (English-only). Non-English TTS
    /// output (e.g. KokoroAne `mandarin` variant) should use Cohere via
    /// `--asr-backend cohere`.
    private static func resolveAsrChoice(
        skipAsrFlag: Bool,
        backendName: String?,
        cohereModelDir: String?,
        asrLanguage: String?,
        cohereComputeUnits: String?,
        corpusLabel: String,
        ttsBackend: Backend
    ) async throws -> AsrChoice {
        let normalized = backendName?.lowercased()
        if skipAsrFlag || normalized == "none" {
            return .skip
        }
        switch normalized {
        case "cohere":
            let dir = try await resolveCohereModelDir(cohereModelDir)
            let language = inferCohereLanguage(
                explicit: asrLanguage, corpus: corpusLabel)
            let units = try resolveCohereComputeUnits(cohereComputeUnits)
            return .cohere(modelDir: dir, language: language, computeUnits: units)
        case "parakeet":
            return .parakeet
        case nil:
            return .parakeet
        default:
            logger.warning(
                "Unknown --asr-backend value '\(normalized ?? "")', falling back to parakeet.")
            return .parakeet
        }
    }

    /// Resolve a Cohere Transcribe model directory (must contain
    /// `cohere_encoder.mlmodelc`, `cohere_decoder_cache_external_v2.mlmodelc`,
    /// and `vocab.json`).
    ///
    /// Order of resolution:
    ///   1. Explicit `--cohere-model-dir <path>`.
    ///   2. The default cache location at
    ///      `~/Library/Application Support/FluidAudio/Models/cohere-transcribe/q8`,
    ///      matching `Repo.cohereTranscribeCoreml.folderName`. Auto-downloaded
    ///      from HuggingFace if missing.
    ///
    /// `Repo.cohereTranscribeCoreml` ships `vocab.json` in `requiredModels`, and
    /// that file lives at the repo root rather than under the `q8/` subPath.
    /// `ModelHub.download` now sweeps the repo root for required
    /// auxiliary files (issue #649), so auto-download resolves it correctly.
    private static func resolveCohereModelDir(_ override: String?) async throws -> URL {
        if let override {
            return resolveURL(override, isDirectory: true)
        }
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask, appropriateFor: nil, create: true)
        // `ModelHub.download` appends `repo.folderName` (cohere-transcribe/q8), so
        // pass the Models base dir and let it land the bundle at `target`.
        let modelsBase = appSupport.appendingPathComponent("FluidAudio/Models")
        let target = modelsBase.appendingPathComponent("cohere-transcribe/q8")
        let needed = [
            ModelNames.CohereTranscribe.encoderCompiledFile,
            ModelNames.CohereTranscribe.decoderCacheExternalV2CompiledFile,
            "vocab.json",
        ]
        func missingFiles() -> [String] {
            needed.filter { name in
                !FileManager.default.fileExists(
                    atPath: target.appendingPathComponent(name).path)
            }
        }
        if !missingFiles().isEmpty {
            try await ModelHub.download(.cohereTranscribeCoreml, to: modelsBase)
        }
        let missing = missingFiles()
        guard missing.isEmpty else {
            throw NSError(
                domain: "TtsBenchmark", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Cohere model dir incomplete at \(target.path). "
                        + "Missing: \(missing.joined(separator: ", ")). "
                        + "Pass --cohere-model-dir <dir> with the required files, or "
                        + "pre-populate the cache via `fluidaudio cohere-transcribe`."
                ])
        }
        return target
    }

    /// Pick a `CohereAsrConfig.Language` from an explicit flag value or by
    /// scanning the corpus label (covers the shipped `minimax-<lang>` set).
    private static func inferCohereLanguage(
        explicit: String?, corpus: String
    ) -> CohereAsrConfig.Language {
        if let explicit,
            let lang = CohereAsrConfig.Language(rawValue: explicit.lowercased())
        {
            return lang
        }
        let lower = corpus.lowercased()
        if lower.contains("chinese") || lower.contains("mandarin") || lower.hasSuffix("-zh") {
            return .chinese
        }
        if lower.contains("japanese") || lower.contains("-ja") { return .japanese }
        if lower.contains("korean") || lower.contains("-ko") { return .korean }
        if lower.contains("vietnamese") || lower.contains("-vi") { return .vietnamese }
        if lower.contains("french") || lower.contains("-fr") { return .french }
        if lower.contains("german") || lower.contains("-de") { return .german }
        if lower.contains("spanish") || lower.contains("-es") { return .spanish }
        if lower.contains("italian") || lower.contains("-it") { return .italian }
        if lower.contains("portuguese") || lower.contains("-pt") { return .portuguese }
        if lower.contains("dutch") || lower.contains("-nl") { return .dutch }
        if lower.contains("polish") || lower.contains("-pl") { return .polish }
        if lower.contains("greek") || lower.contains("-el") { return .greek }
        if lower.contains("arabic") || lower.contains("-ar") { return .arabic }
        return .english
    }

    /// Parse `--cohere-compute-units` into `MLComputeUnits`. Defaults to
    /// `.all` (CoreML decides). Use `cpu-and-gpu` to skip the ANE compile
    /// attempt when the q8 encoder fails ANE compilation (observed:
    /// `MILCompilerForANE error: failed to compile ANE model using ANEF`,
    /// CoreML falls back to CPU+GPU but pays a multi-minute compile cost
    /// on the first call).
    private static func resolveCohereComputeUnits(
        _ flag: String?
    ) throws
        -> MLComputeUnits
    {
        guard let raw = flag?.lowercased(), !raw.isEmpty else { return .all }
        switch raw {
        case "all", "default": return .all
        case "all-ane", "ane", "neural-engine", "cpu-and-ane":
            return .cpuAndNeuralEngine
        case "cpu-and-gpu", "cpuandgpu", "gpu": return .cpuAndGPU
        case "cpu-only", "cpu", "cpuonly": return .cpuOnly
        default:
            throw NSError(
                domain: "TtsBenchmark", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Unknown --cohere-compute-units value '\(raw)'. "
                        + "Expected: all | cpu-and-gpu | cpu-only | all-ane."
                ])
        }
    }

    /// Human-readable label for log lines.
    private static func describeComputeUnits(_ cu: MLComputeUnits) -> String {
        switch cu {
        case .all: return "all (CPU+GPU+ANE)"
        case .cpuAndNeuralEngine: return "cpu-and-ane"
        case .cpuAndGPU: return "cpu-and-gpu"
        case .cpuOnly: return "cpu-only"
        @unknown default: return "unknown"
        }
    }

    /// Build the per-phrase ASR adapter for a resolved choice. Returns
    /// `nil` for `.skip` so the loop can short-circuit.
    private static func buildAsrLoop(_ choice: AsrChoice) async throws -> AsrLoop? {
        switch choice {
        case .skip:
            return nil
        case .parakeet:
            let asrModels = try await AsrModels.downloadAndLoad()
            let asr = AsrManager()
            try await asr.loadModels(asrModels)
            let layers = await asr.decoderLayerCount
            return AsrLoop(
                label: "parakeet-tdt",
                transcribeOne: { url in
                    var state = TdtDecoderState.make(decoderLayers: layers)
                    let r = try await asr.transcribe(url, decoderState: &state)
                    return r.text
                },
                cleanup: { await asr.cleanup() }
            )
        case .cohere(let modelDir, let language, let computeUnits):
            guard #available(macOS 14, iOS 17, *) else {
                throw NSError(
                    domain: "TtsBenchmark", code: 2,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Cohere ASR backend requires macOS 14+ / iOS 17+."
                    ])
            }
            logger.info(
                "Loading Cohere Transcribe (lang=\(language.englishName), "
                    + "compute=\(describeComputeUnits(computeUnits))) from \(modelDir.path)")
            let models = try await CoherePipeline.loadModels(
                encoderDir: modelDir,
                decoderDir: modelDir,
                vocabDir: modelDir,
                decoderVariant: .v2,
                computeUnits: computeUnits)
            let pipeline = CoherePipeline()
            let converter = AudioConverter()
            return AsrLoop(
                label: "cohere-transcribe-\(language.rawValue)",
                transcribeOne: { url in
                    let samples = try converter.resampleAudioFile(path: url.path)
                    let r = try await pipeline.transcribeLong(
                        audio: samples,
                        models: models,
                        language: language,
                        maxNewTokens: 108,
                        repetitionPenalty: 1.1,
                        noRepeatNgram: 3)
                    return r.text
                },
                cleanup: {}
            )
        }
    }

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudio tts-benchmark [options]

            Quantitative TTS benchmark — TTFT, cold/warm split, per-stage timings,
            peak RSS, WER + CER per category, configurable compute-unit preset.

            Backends:
              kokoro-ane    7-stage ANE pipeline (per-stage timings, per-stage CU)
              pocket-tts    Streaming flow-matching (multilingual)
              styletts2     LibriTTS iteration_3, zero-shot, requires --reference
              supertonic3   4-stage multilingual flow-matching (31 langs);
                            requires --voice-style <preset.json>

            Options:
              --backend <name>          See list above (default: kokoro-ane)
              --corpus <name>           MiniMax corpus name: minimax-<lang>
                                        (e.g. minimax-english, minimax-chinese,
                                        minimax-vietnamese — 24 languages total;
                                        see Documentation/TTS/MinimaxCorpus.md)
              --corpus-path <path>      Custom corpus file (overrides --corpus)
              --voice <name>            Voice id (KokoroAne/PocketTTS)
              --language <code>         PocketTTS lang pack code
              --compute-units <preset>  default | all-ane | cpu-and-gpu | cpu-only | ane-tail-gpu
                                        (kokoro-ane on M5/macOS 26.5 needs ane-tail-gpu; see #667)
              --output-json <path>      Write JSON report
              --audio-dir <path>        Keep generated WAVs under this dir
              --skip-asr                Skip ASR roundtrip (no WER/CER)
              --asr-backend <name>      ASR engine for the WER/CER pass:
                                          parakeet  English-only (default for en)
                                          cohere    Multilingual (default for non-en)
                                          none      Same as --skip-asr
              --cohere-model-dir <path> Path to a directory containing Cohere
                                        Transcribe encoder/decoder/vocab.json.
                                        Required when --asr-backend cohere is
                                        active (auto-download is not wired —
                                        vocab.json lives at the repo root, not
                                        under /q8). Default: cache at
                                        ~/Library/Application Support/FluidAudio/
                                        Models/cohere-transcribe/q8
              --asr-language <code>     Override Cohere language code (default:
                                        inferred from corpus name). One of:
                                        en, zh, ja, ko, vi, fr, de, es, it, pt,
                                        nl, pl, el, ar
              --cohere-compute-units <p>  Cohere ASR compute mapping:
                                        all (default; CoreML decides) |
                                        cpu-and-gpu | cpu-only | all-ane.
                                        Use cpu-and-gpu when q8 ANE compile
                                        fails (`MILCompilerForANE error: …`)
                                        — avoids the multi-minute fallback
                                        compile on first call.
              --reference <path>        StyleTTS2 speaker-reference audio
                                        (required for --backend styletts2;
                                        any sample rate / channel layout —
                                        resampled to 24 kHz mono internally)
              --variant <name>          Kokoro ANE variant: english (default) or
                                        mandarin (aliases: zh, chinese)
              --voice-style <path>      Supertonic-3 voice style JSON
                                        (required for --backend supertonic3;
                                        e.g. M1.json shipped under
                                        FluidInference/supertonic-3-coreml).
              --total-steps <n>         Supertonic-3 denoising steps
                                        (default 8 — matches reference CLI).
              --speed <f>               Supertonic-3 speech-rate multiplier
                                        (default 1.05; divides duration).
              --help, -h                Show this help

            Examples:
              fluidaudio tts-benchmark --backend kokoro-ane --output-json bench.json
              fluidaudio tts-benchmark --backend kokoro-ane --variant mandarin \\
                  --voice zf_001 --corpus minimax-chinese --skip-asr
              fluidaudio tts-benchmark --backend pocket-tts --corpus minimax-german --language german
              fluidaudio tts-benchmark --backend styletts2 --reference speaker.wav
              fluidaudio tts-benchmark --backend supertonic3 \\
                  --voice-style M1.json --corpus minimax-english

            Notes:
              For Chinese (zh) and Japanese (ja), WER is meaningless because
              WERCalculator splits on whitespace; trust the CER column instead.
              The summary banner prints an explicit reminder for these langs.
            """
        )
    }
}
#endif
