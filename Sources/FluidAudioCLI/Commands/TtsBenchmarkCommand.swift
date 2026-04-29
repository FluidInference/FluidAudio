#if os(macOS)
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
///   kokoro        — single-graph CPU+GPU (chunk-level only)
///   pocket-tts    — streaming flow-matching (no per-stage timings)
///   magpie        — encoder-decoder + NanoCodec (6-stage timings, slow)
///   cosyvoice3    — Mandarin LLM-based (Mandarin corpus only, no WER)
///   styletts2     — diffusion + HiFi-GAN (one-shot, requires --voice ref_s.bin)
///
/// Usage:
///   fluidaudio tts-benchmark --backend kokoro-ane \
///       --corpus minimax-english \
///       --voice af_heart \
///       --compute-units default \
///       --output-json bench.json
///
/// Corpora ship in `Benchmarks/tts/corpus/minimax/<lang>.txt` —
/// the MiniMax Multilingual TTS Test Set (CC-BY-SA-4.0,
/// 24 languages × 100 phrases). See
/// `Benchmarks/tts/corpus/minimax/README.md` for attribution and
/// the per-backend ↔ language coverage matrix. Reference with
/// `--corpus minimax-<lang>` (e.g. `minimax-english`,
/// `minimax-chinese`, `minimax-vietnamese`, …).
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

    public static func run(arguments: [String]) async {
        var backendName = "kokoro-ane"
        var corpusName: String?
        var corpusPath: String?
        var voice: String?
        var speakerName: String?
        var languageName: String?
        var computeUnitsName = "default"
        var outputJson: String?
        var audioDir: String?
        var skipAsr = false

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
            case "--speaker":
                if i + 1 < arguments.count {
                    speakerName = arguments[i + 1]
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
                "Unknown --compute-units value: \(computeUnitsName). Expected default | all-ane | cpu-and-gpu | cpu-only."
            )
            exit(1)
        }

        // CosyVoice3 has no English ASR pairing in this slice — force --skip-asr.
        var effectiveSkipAsr = skipAsr
        if backend == .cosyVoice3 && !skipAsr {
            logger.info("CosyVoice3: forcing --skip-asr (no Mandarin ASR wired in this slice)")
            effectiveSkipAsr = true
        }

        do {
            switch backend {
            case .kokoroAne:
                try await runKokoroAne(
                    phrases: phrases, corpusLabel: corpusLabel,
                    voice: voice ?? KokoroAneConstants.defaultVoice,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    skipAsr: effectiveSkipAsr)
            case .kokoro:
                try await runKokoro(
                    phrases: phrases, corpusLabel: corpusLabel,
                    voice: voice ?? TtsConstants.recommendedVoice,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    skipAsr: effectiveSkipAsr)
            case .pocketTts:
                try await runPocketTts(
                    phrases: phrases, corpusLabel: corpusLabel,
                    voice: voice ?? PocketTtsConstants.defaultVoice,
                    languageName: languageName,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    skipAsr: effectiveSkipAsr)
            case .magpie:
                try await runMagpie(
                    phrases: phrases, corpusLabel: corpusLabel,
                    speakerName: speakerName, languageName: languageName,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    skipAsr: effectiveSkipAsr)
            case .cosyVoice3:
                try await runCosyVoice3(
                    phrases: phrases, corpusLabel: corpusLabel,
                    voice: voice,
                    preset: preset, outputJson: outputJson, audioDir: audioDir)
            case .styleTts2:
                try await runStyleTts2(
                    phrases: phrases, corpusLabel: corpusLabel,
                    voicePath: voice,
                    preset: preset, outputJson: outputJson, audioDir: audioDir,
                    skipAsr: effectiveSkipAsr)
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
        voice: String,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        skipAsr: Bool
    ) async throws {
        let units = KokoroAneComputeUnits(preset: preset)
        let manager = KokoroAneManager(defaultVoice: voice, computeUnits: units)

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
            skipAsr: skipAsr,
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

    // MARK: - Kokoro driver (single-graph)

    private static func runKokoro(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        voice: String,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        skipAsr: Bool
    ) async throws {
        let units = preset.uniformUnits ?? .all
        let manager = KokoroTtsManager(defaultVoice: voice, computeUnits: units)

        let coldStart = Date()
        try await manager.initialize(preloadVoices: [voice])
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (initialize): %.2fs", coldStartS))

        let firstStart = Date()
        _ = try await manager.synthesizeDetailed(text: "Initialization warm-up.", voice: voice)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(String(format: "First synth: %.0f ms", firstSynthMs))

        try await runPhraseLoop(
            backendId: "kokoro",
            voiceLabel: voice,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            skipAsr: skipAsr,
            extraSummary: ["voice": voice]
        ) { text in
            let t0 = Date()
            let result = try await manager.synthesizeDetailed(text: text, voice: voice)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            let samples = result.chunks.flatMap { $0.samples }
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: samples,
                sampleRate: 24000,
                stageMs: [:],
                extraFields: [
                    "chunk_count": result.chunks.count,
                    "wav_bytes": result.audio.count,
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
        skipAsr: Bool
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
        let firstResult = try await manager.synthesizeDetailed(
            text: "Initialization warm-up.", voice: voice)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(
            String(
                format: "First synth: %.0f ms (frames=%d)",
                firstSynthMs, firstResult.frameCount))

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
            skipAsr: skipAsr,
            extraSummary: ["voice": voice, "language": language.rawValue]
        ) { text in
            let t0 = Date()
            let result = try await manager.synthesizeDetailed(text: text, voice: voice)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: result.samples,
                sampleRate: PocketTtsConstants.audioSampleRate,
                stageMs: [:],
                extraFields: [
                    "frame_count": result.frameCount,
                    "eos_step": result.eosStep ?? -1,
                ]
            )
        }
    }

    // MARK: - Magpie driver

    private static func runMagpie(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        speakerName: String?,
        languageName: String?,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        skipAsr: Bool
    ) async throws {
        let units = preset.uniformUnits ?? .cpuAndNeuralEngine
        let language = parseMagpieLanguage(languageName)
        let speaker = parseMagpieSpeaker(speakerName)
        logger.info("Magpie speaker=\(speaker.displayName) language=\(language.rawValue)")

        let manager = MagpieTtsManager(
            computeUnits: units, preferredLanguages: [language])

        let coldStart = Date()
        try await manager.initialize()
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (initialize): %.2fs", coldStartS))

        let firstStart = Date()
        _ = try await manager.synthesize(
            text: "Initialization warm-up.", speaker: speaker, language: language)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(String(format: "First synth: %.0f ms", firstSynthMs))

        try await runPhraseLoop(
            backendId: "magpie",
            voiceLabel: speaker.displayName,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            skipAsr: skipAsr,
            extraSummary: [
                "speaker": speaker.displayName, "language": language.rawValue,
            ]
        ) { text in
            let t0 = Date()
            let result = try await manager.synthesize(
                text: text, speaker: speaker, language: language)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: result.samples,
                sampleRate: result.sampleRate,
                stageMs: [
                    "text_encoder": result.timings.textEncoderSeconds * 1000,
                    "prefill": result.timings.prefillSeconds * 1000,
                    "ar_loop": result.timings.arLoopSeconds * 1000,
                    "decoder_step": result.timings.decoderStepSeconds * 1000,
                    "sampler": result.timings.samplerSeconds * 1000,
                    "nanocodec": result.timings.nanocodecSeconds * 1000,
                ],
                extraFields: [
                    "code_count": result.codeCount,
                    "finished_on_eos": result.finishedOnEos,
                ]
            )
        }
    }

    // MARK: - CosyVoice3 driver

    private static func runCosyVoice3(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        voice: String?,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?
    ) async throws {
        let units = preset.uniformUnits ?? .cpuAndNeuralEngine
        let voiceId = voice ?? "cosyvoice3-default-zh"

        let coldStart = Date()
        let manager = try await CosyVoice3TtsManager.downloadAndCreate(
            cacheDirectory: nil, includeDefaultVoice: true, computeUnits: units)
        try await manager.initialize()
        let promptAssets = try await manager.loadVoice(voiceId)
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (download+init+voice): %.2fs", coldStartS))

        let firstStart = Date()
        _ = try await manager.synthesize(text: "你好", promptAssets: promptAssets)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(String(format: "First synth: %.0f ms", firstSynthMs))

        try await runPhraseLoop(
            backendId: "cosyvoice3",
            voiceLabel: voiceId,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            skipAsr: true,  // forced upstream
            extraSummary: ["voice": voiceId]
        ) { text in
            let t0 = Date()
            let result = try await manager.synthesize(text: text, promptAssets: promptAssets)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: result.samples,
                sampleRate: result.sampleRate,
                stageMs: [:],
                extraFields: [
                    "generated_token_count": result.generatedTokenCount,
                    "decoded_token_count": result.decodedTokens.count,
                ]
            )
        }
    }

    // MARK: - StyleTTS2 driver

    private static func runStyleTts2(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        voicePath: String?,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        skipAsr: Bool
    ) async throws {
        guard let voicePath, !voicePath.isEmpty else {
            logger.error(
                "StyleTTS2 requires --voice <path/to/ref_s.bin> "
                    + "(256 fp32 LE blob from mobius-styletts2/scripts/06_dump_ref_s.py)")
            exit(1)
        }
        let voiceURL = resolveURL(voicePath, isDirectory: false)
        let voiceLabel = voiceURL.deletingPathExtension().lastPathComponent

        // StyleTTS2 doesn't expose a compute-units knob today; --compute-units
        // is accepted for parity with other backends but only labels the run.
        let manager = StyleTTS2Manager()

        let coldStart = Date()
        try await manager.initialize()
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(String(format: "Cold start (initialize): %.2fs", coldStartS))

        let firstStart = Date()
        _ = try await manager.synthesizeSamples(
            text: "Initialization warm-up.", voiceStyleURL: voiceURL, randomSeed: 42)
        let firstSynthMs = Date().timeIntervalSince(firstStart) * 1000
        logger.info(String(format: "First synth: %.0f ms", firstSynthMs))

        try await runPhraseLoop(
            backendId: "styletts2",
            voiceLabel: voiceLabel,
            corpusLabel: corpusLabel,
            phrases: phrases,
            preset: preset,
            coldStartS: coldStartS,
            firstSynthMs: firstSynthMs,
            outputJson: outputJson,
            audioDir: audioDir,
            skipAsr: skipAsr,
            extraSummary: ["voice": voiceLabel]
        ) { text in
            let t0 = Date()
            let result = try await manager.synthesizeSamples(
                text: text, voiceStyleURL: voiceURL, randomSeed: 42)
            let synthMs = Date().timeIntervalSince(t0) * 1000
            return BackendPhraseSample(
                synthMs: synthMs,
                ttftMs: synthMs,
                samples: result.samples,
                sampleRate: result.sampleRate,
                stageMs: [:],
                extraFields: [:]
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
        skipAsr: Bool,
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

        // Optional ASR.
        var asr: AsrManager? = nil
        var decoderLayers = 0
        if !skipAsr {
            let asrModels = try await AsrModels.downloadAndLoad()
            let asrInstance = AsrManager()
            try await asrInstance.loadModels(asrModels)
            decoderLayers = await asrInstance.decoderLayerCount
            asr = asrInstance
        }

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
            if let asr {
                let asr0 = Date()
                var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                let transcription = try await asr.transcribe(
                    wavURL, decoderState: &decoderState)
                asrMs = Date().timeIntervalSince(asr0) * 1000
                hypothesis = transcription.text
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

        if let asr {
            await asr.cleanup()
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
        if !skipAsr {
            let werVals = perPhrase.compactMap { $0["wer"] as? Double }
            let cerVals = perPhrase.compactMap { $0["cer"] as? Double }
            let macroWer =
                werVals.isEmpty ? 0 : werVals.reduce(0, +) / Double(werVals.count)
            let macroCer =
                cerVals.isEmpty ? 0 : cerVals.reduce(0, +) / Double(cerVals.count)
            logger.info(String(format: "  macro WER:      %.2f%%", macroWer * 100))
            logger.info(String(format: "  macro CER:      %.2f%%", macroCer * 100))
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
                "asr_skipped": skipAsr,
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
        case kokoro
        case pocketTts
        case magpie
        case cosyVoice3
        case styleTts2

        var defaultCorpus: String {
            switch self {
            case .cosyVoice3: return "minimax-chinese"
            default: return "minimax-english"
            }
        }
    }

    private static func parseBackend(_ name: String) -> Backend {
        switch name.lowercased() {
        case "kokoro-ane", "kokoroane", "kokoro_ane", "lai":
            return .kokoroAne
        case "kokoro":
            return .kokoro
        case "pocket-tts", "pockettts", "pocket":
            return .pocketTts
        case "magpie":
            return .magpie
        case "cosyvoice3", "cosyvoice", "cosy":
            return .cosyVoice3
        case "styletts2", "style-tts2", "styletts", "style":
            return .styleTts2
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

    private static func parseMagpieLanguage(_ name: String?) -> MagpieLanguage {
        guard let name, let l = MagpieLanguage(rawValue: name.lowercased()) else {
            return .english
        }
        return l
    }

    private static func parseMagpieSpeaker(_ name: String?) -> MagpieSpeaker {
        switch name?.lowercased() {
        case "sofia": return .sofia
        case "aria": return .aria
        case "jason": return .jason
        case "leo": return .leo
        case "john", nil, "": return .john
        default: return .john
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

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudio tts-benchmark [options]

            Quantitative TTS benchmark — TTFT, cold/warm split, per-stage timings,
            peak RSS, WER + CER per category, configurable compute-unit preset.

            Backends:
              kokoro-ane    7-stage ANE pipeline (per-stage timings, per-stage CU)
              kokoro        Single-graph CPU+GPU
              pocket-tts    Streaming flow-matching (multilingual)
              magpie        Encoder-decoder + NanoCodec (per-stage, slow)
              cosyvoice3    Mandarin LLM-based (forces --skip-asr)

            Options:
              --backend <name>          See list above (default: kokoro-ane)
              --corpus <name>           MiniMax corpus name: minimax-<lang>
                                        (e.g. minimax-english, minimax-chinese,
                                        minimax-vietnamese — 24 languages total;
                                        see Benchmarks/tts/corpus/minimax/README.md)
              --corpus-path <path>      Custom corpus file (overrides --corpus)
              --voice <name>            Voice id (Kokoro/PocketTTS/CosyVoice3)
              --speaker <name>          Magpie speaker: john|sofia|aria|jason|leo
              --language <code>         PocketTTS lang pack or Magpie language code
              --compute-units <preset>  default | all-ane | cpu-and-gpu | cpu-only
              --output-json <path>      Write JSON report
              --audio-dir <path>        Keep generated WAVs under this dir
              --skip-asr                Skip Parakeet roundtrip (no WER/CER)
              --help, -h                Show this help

            Examples:
              fluidaudio tts-benchmark --backend kokoro-ane --output-json bench.json
              fluidaudio tts-benchmark --backend kokoro --corpus minimax-english
              fluidaudio tts-benchmark --backend pocket-tts --corpus minimax-german --language german
              fluidaudio tts-benchmark --backend magpie --speaker sofia --language en
              fluidaudio tts-benchmark --backend cosyvoice3 --corpus minimax-chinese
            """
        )
    }
}
#endif
