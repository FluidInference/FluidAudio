import CoreML
import FluidAudio
import Foundation

public struct TTS {

    private static let logger = AppLogger(category: "TTSCommand")
    private static let artifactsDirectoryName = "fluidaudio_cli"

    private static func ensureArtifactsRoot() throws -> URL {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        let root = cwd.appendingPathComponent(artifactsDirectoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static func resolveOutputURL(
        _ suppliedPath: String,
        artifactsRoot: URL,
        expectsDirectory: Bool
    ) -> URL {
        let expanded = (suppliedPath as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded, isDirectory: expectsDirectory)
        }
        return artifactsRoot.appendingPathComponent(expanded, isDirectory: expectsDirectory)
    }

    private static func resolveInputURL(_ suppliedPath: String) -> URL {
        let expanded = (suppliedPath as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded)
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return cwd.appendingPathComponent(expanded)
    }

    /// Mandarin lexicon loader for KokoroAne `--variant zh`. See
    /// ``MandarinCustomLexicon/parse(_:)`` for the line spec.
    private static func loadMandarinLexicon(from path: String?) throws -> MandarinCustomLexicon? {
        guard let path = path else { return nil }
        let url = resolveLexiconURL(path)
        let lexicon = try MandarinCustomLexicon.load(from: url)
        logger.info(
            "Loaded Mandarin custom lexicon with \(lexicon.count) entries from \(url.path)")
        return lexicon
    }

    private static func resolveLexiconURL(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded)
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return cwd.appendingPathComponent(expanded)
    }

    public static func run(arguments: [String]) async {
        var output = "output.wav"
        var voice = TtsConstants.recommendedVoice
        var metricsPath: String? = nil
        // KokoroAne language variant — only consulted when backend == .kokoroAne.
        // Parsed from the `--variant` flag (en/english/zh/mandarin).
        var kokoroAneVariant: KokoroAneVariant = .english
        var lexiconPath: String? = nil
        var text: String? = nil
        // KokoroAne: treat the positional/`--text` value as a pre-computed
        // phoneme string (IPA for en/ja, Bopomofo+tone for zh) and bypass
        // G2P via synthesizeFromPhonemes. Required for `.japanese`, which
        // ships no text frontend (issue #698).
        var treatAsPhonemes = false
        var deEss = true
        var backend: TtsBackend = .kokoroAne
        var cloneVoicePath: String? = nil
        var voiceFilePath: String? = nil
        var saveVoicePath: String? = nil
        var pocketLanguage: PocketTtsLanguage = .english
        var pocketPlacement: PocketTtsModelPlacement = .gpu
        // PocketTTS deterministic-seed mode (uses session API for fixed RNG).
        var pocketSeed: UInt64? = nil
        // StyleTTS2 zero-shot args.
        var styletts2ReferencePath: String? = nil
        var styletts2Seed: UInt64 = 42
        var cpuOnly: Bool = false
        var styletts2Alpha: Float = StyleTTS2Constants.defaultAlpha
        var styletts2Beta: Float = StyleTTS2Constants.defaultBeta
        // Optional pre-computed IPA passed via `--ipa "…"`. Bypasses
        // CharsiuG2P entirely (the espeak-parity escape hatch).
        var styletts2Ipa: String? = nil
        // Supertonic-3 args.
        var supertonicLanguage: String = "en"
        var supertonicVoiceStylePath: String? = nil
        var supertonicTotalSteps: Int = Supertonic3Constants.defaultTotalSteps
        var supertonicSpeed: Float = Supertonic3Constants.defaultSpeed
        // VectorEstimator build: fp16 | int8/int6/int4 (ANE-bucketed) |
        // dyn-int8/dyn-int6/dyn-int4 (dynamic CPU/GPU). Default fp16.
        var supertonicVE: Supertonic3VectorEstimator = .aneBucketed(.int4)

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
            case "--metrics":
                if i + 1 < arguments.count {
                    metricsPath = arguments[i + 1]
                    i += 1
                }
            case "--phonemes":
                treatAsPhonemes = true
            case "--variant", "--model-variant":
                if i + 1 < arguments.count {
                    let value = arguments[i + 1].lowercased()
                    switch value {
                    case "en", "english":
                        kokoroAneVariant = .english
                    case "zh", "mandarin", "zh-cn", "zh_cn":
                        kokoroAneVariant = .mandarin
                    case "ja", "japanese", "jp":
                        kokoroAneVariant = .japanese
                    default:
                        logger.warning("Unknown variant preference '\(arguments[i + 1])'; ignoring")
                    }
                    i += 1
                }
            case "--lexicon", "-l":
                if i + 1 < arguments.count {
                    lexiconPath = arguments[i + 1]
                    i += 1
                }
            case "--backend":
                if i + 1 < arguments.count {
                    let value = arguments[i + 1].lowercased()
                    switch value {
                    case "pocket", "pockettts":
                        backend = .pocketTts
                    case "kokoro-ane", "kokoroane", "kokoro", "lai":
                        backend = .kokoroAne
                    case "styletts2", "style-tts2", "stts2":
                        backend = .styletts2
                    case "supertonic3", "supertonic-3", "sup3":
                        backend = .supertonic3
                    default:
                        logger.warning("Unknown backend '\(arguments[i + 1])'; using kokoro-ane")
                    }
                    i += 1
                }
            case "--lang":
                if i + 1 < arguments.count {
                    supertonicLanguage = arguments[i + 1].lowercased()
                    i += 1
                }
            case "--voice-style":
                if i + 1 < arguments.count {
                    supertonicVoiceStylePath = arguments[i + 1]
                    i += 1
                }
            case "--ve-variant", "--vector-estimator":
                if i + 1 < arguments.count {
                    let raw = arguments[i + 1].lowercased()
                    if let v = Self.parseSupertonicVE(raw) {
                        supertonicVE = v
                    } else {
                        logger.warning(
                            "Unknown --ve-variant '\(raw)'; using fp16. "
                                + "Valid: fp16, int8/int6/int4 (ANE), dyn-int8/dyn-int6/dyn-int4.")
                    }
                    i += 1
                }
            case "--total-steps":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    supertonicTotalSteps = v
                    i += 1
                }
            case "--speed":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    supertonicSpeed = v
                    i += 1
                }
            case "--alpha":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    styletts2Alpha = v
                    i += 1
                }
            case "--beta":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    styletts2Beta = v
                    i += 1
                }
            case "--ipa":
                if i + 1 < arguments.count {
                    styletts2Ipa = arguments[i + 1]
                    i += 1
                }
            case "--reference":
                if i + 1 < arguments.count {
                    styletts2ReferencePath = arguments[i + 1]
                    i += 1
                }
            case "--seed":
                if i + 1 < arguments.count, let parsed = UInt64(arguments[i + 1]) {
                    styletts2Seed = parsed
                    pocketSeed = parsed
                    i += 1
                }
            case "--cpu-only":
                cpuOnly = true
            case "--text":
                if i + 1 < arguments.count {
                    text = arguments[i + 1]
                    i += 1
                }
            case "--auto-download":
                // No-op: downloads are always ensured by the CLI. Accepted
                // for backward compatibility with documented examples.
                ()
            case "--no-deess":
                deEss = false
            case "--clone-voice":
                if i + 1 < arguments.count {
                    cloneVoicePath = arguments[i + 1]
                    i += 1
                }
            case "--voice-file":
                if i + 1 < arguments.count {
                    voiceFilePath = arguments[i + 1]
                    i += 1
                }
            case "--save-voice":
                if i + 1 < arguments.count {
                    saveVoicePath = arguments[i + 1]
                    i += 1
                }
            case "--placement":
                if i + 1 < arguments.count {
                    let raw = arguments[i + 1].lowercased()
                    if let parsed = PocketTtsModelPlacement(rawValue: raw) {
                        pocketPlacement = parsed
                    } else {
                        logger.error(
                            "Unknown PocketTTS placement '\(arguments[i + 1])'. Supported: gpu, ane, ane-state"
                        )
                        return
                    }
                    i += 1
                }
            case "--language":
                if i + 1 < arguments.count {
                    let raw = arguments[i + 1].lowercased()
                    if let parsed = PocketTtsLanguage(rawValue: raw) {
                        pocketLanguage = parsed
                    } else {
                        let supported = PocketTtsLanguage.allCases
                            .map { $0.rawValue }
                            .joined(separator: ", ")
                        logger.error(
                            "Unknown PocketTTS language '\(arguments[i + 1])'. Supported: \(supported)"
                        )
                        return
                    }
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

        guard let text = text else {
            printUsage()
            return
        }

        switch backend {
        case .pocketTts:
            await runPocketTts(
                text: text, output: output, voice: voice, deEss: deEss,
                metricsPath: metricsPath, cloneVoicePath: cloneVoicePath,
                voiceFilePath: voiceFilePath, saveVoicePath: saveVoicePath,
                language: pocketLanguage, seed: pocketSeed,
                placement: pocketPlacement)
        case .kokoroAne:
            await runKokoroAne(
                text: text, output: output, voice: voice, metricsPath: metricsPath,
                variant: kokoroAneVariant, lexiconPath: lexiconPath,
                treatAsPhonemes: treatAsPhonemes)
        case .styletts2:
            await runStyleTTS2(
                text: text, ipa: styletts2Ipa,
                referencePath: styletts2ReferencePath,
                output: output,
                alpha: styletts2Alpha, beta: styletts2Beta,
                seed: styletts2Seed,
                metricsPath: metricsPath,
                cpuOnly: cpuOnly)
        case .supertonic3:
            await runSupertonic3(
                text: text, output: output, language: supertonicLanguage,
                voiceStylePath: supertonicVoiceStylePath, voiceName: voice,
                totalSteps: supertonicTotalSteps, speed: supertonicSpeed,
                vectorEstimator: supertonicVE,
                metricsPath: metricsPath, cpuOnly: cpuOnly)
        }
    }

    /// Run PocketTTS in deterministic-seed mode through the session API,
    /// applying the same de-essing post-processing as the non-seed path.
    private static func runPocketSeededSynthesis(
        manager: PocketTtsManager,
        text: String,
        voice: String,
        voiceData: PocketTtsVoiceData?,
        seed: UInt64,
        deEss: Bool
    ) async throws -> Data {
        logger.info("PocketTTS deterministic mode: seed=\(seed)")
        let session = try await makePocketSeededSession(
            manager: manager, voice: voice, voiceData: voiceData, seed: seed)
        session.enqueue(text)
        session.finish()
        var allSamples: [Float] = []
        for try await frame in session.frames {
            allSamples.append(contentsOf: frame.samples)
        }
        if deEss {
            AudioPostProcessor.applyTtsPostProcessing(
                &allSamples,
                sampleRate: Float(PocketTtsConstants.audioSampleRate),
                deEssAmount: -3.0,
                smoothing: false)
        }
        return try AudioWAV.data(
            from: allSamples,
            sampleRate: Double(PocketTtsConstants.audioSampleRate))
    }

    /// Pick the right `makeSession` overload based on whether a custom
    /// `PocketTtsVoiceData` was supplied (cloned/loaded voice) or we should
    /// fall back to a named voice from the language pack.
    private static func makePocketSeededSession(
        manager: PocketTtsManager,
        voice: String,
        voiceData: PocketTtsVoiceData?,
        seed: UInt64
    ) async throws -> PocketTtsSession {
        if let voiceData = voiceData {
            return try await manager.makeSession(
                voiceData: voiceData,
                temperature: PocketTtsConstants.temperature,
                seed: seed)
        }
        return try await manager.makeSession(
            voice: voice,
            temperature: PocketTtsConstants.temperature,
            seed: seed)
    }

    private static func runPocketTts(
        text: String, output: String, voice: String, deEss: Bool,
        metricsPath: String?, cloneVoicePath: String?,
        voiceFilePath: String?, saveVoicePath: String?,
        language: PocketTtsLanguage,
        seed: UInt64? = nil,
        placement: PocketTtsModelPlacement = .gpu
    ) async {
        do {
            let tStart = Date()
            let pocketVoice =
                voice == TtsConstants.recommendedVoice
                ? PocketTtsConstants.defaultVoice : voice
            let manager = PocketTtsManager(
                defaultVoice: pocketVoice, language: language, placement: placement)
            logger.info(
                "PocketTTS language: \(language.rawValue), placement: \(placement.rawValue)")

            let tLoad0 = Date()
            try await manager.initialize()
            let tLoad1 = Date()

            // Handle voice cloning options
            var voiceData: PocketTtsVoiceData? = nil

            if let cloneVoicePath = cloneVoicePath {
                let cloneURL = resolveInputURL(cloneVoicePath)
                logger.info("Cloning voice from: \(cloneURL.path)")
                voiceData = try await manager.cloneVoice(from: cloneURL)
                logger.info("Voice cloned successfully")

                if let saveVoicePath = saveVoicePath {
                    let saveURL = resolveInputURL(saveVoicePath)
                    try manager.saveClonedVoice(voiceData!, to: saveURL)
                    logger.info("Saved cloned voice to: \(saveURL.path)")
                }
            } else if let voiceFilePath = voiceFilePath {
                let voiceURL = resolveInputURL(voiceFilePath)
                logger.info("Loading voice from: \(voiceURL.path)")
                voiceData = try manager.loadClonedVoice(from: voiceURL)
                logger.info("Voice loaded successfully")
            }

            let tSynth0 = Date()
            let wav: Data
            if let seed = seed {
                wav = try await runPocketSeededSynthesis(
                    manager: manager,
                    text: text,
                    voice: pocketVoice,
                    voiceData: voiceData,
                    seed: seed,
                    deEss: deEss)
            } else if let voiceData = voiceData {
                wav = try await manager.synthesize(
                    text: text, voiceData: voiceData, deEss: deEss)
            } else {
                wav = try await manager.synthesize(
                    text: text, voice: pocketVoice, deEss: deEss)
            }
            let tSynth1 = Date()

            let outURL = resolveInputURL(output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try wav.write(to: outURL)

            let loadS = tLoad1.timeIntervalSince(tLoad0)
            let synthS = tSynth1.timeIntervalSince(tSynth0)
            let totalS = tSynth1.timeIntervalSince(tStart)
            let sampleRate = Double(PocketTtsConstants.audioSampleRate)
            let payload = max(0, wav.count - 44)
            let audioSecs = Double(payload) / (sampleRate * 2.0)
            let rtfx = synthS > 0 ? audioSecs / synthS : 0

            logger.info("PocketTTS synthesis complete")
            logger.info("  Load: \(String(format: "%.3f", loadS))s")
            logger.info("  Synthesis: \(String(format: "%.3f", synthS))s")
            logger.info("  Audio: \(String(format: "%.3f", audioSecs))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
            logger.info("  Total: \(String(format: "%.3f", totalS))s")
            logger.info("  Output: \(outURL.path)")

            // ASR round-trip evaluation
            if metricsPath != nil {
                logger.info("--- Running ASR for TTS→STT evaluation ---")
                var asrHypothesis: String? = nil
                var werValue: Double? = nil

                do {
                    let asrModels = try await AsrModels.downloadAndLoad()
                    let asr = AsrManager()
                    try await asr.loadModels(asrModels)

                    var decoderState = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
                    let transcription = try await asr.transcribe(outURL, decoderState: &decoderState)
                    asrHypothesis = transcription.text

                    let werMetrics = WERCalculator.calculateWERMetrics(
                        hypothesis: transcription.text, reference: text)
                    werValue = werMetrics.wer

                    logger.info("Reference:  \(text)")
                    logger.info("Hypothesis: \(transcription.text)")
                    logger.info(String(format: "WER: %.1f%%", werValue! * 100))

                    await asr.cleanup()
                } catch {
                    logger.warning("ASR evaluation failed: \(error.localizedDescription)")
                }

                if let metricsPath {
                    var metricsDict: [String: Any] = [
                        "backend": "pockettts",
                        "text": text,
                        "voice": pocketVoice,
                        "output": outURL.path,
                        "model_load_time_s": loadS,
                        "inference_time_s": synthS,
                        "audio_duration_s": audioSecs,
                        "realtime_speed": rtfx,
                        "total_time_s": totalS,
                    ]
                    if let asrHypothesis {
                        metricsDict["asr_hypothesis"] = asrHypothesis
                    }
                    if let werValue {
                        metricsDict["wer"] = werValue
                    }

                    let artifactsRoot = try ensureArtifactsRoot()
                    let mURL = resolveOutputURL(
                        metricsPath, artifactsRoot: artifactsRoot, expectsDirectory: false)
                    try FileManager.default.createDirectory(
                        at: mURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                    let json = try JSONSerialization.data(
                        withJSONObject: metricsDict, options: [.prettyPrinted])
                    try json.write(to: mURL)
                    logger.info("Metrics saved: \(mURL.path)")
                }
            }
        } catch {
            logger.error("PocketTTS Error: \(error)")
            print("PocketTTS failed: \(error)")
            exit(1)
        }
    }

    private static func runKokoroAne(
        text: String, output: String, voice: String, metricsPath: String?,
        variant: KokoroAneVariant, lexiconPath: String?, treatAsPhonemes: Bool
    ) async {
        do {
            let tStart = Date()
            // When the caller didn't pass `--voice`, pick the variant default
            // (af_heart for English, zf_001 for Mandarin) instead of the
            // shared TtsConstants.recommendedVoice (which is af_heart and
            // wouldn't exist in the Mandarin bundle).
            let resolvedVoice =
                voice == TtsConstants.recommendedVoice
                ? variant.defaultVoice : voice
            let manager = KokoroAneManager(
                variant: variant, defaultVoice: resolvedVoice)

            // --lexicon is Mandarin-only. For English, log + ignore so users
            // aren't silently surprised by a flag with no effect.
            if let lexiconPath {
                switch variant {
                case .mandarin:
                    if let lex = try loadMandarinLexicon(from: lexiconPath) {
                        await manager.setMandarinCustomLexicon(lex)
                    }
                case .english, .japanese:
                    logger.warning(
                        "--lexicon ignored: only the KokoroAne Mandarin variant "
                            + "supports a custom lexicon.")
                }
            }

            let tLoad0 = Date()
            try await manager.initialize()
            let tLoad1 = Date()

            let tSynth0 = Date()
            // synthesizeDetailed handles English (G2PModel) and Mandarin
            // (MandarinG2P, with pass-through for pre-computed Bopomofo).
            // With --phonemes, or for Japanese (no text frontend), bypass
            // G2P and feed the input as a pre-computed phoneme string.
            let detailed: KokoroAneSynthesisResult
            if treatAsPhonemes || variant == .japanese {
                detailed = try await manager.synthesizeFromPhonemesDetailed(
                    text, voice: resolvedVoice, speed: 1.0)
            } else {
                detailed = try await manager.synthesizeDetailed(
                    text: text, voice: resolvedVoice, speed: 1.0)
            }
            let wav = try AudioWAV.data(
                from: detailed.samples,
                sampleRate: Double(detailed.sampleRate),
                normalize: variant != .japanese)
            let tSynth1 = Date()

            let outURL = resolveInputURL(output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try wav.write(to: outURL)

            let loadS = tLoad1.timeIntervalSince(tLoad0)
            let synthS = tSynth1.timeIntervalSince(tSynth0)
            let totalS = tSynth1.timeIntervalSince(tStart)
            let audioSecs = Double(detailed.samples.count) / Double(detailed.sampleRate)
            let rtfx = synthS > 0 ? audioSecs / synthS : 0

            logger.info("KokoroAne synthesis complete")
            logger.info("  Load: \(String(format: "%.3f", loadS))s")
            logger.info("  Synthesis: \(String(format: "%.3f", synthS))s")
            logger.info("  Audio: \(String(format: "%.3f", audioSecs))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
            logger.info("  Total: \(String(format: "%.3f", totalS))s")
            logger.info("  Output: \(outURL.path)")
            logger.info(
                "  Stages (ms): albert=\(String(format: "%.1f", detailed.timings.albert))"
                    + " postAlbert=\(String(format: "%.1f", detailed.timings.postAlbert))"
                    + " alignment=\(String(format: "%.1f", detailed.timings.alignment))"
                    + " prosody=\(String(format: "%.1f", detailed.timings.prosody))"
                    + " noise=\(String(format: "%.1f", detailed.timings.noise))"
                    + " vocoder=\(String(format: "%.1f", detailed.timings.vocoder))"
                    + " tail=\(String(format: "%.1f", detailed.timings.tail))"
                    + " total=\(String(format: "%.1f", detailed.timings.totalMs))"
            )

            // ASR round-trip evaluation (only when metrics requested).
            guard let metricsPath else { return }

            logger.info("--- Running ASR for TTS→STT evaluation ---")
            var asrHypothesis: String? = nil
            var werValue: Double? = nil

            do {
                let asrModels = try await AsrModels.downloadAndLoad()
                let asr = AsrManager()
                try await asr.loadModels(asrModels)

                var decoderState = TdtDecoderState.make(
                    decoderLayers: await asr.decoderLayerCount)
                let transcription = try await asr.transcribe(
                    outURL, decoderState: &decoderState)
                asrHypothesis = transcription.text

                let werMetrics = WERCalculator.calculateWERMetrics(
                    hypothesis: transcription.text, reference: text)
                werValue = werMetrics.wer

                logger.info("Reference:  \(text)")
                logger.info("Hypothesis: \(transcription.text)")
                logger.info(String(format: "WER: %.1f%%", werValue! * 100))

                await asr.cleanup()
            } catch {
                logger.warning("ASR evaluation failed: \(error.localizedDescription)")
            }

            var metricsDict: [String: Any] = [
                "backend": "kokoro-ane",
                "text": text,
                "voice": resolvedVoice,
                "output": outURL.path,
                "model_load_time_s": loadS,
                "inference_time_s": synthS,
                "audio_duration_s": audioSecs,
                "realtime_speed": rtfx,
                "total_time_s": totalS,
                "encoder_tokens": detailed.encoderTokens,
                "acoustic_frames": detailed.acousticFrames,
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
            ]
            if let asrHypothesis {
                metricsDict["asr_hypothesis"] = asrHypothesis
            }
            if let werValue {
                metricsDict["wer"] = werValue
            }

            let artifactsRoot = try ensureArtifactsRoot()
            let mURL = resolveOutputURL(
                metricsPath, artifactsRoot: artifactsRoot, expectsDirectory: false)
            try FileManager.default.createDirectory(
                at: mURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let json = try JSONSerialization.data(
                withJSONObject: metricsDict, options: [.prettyPrinted])
            try json.write(to: mURL)
            logger.info("Metrics saved: \(mURL.path)")
        } catch {
            logger.error("KokoroAne Error: \(error)")
            print("KokoroAne failed: \(error)")
            exit(1)
        }
    }

    /// Run StyleTTS2 LibriTTS zero-shot TTS. Requires a reference audio
    /// file (any sample rate / channel layout — resampled to 24 kHz mono
    /// internally) and either a text prompt or a pre-computed IPA string.
    private static func runStyleTTS2(
        text: String, ipa: String?,
        referencePath: String?,
        output: String,
        alpha: Float, beta: Float, seed: UInt64,
        metricsPath: String?, cpuOnly: Bool
    ) async {
        guard let referencePath else {
            logger.error(
                "styletts2 backend requires --reference <speaker-audio-file>")
            return
        }
        do {
            let tStart = Date()
            let computeUnits: MLComputeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
            let manager = StyleTTS2Manager(computeUnits: computeUnits)

            let tLoad0 = Date()
            try await manager.initialize()
            let tLoad1 = Date()

            let referenceURL = resolveInputURL(referencePath)
            logger.info("StyleTTS2 reference audio: \(referenceURL.path)")
            logger.info(
                "StyleTTS2 alpha=\(String(format: "%.2f", alpha)) "
                    + "beta=\(String(format: "%.2f", beta)) seed=\(seed)")

            let tSynth0 = Date()
            let samples: [Float]
            if let ipa, !ipa.isEmpty {
                logger.info("StyleTTS2 IPA override: \(ipa.prefix(60))…")
                samples = try await manager.synthesize(
                    ipa: ipa, referenceAudioURL: referenceURL,
                    alpha: alpha, beta: beta, noiseSeed: seed)
            } else {
                samples = try await manager.synthesize(
                    text: text, referenceAudioURL: referenceURL,
                    alpha: alpha, beta: beta, noiseSeed: seed)
            }
            let tSynth1 = Date()

            let outURL = resolveInputURL(output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let wav = try AudioWAV.data(
                from: samples,
                sampleRate: Double(StyleTTS2Constants.sampleRate))
            try wav.write(to: outURL)

            let loadS = tLoad1.timeIntervalSince(tLoad0)
            let synthS = tSynth1.timeIntervalSince(tSynth0)
            let totalS = tSynth1.timeIntervalSince(tStart)
            let audioSecs = Double(samples.count) / Double(StyleTTS2Constants.sampleRate)
            let rtfx = synthS > 0 ? audioSecs / synthS : 0

            logger.info("StyleTTS2 synthesis complete")
            logger.info("  Load: \(String(format: "%.3f", loadS))s")
            logger.info("  Synthesis: \(String(format: "%.3f", synthS))s")
            logger.info("  Audio: \(String(format: "%.3f", audioSecs))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
            logger.info("  Total: \(String(format: "%.3f", totalS))s")
            logger.info("  Output: \(outURL.path)")

            if let metricsPath {
                let metricsDict: [String: Any] = [
                    "backend": "styletts2",
                    "text": text,
                    "reference": referenceURL.path,
                    "alpha": Double(alpha),
                    "beta": Double(beta),
                    "seed": seed,
                    "output": outURL.path,
                    "model_load_time_s": loadS,
                    "inference_time_s": synthS,
                    "audio_duration_s": audioSecs,
                    "realtime_speed": rtfx,
                    "total_time_s": totalS,
                ]
                let artifactsRoot = try ensureArtifactsRoot()
                let mURL = resolveOutputURL(
                    metricsPath, artifactsRoot: artifactsRoot, expectsDirectory: false)
                try FileManager.default.createDirectory(
                    at: mURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                let json = try JSONSerialization.data(
                    withJSONObject: metricsDict, options: [.prettyPrinted])
                try json.write(to: mURL)
                logger.info("Metrics saved: \(mURL.path)")
            }
        } catch {
            logger.error("StyleTTS2 Error: \(error)")
            print("StyleTTS2 failed: \(error)")
            exit(1)
        }
    }

    /// Run Supertonic-3 multilingual TTS. Voice comes from a built-in style
    /// (`--voice F1`..`M5`, downloaded on demand, default `M1`) or an explicit
    /// `--voice-style <file.json>`, which overrides `--voice`.
    /// Map a `--ve-variant` token to a `Supertonic3VectorEstimator`.
    private static func parseSupertonicVE(_ raw: String) -> Supertonic3VectorEstimator? {
        func q(_ s: String) -> Supertonic3Quantization? { Supertonic3Quantization(rawValue: s) }
        switch raw {
        case "fp16", "fp16dynamic": return .fp16Dynamic
        case "default", "": return .aneBucketed(.int4)
        case "int8", "int6", "int4", "ane-int8", "ane-int6", "ane-int4":
            return q(String(raw.split(separator: "-").last!)).map { .aneBucketed($0) }
        case "dyn-int8", "dyn-int6", "dyn-int4", "dynamic-int8", "dynamic-int6", "dynamic-int4":
            return q("int" + String(raw.suffix(1))).map { .dynamic($0) }
        default: return nil
        }
    }

    private static func runSupertonic3(
        text: String, output: String, language: String,
        voiceStylePath: String?, voiceName: String,
        totalSteps: Int, speed: Float,
        vectorEstimator: Supertonic3VectorEstimator,
        metricsPath: String?, cpuOnly: Bool
    ) async {
        do {
            let tStart = Date()
            let computeUnits: MLComputeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
            let manager = Supertonic3Manager(
                computeUnits: computeUnits, vectorEstimator: vectorEstimator)

            let tLoad0 = Date()
            try await manager.initialize()
            let tLoad1 = Date()

            // Voice resolution: an explicit --voice-style <path> wins; otherwise
            // --voice names a built-in (F1-F5, M1-M5), defaulting to M1.
            let style: Supertonic3VoiceStyle
            if let voiceStylePath {
                let voiceStyleURL = resolveInputURL(voiceStylePath)
                style = try Supertonic3VoiceStyle.load(from: voiceStyleURL)
                logger.info("Supertonic-3 voice style (file): \(voiceStyleURL.path)")
            } else {
                let selected = Supertonic3Voice(name: voiceName) ?? .default
                if Supertonic3Voice(name: voiceName) == nil
                    && voiceName != TtsConstants.recommendedVoice
                {
                    logger.warning(
                        "Unknown Supertonic-3 voice '\(voiceName)'; using "
                            + "\(Supertonic3Voice.default.rawValue). Valid voices: "
                            + Supertonic3Voice.allCases.map(\.rawValue).joined(separator: ", ")
                            + ".")
                }
                style = try await Supertonic3ResourceDownloader.loadVoiceStyle(selected)
                logger.info("Supertonic-3 voice: \(selected.rawValue) (built-in)")
            }
            logger.info(
                "Supertonic-3 lang=\(language) totalSteps=\(totalSteps) "
                    + "speed=\(String(format: "%.2f", speed))")

            let tSynth0 = Date()
            let result = try await manager.synthesize(
                text: text, language: language, style: style,
                totalSteps: totalSteps, speed: speed)
            let tSynth1 = Date()

            let outURL = resolveInputURL(output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let wav = try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(Supertonic3Constants.sampleRate))
            try wav.write(to: outURL)

            let loadS = tLoad1.timeIntervalSince(tLoad0)
            let synthS = tSynth1.timeIntervalSince(tSynth0)
            let totalS = tSynth1.timeIntervalSince(tStart)
            let audioSecs =
                Double(result.samples.count) / Double(Supertonic3Constants.sampleRate)
            let rtfx = synthS > 0 ? audioSecs / synthS : 0

            logger.info("Supertonic-3 synthesis complete")
            logger.info("  Load: \(String(format: "%.3f", loadS))s")
            logger.info("  Synthesis: \(String(format: "%.3f", synthS))s")
            logger.info("  Audio: \(String(format: "%.3f", audioSecs))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
            logger.info("  Total: \(String(format: "%.3f", totalS))s")
            logger.info("  Output: \(outURL.path)")

            if let metricsPath {
                let metricsDict: [String: Any] = [
                    "backend": "supertonic3",
                    "text": text,
                    "language": language,
                    "voice_style": style.name,
                    "total_steps": totalSteps,
                    "speed": Double(speed),
                    "output": outURL.path,
                    "model_load_time_s": loadS,
                    "inference_time_s": synthS,
                    "audio_duration_s": audioSecs,
                    "realtime_speed": rtfx,
                    "total_time_s": totalS,
                ]
                let artifactsRoot = try ensureArtifactsRoot()
                let mURL = resolveOutputURL(
                    metricsPath, artifactsRoot: artifactsRoot, expectsDirectory: false)
                try FileManager.default.createDirectory(
                    at: mURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                let json = try JSONSerialization.data(
                    withJSONObject: metricsDict, options: [.prettyPrinted])
                try json.write(to: mURL)
                logger.info("Metrics saved: \(mURL.path)")
            }
        } catch {
            logger.error("Supertonic-3 Error: \(error)")
            print("Supertonic-3 failed: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts "text" [--output file.wav] [--voice af_heart] [--metrics metrics.json]

            Options:
              --output, -o         Output WAV path (default: output.wav)
              --voice, -v          Voice name (default: af_heart for KokoroAne, alba for PocketTTS)
              --backend            TTS backend: kokoro-ane (default), pocket, styletts2, supertonic3
                                   StyleTTS2 (zero-shot, English):
                                     --reference <speaker.wav>  required
                                     --alpha 0.3                ref-side blend (default 0.3)
                                     --beta 0.7                 prosody-side blend (default 0.7)
                                     --seed N                   RNG seed for fused sampler
                                     --ipa "…"                  bypass G2P, feed raw IPA
                                   Supertonic-3 (multilingual, 31 langs, 44.1 kHz):
                                     --voice F3                 built-in voice F1-F5/M1-M5 (default M1)
                                     --voice-style <file.json>  custom style file (overrides --voice)
                                     --lang en                  ISO-639-1 language code (default en)
                                     --total-steps 8            denoising step count (default 8)
                                     --speed 1.05               duration multiplier (default 1.05)
                                     --cpu-only                 disable Neural Engine
              --lexicon, -l        Custom pronunciation lexicon file (KokoroAne --variant zh only):
                                     word  pinyin1 pinyin2   (e.g. zi4 jie2)
                                     word  @bopomofo1        (escape: @-prefixed,
                                                              bypasses tone sandhi)
                                   Ignored for KokoroAne English (no lexicon support yet).
              --variant            KokoroAne language (values: en,zh).
                                   For --backend kokoro-ane --variant zh, Hanzi
                                   input is auto-phonemized through the bundled
                                   Mandarin G2P pipeline (FMM segmentation +
                                   diacritic→digit + 3+3 / 不 / 一 sandhi +
                                   bopomofo encoding). Pre-computed bopomofo
                                   (no Hanzi present) is also accepted and
                                   passes through unchanged.
              --metrics            Write timing metrics to a JSON file (also runs ASR for evaluation)
              --no-deess           Disable de-essing (sibilance reduction, enabled by default)
              (models/dictionary auto-download is always on in CLI)
              --help, -h           Show this help

            Voice Cloning (PocketTTS only):
              --clone-voice FILE   Clone voice from audio file (WAV, MP3, M4A, etc.)
              --voice-file FILE    Load previously saved voice .bin file
              --save-voice FILE    Save cloned voice to .bin file for later use

            PocketTTS Language Packs:
              --language ID        Language pack (default: english)
                                   Supported: english, french_24l,
                                   german, german_24l, italian, italian_24l,
                                   portuguese, portuguese_24l, spanish, spanish_24l
                                   Note: French is 24-layer only (no 6-layer pack upstream)
              --seed N             Deterministic-mode seed (uses session API for fixed RNG)
              --placement P        Model placement: gpu (default), ane (rank-4 ANE models),
                                   ane-state (Trial 23 MLState multifunction pipeline;
                                   macOS 15+/iOS 18+, requires pocket_state.mlmodelc)

            Voice Cloning examples:
              # Clone and synthesize in one step
              fluidaudio tts "Hello world" --backend pocket --clone-voice speaker.wav

              # Clone, save, and synthesize
              fluidaudio tts "Hello world" --backend pocket --clone-voice speaker.wav --save-voice my_voice.bin

              # Use previously saved voice
              fluidaudio tts "Hello world" --backend pocket --voice-file my_voice.bin
            """
        )
    }
}
