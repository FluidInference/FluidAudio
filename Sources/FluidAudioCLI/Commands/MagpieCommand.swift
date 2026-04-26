#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// CLI surface for the Magpie TTS Multilingual Swift port.
///
/// Subcommands:
///   - `download`             Fetch models + constants + tokenizer data from HuggingFace.
///   - `text`                 Synthesize text → WAV.
///   - `parity`               Compare Swift intermediates against a Python fixture (Phase 5).
///   - `tokenizer-parity`     Compare Swift tokenizer output against a language fixture.
public enum MagpieCommand {

    private static let logger = AppLogger(category: "MagpieCommand")

    public static func run(arguments: [String]) async {
        guard let sub = arguments.first else {
            printUsage()
            return
        }
        let rest = Array(arguments.dropFirst())
        switch sub {
        case "download":
            await runDownload(arguments: rest)
        case "text":
            await runText(arguments: rest)
        case "parity":
            await runParity(arguments: rest)
        case "tokenizer-parity":
            await runTokenizerParity(arguments: rest)
        case "probe":
            await MagpieProbeCommand.run(arguments: rest)
        case "compute-plan":
            if #available(macOS 14.4, *) {
                await MagpieComputePlanCommand.run(arguments: rest)
            } else {
                logger.error("compute-plan requires macOS 14.4+")
                exit(1)
            }
        case "help", "--help", "-h":
            printUsage()
        default:
            logger.error("Unknown magpie subcommand: \(sub)")
            printUsage()
            exit(1)
        }
    }

    // MARK: - download

    private static func runDownload(arguments: [String]) async {
        var languageCodes: [String] = ["en"]
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            if arg == "--languages" || arg == "-l", i + 1 < arguments.count {
                languageCodes = arguments[i + 1].split(separator: ",").map(String.init)
                i += 1
            }
            i += 1
        }
        let langs: Set<MagpieLanguage> = Set(languageCodes.compactMap { MagpieLanguage(rawValue: $0) })
        if langs.isEmpty {
            logger.error("No valid language codes provided")
            exit(1)
        }
        do {
            let repoDir = try await MagpieResourceDownloader.ensureAssets(languages: langs)
            logger.info("Magpie assets ready at: \(repoDir.path)")
        } catch {
            logger.error("Magpie download failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - text

    private static func runText(arguments: [String]) async {
        var text: String? = nil
        var output = "magpie.wav"
        var speakerIdx = MagpieSpeaker.john.rawValue
        var languageCode = "en"
        var cfg: Float = MagpieConstants.defaultCfgScale
        var topK = MagpieConstants.defaultTopK
        var temperature = MagpieConstants.defaultTemperature
        var seed: UInt64? = nil
        var allowIpa = true

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--speaker":
                if i + 1 < arguments.count, let idx = Int(arguments[i + 1]) {
                    speakerIdx = idx
                    i += 1
                }
            case "--language", "-L":
                if i + 1 < arguments.count {
                    languageCode = arguments[i + 1]
                    i += 1
                }
            case "--cfg":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    cfg = v
                    i += 1
                }
            case "--topk":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    topK = v
                    i += 1
                }
            case "--temperature":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    temperature = v
                    i += 1
                }
            case "--seed":
                if i + 1 < arguments.count, let v = UInt64(arguments[i + 1]) {
                    seed = v
                    i += 1
                }
            case "--no-ipa-override":
                allowIpa = false
            default:
                if text == nil { text = arg }
            }
            i += 1
        }

        guard let text = text, !text.isEmpty else {
            logger.error("Missing text argument")
            printUsage()
            exit(1)
        }
        guard let speaker = MagpieSpeaker(rawValue: speakerIdx) else {
            logger.error("Invalid speaker index \(speakerIdx); valid range 0..<\(MagpieConstants.numSpeakers)")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language code '\(languageCode)'")
            exit(1)
        }

        do {
            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            let opts = MagpieSynthesisOptions(
                temperature: temperature,
                topK: topK,
                maxSteps: MagpieConstants.maxSteps,
                minFrames: MagpieConstants.minFrames,
                cfgScale: cfg,
                seed: seed,
                peakNormalize: true,
                allowIpaOverride: allowIpa)
            let start = Date()
            let result = try await manager.synthesize(
                text: text, speaker: speaker, language: language, options: opts)
            let elapsed = Date().timeIntervalSince(start)

            let wav = try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(result.sampleRate))
            let outURL = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try wav.write(to: outURL)

            let audioSecs = result.durationSeconds
            let rtfx = elapsed > 0 ? audioSecs / elapsed : 0
            logger.info("Magpie synthesis complete")
            logger.info("  Speaker: \(speaker.displayName), Language: \(language.rawValue)")
            logger.info("  Codes: \(result.codeCount), EOS: \(result.finishedOnEos)")
            logger.info(
                "  Audio: \(String(format: "%.3f", audioSecs))s, Synthesis: \(String(format: "%.3f", elapsed))s, RTFx: \(String(format: "%.2f", rtfx))x"
            )
            logger.info("  Output: \(outURL.path)")
        } catch {
            logger.error("Magpie synthesis failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - parity

    /// Compare Swift synthesis against a mobius-emitted parity fixture.
    ///
    /// Two fixture formats are accepted, matching the two modes of
    /// `mobius/.../emit_parity_fixture.py`:
    ///
    ///   - `.json` — tokenizer-only fixture
    ///     (delegates to the existing `runTokenizerParity` flow).
    ///   - `.npz`  — full pipeline fixture with tensors:
    ///     `textTokens`, `textTokensPadded`, `encoderOutput`, `predictedCodes`,
    ///     `audioPcm`, plus per-layer prefill caches. Synthesis params (text,
    ///     speaker, language, seed, …) must be supplied as CLI overrides since
    ///     the NPZ stores them as numpy unicode scalars that we do not parse.
    ///
    /// Reports MAE, max|Δ|, and SNR for each comparable stage.
    private static func runParity(arguments: [String]) async {
        var fixtureArg: String? = nil
        var text: String? = nil
        var speakerIdx = 0
        var languageCode = "en"
        var seed: UInt64? = nil
        var cfg: Float = MagpieConstants.defaultCfgScale
        var temperature: Float = MagpieConstants.defaultTemperature
        var topK = MagpieConstants.defaultTopK
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--fixture":
                if i + 1 < arguments.count {
                    fixtureArg = arguments[i + 1]
                    i += 1
                }
            case "--text":
                if i + 1 < arguments.count {
                    text = arguments[i + 1]
                    i += 1
                }
            case "--speaker":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    speakerIdx = v
                    i += 1
                }
            case "--language", "-L":
                if i + 1 < arguments.count {
                    languageCode = arguments[i + 1]
                    i += 1
                }
            case "--seed":
                if i + 1 < arguments.count, let v = UInt64(arguments[i + 1]) {
                    seed = v
                    i += 1
                }
            case "--cfg":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    cfg = v
                    i += 1
                }
            case "--temperature":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    temperature = v
                    i += 1
                }
            case "--topk":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    topK = v
                    i += 1
                }
            default:
                break
            }
            i += 1
        }
        guard let fixtureArg = fixtureArg else {
            logger.error("--fixture <path.npz|path.json> is required for magpie parity")
            exit(1)
        }

        let url = URL(fileURLWithPath: fixtureArg)
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.error("Fixture not found at \(url.path)")
            logger.info(
                "Emit one with: uv run python emit_parity_fixture.py \"<text>\" --speaker N --language <code> --seed N --output <path.npz>"
            )
            exit(1)
        }

        // JSON path — tokenizer-only mode.
        if fixtureArg.hasSuffix(".json") {
            await runJsonTokenizerParity(url: url)
            return
        }

        // NPZ path — full mode requires CLI synthesis params.
        guard let text = text, !text.isEmpty else {
            logger.error("--text \"…\" is required when fixture is .npz")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language code '\(languageCode)'")
            exit(1)
        }
        guard let speaker = MagpieSpeaker(rawValue: speakerIdx) else {
            logger.error("Invalid speaker index \(speakerIdx)")
            exit(1)
        }

        do {
            let npz = try MagpieNpzReader.read(from: url)
            logger.info("Loaded NPZ keys: \(npz.keys.sorted().joined(separator: ", "))")

            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            let opts = MagpieSynthesisOptions(
                temperature: temperature,
                topK: topK,
                maxSteps: MagpieConstants.maxSteps,
                minFrames: MagpieConstants.minFrames,
                cfgScale: cfg,
                seed: seed,
                peakNormalize: true,
                allowIpaOverride: true)

            // Stage 1 — token ids parity (mobius emits `textTokens`, padded version
            // available as `textTokensPadded`).
            if let arr = npz["textTokens"] {
                let expected = arr.data.map { Int32($0) }
                try await runTokenizerStage(
                    text: text, expected: expected, language: language, options: opts)
            } else {
                logger.warning("NPZ missing `textTokens`; skipping tokenizer parity stage")
            }

            // Stage 2 — synthesize and compare audio.
            logger.info("Running synthesis…")
            let start = Date()
            let result = try await manager.synthesize(
                text: text, speaker: speaker, language: language, options: opts)
            let elapsed = Date().timeIntervalSince(start)
            let synthLine =
                "  generated \(result.samples.count) samples (\(String(format: "%.3f", result.durationSeconds))s) in \(String(format: "%.3f", elapsed))s, codes=\(result.codeCount), eos=\(result.finishedOnEos)"
            logger.warning("\(synthLine)")
            FileHandle.standardError.write(Data((synthLine + "\n").utf8))

            if let audio = npz["audioPcm"] {
                reportAudioParity(actual: result.samples, expected: audio.data)
            } else {
                logger.info("Skipping audio parity (no `audioPcm` array in NPZ)")
            }
        } catch {
            logger.error("Parity harness failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func runJsonTokenizerParity(url: URL) async {
        do {
            let fixture = try MagpieParityFixture.load(from: url)
            logger.info(
                "Loaded JSON fixture: text=\"\(fixture.text)\" speaker=\(fixture.speakerIndex) language=\(fixture.languageCode)"
            )
            guard let language = MagpieLanguage(rawValue: fixture.languageCode) else {
                logger.error("Fixture language '\(fixture.languageCode)' not supported")
                exit(1)
            }
            try await runTokenizerStage(
                text: fixture.text, expected: fixture.expectedTokenIds, language: language,
                options: MagpieSynthesisOptions())
        } catch {
            logger.error("Parity harness failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    /// Walk Swift tokenizer + compare token ids against the fixture.
    private static func runTokenizerStage(
        text: String, expected: [Int32], language: MagpieLanguage,
        options: MagpieSynthesisOptions
    ) async throws {
        let repoDir = try await MagpieResourceDownloader.ensureAssets(languages: [language])
        let tokenizerDir = MagpieResourceDownloader.tokenizerDirectory(in: repoDir)
        let constantsDir = MagpieResourceDownloader.constantsDirectory(in: repoDir)
        let constants = try MagpieConstantsLoader.load(from: constantsDir)
        let tok = MagpieTokenizer(tokenizerDir: tokenizerDir, eosId: constants.textEosId)
        let tokenized = try await tok.tokenize(text, language: language, options: options)
        let actual = Swift.Array(tokenized.paddedIds.prefix(tokenized.realLength))
        if actual == expected {
            logger.info("Tokenizer parity OK (\(actual.count) tokens)")
        } else {
            logger.error("Tokenizer parity MISMATCH")
            logger.error(
                "  expected (\(expected.count) tokens): \(expected.prefix(32))\(expected.count > 32 ? "…" : "")"
            )
            logger.error(
                "  actual   (\(actual.count) tokens): \(actual.prefix(32))\(actual.count > 32 ? "…" : "")"
            )
        }
    }

    /// Compare two waveforms; print MAE, max|Δ|, and SNR (dB).
    private static func reportAudioParity(actual: [Float], expected: [Float]) {
        let n = Swift.min(actual.count, expected.count)
        if actual.count != expected.count {
            logger.warning(
                "Audio length differs: actual=\(actual.count) expected=\(expected.count); comparing first \(n) samples"
            )
        }
        var sumAbs: Double = 0
        var sumSq: Double = 0
        var sumRefSq: Double = 0
        var maxAbs: Float = 0
        for i in 0..<n {
            let d = actual[i] - expected[i]
            let ad = abs(d)
            sumAbs += Double(ad)
            sumSq += Double(d) * Double(d)
            sumRefSq += Double(expected[i]) * Double(expected[i])
            if ad > maxAbs { maxAbs = ad }
        }
        let mae = sumAbs / Double(n)
        let mse = sumSq / Double(n)
        let refPower = sumRefSq / Double(n)
        let snrDb: Double
        if mse > 0 && refPower > 0 {
            snrDb = 10 * log10(refPower / mse)
        } else if mse == 0 {
            snrDb = .infinity
        } else {
            snrDb = -.infinity
        }
        let parityLine =
            "Audio parity: n=\(n) MAE=\(String(format: "%.6e", mae)) max|Δ|=\(String(format: "%.6e", maxAbs)) SNR=\(String(format: "%.2f", snrDb)) dB"
        logger.warning("\(parityLine)")
        FileHandle.standardError.write(Data((parityLine + "\n").utf8))
    }

    // MARK: - tokenizer-parity (stub)

    private static func runTokenizerParity(arguments: [String]) async {
        var languageCode = "en"
        var fixturePath: String? = nil
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            if arg == "--language" || arg == "-L", i + 1 < arguments.count {
                languageCode = arguments[i + 1]
                i += 1
            } else if arg == "--fixture", i + 1 < arguments.count {
                fixturePath = arguments[i + 1]
                i += 1
            }
            i += 1
        }
        guard let fixturePath = fixturePath else {
            logger.error("--fixture <path> is required")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language '\(languageCode)'")
            exit(1)
        }

        do {
            let url = URL(fileURLWithPath: fixturePath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                logger.error("Fixture not found at \(url.path)")
                exit(1)
            }
            let data = try Data(contentsOf: url)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let text = json["text"] as? String,
                let expected = json["token_ids"] as? [Int]
            else {
                logger.error("Fixture must be a JSON object with keys {text, token_ids}")
                exit(1)
            }

            // Tokenizer is actor-internal; build one against the on-disk tokenizer
            // directory for parity (no need to load the CoreML graph).
            let repoDir = try await MagpieResourceDownloader.ensureAssets(languages: [language])
            let tokenizerDir = MagpieResourceDownloader.tokenizerDirectory(in: repoDir)
            let tok = MagpieTokenizer(tokenizerDir: tokenizerDir, eosId: 0)
            let tokenized = try await tok.tokenize(
                text, language: language, options: MagpieSynthesisOptions())
            let actual = Swift.Array(tokenized.paddedIds.prefix(tokenized.realLength))
            let expectedInt32 = expected.map { Int32($0) }

            let match = actual == expectedInt32
            if match {
                logger.info("Tokenizer parity OK (\(actual.count) tokens)")
            } else {
                logger.error("Tokenizer parity MISMATCH")
                logger.error("  expected: \(expectedInt32.prefix(32))… (\(expectedInt32.count) tokens)")
                logger.error("  actual:   \(actual.prefix(32))… (\(actual.count) tokens)")
                exit(1)
            }
        } catch {
            logger.error("Tokenizer parity failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - usage

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio magpie <subcommand> [options]

            Subcommands:
              download                Download Magpie models + constants + tokenizers
                --languages en,es,de    Comma-separated language codes (default: en)

              text "<text>"           Synthesize text and write a WAV file
                --output, -o PATH       Output WAV path (default: magpie.wav)
                --speaker N             Speaker index 0-4 (default: 0 = John)
                --language CODE         Language code (en, es, de, fr, it, vi, zh, hi)
                --cfg FLOAT             CFG guidance scale (default: 1.0 = off)
                --topk N                Top-K sampling (default: 80)
                --temperature FLOAT     Sampling temperature (default: 0.6)
                --seed N                Deterministic RNG seed
                --no-ipa-override       Disable `|…|` IPA pass-through

              parity --fixture PATH
                                      Compare Swift synthesis to a mobius fixture.
                                      .npz: full pipeline parity (audio MAE/SNR);
                                      .json: tokenizer-only token-id diff.
                For .npz, supply synthesis params:
                  --text "..." --speaker N --language CODE --seed N
                  --cfg X --temperature X --topk N
              tokenizer-parity --fixture PATH --language CODE
                                      Verify tokenizer matches a fixture {text, token_ids}

            IPA override example:
              fluidaudio magpie text "Hello | ˈ n ɛ m o ʊ | Text." --output demo.wav

            """
        )
    }
}

// MARK: - Fixture loader

/// Tokenizer-only fixture emitted by `mobius/.../emit_parity_fixture.py --mode tokenizer`.
///
/// Keys mirror the Python emitter's camelCase output:
///   `{ "text", "speakerIndex", "languageCode", "expectedTokenIds" }`.
private struct MagpieParityFixture: Decodable {
    let text: String
    let speakerIndex: Int
    let languageCode: String
    let expectedTokenIds: [Int32]

    static func load(from url: URL) throws -> MagpieParityFixture {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MagpieParityFixture.self, from: data)
    }
}
#endif
