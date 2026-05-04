@preconcurrency import CoreML
import Foundation

/// Per-stage compute-unit assignment for the laishere chain.
///
/// Mirrors the empirical optima from `iOSDemo` and the conversion script:
/// Albert / PostAlbert / Alignment / Vocoder run on `cpuAndNeuralEngine`,
/// while Prosody / Noise / Tail run on `all` (let the scheduler pick).
public struct KokoroAneComputeUnits: Sendable, Equatable {
    public var albert: MLComputeUnits
    public var postAlbert: MLComputeUnits
    public var alignment: MLComputeUnits
    public var prosody: MLComputeUnits
    public var noise: MLComputeUnits
    public var vocoder: MLComputeUnits
    public var tail: MLComputeUnits

    public init(
        albert: MLComputeUnits = .cpuAndNeuralEngine,
        postAlbert: MLComputeUnits = .cpuAndNeuralEngine,
        alignment: MLComputeUnits = .cpuAndNeuralEngine,
        prosody: MLComputeUnits = .all,
        noise: MLComputeUnits = .all,
        vocoder: MLComputeUnits = .cpuAndNeuralEngine,
        tail: MLComputeUnits = .all
    ) {
        self.albert = albert
        self.postAlbert = postAlbert
        self.alignment = alignment
        self.prosody = prosody
        self.noise = noise
        self.vocoder = vocoder
        self.tail = tail
    }

    /// Empirical default — matches laishere's iOSDemo + this repo's conversion.
    public static let `default` = KokoroAneComputeUnits()

    /// CPU+GPU only (skip ANE entirely). Useful for a baseline / debugging.
    public static let cpuAndGpu = KokoroAneComputeUnits(
        albert: .cpuAndGPU, postAlbert: .cpuAndGPU, alignment: .cpuAndGPU,
        prosody: .cpuAndGPU, noise: .cpuAndGPU, vocoder: .cpuAndGPU, tail: .cpuAndGPU
    )

    /// Force every stage onto `.cpuAndNeuralEngine`. Stages that hit
    /// ANE-incompatible ops will fall back to CPU silently — included
    /// for the benchmark sweep (efficiency vs. latency comparison).
    public static let allAne = KokoroAneComputeUnits(
        albert: .cpuAndNeuralEngine, postAlbert: .cpuAndNeuralEngine,
        alignment: .cpuAndNeuralEngine, prosody: .cpuAndNeuralEngine,
        noise: .cpuAndNeuralEngine, vocoder: .cpuAndNeuralEngine,
        tail: .cpuAndNeuralEngine
    )

    /// CPU-only (no ANE, no GPU). Slowest but most predictable; useful
    /// as a debugging / fallback baseline.
    public static let cpuOnly = KokoroAneComputeUnits(
        albert: .cpuOnly, postAlbert: .cpuOnly, alignment: .cpuOnly,
        prosody: .cpuOnly, noise: .cpuOnly, vocoder: .cpuOnly, tail: .cpuOnly
    )

    /// Build a configuration from a generic preset (used by the
    /// `tts-benchmark` CLI so a single flag maps cleanly across
    /// backends).
    public init(preset: TtsComputeUnitPreset) {
        switch preset {
        case .default:
            self = .default
        case .allAne:
            self = .allAne
        case .cpuAndGpu:
            self = .cpuAndGpu
        case .cpuOnly:
            self = .cpuOnly
        }
    }

    func units(for stage: KokoroAneStage) -> MLComputeUnits {
        switch stage {
        case .albert: return albert
        case .postAlbert: return postAlbert
        case .alignment: return alignment
        case .prosody: return prosody
        case .noise: return noise
        case .vocoder: return vocoder
        case .tail: return tail
        }
    }
}

/// Actor-based store for the laishere Kokoro 7-stage CoreML chain.
///
/// Loads each `.mlmodelc` once with its target compute unit, plus the vocab
/// JSON and the default voice pack `.bin`.
public actor KokoroAneModelStore {

    private let logger = AppLogger(category: "KokoroAneModelStore")

    private var models: [KokoroAneStage: MLModel] = [:]
    private var vocab: KokoroAneVocab?
    private var voicePacks: [String: KokoroAneVoicePack] = [:]
    private var repoDirectory: URL?
    private var mandarinG2P: MandarinG2P?

    private let directory: URL?
    private let computeUnits: KokoroAneComputeUnits
    private let variant: KokoroAneVariant

    public init(
        directory: URL? = nil,
        computeUnits: KokoroAneComputeUnits = .default,
        variant: KokoroAneVariant = .english
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.variant = variant
    }

    /// Download (if missing), load all 7 mlmodelcs, parse vocab + default voice.
    ///
    /// Loads stage models into a local accumulator first and only commits them
    /// to `self.models` once every stage succeeds. This keeps `loadIfNeeded()`
    /// retryable: if any stage throws, `self.models` stays empty and the next
    /// call retries from scratch instead of returning early on a partial state.
    public func loadIfNeeded() async throws {
        guard models.isEmpty else { return }

        let repoDir = try await KokoroAneResourceDownloader.ensureModels(
            variant: variant, directory: directory)

        logger.info("Loading 7 KokoroAne CoreML models from \(repoDir.path)...")
        let loadStart = Date()

        var pendingModels: [KokoroAneStage: MLModel] = [:]
        for stage in KokoroAneStage.allCases {
            let url = repoDir.appendingPathComponent(stage.bundleName)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw KokoroAneError.modelNotLoaded(stage.bundleName)
            }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = computeUnits.units(for: stage)
            cfg.allowLowPrecisionAccumulationOnGPU = true
            let stageStart = Date()
            let model = try MLModel(contentsOf: url, configuration: cfg)
            let stageElapsed = Date().timeIntervalSince(stageStart) * 1000
            pendingModels[stage] = model
            logger.info("  loaded \(stage.bundleName) in \(String(format: "%.0f", stageElapsed)) ms")
        }
        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All 7 KokoroAne models loaded in \(String(format: "%.2f", elapsed))s")

        // Load vocab.json before publishing models so retry semantics stay
        // consistent: vocab failure also leaves `self.models` empty.
        let vocabURL = repoDir.appendingPathComponent(ModelNames.KokoroAne.vocab)
        let loadedVocab = try KokoroAneVocab.load(from: vocabURL)
        logger.info("Loaded vocab (\(loadedVocab.map.count) entries)")

        // Commit. Past this point a partial-failure retry would re-download
        // and recompile, which is OK — that's the documented contract.
        self.models = pendingModels
        self.vocab = loadedVocab
        self.repoDirectory = repoDir

        // Pre-load the default voice. Voice-pack failure does not invalidate
        // the model cache (voices are mutable runtime state).
        _ = try await voicePack(variant.defaultVoice)
    }

    public func model(for stage: KokoroAneStage) throws -> MLModel {
        guard let m = models[stage] else {
            throw KokoroAneError.modelNotLoaded(stage.bundleName)
        }
        return m
    }

    public func vocabulary() throws -> KokoroAneVocab {
        guard let v = vocab else {
            throw KokoroAneError.modelNotLoaded("vocab.json")
        }
        return v
    }

    public func voicePack(_ voice: String) async throws -> KokoroAneVoicePack {
        if let cached = voicePacks[voice] { return cached }
        guard let repoDir = repoDirectory else {
            throw KokoroAneError.modelNotLoaded("voice pack (repo not initialized)")
        }
        let url = try await KokoroAneResourceDownloader.ensureVoicePack(
            voice, repoDirectory: repoDir, variant: variant)
        let pack = try KokoroAneVoicePack.load(from: url)
        voicePacks[voice] = pack
        logger.info("Loaded voice pack '\(voice)'")
        return pack
    }

    public var isLoaded: Bool {
        models.count == KokoroAneStage.allCases.count && vocab != nil
    }

    /// Lazy-load and cache the Mandarin G2P pipeline (binary dicts +
    /// bopomofo mapper). Only used by ``KokoroAneVariant/mandarin``.
    /// Idempotent within a single store lifetime; the dict is held by
    /// value so cleanup() drops it cleanly.
    public func mandarinG2PPipeline() async throws -> MandarinG2P {
        if let g2p = mandarinG2P { return g2p }
        guard variant == .mandarin else {
            throw KokoroAneError.inputProcessingFailed(
                "Mandarin G2P requested on a non-mandarin store")
        }
        guard let repoDir = repoDirectory else {
            throw KokoroAneError.modelNotLoaded("Mandarin G2P (repo not initialized)")
        }
        let g2pDir = try await KokoroAneResourceDownloader.ensureMandarinG2P(
            repoDirectory: repoDir)
        let phrasesURL = g2pDir.appendingPathComponent(
            KokoroAneConstants.g2pPinyinPhrasesFile)
        let singlesURL = g2pDir.appendingPathComponent(
            KokoroAneConstants.g2pPinyinSingleFile)
        let dict = try MandarinPinyinDict.load(
            singlesURL: singlesURL, phrasesURL: phrasesURL)
        let g2pw = await loadG2pwIfAvailable(repoDirectory: repoDir)
        let pipeline = MandarinG2P(dict: dict, g2pw: g2pw)
        mandarinG2P = pipeline
        logger.info(
            "Loaded Mandarin G2P (phrases=\(dict.phrases.count), "
                + "singles=\(dict.singles.count), g2pw=\(g2pw == nil ? "off" : "on"))"
        )
        return pipeline
    }

    /// Best-effort load of the g2pW polyphone disambiguator. Returns
    /// `nil` (and logs) when the assets are missing or fail to load,
    /// so the Mandarin G2P pipeline can keep running on the dict
    /// alone.
    private func loadG2pwIfAvailable(repoDirectory: URL) async -> MandarinG2pwModel? {
        guard
            let g2pwDir = await KokoroAneResourceDownloader.ensureMandarinG2pw(
                repoDirectory: repoDirectory)
        else { return nil }

        let vocabURL = g2pwDir.appendingPathComponent(KokoroAneConstants.g2pwVocabFile)
        let polyURL = g2pwDir.appendingPathComponent(
            KokoroAneConstants.g2pwPolyphonicCharsFile)
        let modelURL = g2pwDir.appendingPathComponent(KokoroAneConstants.g2pwModelBundle)

        do {
            let tokenizer = try MandarinBertTokenizer.load(vocabURL: vocabURL)
            let catalog = try MandarinPolyphoneCatalog.load(fileURL: polyURL)
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            return MandarinG2pwModel(
                model: model, tokenizer: tokenizer, catalog: catalog)
        } catch {
            logger.info(
                "g2pW load failed (\(error.localizedDescription)) — "
                    + "Mandarin G2P will run dict-only")
            return nil
        }
    }

    public func cleanup() {
        models.removeAll()
        voicePacks.removeAll()
        vocab = nil
        repoDirectory = nil
        mandarinG2P = nil
    }
}
