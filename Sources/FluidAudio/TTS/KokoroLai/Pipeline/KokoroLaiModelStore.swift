@preconcurrency import CoreML
import Foundation

/// Per-stage compute-unit assignment for the laishere chain.
///
/// Mirrors the empirical optima from `iOSDemo` and the conversion script:
/// Albert / PostAlbert / Alignment / Vocoder run on `cpuAndNeuralEngine`,
/// while Prosody / Noise / Tail run on `all` (let the scheduler pick).
public struct KokoroLaiComputeUnits: Sendable, Equatable {
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
    public static let `default` = KokoroLaiComputeUnits()

    /// CPU+GPU only (skip ANE entirely). Useful for a baseline / debugging.
    public static let cpuAndGpu = KokoroLaiComputeUnits(
        albert: .cpuAndGPU, postAlbert: .cpuAndGPU, alignment: .cpuAndGPU,
        prosody: .cpuAndGPU, noise: .cpuAndGPU, vocoder: .cpuAndGPU, tail: .cpuAndGPU
    )

    func units(for stage: KokoroLaiStage) -> MLComputeUnits {
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
public actor KokoroLaiModelStore {

    private let logger = AppLogger(category: "KokoroLaiModelStore")

    private var models: [KokoroLaiStage: MLModel] = [:]
    private var vocab: KokoroLaiVocab?
    private var voicePacks: [String: KokoroLaiVoicePack] = [:]
    private var repoDirectory: URL?

    private let directory: URL?
    private let computeUnits: KokoroLaiComputeUnits

    public init(
        directory: URL? = nil,
        computeUnits: KokoroLaiComputeUnits = .default
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
    }

    /// Download (if missing), load all 7 mlmodelcs, parse vocab + default voice.
    public func loadIfNeeded() async throws {
        guard models.isEmpty else { return }

        let repoDir = try await KokoroLaiResourceDownloader.ensureModels(directory: directory)
        self.repoDirectory = repoDir

        logger.info("Loading 7 KokoroLai CoreML models from \(repoDir.path)...")
        let loadStart = Date()

        for stage in KokoroLaiStage.allCases {
            let url = repoDir.appendingPathComponent(stage.bundleName)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw KokoroLaiError.modelNotLoaded(stage.bundleName)
            }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = computeUnits.units(for: stage)
            cfg.allowLowPrecisionAccumulationOnGPU = true
            let stageStart = Date()
            let model = try MLModel(contentsOf: url, configuration: cfg)
            let stageElapsed = Date().timeIntervalSince(stageStart) * 1000
            models[stage] = model
            logger.info("  loaded \(stage.bundleName) in \(String(format: "%.0f", stageElapsed)) ms")
        }
        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All 7 KokoroLai models loaded in \(String(format: "%.2f", elapsed))s")

        // Load vocab.json
        let vocabURL = repoDir.appendingPathComponent(ModelNames.KokoroLai.vocab)
        vocab = try KokoroLaiVocab.load(from: vocabURL)
        logger.info("Loaded vocab (\(self.vocab?.map.count ?? 0) entries)")

        // Pre-load the default voice
        try await loadVoicePackIfNeeded(KokoroLaiConstants.defaultVoice)
    }

    public func model(for stage: KokoroLaiStage) throws -> MLModel {
        guard let m = models[stage] else {
            throw KokoroLaiError.modelNotLoaded(stage.bundleName)
        }
        return m
    }

    public func vocabulary() throws -> KokoroLaiVocab {
        guard let v = vocab else {
            throw KokoroLaiError.modelNotLoaded("vocab.json")
        }
        return v
    }

    public func voicePack(_ voice: String) async throws -> KokoroLaiVoicePack {
        if let cached = voicePacks[voice] { return cached }
        try await loadVoicePackIfNeeded(voice)
        guard let pack = voicePacks[voice] else {
            throw KokoroLaiError.voicePackMissing(URL(fileURLWithPath: "\(voice).bin"))
        }
        return pack
    }

    public var isLoaded: Bool {
        models.count == KokoroLaiStage.allCases.count && vocab != nil
    }

    public func cleanup() {
        models.removeAll()
        voicePacks.removeAll()
        vocab = nil
        repoDirectory = nil
    }

    // MARK: - Private

    private func loadVoicePackIfNeeded(_ voice: String) async throws {
        if voicePacks[voice] != nil { return }
        guard let repoDir = repoDirectory else {
            throw KokoroLaiError.modelNotLoaded("voice pack (repo not initialized)")
        }
        let url = try await KokoroLaiResourceDownloader.ensureVoicePack(
            voice, repoDirectory: repoDir)
        voicePacks[voice] = try KokoroLaiVoicePack.load(from: url)
        logger.info("Loaded voice pack '\(voice)'")
    }
}
