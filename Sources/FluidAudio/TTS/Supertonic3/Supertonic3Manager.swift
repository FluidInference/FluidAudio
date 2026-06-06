@preconcurrency import CoreML
import Foundation

/// Top-level public API for Supertonic-3 multilingual TTS.
///
/// `Supertonic3Manager` orchestrates the four pieces of the on-device
/// pipeline:
///   1. `Supertonic3ModelStore`        — downloads + holds the four
///      `.mlmodelc` bundles (text_encoder, duration_predictor,
///      vector_estimator, vocoder) plus the companion `tts.json` and
///      `unicode_indexer.json`.
///   2. `Supertonic3UnicodeProcessor`  — NFKD text normalization, language
///      tagging, Unicode → indexer ID lookup.
///   3. `Supertonic3TextChunker`       — paragraph / sentence / comma /
///      word chunker used for long-utterance synthesis.
///   4. `Supertonic3Synthesizer`       — drives the 4-stage CoreML graph
///      (text_encoder + duration_predictor → denoising loop → vocoder) and
///      returns 44.1 kHz mono Float32 audio.
///
/// Voice styles are caller-supplied: the upstream repo ships a handful of
/// preset speakers (`M1`, `M2`, `F1`, `F2`, …) as JSON files in
/// `assets/voice_styles/`. Load one via
/// `Supertonic3VoiceStyle.load(from: voiceStyleURL)` and pass it on every
/// synthesize call.
///
/// Usage:
/// ```swift
/// let manager = try await Supertonic3Manager.downloadAndCreate()
/// let style = try Supertonic3VoiceStyle.load(from: voiceStyleURL)
/// let audio = try await manager.synthesize(
///     text: "A gentle breeze moved through the open window.",
///     language: "en",
///     style: style)
/// // `audio.samples` is 44.1 kHz mono Float32 PCM.
/// ```
public actor Supertonic3Manager {

    private let logger = AppLogger(category: "Supertonic3Manager")

    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private let vectorEstimatorOption: Supertonic3VectorEstimator

    private var store: Supertonic3ModelStore?
    private var synthesizer: Supertonic3Synthesizer?

    /// - Parameters:
    ///   - vectorEstimator: which VectorEstimator build to download/run. Default
    ///     `.fp16Dynamic` preserves prior behavior; `.aneBucketed(.int4)` etc.
    ///     opt into the smaller, ANE-resident fixed-length builds.
    public init(
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        vectorEstimator: Supertonic3VectorEstimator = .default
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.vectorEstimatorOption = vectorEstimator
    }

    public var isAvailable: Bool { synthesizer != nil }

    /// Convenience factory: download assets and return a ready-to-use manager.
    public static func downloadAndCreate(
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        vectorEstimator: Supertonic3VectorEstimator = .default
    ) async throws -> Supertonic3Manager {
        let manager = Supertonic3Manager(
            directory: cacheDirectory,
            computeUnits: computeUnits,
            vectorEstimator: vectorEstimator)
        try await manager.initialize()
        return manager
    }

    /// Download (if missing) and load the four Supertonic-3 CoreML stages.
    public func initialize() async throws {
        if synthesizer != nil { return }

        let store = Supertonic3ModelStore(
            directory: directory, computeUnits: computeUnits,
            vectorEstimator: vectorEstimatorOption)
        try await store.loadIfNeeded()
        let indexerURL = try await store.unicodeIndexerURL()
        let processor = try Supertonic3UnicodeProcessor(unicodeIndexerURL: indexerURL)

        self.store = store
        self.synthesizer = Supertonic3Synthesizer(store: store, processor: processor)

        logger.info("Supertonic-3 ready (compute units: \(computeUnits.description))")
    }

    // MARK: - Synthesis

    /// Synthesize a 44.1 kHz mono Float32 utterance.
    ///
    /// - Parameters:
    ///   - text: Source utterance. The text is NFKD-normalized and chunked
    ///     internally; pass the full passage rather than pre-chunking it.
    ///   - language: One of `Supertonic3Constants.availableLanguages` (31
    ///     ISO codes + `"na"` for numeric / language-agnostic input).
    ///   - style: Voice style loaded from a Supertonic preset JSON via
    ///     `Supertonic3VoiceStyle.load(from:)`.
    ///   - totalSteps: Denoising step count (default 8 — mirrors the
    ///     reference CLI). Lower trades quality for latency.
    ///   - speed: Speech-rate multiplier (default 1.05). Divides the
    ///     predicted duration vector.
    ///   - silenceDuration: Silence inserted between chunks when the text
    ///     is split into multiple chunks. Default 0.3 s.
    public func synthesize(
        text: String,
        language: String,
        style: Supertonic3VoiceStyle,
        totalSteps: Int = Supertonic3Constants.defaultTotalSteps,
        speed: Float = Supertonic3Constants.defaultSpeed,
        silenceDuration: Float = Supertonic3Constants.defaultSilenceDuration
    ) async throws -> (samples: [Float], duration: Float) {
        guard let synthesizer = synthesizer else {
            throw Supertonic3Error.notInitialized
        }
        return try await synthesizer.synthesize(
            text: text, language: language, style: style,
            totalSteps: totalSteps, speed: speed,
            silenceDuration: silenceDuration)
    }

    public func cleanup() async {
        if let store = store { await store.unload() }
        store = nil
        synthesizer = nil
    }
}

extension MLComputeUnits {
    fileprivate var description: String {
        switch self {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .all: return "all"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "unknown"
        }
    }
}
