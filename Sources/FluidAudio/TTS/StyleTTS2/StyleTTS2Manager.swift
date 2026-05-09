@preconcurrency import CoreML
import Foundation

/// Top-level public API for StyleTTS2 LibriTTS (iteration_3) zero-shot TTS.
///
/// `StyleTTS2Manager` orchestrates the four moving pieces of the Swift
/// pipeline:
///   1. `StyleTTS2ModelStore`        — downloads + holds the 8 default
///      `.mlmodelc` bundles (plus 6 lazily-loaded bucket variants).
///   2. `StyleTTS2Phonemizer`        — runs CharsiuG2P over the input text.
///   3. `StyleTTS2MelExtractor`      — computes the 80-bin HTK log-mel of
///      the reference audio.
///   4. `StyleTTS2Synthesizer`       — drives the 8-stage CoreML graph and
///      returns 24 kHz mono Float32 audio.
///
/// > Important — Phonemizer parity gap.
/// > The Python reference uses `phonemizer.backend.EspeakBackend(language=
/// > "en-us", with_stress=True)`. FluidAudio cannot ship the espeak C
/// > library, so the default text path uses `MultilingualG2PModel`
/// > (CharsiuG2P ByT5). CharsiuG2P uses a different IPA convention (no
/// > stress markers, slightly different vowels) — output is intelligible but
/// > below the espeak reference. Callers with a reliable phonemizer (e.g.
/// > server-side espeak, or custom IPA) should pass the IPA string directly
/// > via `synthesize(ipa:referenceAudioURL:...)`.
///
/// Usage:
/// ```swift
/// let manager = try await StyleTTS2Manager.downloadAndCreate()
/// let audio = try await manager.synthesize(
///     text: "Hello from StyleTTS2.",
///     referenceAudioURL: refURL)
/// // `audio` is 24 kHz mono Float32 PCM.
/// ```
public actor StyleTTS2Manager {

    private let logger = AppLogger(category: "StyleTTS2Manager")

    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private let g2pLanguage: MultilingualG2PLanguage

    private var store: StyleTTS2ModelStore?
    private var phonemizer: StyleTTS2Phonemizer?
    private var melExtractor: StyleTTS2MelExtractor?
    private var synthesizer: StyleTTS2Synthesizer?

    public init(
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        g2pLanguage: MultilingualG2PLanguage = .americanEnglish
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.g2pLanguage = g2pLanguage
    }

    public var isAvailable: Bool {
        synthesizer != nil
    }

    /// Convenience factory: download assets and return a ready-to-use manager.
    public static func downloadAndCreate(
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        g2pLanguage: MultilingualG2PLanguage = .americanEnglish
    ) async throws -> StyleTTS2Manager {
        let manager = StyleTTS2Manager(
            directory: cacheDirectory,
            computeUnits: computeUnits,
            g2pLanguage: g2pLanguage)
        try await manager.initialize()
        return manager
    }

    /// Download the 8 default-path StyleTTS2 mlmodelc bundles, load them, and
    /// initialize the phonemizer + mel extractor + synthesizer.
    ///
    /// Bucket variants (T = 64 / 128 / 256) are *not* fetched here — they
    /// download lazily the first time a prompt exceeds the previous bucket's
    /// token capacity.
    public func initialize() async throws {
        if synthesizer != nil { return }

        logger.info(
            "StyleTTS2 phonemizer uses Misaki gold/silver lexicon first, "
                + "with CharsiuG2P as a fallback for OOV words. Pass IPA "
                + "directly via synthesize(ipa:...) when you have a "
                + "higher-quality phonemizer.")

        let store = StyleTTS2ModelStore(
            directory: directory, computeUnits: computeUnits)
        try await store.loadIfNeeded()
        self.store = store

        // Eagerly fetch the CharsiuG2P CoreML bundles so the first synthesize
        // call doesn't pay download cost mid-request. `MultilingualG2PModel`
        // is shared and the call is idempotent.
        try await MultilingualG2PModel.shared.ensureModelsAvailable()

        // Fetch Misaki gold (+ best-effort silver) lexicons and load them
        // up-front so per-word lookup is hash-table fast at synthesis time.
        let lexiconDir = try await StyleTTS2ResourceDownloader.ensureLexicons()
        let lexicon = StyleTTS2GoldLexicon()
        do {
            try await lexicon.load(directory: lexiconDir)
        } catch {
            logger.warning(
                "Misaki lexicon load failed (\(error)); falling back to CharsiuG2P-only path")
        }

        self.phonemizer = StyleTTS2Phonemizer(language: g2pLanguage, lexicon: lexicon)
        self.melExtractor = StyleTTS2MelExtractor()
        self.synthesizer = StyleTTS2Synthesizer(store: store)

        logger.info("StyleTTS2 ready (compute units: \(computeUnits.description))")
    }

    // MARK: - Synthesis: text path

    /// Phonemize `text`, extract the reference-audio mel, and synthesize
    /// 24 kHz mono Float32 audio. The reference audio file is decoded and
    /// resampled to 24 kHz mono internally.
    ///
    /// - Parameters:
    ///   - text: Source utterance. Empty strings throw
    ///     `StyleTTS2Error.phonemizationFailed`.
    ///   - referenceAudioURL: WAV / AIFF / CAF / m4a file readable by
    ///     `AVAudioFile`. Any sample rate / channel layout — the loader
    ///     resamples to 24 kHz mono.
    ///   - alpha: Reference-side blend weight (default 0.3 — 30 % diffusion,
    ///     70 % reference style).
    ///   - beta: Prosody-side blend weight (default 0.7 — 70 % diffusion,
    ///     30 % reference prosody).
    ///   - noiseSeed: RNG seed for the fused-sampler aux noises (default 0).
    ///     Same seed → same audio for the same text + reference.
    public func synthesize(
        text: String,
        referenceAudioURL: URL,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        guard let phonemizer = phonemizer else {
            throw StyleTTS2Error.notInitialized
        }
        let tokenIds = try await phonemizer.encode(text)
        return try await synthesize(
            tokenIds: tokenIds,
            referenceAudioURL: referenceAudioURL,
            alpha: alpha, beta: beta, noiseSeed: noiseSeed)
    }

    // MARK: - Synthesis: IPA path (espeak-parity escape hatch)

    /// Synthesize directly from a pre-phonemized IPA string. Bypasses
    /// CharsiuG2P entirely — use this when you have a higher-quality
    /// phonemizer (e.g. server-side espeak) and want to feed the IPA the
    /// model was actually trained against.
    ///
    /// `ipa` is fed verbatim through `StyleTTS2TextCleaner.encode(_)` (which
    /// silently drops any character outside the StyleTTS2 symbol vocab).
    public func synthesize(
        ipa: String,
        referenceAudioURL: URL,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        let tokenIds = StyleTTS2TextCleaner.encode(ipa)
        return try await synthesize(
            tokenIds: tokenIds,
            referenceAudioURL: referenceAudioURL,
            alpha: alpha, beta: beta, noiseSeed: noiseSeed)
    }

    // MARK: - Synthesis: low-level path

    /// Synthesize from already-encoded TextCleaner token IDs. The
    /// reference-audio mel is computed internally from `referenceAudioURL`.
    public func synthesize(
        tokenIds: [Int32],
        referenceAudioURL: URL,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        guard let melExtractor = melExtractor, let synthesizer = synthesizer else {
            throw StyleTTS2Error.notInitialized
        }

        let refSamples = try loadReferenceAudio(url: referenceAudioURL)
        let (mel, frames) = melExtractor.compute(audio: refSamples)
        guard frames > 0 else {
            throw StyleTTS2Error.unsupportedReferenceAudio(
                "reference audio at \(referenceAudioURL.lastPathComponent) yielded 0 mel frames")
        }

        return try await synthesizer.synthesize(
            tokenIds: tokenIds,
            referenceMel: mel, referenceMelFrames: frames,
            alpha: alpha, beta: beta, noiseSeed: noiseSeed)
    }

    /// Synthesize with a caller-provided reference mel. Useful when the same
    /// reference audio is reused for many prompts — compute the mel once and
    /// pass it on every call to skip the FFT / mel work.
    ///
    /// `referenceMel` must be a flat row-major `[1, 1, 80, frames]`
    /// Float32 buffer matching the layout produced by
    /// `StyleTTS2MelExtractor.compute(audio:)`.
    public func synthesize(
        tokenIds: [Int32],
        referenceMel: [Float],
        referenceMelFrames: Int,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        guard let synthesizer = synthesizer else {
            throw StyleTTS2Error.notInitialized
        }
        return try await synthesizer.synthesize(
            tokenIds: tokenIds,
            referenceMel: referenceMel, referenceMelFrames: referenceMelFrames,
            alpha: alpha, beta: beta, noiseSeed: noiseSeed)
    }

    // MARK: - Reference-audio mel preview

    /// Compute (and return) the reference-audio mel without running
    /// synthesis. Intended for callers that want to cache the mel for many
    /// utterances against the same speaker reference.
    public func referenceMel(from url: URL) throws -> (mel: [Float], frames: Int) {
        guard let melExtractor = melExtractor else {
            throw StyleTTS2Error.notInitialized
        }
        let samples = try loadReferenceAudio(url: url)
        return melExtractor.compute(audio: samples)
    }

    // MARK: - Cleanup

    public func cleanup() async {
        if let store = store {
            await store.unload()
        }
        store = nil
        phonemizer = nil
        melExtractor = nil
        synthesizer = nil
    }

    // MARK: - Helpers

    private func loadReferenceAudio(url: URL) throws -> [Float] {
        let converter = AudioConverter(sampleRate: Double(StyleTTS2Constants.sampleRate))
        do {
            return try converter.resampleAudioFile(url)
        } catch {
            throw StyleTTS2Error.unsupportedReferenceAudio("\(error)")
        }
    }
}

// MARK: - Description shim for log lines.
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
