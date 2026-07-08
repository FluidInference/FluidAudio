@preconcurrency import CoreML
import Foundation

/// Top-level public API for LuxTTS (ZipVoice-Distill) zero-shot
/// voice-cloning TTS — 48 kHz output conditioned on a short prompt clip.
///
/// Pipeline pieces:
///   1. `LuxTtsModelStore`   — downloads + holds the CoreML stages
///      (TextEncoder, FmDecoder, fixed-shape Vocos vocoders) and `tokens.txt`.
///   2. `LuxTtsTokenizer`    — espeak-IPA phoneme string → token ids.
///   3. `LuxTtsSynthesizer`  — flow-matching host loop (see its docs).
///
/// Phase 1 input contract: **pre-phonemized espeak IPA** (`en-us`, the token
/// set in `tokens.txt`). There is no in-process text→IPA frontend yet:
/// the model was trained on espeak phonemes, and the KokoroAne/StyleTTS2
/// Misaki frontend output does not map onto them cleanly (Misaki drops
/// espeak length marks `ː`, uses `ʤ`/`ʧ` ligatures and different weak
/// forms — measured 78 vs 85 tokens on the same sentence), which would
/// shift both pronunciation and the ratio-based duration estimate.
/// `synthesize(text:...)` therefore throws `LuxTtsError.g2pUnavailable`
/// until the phase-2 espeak-parity G2P lands.
///
/// Usage:
/// ```swift
/// let manager = try await LuxTtsManager.downloadAndCreate()
/// let result = try await manager.synthesize(
///     phonemes: "ðə kwˈɪk bɹˈaʊn fˈɑːks...",
///     promptAudio: promptWavURL,
///     promptPhonemes: "kwˈɪk bɹˈaʊn fˈɑːks...")
/// // result.samples is 48 kHz mono Float32 PCM.
/// ```
public actor LuxTtsManager {

    private let logger = AppLogger(category: "LuxTtsManager")

    private let directory: URL?
    private let variant: String
    private let computeUnitsOverride: MLComputeUnits?

    private var store: LuxTtsModelStore?
    private var synthesizer: LuxTtsSynthesizer?

    /// - Parameters:
    ///   - directory: Model cache root override (default: shared TTS cache).
    ///   - variant: Graph variant (`ModelNames.LuxTts.gpuVariant` /
    ///     `.aneVariant`). Defaults to the platform-appropriate graph:
    ///     `gpu/` + `.cpuAndGPU` on macOS, `ane/` + `.cpuAndNeuralEngine`
    ///     elsewhere. The `gpu/` graph must never run on the ANE (rel-pos
    ///     attention corrupts audio there).
    ///   - computeUnitsOverride: Force specific compute units for every stage.
    public init(
        directory: URL? = nil,
        variant: String = ModelNames.LuxTts.defaultVariant,
        computeUnitsOverride: MLComputeUnits? = nil
    ) {
        self.directory = directory
        self.variant = variant
        self.computeUnitsOverride = computeUnitsOverride
    }

    public var isAvailable: Bool { synthesizer != nil }

    /// Convenience factory: download assets and return a ready-to-use manager.
    public static func downloadAndCreate(
        cacheDirectory: URL? = nil,
        variant: String = ModelNames.LuxTts.defaultVariant,
        computeUnitsOverride: MLComputeUnits? = nil
    ) async throws -> LuxTtsManager {
        let manager = LuxTtsManager(
            directory: cacheDirectory,
            variant: variant,
            computeUnitsOverride: computeUnitsOverride)
        try await manager.initialize()
        return manager
    }

    /// Download (if missing) and load the LuxTTS CoreML stages.
    public func initialize(progressHandler: ProgressHandler? = nil) async throws {
        if synthesizer != nil { return }

        let store = LuxTtsModelStore(
            directory: directory,
            variant: variant,
            computeUnitsOverride: computeUnitsOverride)
        try await store.loadIfNeeded(progressHandler: progressHandler)

        self.store = store
        self.synthesizer = LuxTtsSynthesizer(store: store)
        logger.info("LuxTTS ready (variant: \(variant))")
    }

    // MARK: - Synthesis

    /// Synthesize from raw text. **Phase 1: always throws
    /// `LuxTtsError.g2pUnavailable`** — see the type-level discussion.
    /// Use `synthesize(phonemes:promptAudio:promptPhonemes:...)`.
    public func synthesize(
        text: String,
        promptAudio: URL,
        promptText: String,
        speed: Float = LuxTtsConstants.defaultSpeed,
        seed: UInt64 = LuxTtsConstants.defaultSeed
    ) async throws -> LuxTtsSynthesisResult {
        _ = (text, promptAudio, promptText, speed, seed)
        throw LuxTtsError.g2pUnavailable
    }

    /// Synthesize from espeak-IPA phoneme strings (the `tokens.txt` set;
    /// one token per Unicode scalar, OOV scalars skipped with a warning).
    ///
    /// - Parameters:
    ///   - phonemes: espeak IPA for the text to speak.
    ///   - promptAudio: Prompt clip (any format/rate; converted to 24 kHz
    ///     mono, capped at `LuxTtsConstants.maxPromptSeconds`). Trim
    ///     leading/trailing silence beforehand (e.g. with `VadManager`) —
    ///     silence inflates the frames-per-token duration ratio.
    ///   - promptPhonemes: espeak IPA of the prompt clip's transcript.
    ///   - speed: Speech-rate divisor for the generated span. Keep 1.0
    ///     (upstream's hidden 1.3 clips sentence onsets).
    ///   - seed: Noise seed for the flow-matching init.
    public func synthesize(
        phonemes: String,
        promptAudio: URL,
        promptPhonemes: String,
        speed: Float = LuxTtsConstants.defaultSpeed,
        seed: UInt64 = LuxTtsConstants.defaultSeed
    ) async throws -> LuxTtsSynthesisResult {
        guard let store = store else { throw LuxTtsError.notInitialized }
        let tokenizer = try await store.tokenizer()
        return try await synthesize(
            tokenIds: tokenizer.tokenIds(phonemes: phonemes),
            promptAudio: promptAudio,
            promptTokenIds: tokenizer.tokenIds(phonemes: promptPhonemes),
            speed: speed,
            seed: seed)
    }

    /// Synthesize from pre-computed token ids (callers running their own
    /// espeak frontend against `tokens.txt`).
    public func synthesize(
        tokenIds: [Int],
        promptAudio: URL,
        promptTokenIds: [Int],
        speed: Float = LuxTtsConstants.defaultSpeed,
        seed: UInt64 = LuxTtsConstants.defaultSeed
    ) async throws -> LuxTtsSynthesisResult {
        guard let synthesizer = synthesizer else { throw LuxTtsError.notInitialized }

        let prompt24k: [Float]
        do {
            let converter = AudioConverter(
                sampleRate: Double(LuxTtsConstants.melSampleRate))
            prompt24k = try converter.resampleAudioFile(promptAudio)
        } catch {
            throw LuxTtsError.invalidPromptAudio(
                "cannot load \(promptAudio.path): \(error.localizedDescription)")
        }

        return try await synthesizer.synthesize(
            promptTokenIds: promptTokenIds,
            textTokenIds: tokenIds,
            promptAudio24k: prompt24k,
            speed: speed,
            seed: seed)
    }

    public func cleanup() async {
        if let store = store { await store.unload() }
        store = nil
        synthesizer = nil
    }
}
