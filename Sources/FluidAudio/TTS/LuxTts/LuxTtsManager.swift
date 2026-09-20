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
/// Text input runs through `LuxTtsG2p` (espeak-parity English G2P from a
/// bundled lexicon — the model was trained on espeak `en-us` phonemes via
/// EmiliaTokenizer, and Misaki-style frontends do not map onto that token
/// set). Pre-phonemized espeak IPA is still accepted via
/// `synthesize(phonemes:...)`.
///
/// Usage:
/// ```swift
/// let manager = try await LuxTtsManager.downloadAndCreate()
/// let result = try await manager.synthesize(
///     text: "The quick brown fox jumps over the lazy dog.",
///     promptAudio: promptWavURL,
///     promptText: "The transcript of the prompt clip.")
/// // result.samples is 48 kHz mono Float32 PCM.
/// ```
public actor LuxTtsManager {

    private let logger = AppLogger(category: "LuxTtsManager")

    private let directory: URL?
    private let variant: String
    private let computeUnitsOverride: MLComputeUnits?

    private var store: LuxTtsModelStore?
    private var synthesizer: LuxTtsSynthesizer?
    private var g2p: LuxTtsG2p?

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

    /// Synthesize from raw English text (espeak-parity G2P, see `LuxTtsG2p`).
    ///
    /// - Parameters:
    ///   - text: English text to speak.
    ///   - promptAudio: Prompt clip (see `synthesize(phonemes:...)`).
    ///   - promptText: Transcript of the prompt clip (raw text).
    public func synthesize(
        text: String,
        promptAudio: URL,
        promptText: String,
        speed: Float = LuxTtsConstants.defaultSpeed,
        seed: UInt64 = LuxTtsConstants.defaultSeed
    ) async throws -> LuxTtsSynthesisResult {
        // Fail fast before the (potentially expensive) G2P lexicon load and
        // phonemization; the phonemes path guards on the same store below.
        guard store != nil else { throw LuxTtsError.notInitialized }
        let g2p = try englishG2p()
        return try await synthesize(
            phonemes: g2p.phonemize(text: text),
            promptAudio: promptAudio,
            promptPhonemes: g2p.phonemize(text: promptText),
            speed: speed,
            seed: seed)
    }

    /// The bundled espeak-parity English G2P (loaded lazily; ~4 MB of
    /// lexicon tables, no network access).
    public func englishG2p() throws -> LuxTtsG2p {
        if let g2p { return g2p }
        let g2p = try LuxTtsG2p()
        self.g2p = g2p
        return g2p
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
        guard let store, let synthesizer else { throw LuxTtsError.notInitialized }

        let prompt24k: [Float]
        do {
            let converter = AudioConverter(
                sampleRate: Double(LuxTtsConstants.melSampleRate))
            prompt24k = try converter.resampleAudioFile(promptAudio)
        } catch {
            throw LuxTtsError.invalidPromptAudio(
                "cannot load \(promptAudio.path): \(error.localizedDescription)")
        }

        let tokenizer = try await store.tokenizer()
        let pauseTokens = Set([",", ".", ";", ":", "!", "?"].compactMap { tokenizer.tokenToId[$0] })
        let boundaryTokens = pauseTokens.union([tokenizer.tokenToId[" "]].compactMap { $0 })
        let maxPromptSamples = Int(
            LuxTtsConstants.maxPromptSeconds * Double(LuxTtsConstants.melSampleRate))
        let promptFrames = LuxTtsMelExtractor().frameCount(
            sampleCount: min(prompt24k.count, maxPromptSamples))

        let spans: [[Int]]
        if LuxTtsContinuation.fitsSinglePass(
            textTokenCount: tokenIds.count,
            promptFrames: promptFrames,
            promptTokenCount: promptTokenIds.count,
            speed: Double(speed))
        {
            spans = [tokenIds]
        } else {
            let maxSpanTokens = LuxTtsContinuation.maxSpanTokens(
                promptFrames: promptFrames,
                promptTokenCount: promptTokenIds.count,
                speed: Double(speed))
            spans = LuxTtsContinuation.chunks(
                tokenIds: tokenIds,
                maxTokens: maxSpanTokens,
                boundaryTokenIds: boundaryTokens)
            logger.info(
                "LuxTTS continuation synthesis: \(tokenIds.count) target tokens in "
                    + "\(spans.count) balanced spans (≤ \(maxSpanTokens) tokens each)")
        }

        let converter = AudioConverter(sampleRate: Double(LuxTtsConstants.melSampleRate))
        let crossfadeSamples = Int(
            LuxTtsConstants.continuationCrossfadeSeconds
                * Double(LuxTtsConstants.outputSampleRate))
        var currentPromptAudio = prompt24k
        var currentPromptTokens = promptTokenIds
        var samples: [Float] = []
        var originalPromptFrames = 0
        var totalGeneratedFrames = 0
        var previousSpeedFactor: Float = 1

        for (index, span) in spans.enumerated() {
            // A continuation prompt already speaks at the requested rate;
            // applying `speed` again would compound it on every span. If the
            // previous span was compressed by a retry, undo that here.
            let (result, speedFactor) = try await synthesizeSpan(
                synthesizer,
                textTokenIds: span,
                promptTokenIds: currentPromptTokens,
                promptAudio24k: currentPromptAudio,
                speed: index == 0 ? speed : 1 / previousSpeedFactor,
                seed: seed &+ UInt64(index) &* LuxTtsConstants.continuationSeedStride,
                allowedPauses: LuxTtsContinuation.expectedPauseCount(
                    in: span, pauseTokenIds: pauseTokens, boundaryTokenIds: boundaryTokens),
                label: "span \(index + 1)/\(spans.count)")
            if spans.count == 1 { return result }

            previousSpeedFactor = speedFactor
            if index == 0 { originalPromptFrames = result.promptFrames }
            totalGeneratedFrames += result.generatedFrames
            var spanSamples = result.samples
            if index > 0 {
                spanSamples = LuxTtsContinuation.trimmingLeadingPadding(
                    spanSamples, sampleRate: result.sampleRate)
            }
            let hasNextSpan = index + 1 < spans.count
            if hasNextSpan,
                !LuxTtsContinuation.endsWithPausePunctuation(
                    span, pauseTokenIds: pauseTokens, boundaryTokenIds: boundaryTokens)
            {
                spanSamples = LuxTtsContinuation.trimmingTrailingPadding(
                    spanSamples, sampleRate: result.sampleRate)
            }
            LuxTtsContinuation.appendWithCrossfade(
                spanSamples, to: &samples, crossfadeSamples: crossfadeSamples)

            guard hasNextSpan else { continue }
            // Prompt with the untrimmed span so its frames-per-token ratio is
            // the one the model actually produced for these tokens.
            currentPromptAudio = try converter.resample(
                result.samples, from: Double(result.sampleRate))
            currentPromptTokens = span
        }

        return LuxTtsSynthesisResult(
            samples: samples,
            sampleRate: LuxTtsConstants.outputSampleRate,
            promptFrames: originalPromptFrames,
            generatedFrames: totalGeneratedFrames,
            featuresLength: originalPromptFrames + totalGeneratedFrames)
    }

    /// One flow-matching pass with a bounded re-seed ladder. The model can
    /// drop a spurious mid-phrase pause whose position depends on the exact
    /// (length, noise) draw (issue #937; the PyTorch reference does the
    /// same), so a pass whose silences exceed the span's punctuation is
    /// re-drawn with the next seed, later attempts also slightly compressed
    /// in duration. The cleanest attempt is kept. Returns the pass and the
    /// speed factor it was rendered with.
    private func synthesizeSpan(
        _ synthesizer: LuxTtsSynthesizer,
        textTokenIds: [Int],
        promptTokenIds: [Int],
        promptAudio24k: [Float],
        speed: Float,
        seed: UInt64,
        allowedPauses: Int,
        label: String
    ) async throws -> (LuxTtsSynthesisResult, Float) {
        let factors = LuxTtsConstants.spuriousPauseRetrySpeedFactors
        func render(_ attempt: Int) async throws -> (LuxTtsSynthesisResult, Float, Int) {
            let factor = factors[min(attempt, factors.count - 1)]
            let result = try await synthesizer.synthesize(
                promptTokenIds: promptTokenIds,
                textTokenIds: textTokenIds,
                promptAudio24k: promptAudio24k,
                speed: speed * factor,
                seed: seed &+ UInt64(attempt))
            let pauses = LuxTtsContinuation.innerPauseCount(
                result.samples, sampleRate: result.sampleRate)
            return (result, factor, max(0, pauses - allowedPauses))
        }

        var (best, bestFactor, bestSpurious) = try await render(0)
        guard bestSpurious > 0 else { return (best, bestFactor) }

        for attempt in 1...max(1, LuxTtsConstants.spuriousPauseRetries) {
            logger.info(
                "LuxTTS \(label): \(bestSpurious) spurious pause(s) beyond the "
                    + "\(allowedPauses) the text allows; re-drawing (attempt \(attempt))")
            let (candidate, factor, spurious) = try await render(attempt)
            if spurious < bestSpurious {
                (best, bestFactor, bestSpurious) = (candidate, factor, spurious)
            }
            if spurious == 0 { return (candidate, factor) }
        }
        logger.warning(
            "LuxTTS \(label): \(bestSpurious) spurious pause(s) remain after "
                + "\(LuxTtsConstants.spuriousPauseRetries) re-draws; keeping the cleanest pass")
        return (best, bestFactor)
    }

    public func cleanup() async {
        if let store = store { await store.unload() }
        store = nil
        synthesizer = nil
    }
}
