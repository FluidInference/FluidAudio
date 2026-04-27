import Foundation

/// High-level facade for the laishere/kokoro 7-stage CoreML chain.
///
/// Mirrors the public surface of `KokoroTtsManager` so callers can swap
/// backends with minimal churn. Internally:
///   * Text → IPA via the existing `G2PModel` (per-word, joined with " ")
///   * IPA → input ids via `KokoroLaiVocab`
///   * Voice pack slice via `KokoroLaiVoicePack`
///   * 7 stages via `KokoroLaiSynthesizer`
///   * Float samples → WAV via `AudioWAV`
///
/// Concurrency: actor-isolated. `KokoroLaiModelStore` is an actor too, so all
/// model access flows through an awaited boundary — no shared mutable state
/// is exposed.
public actor KokoroLaiManager {

    private let logger = AppLogger(category: "KokoroLaiManager")
    private let store: KokoroLaiModelStore
    private var defaultVoice: String

    public init(
        defaultVoice: String = KokoroLaiConstants.defaultVoice,
        directory: URL? = nil,
        computeUnits: KokoroLaiComputeUnits = .default,
        modelStore: KokoroLaiModelStore? = nil
    ) {
        self.defaultVoice = defaultVoice
        self.store =
            modelStore
            ?? KokoroLaiModelStore(
                directory: directory, computeUnits: computeUnits)
    }

    // MARK: - Lifecycle

    /// Download (if missing), load all 7 mlmodelcs + vocab + default voice
    /// pack. Optionally pre-warm additional voice packs.
    public func initialize(preloadVoices: Set<String>? = nil) async throws {
        try await store.loadIfNeeded()
        // G2P models are auto-fetched lazily on first use; warm them here so
        // initialization surfaces failures up-front.
        try await G2PModel.shared.ensureModelsAvailable()
        if let voices = preloadVoices {
            for voice in voices {
                _ = try await store.voicePack(voice)
            }
        }
    }

    /// `true` once the 7 mlmodelcs + vocab are resident.
    public func isAvailable() async -> Bool {
        await store.isLoaded
    }

    /// Override the voice used by default.
    public func setDefaultVoice(_ voice: String) {
        self.defaultVoice = voice
    }

    /// Drop loaded mlmodelcs + voice packs. The store reloads on next call.
    public func cleanup() async {
        await store.cleanup()
    }

    // MARK: - Synthesis

    /// One-shot text → 24 kHz mono 16-bit PCM WAV.
    public func synthesize(
        text: String,
        voice: String? = nil,
        speed: Float = Float(KokoroLaiConstants.defaultSpeed)
    ) async throws -> Data {
        let result = try await synthesizeDetailed(text: text, voice: voice, speed: speed)
        return try wavData(from: result)
    }

    /// Text → samples + per-stage timings.
    public func synthesizeDetailed(
        text: String,
        voice: String? = nil,
        speed: Float = Float(KokoroLaiConstants.defaultSpeed)
    ) async throws -> KokoroLaiSynthesisResult {
        let phonemes = try await phonemize(text: text)
        return try await runChain(phonemes: phonemes, voice: voice, speed: speed)
    }

    /// Bypass G2P; feed an already-IPA phoneme string directly.
    public func synthesizeFromPhonemes(
        _ phonemes: String,
        voice: String? = nil,
        speed: Float = Float(KokoroLaiConstants.defaultSpeed)
    ) async throws -> Data {
        let result = try await runChain(phonemes: phonemes, voice: voice, speed: speed)
        return try wavData(from: result)
    }

    /// Bypass G2P; return samples + timings.
    public func synthesizeFromPhonemesDetailed(
        _ phonemes: String,
        voice: String? = nil,
        speed: Float = Float(KokoroLaiConstants.defaultSpeed)
    ) async throws -> KokoroLaiSynthesisResult {
        try await runChain(phonemes: phonemes, voice: voice, speed: speed)
    }

    // MARK: - Private

    private func runChain(
        phonemes: String,
        voice: String?,
        speed: Float
    ) async throws -> KokoroLaiSynthesisResult {
        try await store.loadIfNeeded()
        let vocab = try await store.vocabulary()
        let voiceName = voice ?? defaultVoice
        let pack = try await store.voicePack(voiceName)

        let inputIds = try vocab.encode(phonemes)
        // Voice pack indexing matches `convert.py:get_ref_data` — row is the
        // raw phoneme-string length (BOS/EOS not counted).
        let phonemeCount = phonemes.count
        let (styleS, styleTimbre) = pack.slice(for: phonemeCount)

        return try await KokoroLaiSynthesizer.synthesize(
            inputIds: inputIds,
            phonemeCount: phonemeCount,
            styleS: styleS,
            styleTimbre: styleTimbre,
            speed: speed,
            store: store
        )
    }

    /// Whitespace-split, per-word G2P, joined with " ". Punctuation is
    /// stripped because the laishere vocab is IPA-only — punctuation chars
    /// would just be dropped at `KokoroLaiVocab.encode` anyway.
    private func phonemize(text: String) async throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw KokoroLaiError.inputProcessingFailed("(empty input)")
        }

        let words = trimmed.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        var parts: [String] = []
        parts.reserveCapacity(words.count)

        for word in words {
            let cleaned = word.trimmingCharacters(in: .punctuationCharacters).lowercased()
            guard !cleaned.isEmpty else { continue }
            do {
                if let ipa = try await G2PModel.shared.phonemize(word: cleaned) {
                    parts.append(ipa.joined())
                } else {
                    logger.warning("G2P returned nil for word '\(cleaned)' — skipping")
                }
            } catch {
                logger.warning(
                    "G2P failed on word '\(cleaned)': \(error.localizedDescription)")
                throw error
            }
        }

        let joined = parts.joined(separator: " ")
        if joined.isEmpty {
            throw KokoroLaiError.inputProcessingFailed(
                "G2P produced no phonemes for input '\(trimmed)'")
        }
        return joined
    }

    private func wavData(from result: KokoroLaiSynthesisResult) throws -> Data {
        do {
            return try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(result.sampleRate))
        } catch {
            throw KokoroLaiError.audioConversionFailed(error.localizedDescription)
        }
    }
}
