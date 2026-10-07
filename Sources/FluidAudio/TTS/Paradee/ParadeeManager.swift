@preconcurrency import CoreML
import Foundation

/// Public API for the Paradee-8M CoreML TTS backend.
///
/// Paradee (Sahil Mahendrakar, Apache-2.0) is Kokoro-82M distilled into an
/// 8.07M-parameter single-voice (`af_heart`) English model, 24 kHz. Two CoreML
/// graphs (`ParadeeText`, `ParadeeAcoustic`) with host-side duration
/// expansion; see `FluidInference/paradee-8m-coreml`.
///
/// Paradee uses Kokoro's phoneme vocabulary and was trained on misaki en-US
/// phonemes, so the text path reuses the KokoroAne English frontend (NeMo
/// text normalization, Misaki lexicon, per-word BART G2P fallback). Like the
/// upstream package, text is synthesized one sentence at a time.
///
/// Run on `.cpuOnly` (default) or `.cpuAndNeuralEngine`; the LSTMs abort on
/// the GPU, so `.all` / `.cpuAndGPU` are rejected.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor ParadeeManager {

    private let logger = AppLogger(category: "ParadeeManager")

    private let store: ParadeeModelStore
    private let englishLexiconCache = LexiconAssetCache()
    private var englishPhonemizer: KokoroAneEnglishPhonemizer?
    private var englishFrontendReady = false

    public nonisolated var sampleRate: Int { ParadeeConstants.sampleRate }

    public init(
        variant: ParadeeVariant = .int8,
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly
    ) {
        self.store = ParadeeModelStore(variant: variant, directory: directory, computeUnits: computeUnits)
    }

    /// Download (if missing) and load both models, the vocab, and the
    /// English frontend assets.
    public func initialize() async throws {
        try await store.loadIfNeeded()
        guard !englishFrontendReady else { return }
        // G2PModel.loadIfNeeded only reads from cache, so fetch the assets
        // explicitly first. They live at the default kokoro cache path
        // (G2PModel.shared hardcodes it), not the store's `directory`.
        try await KokoroAneResourceDownloader.ensureG2PAssets(directory: nil)
        try await G2PModel.shared.ensureModelsAvailable()
        _ = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: nil)
        englishFrontendReady = true
    }

    public func isAvailable() async -> Bool {
        (try? await store.models()) != nil
    }

    // MARK: - Synthesis

    /// Text → 24 kHz mono Float32 PCM. Sentences are synthesized separately
    /// and concatenated, as in the upstream `Paradee.__call__`.
    ///
    /// - Parameters:
    ///   - speed: speech-rate multiplier (> 1 is faster).
    ///   - noiseSeed: seed for the harmonic source noise; equal seeds give
    ///     identical audio.
    public func synthesize(
        text: String,
        speed: Float = ParadeeConstants.defaultSpeed,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        var phonemeChunks: [String] = []
        for sentence in Self.sentences(in: text) {
            phonemeChunks.append(contentsOf: Self.chunks(try await phonemes(for: sentence)))
        }
        return try await synthesize(chunks: phonemeChunks, speed: speed, noiseSeed: noiseSeed)
    }

    /// Misaki-style phonemes → audio, bypassing text normalization and G2P.
    /// Input longer than 510 phonemes is split at punctuation/whitespace.
    public func synthesize(
        phonemes: String,
        speed: Float = ParadeeConstants.defaultSpeed,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        try await synthesize(chunks: Self.chunks(phonemes), speed: speed, noiseSeed: noiseSeed)
    }

    /// The phoneme string ``synthesize(text:speed:noiseSeed:)`` feeds the
    /// model for `text` (NeMo normalization, Misaki lexicon, BART G2P).
    public func phonemes(for text: String) async throws -> String {
        guard englishFrontendReady else { throw ParadeeError.notInitialized }
        let phonemizer = await ensureEnglishPhonemizer()
        let raw = try await phonemizer.phonemize(EnglishTextNormalizer.normalizeForFrontend(text)) { word in
            try await G2PModel.shared.phonemize(word: word)
        }
        return Self.misakiOutputForm(raw)
    }

    public func cleanup() async {
        await store.unload()
        englishPhonemizer = nil
    }

    // MARK: - Helpers

    private func synthesize(chunks: [String], speed: Float, noiseSeed: UInt64) async throws -> [Float] {
        let (text, acoustic) = try await store.models()
        let vocab = try await store.vocabulary()
        var samples: [Float] = []
        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            let ids = try vocab.encode(chunk)
            guard ids.count > 2 else { continue }
            samples += try ParadeeSynthesizer.synthesize(
                inputIds: ids, speed: speed, noiseSeed: noiseSeed &+ UInt64(index),
                text: text, acoustic: acoustic)
        }
        guard !samples.isEmpty else {
            throw ParadeeError.inputProcessingFailed("no speakable phonemes in input")
        }
        return samples
    }

    /// Split like upstream `re.split(r"(?<=[.!?…])\s+|\n+", text)`: after
    /// sentence-final punctuation followed by whitespace, and at newlines.
    static func sentences(in text: String) -> [String] {
        var out: [String] = []
        var current = ""
        var previous: Character?
        var index = text.startIndex
        while index < text.endIndex {
            let ch = text[index]
            if ch.isNewline || (ch.isWhitespace && previous.map { ".!?…".contains($0) } == true) {
                out.append(current)
                current = ""
                // Consume the whole whitespace run.
                while index < text.endIndex, text[index].isWhitespace {
                    index = text.index(after: index)
                }
                previous = nil
                continue
            }
            current.append(ch)
            previous = ch
            index = text.index(after: index)
        }
        out.append(current)
        return out.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    /// misaki's last step for Kokoro v1.0 (`G2P.__call__`): flap `ɾ` → `T`,
    /// glottal stop `ʔ` → `t`. The lexicon stores the raw forms, and Paradee
    /// never saw `ɾ`/`ʔ` in training ("kittens", "satellite" come out garbled).
    static func misakiOutputForm(_ phonemes: String) -> String {
        phonemes.replacingOccurrences(of: "ɾ", with: "T").replacingOccurrences(of: "ʔ", with: "t")
    }

    static func chunks(_ phonemes: String) -> [String] {
        PhonemeChunker.chunk(
            phonemes, maxLength: ParadeeConstants.maxPhonemeLength, countsUnicodeScalars: true)
    }

    /// Build (and cache) the English frontend from the model vocab and the
    /// Misaki lexicon cache. Without the lexicon it returns a transient
    /// G2P-only frontend so the lexicon is retried on the next call.
    private func ensureEnglishPhonemizer() async -> KokoroAneEnglishPhonemizer {
        if let cached = englishPhonemizer { return cached }
        var lower: [String: [String]] = [:]
        var caseSensitive: [String: [String]] = [:]
        var punctuation: Set<Character> = []
        var lexiconLoaded = false
        do {
            let vocab = try await store.vocabulary()
            punctuation = Set(vocab.map.keys.filter { !$0.isLetter && !$0.isNumber && !$0.isWhitespace })
            if let kokoroDir = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: nil) {
                try await englishLexiconCache.ensureLoaded(
                    kokoroDirectory: kokoroDir, allowedTokens: Set(vocab.map.keys.map(String.init)))
                let maps = await englishLexiconCache.lexicons()
                lower = maps.word
                caseSensitive = maps.caseSensitive
                lexiconLoaded = true
            }
        } catch {
            logger.warning("English lexicon unavailable (\(error.localizedDescription)); using BART G2P only")
        }
        let phonemizer = KokoroAneEnglishPhonemizer(
            wordToPhonemes: lower, caseSensitiveWordToPhonemes: caseSensitive,
            allowedPunctuation: punctuation)
        if lexiconLoaded {
            englishPhonemizer = phonemizer
        }
        return phonemizer
    }
}
