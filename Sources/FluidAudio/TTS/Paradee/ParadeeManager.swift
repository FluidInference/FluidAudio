#if TTS
@preconcurrency import CoreML
import Foundation

/// Public API for the Paradee-8M CoreML TTS backend.
///
/// Paradee (Sahil Mahendrakar, Apache-2.0) is Kokoro-82M distilled into an
/// 8M-parameter English model, 24 kHz. v1.2 (default) speaks four voices
/// (``ParadeeVoice``); v1.0 speaks `af_heart` only. Two CoreML graphs
/// (`ParadeeText`, `ParadeeAcoustic`) with host-side duration expansion; see
/// `FluidInference/paradee-8m-coreml`.
///
/// Paradee uses Kokoro's phoneme vocabulary and was trained on misaki
/// phonemes, so the text path reuses the KokoroAne English frontend (NeMo
/// text normalization, Misaki lexicon, per-word BART G2P fallback). The
/// British voices read misaki's British lexicon, as upstream does; it is
/// downloaded the first time a British voice is used. Like the upstream
/// package, text is synthesized one sentence at a time.
///
/// Run on `.cpuOnly` (default) or `.cpuAndNeuralEngine`; the LSTMs abort on
/// the GPU, so `.all` / `.cpuAndGPU` are rejected.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor ParadeeManager {

    private let logger = AppLogger(category: "ParadeeManager")

    private let store: ParadeeModelStore
    /// BART G2P fallback bound to the store's models root (#1023).
    nonisolated let englishG2P: G2PModel
    private let americanLexiconCache = LexiconAssetCache()
    private let britishLexiconCache = LexiconAssetCache()
    private var americanPhonemizer: KokoroAneEnglishPhonemizer?
    private var britishPhonemizer: KokoroAneEnglishPhonemizer?
    private var englishFrontendReady = false

    /// The Paradee release this manager loads.
    public nonisolated let version: ParadeeVersion

    public nonisolated var sampleRate: Int { ParadeeConstants.sampleRate }

    /// Voices of the loaded release.
    public nonisolated var voices: [ParadeeVoice] { version.voices }

    public init(
        version: ParadeeVersion = .v12,
        variant: ParadeeVariant = .int8,
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly
    ) {
        self.version = version
        self.store = ParadeeModelStore(
            version: version, variant: variant, directory: directory, computeUnits: computeUnits)
        self.englishG2P = G2PModel.shared(directory: directory)
    }

    /// Download (if missing) and load both models, the vocab, and the
    /// English frontend assets. The British lexicon is fetched on first use
    /// of a British voice.
    public func initialize() async throws {
        try await store.loadIfNeeded()
        guard !englishFrontendReady else { return }
        // G2PModel.loadIfNeeded only reads from disk, so fetch the assets
        // explicitly first. They live under the store's `directory`
        // (`<directory>/kokoro/`), next to the Paradee models.
        try await KokoroAneResourceDownloader.ensureG2PAssets(directory: store.directory)
        try await englishG2P.ensureModelsAvailable()
        _ = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: store.directory)
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
    ///   - voice: one of ``voices``; British voices get British phonemes.
    ///   - speed: speech-rate multiplier (> 1 is faster).
    ///   - noiseSeed: seed for the harmonic source noise; equal seeds give
    ///     identical audio.
    public func synthesize(
        text: String,
        voice: ParadeeVoice = .afHeart,
        speed: Float = ParadeeConstants.defaultSpeed,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        try checkVoice(voice)
        var phonemeChunks: [String] = []
        for sentence in Self.sentences(in: text) {
            phonemeChunks.append(contentsOf: Self.chunks(try await phonemes(for: sentence, voice: voice)))
        }
        return try await synthesize(chunks: phonemeChunks, voice: voice, speed: speed, noiseSeed: noiseSeed)
    }

    /// Misaki-style phonemes → audio, bypassing text normalization and G2P.
    /// Input longer than 510 phonemes is split at punctuation/whitespace.
    /// Write British phonemes for the British voices.
    public func synthesize(
        phonemes: String,
        voice: ParadeeVoice = .afHeart,
        speed: Float = ParadeeConstants.defaultSpeed,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        try checkVoice(voice)
        return try await synthesize(chunks: Self.chunks(phonemes), voice: voice, speed: speed, noiseSeed: noiseSeed)
    }

    /// The phoneme string ``synthesize(text:voice:speed:noiseSeed:)`` feeds
    /// the model for `text` (NeMo normalization, Misaki lexicon, BART G2P):
    /// British for `bf_` / `bm_` voices, American otherwise.
    public func phonemes(for text: String, voice: ParadeeVoice = .afHeart) async throws -> String {
        guard englishFrontendReady else { throw ParadeeError.notInitialized }
        let british = voice.isBritish
        let phonemizer = await ensurePhonemizer(british: british)
        let raw = try await phonemizer.phonemize(EnglishTextNormalizer.normalizeForFrontend(text)) {
            [englishG2P] word in
            // BART G2P only knows American phonemes.
            let american = try await englishG2P.phonemize(word: word)
            guard british, let american else { return american }
            return [MisakiBritishFallback.fromAmerican(american.joined())]
        }
        return Self.misakiOutputForm(raw)
    }

    public func cleanup() async {
        await store.unload()
        americanPhonemizer = nil
        britishPhonemizer = nil
    }

    // MARK: - Helpers

    private func checkVoice(_ voice: ParadeeVoice) throws {
        guard version.voices.contains(voice) else {
            throw ParadeeError.unsupportedVoice(voice: voice.rawValue, version: version.rawValue)
        }
    }

    private func synthesize(
        chunks: [String], voice: ParadeeVoice, speed: Float, noiseSeed: UInt64
    ) async throws -> [Float] {
        let (text, acoustic) = try await store.models()
        let vocab = try await store.vocabulary()
        let voiceIndex = version.hasVoiceInput ? voice.modelIndex : nil
        var samples: [Float] = []
        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            let ids = try vocab.encode(chunk)
            guard ids.count > 2 else { continue }
            samples += try ParadeeSynthesizer.synthesize(
                inputIds: ids, voice: voiceIndex, speed: speed, noiseSeed: noiseSeed &+ UInt64(index),
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

    /// Build (and cache) the American or British English frontend from the
    /// model vocab and that dialect's Misaki lexicon cache. Without the
    /// lexicon it returns a transient G2P-only frontend so the lexicon is
    /// retried on the next call.
    private func ensurePhonemizer(british: Bool) async -> KokoroAneEnglishPhonemizer {
        if let cached = british ? britishPhonemizer : americanPhonemizer { return cached }
        var lower: [String: [String]] = [:]
        var caseSensitive: [String: [String]] = [:]
        var punctuation: Set<Character> = []
        var lexiconLoaded = false
        do {
            let vocab = try await store.vocabulary()
            punctuation = Set(vocab.map.keys.filter { !$0.isLetter && !$0.isNumber && !$0.isWhitespace })
            if let kokoroDir = await KokoroAneResourceDownloader.ensureEnglishLexicon(
                directory: store.directory, british: british)
            {
                let cache = british ? britishLexiconCache : americanLexiconCache
                try await cache.ensureLoaded(
                    kokoroDirectory: kokoroDir, allowedTokens: Set(vocab.map.keys.map(String.init)),
                    fileName: LexiconAssetCache.englishFileName(british: british))
                let maps = await cache.lexicons()
                lower = maps.word
                caseSensitive = maps.caseSensitive
                lexiconLoaded = true
            }
        } catch {
            logger.warning(
                "\(british ? "British" : "American") English lexicon unavailable (\(error.localizedDescription)); "
                    + "using BART G2P only")
        }
        let phonemizer = KokoroAneEnglishPhonemizer(
            wordToPhonemes: lower, caseSensitiveWordToPhonemes: caseSensitive,
            allowedPunctuation: punctuation, british: british)
        if lexiconLoaded {
            if british {
                britishPhonemizer = phonemizer
            } else {
                americanPhonemizer = phonemizer
            }
        }
        return phonemizer
    }
}
#endif
