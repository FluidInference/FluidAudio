import Foundation

/// High-level facade for the Kokoro 82M 7-stage CoreML chain
/// (ANE-resident, derived from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml)).
///
/// Splits the model into 7 CoreML graphs with per-stage compute-unit
/// placement (``KokoroAneComputeUnits``). The default routing keeps the
/// RNN-bearing stages (Albert / PostAlbert / Alignment / Prosody / Vocoder)
/// resident on the Neural Engine and sends the all-fp32 stages
/// (Noise + Tail iSTFT) to the GPU — the only placement that runs on every
/// Apple Silicon generation (see #667). Multi-graph splitting yields a large
/// RTFx win over a single-graph CPU+GPU Kokoro implementation.
///
/// Constraints:
///   * One default voice per variant (`af_heart` for English, `zf_001` for
///     Mandarin); additional voices download on demand via ``setDefaultVoice``
///     / `voice:` / `initialize(preloadVoices:)`.
///   * IPA input capped at 512 tokens — chunk longer prompts upstream.
///   * Loads from HF path `kokoro-82m-coreml/ANE/` (English) or
///     `ANE-zh/` (Mandarin).
///
/// Pipeline:
///   * Text → IPA via ``KokoroAneEnglishPhonemizer`` (Misaki lexicon first
///     — weak function-word forms, vocab punctuation kept as prosody
///     tokens — with per-word BART `G2PModel` fallback for OOV words)
///   * IPA → input ids via `KokoroAneVocab`
///   * Voice pack slice via `KokoroAneVoicePack`
///   * 7 stages via `KokoroAneSynthesizer`
///   * Float samples → WAV via `AudioWAV`
///
/// Concurrency: actor-isolated. `KokoroAneModelStore` is an actor too, so all
/// model access flows through an awaited boundary — no shared mutable state
/// is exposed.
public actor KokoroAneManager {

    private let logger = AppLogger(category: "KokoroAneManager")
    private let store: KokoroAneModelStore
    private let variant: KokoroAneVariant
    private var defaultVoice: String

    /// English frontend: Misaki lexicon + custom overrides + punctuation
    /// pass-through. Built lazily (needs the chain vocab + lexicon asset);
    /// cached only after a successful lexicon load so a transient download
    /// failure doesn't pin the degraded G2P-only path for the session.
    private var englishPhonemizer: KokoroAneEnglishPhonemizer?
    private var englishCustomLexicon: [String: String] = [:]
    private let englishLexiconCache = LexiconAssetCache()

    public init(
        variant: KokoroAneVariant = .english,
        defaultVoice: String? = nil,
        directory: URL? = nil,
        computeUnits: KokoroAneComputeUnits = .default,
        modelStore: KokoroAneModelStore? = nil
    ) {
        self.variant = variant
        self.defaultVoice = defaultVoice ?? variant.defaultVoice
        self.store =
            modelStore
            ?? KokoroAneModelStore(
                directory: directory, computeUnits: computeUnits, variant: variant)
    }

    // MARK: - Lifecycle

    /// Download (if missing), load all 7 mlmodelcs + vocab + default voice
    /// pack. Optionally pre-warm additional voice packs.
    public func initialize(preloadVoices: Set<String>? = nil) async throws {
        try await store.loadIfNeeded()
        // English G2P CoreML assets live in the kokoro repo and are loaded
        // from ~/.cache/fluidaudio/Models/kokoro/. The Mandarin variant
        // routes through the in-process MandarinG2P pipeline (loaded by
        // store.loadIfNeeded()) and never calls G2PModel.shared, so the
        // English G2P bundle would just be wasted bandwidth + memory.
        //
        // For English: G2PModel.loadIfNeeded only reads from cache (it
        // never downloads), so first-time KokoroAne users who have never
        // run the regular kokoro backend would otherwise hit a cryptic
        // G2PModelError.vocabLoadFailed. Fetch G2P assets explicitly
        // before warming the in-process G2P model.
        //
        // NOTE: pass nil (not `directory`) — `G2PModel.shared` is a singleton
        // that hardcodes the default cache path (TtsCacheDirectory.ensure()
        // /Models/kokoro). If we honoured the caller's custom `directory` here
        // we'd download to a path G2PModel can't see and still hit
        // vocabLoadFailed. The KokoroAne mlmodelc chain itself does respect
        // `directory` (via store), only the shared G2P assets are pinned.
        if variant == .english {
            try await KokoroAneResourceDownloader.ensureG2PAssets(directory: nil)
            try await G2PModel.shared.ensureModelsAvailable()
            // Best-effort pre-fetch of the Misaki lexicon cache (weak
            // function-word forms, issue #691). Missing lexicon degrades
            // to the BART-G2P-only path rather than failing initialize.
            _ = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: nil)
        }
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

    /// Install (or clear) a user-supplied Mandarin pronunciation override.
    ///
    /// Slots in **at the front** of ``MandarinG2P``'s segmentation cascade:
    /// longest-prefix match against the user lexicon runs before the
    /// bundled `pinyin_phrases.bin` / `pinyin_single.bin` lookup. User
    /// entries of equal length to a dict entry win. Pinyin-form tokens
    /// (`zi4`) participate in tone sandhi with surrounding context;
    /// `@`-bopomofo tokens (`@ㄈㄨ4`) bypass sandhi.
    ///
    /// Pass ``MandarinCustomLexicon/empty`` to clear. Only meaningful
    /// for ``KokoroAneVariant/mandarin`` — calling on the English variant
    /// stores the value but has no synthesis effect.
    public func setMandarinCustomLexicon(_ lexicon: MandarinCustomLexicon) async {
        await store.setMandarinCustomLexicon(lexicon)
    }

    /// Install (or clear) a user-supplied English pronunciation override.
    ///
    /// Entries map a word to a Misaki-style IPA string (e.g.
    /// `["to": "tə", "GIF": "ʤˈɪf"]`). The exact spelling is checked
    /// first, then the lower-cased form, before the bundled Misaki
    /// lexicon and the BART G2P fallback. Pass `[:]` to clear.
    ///
    /// Only meaningful for ``KokoroAneVariant/english`` — calling on the
    /// Mandarin variant stores the value but has no synthesis effect
    /// (use ``setMandarinCustomLexicon(_:)`` there).
    public func setEnglishCustomLexicon(_ entries: [String: String]) {
        englishCustomLexicon = entries
        // Rebuild the cached frontend with the new overrides on next use.
        englishPhonemizer = nil
    }

    /// Drop loaded mlmodelcs + voice packs. The store reloads on next call.
    public func cleanup() async {
        await store.cleanup()
        englishPhonemizer = nil
    }

    // MARK: - Synthesis

    /// One-shot text → 24 kHz mono 16-bit PCM WAV.
    public func synthesize(
        text: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> Data {
        let result = try await synthesizeDetailed(text: text, voice: voice, speed: speed)
        return try wavData(from: result)
    }

    /// Text → samples + per-stage timings.
    ///
    /// For ``KokoroAneVariant/mandarin`` the input is routed through
    /// ``MandarinG2P``: Hanzi → forward-max-match segmentation
    /// (`pinyin_phrases.bin` + `pinyin_single.bin`) → diacritic
    /// → tone-digit normalization → 3+3 / 不 / 一 sandhi → bopomofo +
    /// tone-digit string. Strings that already look like phonemes
    /// (no Hanzi) bypass the pipeline and are forwarded as-is, so
    /// callers can still feed pre-computed bopomofo when they want
    /// to override the bundled lexicon.
    public func synthesizeDetailed(
        text: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> KokoroAneSynthesisResult {
        let resolved = try await phonemes(for: text)
        return try await runChain(phonemes: resolved, voice: voice, speed: speed)
    }

    /// Resolve the exact phoneme string ``synthesize(text:voice:speed:)``
    /// would feed the 7-stage chain — for diagnostics, tests, and
    /// caller-side phoneme caching (issue #691).
    ///
    /// English: Misaki-lexicon-first with BART G2P fallback. Mandarin:
    /// the ``MandarinG2P`` pipeline for Hanzi input, pass-through for
    /// strings that already look like phonemes. Japanese: no text frontend
    /// — throws (use ``synthesizeFromPhonemes(_:voice:speed:)`` with
    /// pre-computed IPA, issue #698).
    public func phonemes(for text: String) async throws -> String {
        switch variant {
        case .english:
            // Byte-exact NeMo TN before G2P via the shared frontend entry
            // point: "$5" → "five dollars", "2024" → "twenty twenty four".
            // No-op for plain prose.
            let normalized = EnglishTextNormalizer.normalizeForFrontend(text)
            return try await phonemize(text: normalized)
        case .mandarin:
            try await store.loadIfNeeded()
            // Normalize written forms to their Mandarin reading before
            // segmentation — e.g. "$5" → "五美元", "2024年" → "二零二四年" —
            // so the numeric/semiotic tokens reach MandarinG2P as Hanzi.
            let normalized = NemoTextNormalizer.normalize(text, language: .mandarin)
            if MandarinG2P.looksLikeHanzi(normalized) {
                let g2p = try await store.mandarinG2PPipeline()
                return try await g2p.phonemize(normalized)
            } else {
                // No Hanzi present → caller already supplied bopomofo /
                // ASCII punctuation. Pass through so power users can
                // still override pronunciation manually.
                return normalized
            }
        case .japanese:
            // The Japanese variant ships no in-process kana/kanji → IPA
            // frontend. Text synthesis isn't supported; callers feed
            // pre-computed IPA via synthesizeFromPhonemes(_:voice:speed:),
            // which bypasses phonemes(for:) entirely.
            throw KokoroAneError.inputProcessingFailed(
                "Japanese variant has no text G2P frontend; call "
                    + "synthesizeFromPhonemes(_:voice:speed:) with pre-computed IPA (see #698).")
        }
    }

    /// Bypass G2P; feed an already-IPA phoneme string directly.
    ///
    /// For the ``KokoroAneVariant/mandarin`` variant the `phonemes` argument
    /// must be Bopomofo + tone digits + IPA punctuation matching the
    /// `kokoro-82m-coreml/ANE-zh/vocab.json` token set.
    public func synthesizeFromPhonemes(
        _ phonemes: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> Data {
        let result = try await runChain(phonemes: phonemes, voice: voice, speed: speed)
        return try wavData(from: result)
    }

    /// Bypass G2P; return samples + timings.
    public func synthesizeFromPhonemesDetailed(
        _ phonemes: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> KokoroAneSynthesisResult {
        try await runChain(phonemes: phonemes, voice: voice, speed: speed)
    }

    // MARK: - Private

    private func runChain(
        phonemes: String,
        voice: String?,
        speed: Float
    ) async throws -> KokoroAneSynthesisResult {
        try await store.loadIfNeeded()
        let vocab = try await store.vocabulary()
        let voiceName = voice ?? defaultVoice
        let pack = try await store.voicePack(voiceName)

        let inputIds = try vocab.encode(phonemes)
        // Voice pack indexing matches `convert.py:get_ref_data` — row is the
        // raw phoneme-string length (BOS/EOS not counted).
        let phonemeCount = phonemes.count
        let (styleS, styleTimbre) = pack.slice(for: phonemeCount)

        return try await KokoroAneSynthesizer.synthesize(
            inputIds: inputIds,
            styleS: styleS,
            styleTimbre: styleTimbre,
            speed: speed,
            store: store
        )
    }

    /// English text → Misaki-style IPA. Lexicon-first resolution (weak
    /// function-word forms — `to` → `tu`, not the stressed BART citation
    /// form `tˈO`, issue #691), per-word BART G2P fallback for OOV words,
    /// and vocab-supported punctuation kept as prosody/pause tokens.
    private func phonemize(text: String) async throws -> String {
        let phonemizer = await ensureEnglishPhonemizer()
        return try await phonemizer.phonemize(text) { word in
            try await G2PModel.shared.phonemize(word: word)
        }
    }

    /// Build (and cache) the English frontend: chain vocab → allowed
    /// token/punctuation sets, Misaki lexicon cache → weak-form maps.
    /// On any failure returns a transient G2P-only frontend (current
    /// pre-#691 behavior) without caching it, so the lexicon is retried
    /// on the next call.
    private func ensureEnglishPhonemizer() async -> KokoroAneEnglishPhonemizer {
        if let cached = englishPhonemizer { return cached }

        var lower: [String: [String]] = [:]
        var caseSensitive: [String: [String]] = [:]
        var punctuation: Set<Character> = []
        var lexiconLoaded = false

        do {
            try await store.loadIfNeeded()
            let vocab = try await store.vocabulary()
            // Stress/length marks (ˈ ˌ ː) are Unicode modifier letters, so
            // `isLetter` keeps them out of the punctuation set.
            punctuation = Set(
                vocab.map.keys.filter { !$0.isLetter && !$0.isNumber && !$0.isWhitespace })

            if let kokoroDir = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: nil) {
                let allowedTokens = Set(vocab.map.keys.map(String.init))
                try await englishLexiconCache.ensureLoaded(
                    kokoroDirectory: kokoroDir, allowedTokens: allowedTokens)
                let maps = await englishLexiconCache.lexicons()
                lower = maps.word
                caseSensitive = maps.caseSensitive
                lexiconLoaded = true
            }
        } catch {
            logger.warning(
                "English lexicon unavailable (\(error.localizedDescription)); using BART G2P only")
        }

        let phonemizer = KokoroAneEnglishPhonemizer(
            wordToPhonemes: lower,
            caseSensitiveWordToPhonemes: caseSensitive,
            customLexicon: englishCustomLexicon,
            allowedPunctuation: punctuation
        )
        if lexiconLoaded {
            englishPhonemizer = phonemizer
        }
        return phonemizer
    }

    private func wavData(from result: KokoroAneSynthesisResult) throws -> Data {
        do {
            // Japanese writes at the model's native level (no peak-normalization)
            // so the output matches the PyTorch reference instead of being
            // slammed to 0 dBFS. English/Mandarin keep peak-normalization until
            // their tails get the same COLA-corrected iSTFT (#698 follow-up).
            return try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(result.sampleRate),
                normalize: variant != .japanese)
        } catch {
            throw KokoroAneError.audioConversionFailed(error.localizedDescription)
        }
    }
}
