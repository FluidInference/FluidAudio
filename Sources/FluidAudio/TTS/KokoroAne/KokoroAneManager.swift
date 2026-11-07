#if TTS
import Foundation

/// High-level facade for the legacy Kokoro chain or opt-in ANE-v3 runtime
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
///   * IPA input capped at 512 tokens. The high-level text API
///     (``synthesize(text:voice:speed:)`` / ``synthesizeDetailed(text:voice:speed:)``)
///     auto-chunks longer prompts at whitespace / pause punctuation (#712, #940);
///     the low-level ``synthesizeFromPhonemes(_:voice:speed:)`` stays strict
///     and throws ``KokoroAneError/phonemeSequenceTooLong(_:)`` past the cap.
///   * Loads from HF path `kokoro-82m-coreml/ANE/` (English, Spanish,
///     French), `ANE-zh/` (Mandarin) or `ANE-ja/` (Japanese).
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
    /// Set once ``prepareEnglishFrontend()`` has fetched and loaded the
    /// English G2P assets.
    private var englishFrontendReady = false
    /// BART G2P fallback bound to the store's models root (#1023).
    nonisolated let englishG2P: G2PModel
    /// ``setPunctuationPauses(_:)``, in acoustic frames.
    private var punctuationPauses: [Character: Int32] = [:]

    /// Select `.v3` for the pinned static-bucket pipeline (macOS 15 / iOS 18+).
    /// A supplied model store owns the runtime version and compute policy.
    public init(
        variant: KokoroAneVariant = .english,
        defaultVoice: String? = nil,
        directory: URL? = nil,
        computeUnits: KokoroAneComputeUnits = .default,
        modelStore: KokoroAneModelStore? = nil,
        version: KokoroAneVersion = .legacy
    ) {
        self.variant = variant
        self.defaultVoice = defaultVoice ?? variant.defaultVoice
        let store =
            modelStore
            ?? KokoroAneModelStore(
                directory: directory, computeUnits: computeUnits, variant: variant, version: version)
        self.store = store
        self.englishG2P = G2PModel.shared(directory: store.directory)
    }

    // MARK: - Lifecycle

    /// Download (if missing), load the selected runtime + vocab + default voice
    /// pack. Optionally pre-warm additional voice packs.
    public func initialize(preloadVoices: Set<String>? = nil) async throws {
        if let advisory = Self.osAdvisory(for: ProcessInfo.processInfo.operatingSystemVersion) {
            logger.warning(advisory)
        }
        try await store.loadIfNeeded()
        // English G2P CoreML assets live in the kokoro repo under
        // `<models root>/kokoro/`. The Mandarin variant routes through the
        // in-process MandarinG2P pipeline (loaded by store.loadIfNeeded())
        // and never touches the BART G2P model, so the English G2P bundle
        // would just be wasted bandwidth + memory.
        if variant == .english {
            try await prepareEnglishFrontend()
        }
        switch variant {
        case .french:
            // Lexicon + CharsiuG2P fallback, fetched up front so the first
            // synthesis does not stall on (or fail without) the network.
            _ = try await store.frenchG2PPipeline()
            try await KokoroAneResourceDownloader.ensureMultilingualG2PAssets(directory: store.directory)
        case .spanish:
            _ = await store.spanishLexicon()
        case .english, .mandarin, .japanese:
            break
        }
        if let voices = preloadVoices {
            for voice in voices {
                _ = try await store.voicePack(voice)
            }
        }
    }

    #if os(macOS)
    private static let runningOnMacOS = true
    #else
    private static let runningOnMacOS = false
    #endif

    /// The 26.4+ OS line carries an Apple BNNS bug that can intermittently
    /// crash synthesis in libBNNS on any compute-unit routing
    /// (#328/#587/#667/#817). macOS 26.6 fixes it (verified, #817); iOS 26.6
    /// still crashes with the identical signature (#844), and iOS 27.0 crashes
    /// in libBNNS (`vadd_fp16_sme_internal`) on the Metal-free route that is
    /// the 27 default, while the Metal route aborts in MPSGraph (#843, #889).
    /// So on non-macOS everything from 26.4 on stays flagged until a build is
    /// shown to be safe. macOS 27 has no BNNS report and is not flagged; its
    /// MPSGraph abort (#1103) is on the GPU route, which the 27 default avoids.
    static func isBnnsCrashProneOS(
        _ version: OperatingSystemVersion, onMacOS: Bool = runningOnMacOS
    ) -> Bool {
        if version.majorVersion >= 27 { return !onMacOS }
        guard version.majorVersion == 26, version.minorVersion >= 4 else { return false }
        return onMacOS ? version.minorVersion <= 5 : true
    }

    /// The warning `initialize()` logs on a crash-prone OS build, or nil.
    /// Route-aware: on the iOS 27 line neither Core ML route is known to be
    /// safe (#889), which is a different message from the 26.x BNNS bug.
    static func osAdvisory(
        for version: OperatingSystemVersion, onMacOS: Bool = runningOnMacOS
    ) -> String? {
        guard isBnnsCrashProneOS(version, onMacOS: onMacOS) else { return nil }
        if version.majorVersion >= 27 {
            return
                "iOS/iPadOS 27: no Core ML route for Kokoro ANE is known to be safe. "
                + "The default Metal-free route (noise + tail on CPU) has crashed in libBNNS "
                + "(vadd_fp16_sme_internal SIGSEGV) after ~1 h of synthesis, and the Metal "
                + "route aborts in MPSGraph within minutes. Both are uncatchable in-process. "
                + "Consider disabling Kokoro ANE on this OS line until a safe route is shown. "
                + "See https://github.com/FluidInference/FluidAudio/issues/889"
        }
        return
            "This OS build has a known Apple BNNS bug that can "
            + "intermittently crash Kokoro synthesis (EXC_BAD_ACCESS in libBNNS) "
            + "regardless of compute-unit routing. macOS 26.6 fixes it; on iOS "
            + "the 26.6 line still crashes. "
            + "See https://github.com/FluidInference/FluidAudio/issues/844"
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

    /// Set how long the pause at a punctuation mark lasts, replacing the
    /// pause Kokoro predicts (Mandarin `、` and `，` get only 50–125 ms).
    ///
    /// Keys are marks as written in the input text. Mandarin text keeps
    /// `、` and `，` apart even though both reach the model as `,`; phoneme
    /// input and the other languages are matched on the marks in the
    /// phoneme string. A mark without an entry of its own uses the entry
    /// for the mark the model sees, so `","` also covers `，` and `、`.
    /// Consecutive marks share the longest of their pauses.
    ///
    /// Marks the frontend adds (the Mandarin `/` between words) and
    /// whitespace are not keys.
    ///
    /// The duration is rounded to Kokoro's 25 ms acoustic frame and is not
    /// scaled by `speed`. A longer pause than the predicted one is rendered
    /// as silence; a shorter one trims the predicted pause. When the pauses
    /// make a pass longer than its 2000-frame (50 s) cap, the text is split
    /// at a pause and rendered in parts. Applies to all later synthesis on
    /// this manager; pass `[:]` to restore the predicted pauses.
    public func setPunctuationPauses(_ pauses: [Character: Duration]) {
        punctuationPauses = pauses.mapValues(KokoroAnePauses.frames(for:))
    }

    /// Install (or clear) a user-supplied English pronunciation override.
    ///
    /// Entries map a word to a Misaki-style IPA string (e.g.
    /// `["to": "tə", "GIF": "ʤˈɪf"]`). The exact spelling is checked
    /// first, then the lower-cased form, before the bundled Misaki
    /// lexicon and the BART G2P fallback. Pass `[:]` to clear.
    ///
    /// Used by the English variant's text path, and on any variant by
    /// ``englishPhonemes(for:)``. It does not change how Mandarin text is
    /// read (use ``setMandarinCustomLexicon(_:)`` there).
    public func setEnglishCustomLexicon(_ entries: [String: String]) {
        englishCustomLexicon = entries
        // Rebuild the cached frontend with the new overrides on next use.
        englishPhonemizer = nil
    }

    /// English text → phonemes in this variant's vocab, through the English
    /// frontend (NeMo text normalization, Misaki lexicon, per-word BART G2P
    /// fallback), without loading the English chain.
    ///
    /// For callers that mix English into another variant's input and feed
    /// the result to ``synthesizeFromPhonemes(_:voice:speed:)``: Kokoro
    /// v1.1-zh was trained with English runs phonemized this way (misaki's
    /// `ZHG2P(version="1.1", en_callable=…)`). How to split mixed text is
    /// left to the caller. Punctuation and lexicon entries are limited to
    /// this variant's vocab, so every token is one its chain can encode.
    /// The first call fetches the English G2P assets
    /// (``prepareEnglishFrontend()``); ``setEnglishCustomLexicon(_:)``
    /// applies.
    public func englishPhonemes(for text: String) async throws -> String {
        try await prepareEnglishFrontend()
        return try await phonemize(text: EnglishTextNormalizer.normalizeForFrontend(text))
    }

    /// Fetch (if missing) and load the English G2P assets that
    /// ``englishPhonemes(for:)`` needs. The English variant does this in
    /// ``initialize(preloadVoices:)``; other variants can call it up front so
    /// the first English text doesn't wait for it. Idempotent.
    ///
    /// The assets live under the store's `directory` (`<directory>/kokoro/`),
    /// next to the CoreML chain, or the default TTS cache when none was given.
    public func prepareEnglishFrontend() async throws {
        guard !englishFrontendReady else { return }
        // G2PModel.loadIfNeeded only reads from disk (it never downloads),
        // so fetch the assets explicitly before warming the in-process model.
        try await KokoroAneResourceDownloader.ensureG2PAssets(directory: store.directory)
        try await englishG2P.ensureModelsAvailable()
        // Best-effort pre-fetch of the Misaki lexicon cache (weak
        // function-word forms, issue #691). Missing lexicon degrades
        // to the BART-G2P-only path rather than failing.
        _ = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: store.directory)
        englishFrontendReady = true
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
        let frontend = try await resolveFrontend(for: text)
        // Chunk after normalization / G2P so written forms are never split; the
        // cap is in Unicode scalars, as the vocab encoder counts it (#712, #940).
        let chunks = Self.chunks(of: frontend.phonemes, sources: frontend.punctuationSources)
        guard chunks.count > 1 else {
            return try await runChain(
                phonemes: frontend.phonemes, normalizedText: frontend.normalizedText, voice: voice, speed: speed,
                punctuationSources: frontend.punctuationSources)
        }
        var parts: [KokoroAneSynthesisResult] = []
        for chunk in chunks {
            try Task.checkCancellation()
            parts.append(
                try await runChain(
                    phonemes: chunk.phonemes, normalizedText: nil, voice: voice, speed: speed,
                    punctuationSources: chunk.sources))
        }
        var result = KokoroAneSynthesisResult.concatenating(parts)
        result.normalizedText = frontend.normalizedText
        result.phonemes = frontend.phonemes
        return result
    }

    /// `phonemes` split to the per-pass cap, each chunk with the
    /// `sources` inside it rebased to the chunk.
    static func chunks(
        of phonemes: String, sources: [Int: Character]?
    ) -> [(phonemes: String, sources: [Int: Character]?)] {
        let scalars = phonemes.unicodeScalars.map(Character.init)
        return PhonemeChunker.chunkRanges(scalars, maxLength: KokoroAneConstants.maxPhonemeLength).map {
            KokoroAnePauses.piece(of: scalars, $0, sources: sources)
        }
    }

    /// Resolve the exact phoneme string ``synthesize(text:voice:speed:)``
    /// would feed the 7-stage chain — for diagnostics, tests, and
    /// caller-side phoneme caching (issue #691).
    ///
    /// English: Misaki-lexicon-first with BART G2P fallback. Mandarin:
    /// the ``MandarinG2P`` pipeline for Hanzi input, pass-through for
    /// strings that already look like phonemes. Japanese: half-width kana
    /// and range-tilde folding, NeMo written-form normalization, then the
    /// in-process Cutlet port (MeCab over unidic-lite + Cutlet rules, the
    /// Kokoro training frontend). A string made only of phoneme-alphabet
    /// scalars is treated as pre-computed IPA and passed through (issue
    /// #698); digits, kana and kanji always go through the frontend.
    /// Spanish: NeMo normalization, then ``SpanishG2P`` spelling rules with
    /// the `es_lexicon_cache.json` exceptions. French: NeMo normalization,
    /// then `fr_lexicon_cache.json` with a CharsiuG2P fallback (``FrenchG2P``). Both emit espeak-ng-style IPA;
    /// pre-computed IPA goes through ``synthesizeFromPhonemes(_:voice:speed:)``.
    public func phonemes(for text: String) async throws -> String {
        try await resolveFrontend(for: text).phonemes
    }

    /// Text normalization + G2P, returning both halves so
    /// ``synthesizeDetailed(text:voice:speed:)`` can report the normalized
    /// text it actually spoke (issue #943). `normalizedText` is `nil` when the
    /// input was treated as pre-computed phonemes.
    ///
    /// `punctuationSources` maps the Unicode-scalar offset of each
    /// punctuation mark from the input to the mark as written (Mandarin
    /// text only; `nil` elsewhere, see ``KokoroAnePauses/spans``).
    func resolveFrontend(
        for text: String
    ) async throws -> (normalizedText: String?, phonemes: String, punctuationSources: [Int: Character]?) {
        switch variant {
        case .english:
            // Byte-exact NeMo TN before G2P via the shared frontend entry
            // point: "$5" → "five dollars", "2024" → "twenty twenty four".
            // No-op for plain prose.
            let normalized = EnglishTextNormalizer.normalizeForFrontend(text)
            return (normalized, try await phonemize(text: normalized), nil)
        case .mandarin:
            try await store.loadIfNeeded()
            // Normalize written forms to their Mandarin reading before
            // segmentation — e.g. "$5" → "五美元", "2024年" → "二零二四年" —
            // so the numeric/semiotic tokens reach MandarinG2P as Hanzi.
            var normalized = NemoTextNormalizer.normalize(text, language: .mandarin)
            // Without the engine linked (`NemoTextProcessing` trait off), a
            // numeric-only input ("$5.50", "99%") has no Hanzi and would fall
            // into the bopomofo passthrough below, reading digits as tones.
            // MandarinNumberNormalizer covers those forms so the gate sees Hanzi.
            if !NemoTextNormalizer.isAvailable, !MandarinG2P.looksLikeHanzi(normalized) {
                normalized = MandarinNumberNormalizer.normalize(normalized)
            }
            if MandarinG2P.looksLikeHanzi(normalized) {
                let g2p = try await store.mandarinG2PPipeline()
                let traced = try await g2p.phonemizeTracingPunctuation(normalized)
                return (normalized, traced.phonemes, traced.punctuationSources)
            } else {
                // No Hanzi present → caller already supplied bopomofo /
                // ASCII punctuation. Pass through so power users can
                // still override pronunciation manually.
                return (nil, normalized, nil)
            }
        case .japanese:
            // Pre-computed IPA (issue #698) passes through untouched: NFKC
            // would fold its modifier letters (ʲ → j). Anything outside the
            // phoneme alphabet — kana, kanji, half-width kana, digits — is
            // text and goes through normalization and the frontend.
            guard !Self.looksLikePrecomputedJapaneseIPA(text) else { return (nil, text, nil) }
            // The NeMo FST drops half-width dakuten (ｶﾞ → カ) and reads the
            // full-width tilde as a symbol, so fold both before it runs.
            let folded = JapaneseCutlet.foldingHalfWidthForms(text)
            let normalized = NemoTextNormalizer.normalize(folded, language: .japanese)
            let g2p = try await store.japaneseG2PPipeline()
            return (normalized, try await g2p.phonemize(normalized), nil)
        case .spanish:
            var normalized = NemoTextNormalizer.normalize(text, language: .spanish)
            // Without the engine linked, read numbers here rather than let the
            // tokenizer drop them.
            if !NemoTextNormalizer.isAvailable {
                normalized = RomanceNumberNormalizer.normalize(normalized, language: .spanish)
            }
            let lexicon = await store.spanishLexicon()
            return (normalized, try Self.nonEmpty(SpanishG2P.phonemize(normalized, lexicon: lexicon), for: text), nil)
        case .french:
            var normalized = NemoTextNormalizer.normalize(text, language: .french)
            if !NemoTextNormalizer.isAvailable {
                normalized = RomanceNumberNormalizer.normalize(normalized, language: .french)
            }
            let g2p = try await store.frenchG2PPipeline()
            return (normalized, try Self.nonEmpty(await g2p.phonemize(normalized), for: text), nil)
        }
    }

    private static func nonEmpty(_ phonemes: String, for text: String) throws -> String {
        guard phonemes.contains(where: { $0.isLetter }) else {
            throw KokoroAneError.inputProcessingFailed("G2P produced no phonemes for '\(text)'.")
        }
        return phonemes
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
        let result = try await runChain(phonemes: phonemes, normalizedText: nil, voice: voice, speed: speed)
        return try wavData(from: result)
    }

    /// Bypass G2P; return samples + timings.
    public func synthesizeFromPhonemesDetailed(
        _ phonemes: String,
        voice: String? = nil,
        speed: Float = KokoroAneConstants.defaultSpeed
    ) async throws -> KokoroAneSynthesisResult {
        try await runChain(phonemes: phonemes, normalizedText: nil, voice: voice, speed: speed)
    }

    // MARK: - Private

    private func runChain(
        phonemes: String,
        normalizedText: String?,
        voice: String?,
        speed: Float,
        punctuationSources: [Int: Character]? = nil
    ) async throws -> KokoroAneSynthesisResult {
        try await store.loadIfNeeded()
        let vocab = try await store.vocabulary()
        let voiceName = voice ?? defaultVoice
        let pack = try await store.voicePack(voiceName)

        let inputIds = try vocab.encode(phonemes)
        // Voice pack indexing matches `convert.py:get_ref_data` — row is the
        // raw phoneme-string length in scalars (BOS/EOS not counted).
        let phonemeCount = KokoroAneVocab.phonemeLength(phonemes)
        let (styleS, styleTimbre) = pack.slice(for: phonemeCount)
        let pauses = KokoroAnePauses.spans(
            phonemes: phonemes, vocab: vocab, sources: punctuationSources, pauses: punctuationPauses)

        var result: KokoroAneSynthesisResult
        do {
            if store.version == .v3 {
                result = try await store.synthesizeV3(
                    inputIds: inputIds, styleS: styleS, styleTimbre: styleTimbre, speed: speed, pauses: pauses)
            } else {
                result = try await KokoroAneSynthesizer.synthesize(
                    inputIds: inputIds, styleS: styleS, styleTimbre: styleTimbre, speed: speed, pauses: pauses,
                    store: store)
            }
        } catch let error as KokoroAneError {
            // Long pauses can push a pass past the frame cap; render the
            // halves on either side of a pause separately.
            guard case .acousticFramesExceedCap = error,
                let pieces = KokoroAnePauses.split(phonemes, atPauseNearestMiddle: pauses, sources: punctuationSources)
            else { throw error }
            var parts: [KokoroAneSynthesisResult] = []
            for piece in pieces {
                try Task.checkCancellation()
                parts.append(
                    try await runChain(
                        phonemes: piece.phonemes, normalizedText: nil, voice: voice, speed: speed,
                        punctuationSources: piece.sources))
            }
            result = KokoroAneSynthesisResult.concatenating(parts)
        }
        result.normalizedText = normalizedText
        result.phonemes = phonemes
        return result
    }

    /// English text → Misaki-style IPA. Lexicon-first resolution (weak
    /// function-word forms — `to` → `tu`, not the stressed BART citation
    /// form `tˈO`, issue #691), per-word BART G2P fallback for OOV words,
    /// and vocab-supported punctuation kept as prosody/pause tokens.
    private func phonemize(text: String) async throws -> String {
        let phonemizer = await ensureEnglishPhonemizer()
        let raw = try await phonemizer.phonemize(text) { [englishG2P] word in
            try await englishG2P.phonemize(word: word)
        }
        return Self.misakiOutputForm(raw)
    }

    /// misaki's flap step for Kokoro v1.0 (`G2P.__call__`): `ɾ` → `T`. The
    /// lexicon stores the raw flap ("metal" was read as "mattle"). misaki also
    /// maps the glottal stop `ʔ` → `t`, but this chain reads `ʔn` better than
    /// `tn` ("button" became "butt"), so `ʔ` is kept.
    static func misakiOutputForm(_ phonemes: String) -> String {
        phonemes.replacingOccurrences(of: "ɾ", with: "T")
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

            if let kokoroDir = await KokoroAneResourceDownloader.ensureEnglishLexicon(directory: store.directory) {
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
            // All variants write at the model's native level (no
            // peak-normalization) so the output matches the PyTorch reference
            // instead of being slammed to 0 dBFS. Requires the COLA-corrected
            // KokoroTail_v2 (#852).
            return try AudioWAV.data(
                from: result.samples,
                sampleRate: Double(result.sampleRate),
                normalize: false)
        } catch {
            throw KokoroAneError.audioConversionFailed(error.localizedDescription)
        }
    }

    /// Every scalar is one Kokoro's Japanese phoneme strings can contain:
    /// ASCII letters, punctuation and space, IPA and modifier letters,
    /// combining marks, and the quotes/dashes the frontend emits. Digits,
    /// kana (full- or half-width) and kanji are text, not phonemes.
    static func looksLikePrecomputedJapaneseIPA(_ text: String) -> Bool {
        !text.isEmpty
            && text.unicodeScalars.allSatisfy { scalar in
                switch scalar.value {
                case 0x20, 0x21...0x2F, 0x3A...0x40, 0x5B...0x60, 0x7B...0x7E: return true  // ASCII punctuation, space
                case 0x41...0x5A, 0x61...0x7A: return true  // ASCII letters
                case 0x00C0...0x024F: return true  // Latin-1 / Extended-A/B (ɡ ǀ ß …)
                case 0x0250...0x02AF: return true  // IPA extensions
                case 0x02B0...0x02FF: return true  // spacing modifier letters (ʲ ʰ ː)
                case 0x0300...0x036F: return true  // combining diacritics
                case 0x0370...0x03FF: return true  // Greek (β)
                case 0x1D00...0x1DBF: return true  // phonetic extensions (ᵝ)
                case 0x2010...0x2027, 0x2039...0x203A, 0x00AB, 0x00BB: return true  // dashes, quotes, ellipsis
                default: return false
                }
            }
    }
}
#endif
