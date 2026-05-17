import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Callback invoked when new tokens are decoded (for live transcription updates).
/// Fires with the running transcript text only — the language tag, if any,
/// is surfaced via `detectedLanguage()`.
public typealias NemotronMultilingualPartialCallback = @Sendable (String) -> Void

/// High-level manager for the Nemotron Speech Streaming Multilingual 0.6B pipeline.
///
/// Distinct from the English `StreamingNemotronAsrManager` because:
///   1. The encoder takes an extra `prompt_id` int32 [1] input per chunk.
///   2. The vocab is ~13k tokens and includes language-tag pieces like
///      `<en-US>` which are filtered from the transcript.
///   3. The channel cache shape is `[1, 24, 56, 1024]` (att_context_size=[56,0]).
///
/// **Local-path-only**: this manager intentionally does not auto-download
/// from HuggingFace because the multilingual model has not been uploaded yet.
/// Callers must point at a directory containing the compiled `.mlmodelc`
/// bundles (or `.mlpackage` archives) plus `metadata.json` and
/// `tokenizer.json`.
public actor StreamingNemotronMultilingualAsrManager {
    private let logger = AppLogger(category: "NemotronMultilingualStreaming")

    // Models
    internal var preprocessor: MLModel?
    internal var encoder: MLModel?
    internal var decoder: MLModel?
    internal var joint: MLModel?

    // Components
    private let audioConverter = AudioConverter()
    internal var tokenizer: NemotronMultilingualTokenizer?

    // Configuration (loaded from metadata.json)
    public private(set) var config: NemotronMultilingualStreamingConfig

    // Audio Buffer
    private var audioBuffer: [Float] = []

    // Accumulated token IDs (raw, including any lang-tag tokens)
    internal var accumulatedTokenIds: [Int] = []

    // First lang-tag piece encountered this session (without angle brackets).
    private var firstDetectedLanguage: String?

    // Encoder cache states
    internal var cacheChannel: MLMultiArray?
    internal var cacheTime: MLMultiArray?
    internal var cacheLen: MLMultiArray?

    // Mel cache (last 9 frames from previous chunk)
    internal var melCache: MLMultiArray?

    // Decoder LSTM states
    internal var hState: MLMultiArray?
    internal var cState: MLMultiArray?
    internal var lastToken: Int32

    // Current prompt id (the language hint). Defaults to `defaultPromptId`
    // ("auto" mode) until the caller sets a specific language.
    private var currentPromptId: Int32

    // Current language code requested by the caller (e.g. "en-US"). Used
    // to look up the matching lang-tag token id when forced-prefix decoding
    // is enabled.
    private var currentLanguageCode: String?

    // When true, after each reset / language change, run the decoder once
    // with the lang-tag token id for `currentLanguageCode` to seed the
    // LSTM state. This is the Whisper-style hard language lock; the
    // encoder still receives `prompt_id` as usual.
    private var useForcedPrefix: Bool = false

    // Callbacks
    internal var partialCallback: NemotronMultilingualPartialCallback?

    // Stats
    internal var processedChunks: Int = 0

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(configuration: MLModelConfiguration? = nil) {
        // Default to `.cpuAndNeuralEngine`: the int8 encoder is ANE-targeted.
        // `MLModelConfiguration()`'s default `.all` routes int8 ops to GPU
        // and runs ~10× slower than the ANE path.
        self.mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        self.config = NemotronMultilingualStreamingConfig()
        self.lastToken = Int32(config.blankIdx)
        self.currentPromptId = Int32(config.defaultPromptId)
    }

    /// Set callback for partial transcription updates
    public func setPartialCallback(_ callback: @escaping NemotronMultilingualPartialCallback) {
        self.partialCallback = callback
    }

    /// Set the language hint by code (e.g. `"en-US"`, `"zh-CN"`, `"auto"`).
    /// Falls back to the model's `default_prompt_id` if the code is unknown.
    public func setLanguage(_ language: String?) async {
        let id = config.promptId(forLanguage: language)
        self.currentPromptId = Int32(id)
        self.currentLanguageCode = language
        logger.info("Prompt id set to \(id) for language \(language ?? "auto")")
        if useForcedPrefix {
            do {
                try await applyForcedPrefixIfNeeded()
            } catch {
                logger.error("Forced prefix seeding failed: \(error.localizedDescription)")
            }
        }
    }

    /// Enable or disable Whisper-style forced-prefix decoding. When enabled,
    /// after each `reset()` or `setLanguage(_:)` call we run the decoder once
    /// with the lang-tag token id matching `currentLanguageCode`, threading
    /// its output `h_out`/`c_out` into the LSTM state and setting
    /// `lastToken` to the lang-tag id. The encoder still gets `prompt_id`.
    public func setForcedPrefix(_ enabled: Bool) async {
        self.useForcedPrefix = enabled
        logger.info("Forced prefix \(enabled ? "enabled" : "disabled")")
        if enabled {
            do {
                try await applyForcedPrefixIfNeeded()
            } catch {
                logger.error("Forced prefix seeding failed: \(error.localizedDescription)")
            }
        }
    }

    /// Whether forced-prefix decoding is currently enabled.
    public func forcedPrefixEnabled() -> Bool { useForcedPrefix }

    /// Set the language hint by raw prompt id (advanced users).
    /// The caller is responsible for ensuring the id is in `[0, numPrompts)`.
    public func setPromptId(_ promptId: Int) {
        self.currentPromptId = Int32(promptId)
    }

    /// Current prompt id (the language hint fed to the encoder).
    public func promptId() -> Int { Int(currentPromptId) }

    /// First language-tag piece (e.g. `"en-US"`) emitted by the decoder this
    /// session, or `nil` if no tag has been seen yet.
    public func detectedLanguage() -> String? { firstDetectedLanguage }

    /// Load models from a directory containing preprocessor, encoder, decoder,
    /// joint, plus `metadata.json` and `tokenizer.json`. Accepts either
    /// `.mlmodelc` (preferred) or uncompiled `.mlpackage` bundles.
    public func loadModels(from directory: URL) async throws {
        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform(
                "Nemotron multilingual int8 streaming models require Apple Silicon (ANE). Intel Macs are not supported."
            )
        }

        logger.info("Loading Nemotron multilingual CoreML models from \(directory.path)...")

        // Load config from metadata.json (required — the prompt dictionary lives here)
        let metadataPath = directory.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.metadata)
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw ASRError.processingFailed(
                "metadata.json not found at \(metadataPath.path). The multilingual variant requires it for prompt_dictionary and lang_tag_token_ids."
            )
        }
        self.config = try NemotronMultilingualStreamingConfig(from: metadataPath)
        self.lastToken = Int32(config.blankIdx)
        self.currentPromptId = Int32(config.defaultPromptId)
        logger.info(
            "Loaded multilingual config: \(config.chunkMs)ms chunks, vocab=\(config.vocabSize), \(config.numPrompts) prompts, default=\(config.defaultPromptId)"
        )

        // Load model bundles (prefer .mlmodelc, fall back to .mlpackage with on-demand compile)
        let preprocessorURL = try await locateModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.preprocessorFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.preprocessorPackage
        )
        self.preprocessor = try await MLModel.load(contentsOf: preprocessorURL, configuration: mlConfiguration)

        let encoderURL = try await locateModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.encoderFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.encoderPackage
        )
        self.encoder = try await MLModel.load(contentsOf: encoderURL, configuration: mlConfiguration)

        let decoderURL = try await locateModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.decoderFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.decoderPackage
        )
        self.decoder = try await MLModel.load(contentsOf: decoderURL, configuration: mlConfiguration)

        let jointURL = try await locateModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.jointFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.jointPackage
        )
        self.joint = try await MLModel.load(contentsOf: jointURL, configuration: mlConfiguration)

        // Load tokenizer with lang-tag filter set
        let tokenizerURL = directory.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.tokenizer)
        self.tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tokenizerURL,
            langTagTokenIds: config.langTagTokenIds
        )

        // Initialize states
        try resetStates()

        logger.info(
            "Nemotron multilingual models loaded successfully (\(config.chunkMs)ms chunks)."
        )
    }

    private func locateModelBundle(in directory: URL, compiled: String, uncompiled: String) async throws -> URL {
        let compiledURL = directory.appendingPathComponent(compiled)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }
        let uncompiledURL = directory.appendingPathComponent(uncompiled)
        if FileManager.default.fileExists(atPath: uncompiledURL.path) {
            // `MLModel.load(contentsOf:)` requires a compiled `.mlmodelc`. Compile
            // the `.mlpackage` to a sibling `.mlmodelc` (cached for reuse).
            let baseName = (compiled as NSString).deletingPathExtension
            let cachedCompiledURL = directory.appendingPathComponent("\(baseName).mlmodelc")
            if FileManager.default.fileExists(atPath: cachedCompiledURL.path) {
                return cachedCompiledURL
            }
            logger.info("Compiling \(uncompiled) to .mlmodelc (first run only)...")
            let tempCompiledURL = try await MLModel.compileModel(at: uncompiledURL)
            // Try to cache next to the .mlpackage so subsequent loads skip
            // compilation. Falls back to the temp URL if the directory isn't
            // writable.
            do {
                if FileManager.default.fileExists(atPath: cachedCompiledURL.path) {
                    try FileManager.default.removeItem(at: cachedCompiledURL)
                }
                try FileManager.default.moveItem(at: tempCompiledURL, to: cachedCompiledURL)
                return cachedCompiledURL
            } catch {
                logger.warning(
                    "Could not cache compiled model next to .mlpackage (\(error.localizedDescription)); using temp path."
                )
                return tempCompiledURL
            }
        }
        throw ASRError.processingFailed(
            "Could not find \(compiled) or \(uncompiled) in \(directory.path)"
        )
    }

    /// Reset all states for a new transcription session.
    /// Preserves the currently selected prompt id and ml configuration.
    public func reset() async {
        StreamingAsrUtils.resetSharedState(
            audioBuffer: &audioBuffer,
            accumulatedTokenIds: &accumulatedTokenIds,
            processedChunks: &processedChunks
        )
        firstDetectedLanguage = nil
        do {
            try resetStates()
        } catch {
            logger.error("Failed to reset states: \(error.localizedDescription)")
        }
        if useForcedPrefix {
            do {
                try await applyForcedPrefixIfNeeded()
            } catch {
                logger.error("Forced prefix seeding failed: \(error.localizedDescription)")
            }
        }
    }

    /// Run the decoder once with the lang-tag token for the currently selected
    /// language and write the resulting state back to `hState`/`cState`/`lastToken`.
    /// No-op if forced-prefix is disabled, no language is set, the tokenizer/
    /// decoder isn't loaded, or the language has no matching lang-tag token.
    private func applyForcedPrefixIfNeeded() async throws {
        guard useForcedPrefix,
            let language = currentLanguageCode,
            let tokenizer = tokenizer,
            let decoder = decoder,
            let h = hState,
            let c = cState
        else { return }

        guard let langTagId = tokenizer.langTagTokenId(forLanguage: language) else {
            logger.info("Forced prefix: no lang-tag token for \(language); skipping seed")
            return
        }

        let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenInput[0] = NSNumber(value: langTagId)

        let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
        tokenLen[0] = 1

        let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokenInput),
            "token_length": MLFeatureValue(multiArray: tokenLen),
            "h_in": MLFeatureValue(multiArray: h),
            "c_in": MLFeatureValue(multiArray: c),
        ])

        let decoderOutput = try await decoder.prediction(from: decoderInput)
        guard let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
            let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
        else {
            logger.warning("Forced prefix: decoder did not return h_out/c_out; skipping")
            return
        }

        self.hState = hOut
        self.cState = cOut
        self.lastToken = Int32(langTagId)
        // Mirror what the pipeline would do when it observes a lang-tag token.
        recordDetectedLanguage(language)
        logger.info("Forced prefix: seeded decoder with lang-tag token id \(langTagId) for \(language)")
    }

    public func cleanup() async {
        await reset()
        preprocessor = nil
        encoder = nil
        decoder = nil
        joint = nil
        tokenizer = nil
        cacheChannel = nil
        cacheTime = nil
        cacheLen = nil
        melCache = nil
        hState = nil
        cState = nil
        logger.info("StreamingNemotronMultilingualAsrManager resources cleaned up")
    }

    private func resetStates() throws {
        let cacheConfig = EncoderCacheManager.CacheConfig(
            channelShape: config.cacheChannelShape,
            timeShape: config.cacheTimeShape,
            lenShape: [1]
        )
        let caches = try EncoderCacheManager.createInitialCaches(config: cacheConfig)
        cacheChannel = caches.channel
        cacheTime = caches.time
        cacheLen = caches.len
        // Seed cache_len with 1 instead of 0 so the encoder's
        // `ios17.slice_by_index` op never sees a zero-length slice, which would
        // fail CoreML shape inference and skip MPSGraph caching on every
        // session start. The cache buffers are zero, so this is equivalent to
        // 1 frame of silence preamble.
        cacheLen?[0] = 1

        // Mel cache (will be initialized on first chunk)
        melCache = nil

        // Decoder LSTM states
        hState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden]
        )

        cState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden]
        )

        lastToken = Int32(config.blankIdx)
    }

    /// Append audio buffer for processing
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        try StreamingAsrUtils.appendAudio(buffer, using: audioConverter, to: &audioBuffer)
    }

    /// Process audio. Returns the empty string because the partial transcript
    /// is delivered via the partial callback or `getPartialTranscript()`.
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        guard preprocessor != nil, encoder != nil, decoder != nil, joint != nil else {
            throw ASRError.notInitialized
        }

        let samples = try audioConverter.resampleBuffer(audioBuffer)
        self.audioBuffer.append(contentsOf: samples)

        while self.audioBuffer.count >= config.chunkSamples {
            let chunk = Array(self.audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            // Recheck buffer count after await to handle actor reentrancy
            let samplesToRemove = min(config.chunkSamples, self.audioBuffer.count)
            self.audioBuffer.removeFirst(samplesToRemove)
        }

        return ""
    }

    /// Finish processing remaining audio (padded if needed) and return the
    /// final transcript text. The detected language is available via
    /// `detectedLanguage()` after this returns.
    public func finish() async throws -> String {
        guard let tokenizer = tokenizer,
            preprocessor != nil,
            encoder != nil,
            decoder != nil,
            joint != nil
        else {
            throw ASRError.notInitialized
        }

        if !audioBuffer.isEmpty {
            let paddingNeeded = config.chunkSamples - audioBuffer.count
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }

            let chunk = Array(audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            audioBuffer.removeAll()
        }

        let decoded = tokenizer.decode(ids: accumulatedTokenIds)
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = decoded.detectedLanguage
        }
        accumulatedTokenIds.removeAll()

        return decoded.text
    }

    /// Get current partial transcript without finishing
    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        let decoded = tokenizer.decode(ids: accumulatedTokenIds)
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = decoded.detectedLanguage
        }
        return decoded.text
    }

    /// Internal getter for the current prompt id, used by the pipeline.
    internal func currentPromptIdValue() -> Int32 { currentPromptId }

    /// Internal setter used by the pipeline when it encounters a lang-tag
    /// token in the decoder output.
    internal func recordDetectedLanguage(_ language: String) {
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = language
        }
    }
}
