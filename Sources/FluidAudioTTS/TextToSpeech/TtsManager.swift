import FluidAudio
import Foundation
import OSLog

/// Manages text-to-speech synthesis using the Kokoro CoreML model.
///
/// - Note: **Beta:** The TTS system is currently in beta and only supports American English.
///   Additional language support is planned for future releases.
///
/// Example usage:
/// ```swift
/// let manager = TtSManager()
/// try await manager.initialize()
/// let audioData = try await manager.synthesize(text: "Hello, world!")
/// ```
public final class TtSManager {

    private let logger = AppLogger(category: "TtSManager")
    private let modelCache: KokoroModelCache
    private let lexiconAssets: LexiconAssetManager

    private var ttsModels: TtsModels?
    private var isInitialized = false
    private var assetsReady = false
    private var defaultVoice: String
    private var defaultSpeakerId: Int
    private var ensuredVoices: Set<String> = []

    /// Custom pronunciation dictionary that takes precedence over built-in lexicons.
    private var customLexicon: TtsCustomLexicon?

    /// Creates a new TTS manager.
    ///
    /// - Parameters:
    ///   - defaultVoice: Default voice identifier for synthesis.
    ///   - defaultSpeakerId: Default speaker ID for multi-speaker voices.
    ///   - modelCache: Cache for loaded CoreML models.
    ///   - customLexicon: Optional custom pronunciation dictionary. Entries in this dictionary
    ///     take precedence over all built-in dictionaries and grapheme-to-phoneme conversion.
    public init(
        defaultVoice: String = TtsConstants.recommendedVoice,
        defaultSpeakerId: Int = 0,
        modelCache: KokoroModelCache = KokoroModelCache(),
        customLexicon: TtsCustomLexicon? = nil
    ) {
        self.modelCache = modelCache
        self.lexiconAssets = LexiconAssetManager()
        self.defaultVoice = Self.normalizeVoice(defaultVoice)
        self.defaultSpeakerId = defaultSpeakerId
        self.customLexicon = customLexicon
    }

    init(
        defaultVoice: String = TtsConstants.recommendedVoice,
        defaultSpeakerId: Int = 0,
        modelCache: KokoroModelCache = KokoroModelCache(),
        lexiconAssets: LexiconAssetManager,
        customLexicon: TtsCustomLexicon? = nil
    ) {
        self.modelCache = modelCache
        self.lexiconAssets = lexiconAssets
        self.defaultVoice = Self.normalizeVoice(defaultVoice)
        self.defaultSpeakerId = defaultSpeakerId
        self.customLexicon = customLexicon
    }

    public var isAvailable: Bool {
        isInitialized
    }

    public func initialize(
        models: TtsModels,
        preloadVoices: Set<String>? = nil
    ) async throws {
        self.ttsModels = models

        await modelCache.registerPreloadedModels(models)
        try await prepareLexiconAssetsIfNeeded()
        try await preloadVoiceEmbeddings(preloadVoices)
        try await KokoroSynthesizer.loadSimplePhonemeDictionary()
        try await modelCache.loadModelsIfNeeded(variants: models.availableVariants)
        isInitialized = true
        logger.notice("TtSManager initialized with provided models")
    }

    public func initialize(preloadVoices: Set<String>? = nil) async throws {
        let models = try await TtsModels.download()
        try await initialize(models: models, preloadVoices: preloadVoices)
    }

    public func synthesize(
        text: String,
        voice: String? = nil,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0,
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws -> Data {
        let detailed = try await synthesizeDetailed(
            text: text,
            voice: voice,
            voiceSpeed: voiceSpeed,
            speakerId: speakerId,
            variantPreference: variantPreference
        )
        return detailed.audio
    }

    public func synthesizeDetailed(
        text: String,
        voice: String? = nil,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0,
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws -> KokoroSynthesizer.SynthesisResult {
        guard isInitialized else {
            throw TTSError.modelNotFound("Kokoro model not initialized")
        }

        try await prepareLexiconAssetsIfNeeded()

        let preprocessing = TtsTextPreprocessor.preprocessDetailed(text)
        let cleanedText = try KokoroSynthesizer.sanitizeInput(preprocessing.text)
        let selectedVoice = resolveVoice(voice, speakerId: speakerId)
        try await ensureVoiceEmbeddingIfNeeded(for: selectedVoice)

        return try await KokoroSynthesizer.withLexiconAssets(lexiconAssets) {
            try await KokoroSynthesizer.withModelCache(modelCache) {
                try await KokoroSynthesizer.withCustomLexicon(customLexicon) {
                    try await KokoroSynthesizer.synthesizeDetailed(
                        text: cleanedText,
                        voice: selectedVoice,
                        voiceSpeed: voiceSpeed,
                        variantPreference: variantPreference,
                        phoneticOverrides: preprocessing.phoneticOverrides
                    )
                }
            }
        }
    }

    public func synthesizeToFile(
        text: String,
        outputURL: URL,
        voice: String? = nil,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0,
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let audioData = try await synthesize(
            text: text,
            voice: voice,
            voiceSpeed: voiceSpeed,
            speakerId: speakerId,
            variantPreference: variantPreference
        )

        try audioData.write(to: outputURL)
        logger.notice("Saved synthesized audio to: \(outputURL.lastPathComponent)")
    }

    public func setDefaultVoice(_ voice: String, speakerId: Int = 0) async throws {
        let normalized = Self.normalizeVoice(voice)
        try await ensureVoiceEmbeddingIfNeeded(for: normalized)
        defaultVoice = normalized
        defaultSpeakerId = speakerId
        ensuredVoices.insert(normalized)
    }

    /// Sets or updates the custom pronunciation dictionary.
    ///
    /// Custom lexicon entries take precedence over built-in dictionaries and
    /// grapheme-to-phoneme conversion. Pass `nil` to clear the custom lexicon.
    ///
    /// - Parameter lexicon: The custom lexicon to use, or `nil` to clear.
    public func setCustomLexicon(_ lexicon: TtsCustomLexicon?) {
        customLexicon = lexicon
    }

    /// Returns the current custom lexicon, if any.
    public var currentCustomLexicon: TtsCustomLexicon? {
        customLexicon
    }

    private func resolveVoice(_ requested: String?, speakerId: Int) -> String {
        guard let requested = requested?.trimmingCharacters(in: .whitespacesAndNewlines), !requested.isEmpty else {
            return voiceName(for: speakerId)
        }
        return requested
    }

    public func cleanup() {
        ttsModels = nil
        isInitialized = false
        assetsReady = false
        ensuredVoices.removeAll(keepingCapacity: false)
    }

    private func voiceName(for speakerId: Int) -> String {
        if speakerId == defaultSpeakerId {
            return defaultVoice
        }
        let voices = TtsConstants.availableVoices
        guard !voices.isEmpty else { return defaultVoice }
        let index = abs(speakerId) % voices.count
        return voices[index]
    }

    private func prepareLexiconAssetsIfNeeded() async throws {
        if assetsReady { return }
        try await lexiconAssets.ensureCoreAssets()
        assetsReady = true
    }

    private func ensureVoiceEmbeddingIfNeeded(for voice: String) async throws {
        if ensuredVoices.contains(voice) { return }
        try await TtsResourceDownloader.ensureVoiceEmbedding(voice: voice)
        ensuredVoices.insert(voice)
    }

    private func preloadVoiceEmbeddings(_ requestedVoices: Set<String>?) async throws {
        var voices = requestedVoices ?? Set<String>()
        voices.insert(defaultVoice)

        for voice in voices {
            let normalized = Self.normalizeVoice(voice)
            try await ensureVoiceEmbeddingIfNeeded(for: normalized)
        }
    }

    private static func normalizeVoice(_ voice: String) -> String {
        let trimmed = voice.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? TtsConstants.recommendedVoice : trimmed
    }
}
