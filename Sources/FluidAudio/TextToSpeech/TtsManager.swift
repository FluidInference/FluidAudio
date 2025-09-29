import Foundation
import OSLog

@available(macOS 13.0, *)
public final class TtSManager {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "TtSManager")
    private let modelCache: KokoroModelCache

    private var ttsModels: TtsModels?
    private var isInitialized = false
    private var assetsReady = false
    private var defaultVoice: String
    private var defaultSpeakerId: Int
    private var ensuredVoices: Set<String> = []

    public init(
        defaultVoice: String = "af_heart",
        defaultSpeakerId: Int = 0,
        modelCache: KokoroModelCache = KokoroModelCache()
    ) {
        self.modelCache = modelCache
        self.defaultVoice = Self.normalizeVoice(defaultVoice)
        self.defaultSpeakerId = defaultSpeakerId
        KokoroSynthesizer.configure(modelCache: modelCache)
    }

    public var isAvailable: Bool {
        isInitialized
    }

    public func initialize(models: TtsModels) async throws {
        self.ttsModels = models

        await modelCache.registerPreloadedModels(models)
        try await prepareLexiconAssetsIfNeeded()
        try await ensureVoiceEmbeddingIfNeeded(for: defaultVoice)
        try await KokoroSynthesizer.loadSimplePhonemeDictionary()
        try await modelCache.loadModelsIfNeeded(variants: models.availableVariants)
        isInitialized = true
        logger.notice("TtSManager initialized with provided models")
    }

    public func initialize() async throws {
        let models = try await TtsModels.download()
        try await initialize(models: models)
    }

    public func synthesize(
        text: String,
        voice: String? = nil,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0,
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws -> Data {
        guard isInitialized else {
            throw TTSError.modelNotFound("Kokoro model not initialized")
        }

        try await prepareLexiconAssetsIfNeeded()

        let cleanedText = try KokoroSynthesizer.sanitizeInput(text)
        let selectedVoice = resolveVoice(voice, speakerId: speakerId)
        try await ensureVoiceEmbeddingIfNeeded(for: selectedVoice)

        let synthesis = try await KokoroSynthesizer.synthesizeDetailed(
            text: cleanedText,
            voice: selectedVoice,
            voiceSpeed: voiceSpeed,
            variantPreference: variantPreference
        )

        return synthesis.audio
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
        try await LexiconAssetManager.ensureCoreAssets()
        assetsReady = true
    }

    private func ensureVoiceEmbeddingIfNeeded(for voice: String) async throws {
        if ensuredVoices.contains(voice) { return }
        try await TtsResourceDownloader.ensureVoiceEmbedding(voice: voice)
        ensuredVoices.insert(voice)
    }

    private static func normalizeVoice(_ voice: String) -> String {
        let trimmed = voice.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "af_heart" : trimmed
    }
}
