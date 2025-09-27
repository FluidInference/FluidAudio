import Foundation
import OSLog

@available(macOS 13.0, *)
public final class TtSManager {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "TtSManager")

    private var ttsModels: TtsModels?
    private var isInitialized = false

    private let availableVoices = [
        "af_heart",
        "am_adam",
        "af_alloy",
    ]

    public init() {}

    public var isAvailable: Bool {
        isInitialized
    }

    public func initialize(models: TtsModels) async throws {
        logger.info("Initializing TtSManager with provided models")

        self.ttsModels = models

        await KokoroModelCache.shared.registerPreloadedModels(models)
        try await LexiconAssetManager.ensureCoreAssets()
        try await KokoroSynthesizer.loadSimplePhonemeDictionary()
        try await KokoroModelCache.shared.loadModelsIfNeeded()
        isInitialized = true

        logger.info("TtSManager initialized successfully with preloaded models")
    }

    public func initialize() async throws {
        logger.info("Initializing TtSManager with downloaded models")

        let models = try await TtsModels.download()
        try await initialize(models: models)
    }

    public func synthesize(
        text: String,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0,
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws -> Data {
        let detailed = try await synthesizeDetailed(
            text: text,
            voice: nil,
            voiceSpeed: voiceSpeed,
            speakerId: speakerId,
            variantPreference: variantPreference
        )

        logger.info("Successfully synthesized \(detailed.audio.count) bytes of audio")
        return detailed.audio
    }

    public func synthesizeToFile(
        text: String,
        outputURL: URL,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0,
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let audioData = try await synthesize(
            text: text,
            voiceSpeed: voiceSpeed,
            speakerId: speakerId,
            variantPreference: variantPreference
        )

        try audioData.write(to: outputURL)
        logger.info("Saved synthesized audio to: \(outputURL.path)")
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

        let cleanedText = try sanitizeInput(text)
        let selectedVoice = resolveVoice(voice, speakerId: speakerId)

        try await LexiconAssetManager.ensureCoreAssets()
        try await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: selectedVoice)

        let synthesis = try await KokoroSynthesizer.synthesizeDetailed(
            text: cleanedText,
            voice: selectedVoice,
            variantPreference: variantPreference
        )
        let factor = max(0.1, voiceSpeed)

        if abs(factor - 1.0) < 0.01 {
            return synthesis
        }

        let adjustedChunks = synthesis.chunks.map { chunk -> KokoroSynthesizer.ChunkInfo in
            let stretched = adjustSamples(chunk.samples, factor: factor)
            return KokoroSynthesizer.ChunkInfo(
                index: chunk.index,
                text: chunk.text,
                wordCount: chunk.wordCount,
                words: chunk.words,
                atoms: chunk.atoms,
                pauseAfterMs: chunk.pauseAfterMs,
                tokenCount: chunk.tokenCount,
                samples: stretched,
                variant: chunk.variant
            )
        }

        let combinedSamples = adjustedChunks.flatMap { $0.samples }
        let audioData = try AudioWAV.data(from: combinedSamples, sampleRate: 24_000)

        return KokoroSynthesizer.SynthesisResult(audio: audioData, chunks: adjustedChunks)
    }

    private func sanitizeInput(_ text: String) throws -> String {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !cleanText.isEmpty else {
            throw TTSError.processingFailed("Input text is empty")
        }

        return cleanText
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
        logger.info("TtSManager cleaned up")
    }

    private func voiceName(for speakerId: Int) -> String {
        guard !availableVoices.isEmpty else { return "af_heart" }
        let index = abs(speakerId) % availableVoices.count
        return availableVoices[index]
    }

    private func adjustSamples(_ samples: [Float], factor: Float) -> [Float] {
        let clamped = max(0.1, factor)
        if abs(clamped - 1.0) < 0.01 { return samples }

        if clamped < 1.0 {
            let repeatCount = max(1, Int(round(1.0 / clamped)))
            var stretched: [Float] = []
            stretched.reserveCapacity(samples.count * repeatCount)
            for sample in samples {
                for _ in 0..<repeatCount {
                    stretched.append(sample)
                }
            }
            return stretched
        } else {
            let skip = max(1, Int(round(clamped)))
            var reduced: [Float] = []
            reduced.reserveCapacity(samples.count / skip)
            var index = 0
            while index < samples.count {
                reduced.append(samples[index])
                index += skip
            }
            return reduced.isEmpty ? samples : reduced
        }
    }
}
