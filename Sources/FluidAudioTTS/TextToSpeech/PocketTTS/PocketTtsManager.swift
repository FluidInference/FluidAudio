import FluidAudio
import Foundation
import OSLog

/// Manages text-to-speech synthesis using PocketTTS CoreML models.
///
/// PocketTTS uses a flow-matching language model architecture that generates
/// audio autoregressively at 24kHz. Each generation step produces an 80ms
/// audio frame (1920 samples).
///
/// Example usage:
/// ```swift
/// let manager = PocketTtsManager()
/// try await manager.initialize()
/// let audioData = try await manager.synthesize(text: "Hello, world!")
/// ```
public final class PocketTtsManager {

    private let logger = AppLogger(category: "PocketTtsManager")
    private let modelCache: PocketTtsModelCache
    private var defaultVoice: String
    private var isInitialized = false

    /// Creates a new PocketTTS manager.
    ///
    /// - Parameters:
    ///   - defaultVoice: Default voice identifier (default: "alba").
    public init(defaultVoice: String = PocketTtsConstants.defaultVoice) {
        self.modelCache = PocketTtsModelCache()
        self.defaultVoice = defaultVoice
    }

    public var isAvailable: Bool {
        isInitialized
    }

    /// Initialize by downloading and loading all PocketTTS models.
    public func initialize() async throws {
        try await modelCache.loadIfNeeded()
        isInitialized = true
        logger.notice("PocketTtsManager initialized")
    }

    /// Synthesize text to WAV audio data.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice identifier (default: uses the manager's default voice).
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - deEss: Whether to apply de-essing post-processing (default: true).
    /// - Returns: WAV audio data at 24kHz.
    public func synthesize(
        text: String,
        voice: String? = nil,
        temperature: Float = PocketTtsConstants.temperature,
        deEss: Bool = true
    ) async throws -> Data {
        guard isInitialized else {
            throw TTSError.modelNotFound("PocketTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice

        return try await PocketTtsSynthesizer.withModelCache(modelCache) {
            let result = try await PocketTtsSynthesizer.synthesize(
                text: text,
                voice: selectedVoice,
                temperature: temperature,
                deEss: deEss
            )
            return result.audio
        }
    }

    /// Synthesize text and return detailed results including frame count and EOS info.
    public func synthesizeDetailed(
        text: String,
        voice: String? = nil,
        temperature: Float = PocketTtsConstants.temperature,
        deEss: Bool = true
    ) async throws -> PocketTtsSynthesizer.SynthesisResult {
        guard isInitialized else {
            throw TTSError.modelNotFound("PocketTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice

        return try await PocketTtsSynthesizer.withModelCache(modelCache) {
            try await PocketTtsSynthesizer.synthesize(
                text: text,
                voice: selectedVoice,
                temperature: temperature,
                deEss: deEss
            )
        }
    }

    /// Synthesize text and write the result directly to a file.
    public func synthesizeToFile(
        text: String,
        outputURL: URL,
        voice: String? = nil,
        temperature: Float = PocketTtsConstants.temperature,
        deEss: Bool = true
    ) async throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let audioData = try await synthesize(
            text: text,
            voice: voice,
            temperature: temperature,
            deEss: deEss
        )

        try audioData.write(to: outputURL)
        logger.notice("Saved synthesized audio to: \(outputURL.lastPathComponent)")
    }

    /// Update the default voice.
    public func setDefaultVoice(_ voice: String) {
        defaultVoice = voice
    }

    public func cleanup() {
        isInitialized = false
    }
}
