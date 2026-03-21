import Foundation
import OSLog

/// Manages text-to-speech synthesis using KittenTTS CoreML models.
///
/// KittenTTS is a single-shot StyleTTS2-based synthesizer that produces
/// complete utterances in one forward pass at 24kHz. Two variants are available:
/// - **Nano** (15M params): Lightweight, no speed control
/// - **Mini** (80M params): Higher quality, speed control
///
/// Example usage:
/// ```swift
/// let manager = KittenTtsManager(variant: .mini)
/// try await manager.initialize()
/// let audioData = try await manager.synthesize(text: "Hello, world!")
/// ```
public actor KittenTtsManager {

    private let logger = AppLogger(category: "KittenTtsManager")
    private let modelStore: KittenTtsModelStore
    private var defaultVoice: String
    private var isInitialized = false

    /// Creates a new KittenTTS manager.
    ///
    /// - Parameters:
    ///   - variant: Model variant to use (.nano or .mini).
    ///   - defaultVoice: Default voice identifier.
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    public init(
        variant: KittenTtsVariant = .mini,
        defaultVoice: String = KittenTtsConstants.defaultVoice,
        directory: URL? = nil
    ) {
        self.modelStore = KittenTtsModelStore(variant: variant, directory: directory)
        self.defaultVoice = defaultVoice
    }

    public var isAvailable: Bool {
        isInitialized
    }

    /// Initialize by downloading and loading KittenTTS models.
    public func initialize() async throws {
        try await modelStore.loadIfNeeded()
        isInitialized = true
        logger.notice("KittenTtsManager initialized")
    }

    /// Synthesize text to WAV audio data.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice identifier (default: uses the manager's default voice).
    ///   - speed: Speech speed multiplier (Mini only, 1.0 = normal).
    ///   - deEss: Whether to apply de-essing post-processing (default: true).
    /// - Returns: WAV audio data at 24kHz.
    public func synthesize(
        text: String,
        voice: String? = nil,
        speed: Float = 1.0,
        deEss: Bool = true
    ) async throws -> Data {
        guard isInitialized else {
            throw KittenTTSError.modelNotFound("KittenTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice

        return try await KittenTtsSynthesizer.withModelStore(modelStore) {
            let result = try await KittenTtsSynthesizer.synthesize(
                text: text,
                voice: selectedVoice,
                speed: speed,
                deEss: deEss
            )
            return result.audio
        }
    }

    /// Synthesize text and return detailed results.
    public func synthesizeDetailed(
        text: String,
        voice: String? = nil,
        speed: Float = 1.0,
        deEss: Bool = true
    ) async throws -> KittenTtsSynthesizer.SynthesisResult {
        guard isInitialized else {
            throw KittenTTSError.modelNotFound("KittenTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice

        return try await KittenTtsSynthesizer.withModelStore(modelStore) {
            try await KittenTtsSynthesizer.synthesize(
                text: text,
                voice: selectedVoice,
                speed: speed,
                deEss: deEss
            )
        }
    }

    /// Synthesize text and write the result directly to a file.
    public func synthesizeToFile(
        text: String,
        outputURL: URL,
        voice: String? = nil,
        speed: Float = 1.0,
        deEss: Bool = true
    ) async throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let audioData = try await synthesize(
            text: text,
            voice: voice,
            speed: speed,
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
