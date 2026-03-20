import Foundation

/// Manages text-to-speech synthesis using VoxCPM 1.5 CoreML models.
///
/// VoxCPM 1.5 is a diffusion autoregressive TTS that generates 44.1kHz audio
/// with bilingual EN/ZH support and voice cloning.
///
/// Example usage:
/// ```swift
/// let manager = VoxCpmManager()
/// try await manager.initialize()
/// let audioData = try await manager.synthesize(text: "Hello, world!")
/// ```
public actor VoxCpmManager {

    private let logger = AppLogger(category: "VoxCpmManager")
    private let modelStore: VoxCpmModelStore
    private var isInitialized = false

    /// Creates a new VoxCPM manager.
    ///
    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public init(directory: URL? = nil) {
        self.modelStore = VoxCpmModelStore(directory: directory)
    }

    public var isAvailable: Bool {
        isInitialized
    }

    /// Initialize by downloading and loading all VoxCPM models.
    public func initialize() async throws {
        try await modelStore.loadIfNeeded()
        isInitialized = true
        logger.notice("VoxCpmManager initialized")
    }

    /// Synthesize text to WAV audio data at 44.1kHz.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - promptAudioURL: Optional URL to prompt audio for voice cloning.
    ///   - promptText: Transcript of prompt audio (required when using voice cloning).
    ///   - maxLen: Maximum generation steps (default: 200).
    ///   - minLen: Minimum steps before stop head can trigger (default: 5).
    /// - Returns: WAV audio data at 44.1kHz.
    public func synthesize(
        text: String,
        promptAudioURL: URL? = nil,
        promptText: String? = nil,
        maxLen: Int = VoxCpmConstants.defaultMaxLen,
        minLen: Int = VoxCpmConstants.defaultMinLen
    ) async throws -> Data {
        guard isInitialized else {
            throw VoxCpmError.modelNotFound("VoxCPM not initialized")
        }

        return try await VoxCpmSynthesizer.withModelStore(modelStore) {
            let result = try await VoxCpmSynthesizer.synthesize(
                text: text,
                promptAudioURL: promptAudioURL,
                promptText: promptText,
                maxLen: maxLen,
                minLen: minLen
            )
            return result.audio
        }
    }

    /// Synthesize text and return detailed results.
    public func synthesizeDetailed(
        text: String,
        promptAudioURL: URL? = nil,
        promptText: String? = nil,
        maxLen: Int = VoxCpmConstants.defaultMaxLen,
        minLen: Int = VoxCpmConstants.defaultMinLen
    ) async throws -> VoxCpmSynthesizer.SynthesisResult {
        guard isInitialized else {
            throw VoxCpmError.modelNotFound("VoxCPM not initialized")
        }

        return try await VoxCpmSynthesizer.withModelStore(modelStore) {
            try await VoxCpmSynthesizer.synthesize(
                text: text,
                promptAudioURL: promptAudioURL,
                promptText: promptText,
                maxLen: maxLen,
                minLen: minLen
            )
        }
    }

    /// Synthesize text and write the result directly to a file.
    public func synthesizeToFile(
        text: String,
        outputURL: URL,
        promptAudioURL: URL? = nil,
        promptText: String? = nil,
        maxLen: Int = VoxCpmConstants.defaultMaxLen
    ) async throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let audioData = try await synthesize(
            text: text,
            promptAudioURL: promptAudioURL,
            promptText: promptText,
            maxLen: maxLen
        )

        try audioData.write(to: outputURL)
        logger.notice("Saved synthesized audio to: \(outputURL.lastPathComponent)")
    }

    public func cleanup() {
        isInitialized = false
    }
}
