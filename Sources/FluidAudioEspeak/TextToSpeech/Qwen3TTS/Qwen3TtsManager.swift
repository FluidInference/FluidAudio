import FluidAudio
import Foundation
import OSLog

/// Manages text-to-speech synthesis using Qwen3-TTS CoreML models.
///
/// Qwen3-TTS is a large language model-based TTS system that supports
/// multiple languages including English and Chinese. It uses a 4-stage
/// pipeline: prefill → LM decode → code predictor → audio decoder.
///
/// Example usage:
/// ```swift
/// let manager = Qwen3TtsManager()
/// try await manager.loadFromDirectory(modelDirectory)
/// let audioData = try await manager.synthesize(text: "Hello world", tokenIds: [...])
/// ```
///
/// NOTE: This implementation requires pre-tokenized input. The text must be
/// tokenized using the Qwen3 tokenizer externally (e.g., in Python).
public actor Qwen3TtsManager {

    private let logger = AppLogger(category: "Qwen3TtsManager")
    private let modelStore: Qwen3TtsModelStore
    private var isInitialized = false

    /// Creates a new Qwen3-TTS manager.
    public init() {
        self.modelStore = Qwen3TtsModelStore()
    }

    public var isAvailable: Bool {
        isInitialized
    }

    /// Download models from HuggingFace and initialize.
    public func initialize() async throws {
        try await modelStore.loadIfNeeded()
        isInitialized = true
        logger.notice("Qwen3TtsManager initialized (auto-download)")
    }

    /// Load models from a local directory.
    ///
    /// - Parameter directory: Path to directory containing CoreML model bundles.
    public func loadFromDirectory(_ directory: URL) async throws {
        try await modelStore.loadFromDirectory(directory)
        isInitialized = true
        logger.notice("Qwen3TtsManager initialized from \(directory.lastPathComponent)")
    }

    /// Synthesize text to WAV audio data.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize (for logging purposes).
    ///   - tokenIds: Pre-tokenized text IDs from Qwen3 tokenizer.
    ///   - useSpeaker: Whether to use speaker embedding (default: true).
    /// - Returns: WAV audio data at 24kHz.
    public func synthesize(
        text: String,
        tokenIds: [Int],
        useSpeaker: Bool = true
    ) async throws -> Data {
        guard isInitialized else {
            throw TTSError.modelNotFound("Qwen3-TTS models not initialized")
        }

        return try await Qwen3TtsSynthesizer.withModelStore(modelStore) {
            let result = try await Qwen3TtsSynthesizer.synthesize(
                text: text,
                tokenIds: tokenIds,
                useSpeaker: useSpeaker
            )
            return result.audio
        }
    }

    /// Synthesize text and return detailed results.
    public func synthesizeDetailed(
        text: String,
        tokenIds: [Int],
        useSpeaker: Bool = true
    ) async throws -> Qwen3TtsSynthesizer.SynthesisResult {
        guard isInitialized else {
            throw TTSError.modelNotFound("Qwen3-TTS models not initialized")
        }

        return try await Qwen3TtsSynthesizer.withModelStore(modelStore) {
            try await Qwen3TtsSynthesizer.synthesize(
                text: text,
                tokenIds: tokenIds,
                useSpeaker: useSpeaker
            )
        }
    }

    /// Synthesize text and write the result directly to a file.
    public func synthesizeToFile(
        text: String,
        tokenIds: [Int],
        outputURL: URL,
        useSpeaker: Bool = true
    ) async throws {
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let audioData = try await synthesize(
            text: text,
            tokenIds: tokenIds,
            useSpeaker: useSpeaker
        )

        try audioData.write(to: outputURL)
        logger.notice("Saved synthesized audio to: \(outputURL.lastPathComponent)")
    }

    /// Get the underlying model store for advanced usage.
    public func getModelStore() -> Qwen3TtsModelStore {
        modelStore
    }

    public func cleanup() {
        isInitialized = false
    }
}
