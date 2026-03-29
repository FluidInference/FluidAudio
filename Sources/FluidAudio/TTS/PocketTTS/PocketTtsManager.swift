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
public actor PocketTtsManager {

    private let logger = AppLogger(category: "PocketTtsManager")
    private let modelStore: PocketTtsModelStore
    private var defaultVoice: String
    private var isInitialized = false

    /// Creates a new PocketTTS manager.
    ///
    /// - Parameters:
    ///   - defaultVoice: Default voice identifier (default: "alba").
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    public init(
        defaultVoice: String = PocketTtsConstants.defaultVoice,
        directory: URL? = nil
    ) {
        self.modelStore = PocketTtsModelStore(directory: directory)
        self.defaultVoice = defaultVoice
    }

    public var isAvailable: Bool {
        isInitialized
    }

    /// Initialize by downloading and loading all PocketTTS models.
    public func initialize() async throws {
        try await modelStore.loadIfNeeded()
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
            throw PocketTTSError.modelNotFound("PocketTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice

        return try await PocketTtsSynthesizer.withModelStore(modelStore) {
            let result = try await PocketTtsSynthesizer.synthesize(
                text: text,
                voice: selectedVoice,
                temperature: temperature,
                deEss: deEss
            )
            return result.audio
        }
    }

    /// Synthesize text to WAV audio data using custom voice data.
    ///
    /// Use this for cloned voices without saving to disk first.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voiceData: Voice conditioning data (e.g., from cloneVoice).
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - deEss: Whether to apply de-essing post-processing (default: true).
    /// - Returns: WAV audio data at 24kHz.
    ///
    /// Example:
    /// ```swift
    /// let voiceData = try await manager.cloneVoice(from: audioURL)
    /// let audio = try await manager.synthesize(text: "Hello!", voiceData: voiceData)
    /// ```
    public func synthesize(
        text: String,
        voiceData: PocketTtsVoiceData,
        temperature: Float = PocketTtsConstants.temperature,
        deEss: Bool = true
    ) async throws -> Data {
        guard isInitialized else {
            throw PocketTTSError.modelNotFound("PocketTTS model not initialized")
        }

        return try await PocketTtsSynthesizer.withModelStore(modelStore) {
            let result = try await PocketTtsSynthesizer.synthesize(
                text: text,
                voiceData: voiceData,
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
            throw PocketTTSError.modelNotFound("PocketTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice

        return try await PocketTtsSynthesizer.withModelStore(modelStore) {
            try await PocketTtsSynthesizer.synthesize(
                text: text,
                voice: selectedVoice,
                temperature: temperature,
                deEss: deEss
            )
        }
    }

    /// Synthesize text as a stream of 80ms audio frames.
    ///
    /// Each frame contains 1920 Float32 samples at 24kHz. Frames are yielded
    /// as they are generated, enabling playback to start before the full
    /// utterance is complete.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice identifier (default: uses the manager's default voice).
    ///   - temperature: Generation temperature (default: 0.7).
    /// - Returns: An `AsyncThrowingStream` of audio frames. Throws if a model
    ///   inference error occurs during generation.
    ///
    /// Example:
    /// ```swift
    /// let manager = PocketTtsManager()
    /// try await manager.initialize()
    /// let stream = try await manager.synthesizeStreaming(text: "Hello, world!")
    /// for try await frame in stream {
    ///     audioEngine.schedule(frame.samples)
    /// }
    /// ```
    public func synthesizeStreaming(
        text: String,
        voice: String? = nil,
        temperature: Float = PocketTtsConstants.temperature
    ) async throws -> AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error> {
        guard isInitialized else {
            throw PocketTTSError.modelNotFound("PocketTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice

        return try await PocketTtsSynthesizer.withModelStore(modelStore) {
            try await PocketTtsSynthesizer.synthesizeStreaming(
                text: text,
                voice: selectedVoice,
                temperature: temperature
            )
        }
    }

    /// Synthesize text as a stream of audio frames using custom voice data.
    ///
    /// Use this for cloned voices without saving to disk first.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voiceData: Voice conditioning data (e.g., from cloneVoice).
    ///   - temperature: Generation temperature (default: 0.7).
    /// - Returns: An `AsyncThrowingStream` of audio frames. Throws if a model
    ///   inference error occurs during generation.
    public func synthesizeStreaming(
        text: String,
        voiceData: PocketTtsVoiceData,
        temperature: Float = PocketTtsConstants.temperature
    ) async throws -> AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error> {
        guard isInitialized else {
            throw PocketTTSError.modelNotFound("PocketTTS model not initialized")
        }

        return try await PocketTtsSynthesizer.withModelStore(modelStore) {
            try await PocketTtsSynthesizer.synthesizeStreaming(
                text: text,
                voiceData: voiceData,
                temperature: temperature
            )
        }
    }

    // MARK: - Session API

    /// Create a persistent TTS session that keeps the voice KV cache warm.
    ///
    /// The expensive voice prefill (~125 tokens) is performed once during
    /// session creation. Each subsequent `enqueue()` call only pays the
    /// text prefill cost. Mimi decoder state persists across utterances
    /// for seamless audio continuity.
    ///
    /// - Parameters:
    ///   - voice: Voice identifier (default: uses the manager's default voice).
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - seed: Random seed for reproducibility (nil for random).
    /// - Returns: A session ready to accept text via `enqueue()`.
    ///
    /// Example:
    /// ```swift
    /// let session = try await manager.makeSession(voice: "alba")
    /// session.enqueue("Hello there.")
    /// session.enqueue("How are you?")
    /// session.finish()
    /// for try await frame in session.frames {
    ///     audioEngine.schedule(frame.samples)
    /// }
    /// ```
    public func makeSession(
        voice: String? = nil,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil
    ) async throws -> PocketTtsSession {
        guard isInitialized else {
            throw PocketTTSError.modelNotFound("PocketTTS model not initialized")
        }

        let selectedVoice = voice ?? defaultVoice
        let voiceData = try await modelStore.voiceData(for: selectedVoice)

        return try await buildSession(
            voiceData: voiceData, temperature: temperature, seed: seed
        )
    }

    /// Create a persistent TTS session using custom voice data.
    ///
    /// Use this for cloned voices without saving to disk first.
    ///
    /// - Parameters:
    ///   - voiceData: Voice conditioning data (e.g., from cloneVoice).
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - seed: Random seed for reproducibility (nil for random).
    /// - Returns: A session ready to accept text via `enqueue()`.
    public func makeSession(
        voiceData: PocketTtsVoiceData,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil
    ) async throws -> PocketTtsSession {
        guard isInitialized else {
            throw PocketTTSError.modelNotFound("PocketTTS model not initialized")
        }

        return try await buildSession(
            voiceData: voiceData, temperature: temperature, seed: seed
        )
    }

    private func buildSession(
        voiceData: PocketTtsVoiceData,
        temperature: Float,
        seed: UInt64?
    ) async throws -> PocketTtsSession {
        return try await PocketTtsSynthesizer.withModelStore(modelStore) {
            try await PocketTtsSynthesizer.makeSession(
                voiceData: voiceData,
                temperature: temperature,
                seed: seed
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

    // MARK: - Voice Cloning

    /// Check if voice cloning is available (mimi_encoder model present).
    public func isVoiceCloningAvailable() async -> Bool {
        await modelStore.isMimiEncoderAvailable()
    }

    /// Clone a voice from an audio file.
    ///
    /// Creates voice conditioning data that can be used for TTS synthesis.
    ///
    /// - Parameters:
    ///   - audioURL: URL to the source audio file (WAV format, any sample rate).
    /// - Returns: Voice conditioning data.
    /// - Throws: `PocketTTSError.modelNotFound` if the mimi_encoder is not available.
    ///
    /// Example:
    /// ```swift
    /// let voiceData = try await manager.cloneVoice(from: audioURL)
    /// try manager.saveClonedVoice(voiceData, to: outputURL)
    /// ```
    public func cloneVoice(from audioURL: URL) async throws -> PocketTtsVoiceData {
        try await modelStore.loadMimiEncoderIfNeeded()
        return try await modelStore.cloneVoice(from: audioURL)
    }

    /// Clone a voice from audio samples.
    ///
    /// - Parameters:
    ///   - samples: Audio samples at 24kHz mono float32.
    /// - Returns: Voice conditioning data.
    public func cloneVoice(from samples: [Float]) async throws -> PocketTtsVoiceData {
        try await modelStore.loadMimiEncoderIfNeeded()
        return try await modelStore.cloneVoice(from: samples)
    }

    /// Save cloned voice data to a binary file.
    ///
    /// The output file can be placed in the constants_bin directory
    /// and used with the `voice:` parameter in synthesize calls.
    ///
    /// - Parameters:
    ///   - voiceData: The voice conditioning data from `cloneVoice`.
    ///   - url: Destination URL (should end with `_audio_prompt.bin`).
    public nonisolated func saveClonedVoice(_ voiceData: PocketTtsVoiceData, to url: URL) throws {
        try PocketTtsVoiceCloner.saveVoice(voiceData, to: url)
    }

    /// Clone a voice and save it in one step.
    ///
    /// - Parameters:
    ///   - audioURL: URL to the source audio file.
    ///   - outputURL: Destination URL for the voice data file.
    /// - Returns: The cloned voice data.
    @discardableResult
    public func cloneVoiceToFile(from audioURL: URL, to outputURL: URL) async throws -> PocketTtsVoiceData {
        let voiceData = try await cloneVoice(from: audioURL)
        try saveClonedVoice(voiceData, to: outputURL)
        return voiceData
    }

    /// Load previously saved voice data from a binary file.
    ///
    /// - Parameters:
    ///   - url: Path to the .bin file containing voice data.
    /// - Returns: Voice conditioning data ready for TTS.
    public nonisolated func loadClonedVoice(from url: URL) throws -> PocketTtsVoiceData {
        try PocketTtsVoiceCloner.loadVoice(from: url)
    }
}
