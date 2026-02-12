import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "Qwen3StreamingManager")

// MARK: - Streaming Configuration

/// Configuration for Qwen3 streaming transcription.
public struct Qwen3StreamingConfig: Sendable {
    /// Minimum audio duration (seconds) before first transcription.
    public let minAudioSeconds: Double

    /// Chunk duration (seconds) - how often to re-transcribe.
    public let chunkSeconds: Double

    /// Maximum audio duration (seconds) to accumulate.
    public let maxAudioSeconds: Double

    /// Language hint for transcription.
    public let language: Qwen3AsrConfig.Language?

    public init(
        minAudioSeconds: Double = 1.0,
        chunkSeconds: Double = 2.0,
        maxAudioSeconds: Double = 30.0,
        language: Qwen3AsrConfig.Language? = nil
    ) {
        self.minAudioSeconds = minAudioSeconds
        self.chunkSeconds = chunkSeconds
        self.maxAudioSeconds = maxAudioSeconds
        self.language = language
    }

    public static let `default` = Qwen3StreamingConfig()
}

// MARK: - Streaming Result

/// Result from a streaming transcription update.
public struct Qwen3StreamingResult: Sendable {
    /// Current transcript (may change with more audio).
    public let transcript: String

    /// Audio duration processed so far (seconds).
    public let audioDuration: Double

    /// Whether this is a final result (no more audio expected).
    public let isFinal: Bool
}

// MARK: - Streaming Manager

/// Streaming transcription manager for Qwen3-ASR.
///
/// Accumulates audio chunks and provides incremental transcription updates.
/// Uses sliding window approach - re-transcribes accumulated audio each update.
///
/// Usage:
/// ```swift
/// let streaming = Qwen3StreamingManager(asrManager: manager)
/// streaming.configure(config)
///
/// // Feed audio chunks as they arrive
/// for chunk in audioChunks {
///     if let result = try await streaming.addAudio(chunk) {
///         print("Partial: \(result.transcript)")
///     }
/// }
///
/// // Get final result
/// let final = try await streaming.finish()
/// print("Final: \(final.transcript)")
/// ```
@available(macOS 15, iOS 18, *)
public actor Qwen3StreamingManager {
    private let asrManager: Qwen3AsrManager
    private var config: Qwen3StreamingConfig
    private var audioBuffer: [Float] = []
    private var lastTranscript: String = ""
    private var lastTranscribeTime: CFAbsoluteTime = 0
    private var samplesSinceLastTranscribe: Int = 0

    public init(asrManager: Qwen3AsrManager, config: Qwen3StreamingConfig = .default) {
        self.asrManager = asrManager
        self.config = config
    }

    /// Configure streaming parameters.
    public func configure(_ config: Qwen3StreamingConfig) {
        self.config = config
    }

    /// Reset streaming state for a new session.
    public func reset() {
        audioBuffer.removeAll()
        lastTranscript = ""
        lastTranscribeTime = 0
        samplesSinceLastTranscribe = 0
    }

    /// Add audio samples and get transcription update if ready.
    ///
    /// - Parameter samples: 16kHz mono Float32 audio samples.
    /// - Returns: Transcription result if enough audio accumulated, nil otherwise.
    public func addAudio(_ samples: [Float]) async throws -> Qwen3StreamingResult? {
        audioBuffer.append(contentsOf: samples)
        samplesSinceLastTranscribe += samples.count

        let totalSeconds = Double(audioBuffer.count) / 16000.0
        let secondsSinceLastTranscribe = Double(samplesSinceLastTranscribe) / 16000.0

        // Check if we should transcribe
        let hasMinAudio = totalSeconds >= config.minAudioSeconds
        let chunkReady = secondsSinceLastTranscribe >= config.chunkSeconds

        guard hasMinAudio && chunkReady else {
            return nil
        }

        // Trim if exceeding max duration
        let maxSamples = Int(config.maxAudioSeconds * 16000)
        if audioBuffer.count > maxSamples {
            let excess = audioBuffer.count - maxSamples
            audioBuffer.removeFirst(excess)
            logger.debug("Trimmed \(excess) samples to stay under \(self.config.maxAudioSeconds)s limit")
        }

        // Transcribe accumulated audio
        let transcript = try await asrManager.transcribe(
            audioSamples: audioBuffer,
            language: config.language,
            maxNewTokens: 512
        )

        lastTranscript = transcript
        lastTranscribeTime = CFAbsoluteTimeGetCurrent()
        samplesSinceLastTranscribe = 0

        let duration = Double(audioBuffer.count) / 16000.0
        logger.debug("Streaming update: \(String(format: "%.1f", duration))s audio -> \"\(transcript.prefix(50))...\"")

        return Qwen3StreamingResult(
            transcript: transcript,
            audioDuration: duration,
            isFinal: false
        )
    }

    /// Finish streaming and get final transcription.
    ///
    /// Transcribes any remaining audio and returns the final result.
    public func finish() async throws -> Qwen3StreamingResult {
        let duration = Double(audioBuffer.count) / 16000.0

        // If we have audio that hasn't been transcribed, do final transcription
        if samplesSinceLastTranscribe > 0 && audioBuffer.count >= Int(config.minAudioSeconds * 16000) {
            let transcript = try await asrManager.transcribe(
                audioSamples: audioBuffer,
                language: config.language,
                maxNewTokens: 512
            )
            lastTranscript = transcript
        }

        logger.info("Streaming finished: \(String(format: "%.1f", duration))s audio")

        let result = Qwen3StreamingResult(
            transcript: lastTranscript,
            audioDuration: duration,
            isFinal: true
        )

        // Reset for next session
        reset()

        return result
    }

    /// Get current transcript without triggering a new transcription.
    public func currentTranscript() -> String {
        lastTranscript
    }

    /// Get current audio duration in seconds.
    public func currentAudioDuration() -> Double {
        Double(audioBuffer.count) / 16000.0
    }
}
