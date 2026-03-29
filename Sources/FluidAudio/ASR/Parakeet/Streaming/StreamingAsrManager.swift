@preconcurrency import AVFoundation
import Foundation

/// Universal protocol for true streaming ASR engines.
///
/// `StreamingEouAsrManager` and `StreamingNemotronAsrManager` conform to this protocol,
/// enabling polymorphic usage through `StreamingModelVariant.createManager()`.
///
/// Note: `SlidingWindowAsrManager` (TDT) intentionally does **not** conform — it uses
/// an offline encoder with overlapping windows, not a cache-aware streaming architecture.
///
/// Usage pattern:
/// ```swift
/// let manager = StreamingModelVariant.parakeetEou160ms.createManager()
/// try await manager.loadModels()
/// try manager.appendAudio(buffer)
/// try await manager.processBufferedAudio()
/// let text = try await manager.finish()
/// ```
public protocol StreamingAsrManager: Actor {
    /// Human-readable display name (e.g., "Parakeet TDT 0.6B v3 Streaming")
    var displayName: String { get }

    /// Download models from HuggingFace (if needed) and load them into memory.
    /// Each engine knows its own HuggingFace repo and download logic.
    /// Calling multiple times is safe (idempotent after first successful load).
    func loadModels() async throws

    /// Append audio data for transcription.
    ///
    /// The engine buffers audio internally. Audio in any format is accepted
    /// and will be resampled to 16 kHz mono internally.
    /// - Parameter buffer: Audio buffer to process.
    func appendAudio(_ buffer: AVAudioPCMBuffer) throws

    /// Process any buffered audio that has accumulated since the last call.
    ///
    /// For chunk-based engines (EOU, Nemotron), this processes complete chunks
    /// from the internal buffer.
    func processBufferedAudio() async throws

    /// Signal end of audio stream, flush remaining audio, and return the final transcript.
    func finish() async throws -> String

    /// Reset engine state for a new transcription session.
    /// Models remain loaded; only decoding and buffer state is cleared.
    func reset() async throws

    /// Release all loaded models and free memory.
    /// The manager cannot be used for transcription after this until `loadModels()` is called again.
    func cleanup() async

    /// Set a callback invoked when partial transcript text updates are available.
    /// - Parameter callback: Closure receiving the current partial transcript string.
    func setPartialTranscriptCallback(_ callback: @escaping @Sendable (String) -> Void)

    /// Get the current partial transcript without finishing.
    func getPartialTranscript() -> String
}
