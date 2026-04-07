import AVFoundation
import Foundation

/// Shared utilities for streaming ASR implementations.
/// Provides common operations used by both EOU and Nemotron streaming managers.
public struct StreamingAsrUtils {

    /// Process remaining audio at end of stream by padding to chunk size
    /// - Parameters:
    ///   - audioBuffer: Current audio buffer
    ///   - chunkSamples: Required chunk size in samples
    ///   - processChunk: Closure to process the final padded chunk
    /// - Returns: True if audio was processed, false if buffer was empty
    public static func processRemainingAudio(
        audioBuffer: inout [Float],
        chunkSamples: Int,
        processChunk: ([Float]) async throws -> Void
    ) async throws -> Bool {
        guard !audioBuffer.isEmpty else { return false }

        // Pad to chunk size with zeros
        let paddingNeeded = chunkSamples - audioBuffer.count
        if paddingNeeded > 0 {
            audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
        }

        // Process final chunk
        let chunk = Array(audioBuffer.prefix(chunkSamples))
        try await processChunk(chunk)
        audioBuffer.removeAll()

        return true
    }

    /// Decode accumulated token IDs to text using tokenizer
    /// - Parameters:
    ///   - tokenIds: Array of token IDs to decode
    ///   - tokenizer: Tokenizer instance
    /// - Returns: Decoded text string
    public static func decodeTokens(_ tokenIds: [Int], using tokenizer: Tokenizer) -> String {
        tokenizer.decode(ids: tokenIds)
    }

    /// Reset shared streaming state (audio buffer, tokens, counters)
    /// - Parameters:
    ///   - audioBuffer: Audio buffer to clear
    ///   - accumulatedTokenIds: Token accumulator to clear
    ///   - processedChunks: Chunk counter to reset
    public static func resetSharedState(
        audioBuffer: inout [Float],
        accumulatedTokenIds: inout [Int],
        processedChunks: inout Int
    ) {
        audioBuffer.removeAll()
        accumulatedTokenIds.removeAll()
        processedChunks = 0
    }

    /// Append audio samples from buffer after resampling
    /// - Parameters:
    ///   - buffer: Input audio buffer
    ///   - audioConverter: Converter for resampling
    ///   - audioBuffer: Destination buffer
    public static func appendAudio(
        _ buffer: AVAudioPCMBuffer,
        using audioConverter: AudioConverter,
        to audioBuffer: inout [Float]
    ) throws {
        let samples = try audioConverter.resampleBuffer(buffer)
        audioBuffer.append(contentsOf: samples)
    }
}
