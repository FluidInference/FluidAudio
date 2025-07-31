import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public actor RealtimeAsrManager {
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "RealtimeManager")
    private var activeStreams: [UUID: RealtimeAsrStream.StreamProcessor] = [:]
    private let asrModels: AsrModels
    
    public init(models: AsrModels) {
        self.asrModels = models
        logger.info("Initialized RealtimeAsrManager with ASR models")
    }
    
    /// Create a new realtime transcription stream
    /// - Parameters:
    ///   - source: Audio source (microphone or system)
    ///   - config: Realtime ASR configuration
    ///   - resetDecoderState: Whether to reset decoder state for this stream (default: true)
    /// - Returns: A new realtime ASR stream
    public func createStream(
        source: AudioSource = .microphone,
        config: RealtimeAsrConfig = .default,
        resetDecoderState: Bool = true
    ) async throws -> RealtimeAsrStream {
        let streamId = UUID()
        
        // Create dedicated ASR manager for this stream
        let asrManager = AsrManager(config: config.asrConfig)
        try await asrManager.initialize(models: asrModels)
        
        // Reset decoder state if requested
        if resetDecoderState {
            try await asrManager.resetDecoderState(for: source)
            logger.info("Reset decoder state for new stream \(streamId)")
        }
        
        // Create stream processor
        let processor = try await RealtimeAsrStream.StreamProcessor(
            streamId: streamId,
            asrManager: asrManager,
            source: source,
            config: config
        )
        
        // Store processor
        activeStreams[streamId] = processor
        
        logger.info("Created new realtime stream: \(streamId) with source: \(String(describing: source))")
        
        return RealtimeAsrStream(id: streamId, source: source)
    }
    
    /// Process audio samples for a specific stream
    /// - Parameters:
    ///   - streamId: The stream to process audio for
    ///   - samples: Audio samples to process
    /// - Returns: Transcription update if available
    public func processAudio(
        streamId: UUID,
        samples: [Float]
    ) async throws -> TranscriptionUpdate? {
        guard let processor = activeStreams[streamId] else {
            logger.error("Stream not found: \(streamId)")
            throw RealtimeAsrError.streamNotFound
        }
        
        return try await processor.processAudio(samples)
    }
    
    /// Get an async stream of transcription updates for a specific stream
    /// - Parameter streamId: The stream to get updates for
    /// - Returns: AsyncThrowingStream of transcription updates
    public func getTranscriptionStream(
        streamId: UUID
    ) async throws -> AsyncThrowingStream<TranscriptionUpdate, Error> {
        guard let processor = activeStreams[streamId] else {
            logger.error("Stream not found: \(streamId)")
            throw RealtimeAsrError.streamNotFound
        }
        
        return await processor.getTranscriptionStream()
    }
    
    /// Get metrics for a specific stream
    /// - Parameter streamId: The stream to get metrics for
    /// - Returns: Stream metrics
    public func getStreamMetrics(streamId: UUID) async throws -> StreamMetrics {
        guard let processor = activeStreams[streamId] else {
            logger.error("Stream not found: \(streamId)")
            throw RealtimeAsrError.streamNotFound
        }
        
        return await processor.getMetrics()
    }
    
    /// Get the full transcription history for a stream
    /// - Parameter streamId: The stream to get history for
    /// - Returns: Array of transcription segments
    public func getTranscriptionHistory(
        streamId: UUID
    ) async throws -> [(timeRange: ClosedRange<TimeInterval>, text: String, type: TranscriptionUpdate.UpdateType)] {
        guard let processor = activeStreams[streamId] else {
            logger.error("Stream not found: \(streamId)")
            throw RealtimeAsrError.streamNotFound
        }
        
        return await processor.getTranscriptionHistory()
    }
    
    /// Get the final merged transcription for a stream
    /// - Parameter streamId: The stream to get final transcription for
    /// - Returns: Final transcription text
    public func getFinalTranscription(streamId: UUID) async throws -> String {
        guard let processor = activeStreams[streamId] else {
            logger.error("Stream not found: \(streamId)")
            throw RealtimeAsrError.streamNotFound
        }
        
        return await processor.getFinalTranscription()
    }
    
    /// Remove a stream when done
    /// - Parameter streamId: The stream to remove
    public func removeStream(_ streamId: UUID) {
        activeStreams.removeValue(forKey: streamId)
        logger.info("Removed stream: \(streamId)")
    }
    
    /// Get list of active stream IDs
    public func getActiveStreamIds() -> [UUID] {
        return Array(activeStreams.keys)
    }
    
    /// Get count of active streams
    public func activeStreamCount() -> Int {
        return activeStreams.count
    }
    
    /// Remove all active streams
    public func removeAllStreams() {
        let count = activeStreams.count
        activeStreams.removeAll()
        logger.info("Removed all \(count) active streams")
    }
    
    /// Reset decoder state for a specific stream
    /// - Parameters:
    ///   - streamId: The stream to reset decoder state for
    ///   - source: The audio source for the stream
    public func resetDecoderState(for streamId: UUID, source: AudioSource) async throws {
        guard activeStreams[streamId] != nil else {
            logger.error("Stream not found: \(streamId)")
            throw RealtimeAsrError.streamNotFound
        }
        
        // Get the ASR manager from the processor and reset its decoder state
        // Note: We need to access the asrManager from within the processor
        // This requires exposing it or creating a method in the processor
        logger.info("Decoder state reset requested for stream: \(streamId)")
        // TODO: Implement this when StreamProcessor exposes its ASR manager
    }
}