import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct RealtimeAsrStream: Sendable {
    public let id: UUID
    public let source: AudioSource
    
    /// Each stream gets its own isolated state processor
    public actor StreamProcessor {
        private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "RealtimeASR")
        private let asrManager: AsrManager
        private let source: AudioSource
        private let audioBuffer: AudioBuffer
        private var transcriptionHistory: [TranscriptionSegment] = []
        private let config: RealtimeAsrConfig
        private let streamId: UUID
        
        // Metrics tracking
        private var metrics = StreamMetrics()
        private var startTime: Date?
        private var firstWordDetected = false
        private var currentChunkIndex = 0
        private var lastProcessedAudioTime: TimeInterval = 0
        
        // Continuation for async stream
        private var streamContinuation: AsyncThrowingStream<TranscriptionUpdate, Error>.Continuation?
        
        init(
            streamId: UUID,
            asrManager: AsrManager,
            source: AudioSource,
            config: RealtimeAsrConfig
        ) async throws {
            self.streamId = streamId
            self.asrManager = asrManager
            self.source = source
            self.config = config
            self.audioBuffer = AudioBuffer(capacity: config.bufferCapacity)
            
            // Reset decoder state for this new stream
            try await asrManager.resetDecoderState(for: source)
            
            logger.info("Initialized stream processor for stream \(streamId) with source: \(String(describing: source))")
        }
        
        /// Process incoming audio samples
        func processAudio(_ samples: [Float]) async throws -> TranscriptionUpdate? {
            // Track start time for time to first word
            if startTime == nil {
                startTime = Date()
            }
            
            logger.debug("[Stream \(self.streamId)] Processing \(samples.count) new samples")
            
            // Append to buffer
            try await audioBuffer.append(samples)
            
            // Check if we have enough samples for a chunk
            let availableSamples = await audioBuffer.availableSamples()
            let requiredSamples = config.chunkSizeInSamples
            
            logger.debug("[Stream \(self.streamId)] Buffer state: available=\(availableSamples), required=\(requiredSamples), chunkSize=\(self.config.chunkSizeInSamples)")
            
            // Print accumulation progress
            if availableSamples < requiredSamples {
                let progress = Float(availableSamples) / Float(requiredSamples) * 100.0
                print("DEBUG: Accumulating audio: \(availableSamples)/\(requiredSamples) samples (\(String(format: "%.1f", progress))%)")
            }
            
            guard availableSamples >= requiredSamples else {
                logger.debug("[Stream \(self.streamId)] Not enough samples yet, waiting...")
                return nil // Not enough samples yet
            }
            
            // Get chunk
            guard let audioChunk = await audioBuffer.getChunk(size: config.chunkSizeInSamples) else {
                logger.warning("[Stream \(self.streamId)] Failed to get chunk")
                return nil
            }
            
            print("DEBUG: Processing chunk \(currentChunkIndex) with \(audioChunk.count) samples")
            logger.debug("[Stream \(self.streamId)] Got chunk of \(audioChunk.count) samples for chunk index \(self.currentChunkIndex)")
            logger.debug("[Stream \(self.streamId)] Chunk time range: \(self.lastProcessedAudioTime)s - \(self.lastProcessedAudioTime + self.config.chunkDuration)s")
            
            // Process the chunk
            let processingStartTime = Date()
            
            let result = try await transcribeChunk(audioChunk)
            
            let processingTime = Date().timeIntervalSince(processingStartTime)
            
            logger.debug("[Stream \(self.streamId)] Transcription result for chunk \(self.currentChunkIndex): '\(result.text)' (confidence: \(result.confidence))")
            
            // Update metrics
            metrics.chunkCount += 1
            metrics.totalProcessingTime += processingTime
            metrics.totalAudioDuration = lastProcessedAudioTime + config.chunkDuration
            
            // Calculate time range for this chunk
            let startTime = lastProcessedAudioTime
            let endTime = startTime + config.chunkDuration
            lastProcessedAudioTime = endTime // No overlap, so we advance by full chunk duration
            
            // Create transcription segment
            let segment = TranscriptionSegment(
                chunkIndex: currentChunkIndex,
                timeRange: startTime...endTime,
                text: result.text,
                confidence: result.confidence,
                updateType: .pending
            )
            
            transcriptionHistory.append(segment)
            currentChunkIndex += 1
            
            // Stabilize transcriptions
            stabilizeTranscriptionHistory()
            
            // Debug logging
            if config.asrConfig.enableDebug || result.text.isEmpty {
                logger.info("Chunk \(self.currentChunkIndex): '\(result.text)' (confidence: \(result.confidence))")
                print("DEBUG: Chunk \(self.currentChunkIndex) transcription: '\(result.text)'")
                print("DEBUG: Chunk time range: \(startTime)s - \(endTime)s")
            }
            
            // Track time to first word
            if !firstWordDetected && !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                firstWordDetected = true
                if let startTime = self.startTime {
                    metrics.timeToFirstWord = Date().timeIntervalSince(startTime)
                    logger.info("Time to first word: \(String(format: "%.2f", self.metrics.timeToFirstWord ?? 0))s")
                }
            }
            
            // Create update
            let update = TranscriptionUpdate(
                streamId: streamId,
                timestamp: Date(),
                timeRange: startTime...endTime,
                text: result.text,
                type: .pending,
                confidence: result.confidence,
                processingTime: processingTime
            )
            
            // Send to stream if active
            streamContinuation?.yield(update)
            
            return update
        }
        
        /// Create an async stream of transcription updates
        func getTranscriptionStream() -> AsyncThrowingStream<TranscriptionUpdate, Error> {
            AsyncThrowingStream { continuation in
                self.streamContinuation = continuation
                
                // Set up termination handler
                continuation.onTermination = { @Sendable _ in
                    Task {
                        await self.cleanup()
                    }
                }
            }
        }
        
        /// Get current metrics
        func getMetrics() -> StreamMetrics {
            return metrics
        }
        
        /// Get full transcription history
        func getTranscriptionHistory() -> [(timeRange: ClosedRange<TimeInterval>, text: String, type: TranscriptionUpdate.UpdateType)] {
            return transcriptionHistory.map { segment in
                (segment.timeRange, segment.text, segment.updateType)
            }
        }
        
        /// Stabilize transcription history based on chunk age
        private func stabilizeTranscriptionHistory() {
            let currentIndex = currentChunkIndex
            
            for (index, segment) in transcriptionHistory.enumerated() {
                let chunkAge = currentIndex - segment.chunkIndex
                
                // Update segment type based on age
                if chunkAge >= config.stabilizationDelay {
                    transcriptionHistory[index].updateType = .confirmed
                } else if chunkAge >= 1 {
                    transcriptionHistory[index].updateType = .partial
                }
                // Latest chunk remains .pending
            }
        }
        
        /// Merge overlapping transcriptions for final output
        func getFinalTranscription() -> String {
            // Include all segments (confirmed, partial, and pending) for final output
            // In a real-time scenario, you might want to only use confirmed segments
            return transcriptionHistory
                .map { $0.text }
                .joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        /// Transcribe a chunk with the appropriate source
        private func transcribeChunk(_ audioChunk: [Float]) async throws -> ASRResult {
            // Debug: print chunk info
            if config.asrConfig.enableDebug {
                let avgAmplitude = audioChunk.reduce(0.0) { $0 + abs($1) } / Float(audioChunk.count)
                print("DEBUG: Transcribing chunk with \(audioChunk.count) samples, avg amplitude: \(avgAmplitude)")
            }
            
            // Use the source-specific decoder state within AsrManager
            return try await asrManager.transcribe(audioChunk, source: source)
        }
        
        /// Clean up resources
        private func cleanup() {
            streamContinuation?.finish()
            streamContinuation = nil
            logger.info("Cleaned up stream processor for stream \(self.streamId)")
        }
    }
}