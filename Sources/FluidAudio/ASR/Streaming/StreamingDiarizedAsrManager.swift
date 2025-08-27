import AVFoundation
import CoreMedia
import Foundation
import OSLog

/// High-level streaming manager that combines ASR transcription with speaker diarization
///
/// Provides real-time speaker-attributed transcription by coordinating:
/// - StreamingAsrManager for speech-to-text
/// - DiarizerManager for speaker identification
/// - Alignment logic to match transcribed text with speakers
///
/// **Basic Usage:**
/// ```swift
/// let diarizedAsr = StreamingDiarizedAsrManager(config: .default)
/// try await diarizedAsr.start()
///
/// // Listen for speaker-attributed transcriptions
/// for await result in diarizedAsr.results {
///     print("Speaker \(result.speakerId): \(result.attributedText)")
/// }
/// ```
@available(macOS 13.0, iOS 16.0, *)
public actor StreamingDiarizedAsrManager {
    private let logger = Logger(
        subsystem: "com.fluidinfluence.diarized-asr",
        category: "StreamingDiarizedASR"
    )
    private let config: StreamingDiarizedAsrConfig
    private let audioConverter = AudioConverter()

    // Core processing components
    private var streamingAsr: StreamingAsrManager?
    private var diarizer: DiarizerManager?
    private var asrModels: AsrModels?
    private var diarizerModels: DiarizerModels?

    // Audio input stream
    private let inputSequence: AsyncStream<AVAudioPCMBuffer>
    private let inputBuilder: AsyncStream<AVAudioPCMBuffer>.Continuation

    // Result output streams
    private var resultsContinuation: AsyncStream<DiarizedTranscriptionResult>.Continuation?
    private var snapshotsContinuation: AsyncStream<DiarizedTranscriptSnapshot>.Continuation?

    // Processing state
    private var processingTask: Task<Void, Error>?
    private var audioSource: AudioSource = .microphone
    private var isProcessing = false

    // Chunk processing state
    private var sampleBuffer: [Float] = []
    private var bufferStartTimeSeconds: TimeInterval = 0
    private var chunkCounter: Int = 0

    // Speaker and transcript tracking
    private var speakerTranscripts: [String: String] = [:]
    private var volatileTranscripts: [String: String] = [:]
    private var speakerStats: [String: SpeakerSessionStats] = [:]
    private var sessionStartTime: Date?

    // Result alignment
    private var alignmentProcessor: ResultAlignmentProcessor?
    private var asrResultsTask: Task<Void, Error>?

    /// Initialize the diarized ASR manager
    /// - Parameter config: Configuration for the unified pipeline
    public init(config: StreamingDiarizedAsrConfig = .default) {
        self.config = config

        // Create input stream
        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        self.inputSequence = stream
        self.inputBuilder = continuation

        logger.info("Initialized StreamingDiarizedAsrManager")
    }

    /// Start the streaming pipeline with automatic model download
    /// - Parameter source: The audio source to use (default: microphone)
    public func start(source: AudioSource = .microphone) async throws {
        logger.info("Starting diarized ASR pipeline for source: \(String(describing: source))...")

        // Download and load models concurrently
        async let asrModelsTask = AsrModels.downloadAndLoad()
        async let diarizerModelsTask = DiarizerModels.download()

        let asrModels = try await asrModelsTask
        let diarizerModels = try await diarizerModelsTask

        try await start(asrModels: asrModels, diarizerModels: diarizerModels, source: source)
    }

    /// Start the streaming pipeline with pre-loaded models
    /// - Parameters:
    ///   - asrModels: Pre-loaded ASR models
    ///   - diarizerModels: Pre-loaded diarizer models
    ///   - source: The audio source to use
    public func start(
        asrModels: AsrModels,
        diarizerModels: DiarizerModels,
        source: AudioSource = .microphone
    ) async throws {
        logger.info("Starting diarized ASR pipeline with pre-loaded models...")

        self.audioSource = source
        self.asrModels = asrModels
        self.diarizerModels = diarizerModels

        // Initialize streaming ASR
        streamingAsr = StreamingAsrManager(config: config.asrConfig)
        try await streamingAsr?.start(models: asrModels, source: source)

        // Initialize diarizer
        diarizer = DiarizerManager(config: config.diarizerConfig)
        diarizer?.initialize(models: diarizerModels)

        // Initialize alignment processor
        alignmentProcessor = ResultAlignmentProcessor(alignmentTolerance: config.alignmentTolerance)

        // Reset state
        await resetProcessingState()
        sessionStartTime = Date()

        // Start monitoring ASR results
        await startAsrResultsMonitoring()

        // Start background processing
        processingTask = Task {
            logger.info("Starting unified processing task...")
            try await processAudioStream()
        }

        isProcessing = true
        logger.info("Diarized ASR pipeline started successfully")
    }

    /// Stream audio data for processing
    /// - Parameter buffer: Audio buffer in any format (will be converted to 16kHz mono)
    public func streamAudio(_ buffer: AVAudioPCMBuffer) {
        inputBuilder.yield(buffer)
    }

    /// Results stream: speaker-attributed transcription updates
    public var results: AsyncStream<DiarizedTranscriptionResult> {
        AsyncStream { continuation in
            self.resultsContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearResultsContinuation()
                }
            }
        }
    }

    /// Snapshots stream: complete state of all speaker transcripts
    public var snapshots: AsyncStream<DiarizedTranscriptSnapshot> {
        AsyncStream { continuation in
            self.snapshotsContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearSnapshotsContinuation()
                }
            }
        }
    }

    /// Get current session statistics for all speakers
    public var sessionStatistics: [SpeakerSessionStats] {
        Array(speakerStats.values)
    }

    /// Finish processing and get the complete diarized transcript
    /// - Returns: Final transcript organized by speaker
    public func finish() async throws -> [String: String] {
        logger.info("Finishing diarized ASR pipeline...")

        // Signal end of input
        inputBuilder.finish()

        // Wait for processing to complete
        do {
            try await processingTask?.value
        } catch {
            logger.error("Processing task failed: \(error)")
            throw error
        }

        // Process any remaining chunks
        await processRemainingChunks()

        // Merge volatile transcripts into final transcripts for the final result
        for (speakerId, volatileText) in volatileTranscripts {
            if speakerTranscripts[speakerId] == nil {
                speakerTranscripts[speakerId] = volatileText
            } else {
                speakerTranscripts[speakerId]! += " " + volatileText
            }
        }

        // Return final speaker transcripts (including volatile ones that are now finalized)
        logger.info("Diarized ASR finished with \(self.speakerTranscripts.count) speakers")
        return speakerTranscripts
    }

    /// Reset the pipeline for a new session
    public func reset() async throws {
        await resetProcessingState()

        // Reset components
        try await streamingAsr?.reset()
        diarizer?.speakerManager.reset()

        sessionStartTime = Date()
        logger.info("Diarized ASR pipeline reset")
    }

    /// Cancel processing without waiting for results
    public func cancel() async {
        inputBuilder.finish()
        processingTask?.cancel()
        asrResultsTask?.cancel()
        resultsContinuation?.finish()
        snapshotsContinuation?.finish()

        // Cleanup components
        await streamingAsr?.cancel()
        diarizer?.cleanup()
        await audioConverter.cleanup()

        isProcessing = false
        logger.info("Diarized ASR pipeline cancelled")
    }

    // MARK: - Private Methods

    /// Reset all processing state
    private func resetProcessingState() async {
        sampleBuffer.removeAll(keepingCapacity: true)
        bufferStartTimeSeconds = 0
        chunkCounter = 0

        speakerTranscripts.removeAll()
        volatileTranscripts.removeAll()
        speakerStats.removeAll()

        await alignmentProcessor?.reset()
    }

    /// Start monitoring ASR results from the streaming ASR manager
    private func startAsrResultsMonitoring() async {
        guard let streamingAsr = streamingAsr else { return }

        asrResultsTask = Task {
            if config.enableDebug {
                print("DEBUG: Starting ASR results monitoring task")
            }

            // Monitor ASR results and feed them to alignment processor
            var resultCount = 0
            for await asrResult in await streamingAsr.results {
                resultCount += 1
                // DEBUG: Print ASR results
                if config.enableDebug {
                    print("DEBUG: Received ASR result #\(resultCount): '\(asrResult.attributedText)'")
                }

                await alignmentProcessor?.addAsrResults([asrResult])

                // Trigger alignment processing
                await processAlignmentResults()
            }

            if config.enableDebug {
                print("DEBUG: ASR results monitoring completed with \(resultCount) results")
            }
        }
    }

    /// Main audio processing loop
    private func processAudioStream() async throws {
        for await pcmBuffer in inputSequence {
            do {
                // Convert to ASR format
                let samples = try await audioConverter.convertToAsrFormat(pcmBuffer)

                // Add to buffer and process chunks
                await appendSamplesAndProcessChunks(samples)

            } catch {
                logger.error("Audio processing error: \(error)")
                throw DiarizedAsrError.processingFailed(error.localizedDescription)
            }
        }
    }

    /// Add samples to buffer and process complete chunks
    private func appendSamplesAndProcessChunks(_ samples: [Float]) async {
        // Add samples to buffer
        sampleBuffer.append(contentsOf: samples)

        let chunkSize = Int(config.diarizerConfig.chunkDuration * 16000)  // 10 seconds at 16kHz

        // Process complete chunks
        while sampleBuffer.count >= chunkSize {
            let chunkSamples = Array(sampleBuffer.prefix(chunkSize))
            let chunkStartTime = bufferStartTimeSeconds

            // Create processing chunk
            let chunk = ProcessingChunk(
                chunkId: UUID(),
                audioSamples: chunkSamples,
                startTimeSeconds: chunkStartTime
            )

            // Process the chunk
            do {
                try await processChunk(chunk)
            } catch {
                logger.error("Chunk processing failed: \(error)")
            }

            // Remove processed samples
            let stepSize = chunkSize - Int(config.diarizerConfig.chunkOverlap * 16000)
            sampleBuffer.removeFirst(stepSize)
            bufferStartTimeSeconds += Double(stepSize) / 16000.0
            chunkCounter += 1
        }
    }

    /// Process a single audio chunk through both ASR and diarization
    private func processChunk(_ chunk: ProcessingChunk) async throws {
        // DEBUG: Print chunk processing
        if config.enableDebug {
            print("DEBUG: Processing chunk \(self.chunkCounter) with \(chunk.audioSamples.count) samples")
        }

        guard let streamingAsr = streamingAsr,
            let diarizer = diarizer
        else {
            throw DiarizedAsrError.processingFailed("Components not initialized")
        }

        // Process through diarization to get speaker segments
        let diarizationResult = try diarizer.performCompleteDiarization(
            chunk.audioSamples,
            sampleRate: 16000
        )

        // Adjust speaker segment timestamps to absolute time
        let adjustedSpeakerSegments = diarizationResult.segments.map { segment in
            TimedSpeakerSegment(
                speakerId: segment.speakerId,
                embedding: segment.embedding,
                startTimeSeconds: segment.startTimeSeconds + Float(chunk.startTimeSeconds),
                endTimeSeconds: segment.endTimeSeconds + Float(chunk.startTimeSeconds),
                qualityScore: segment.qualityScore
            )
        }

        // Add speaker segments to alignment processor
        await alignmentProcessor?.addSpeakerSegments(adjustedSpeakerSegments)

        // DEBUG: Print speaker segments found
        if config.enableDebug {
            print("DEBUG: Chunk \(self.chunkCounter) found \(adjustedSpeakerSegments.count) speaker segments")
            for segment in adjustedSpeakerSegments {
                print("  - Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
            }
        }

        // Create PCM buffer from samples for streaming ASR
        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        let frameCount = AVAudioFrameCount(chunk.audioSamples.count)
        guard
            let pcmBuffer = AVAudioPCMBuffer(
                pcmFormat: audioFormat,
                frameCapacity: frameCount
            )
        else {
            throw DiarizedAsrError.processingFailed("Failed to create PCM buffer")
        }

        pcmBuffer.frameLength = frameCount
        let channelData = pcmBuffer.floatChannelData![0]
        for i in 0..<chunk.audioSamples.count {
            channelData[i] = chunk.audioSamples[i]
        }

        // Feed to streaming ASR (ASR results will be processed via the monitoring task)
        await streamingAsr.streamAudio(pcmBuffer)

        // Trigger alignment processing
        await processAlignmentResults()

        logger.debug(
            "Processed chunk with \(adjustedSpeakerSegments.count) speaker segments"
        )
    }

    private func clearResultsContinuation() {
        resultsContinuation = nil
    }

    private func clearSnapshotsContinuation() {
        snapshotsContinuation = nil
    }

    /// Process any remaining chunks at the end of the stream
    private func processRemainingChunks() async {
        // Process any remaining audio in the buffer
        if !sampleBuffer.isEmpty {
            let chunk = ProcessingChunk(
                chunkId: UUID(),
                audioSamples: sampleBuffer,
                startTimeSeconds: bufferStartTimeSeconds
            )

            do {
                try await processChunk(chunk)
            } catch {
                logger.error("Failed to process final chunk: \(error)")
            }
        }

        // Final alignment pass
        await processAlignmentResults()
    }
}

// MARK: - Result Processing and Emission

@available(macOS 13.0, iOS 16.0, *)
extension StreamingDiarizedAsrManager {

    /// Process alignment results and emit diarized transcription results
    private func processAlignmentResults() async {
        guard let alignmentProcessor = alignmentProcessor else { return }

        // Get newly aligned results
        let alignedResults = await alignmentProcessor.processAlignment()

        // DEBUG: Print alignment results
        if config.enableDebug && !alignedResults.isEmpty {
            print("DEBUG: Aligned \(alignedResults.count) results")
        }

        // Process each aligned result
        for result in alignedResults {
            // Update speaker transcripts
            await updateSpeakerTranscripts(with: result)

            // Emit the diarized result
            resultsContinuation?.yield(result)
        }

        // Emit updated snapshot if we have new results
        if !alignedResults.isEmpty {
            await emitSnapshot()
        }

        // Log alignment statistics
        let stats = await alignmentProcessor.alignmentStats
        if config.enableDebug {
            print(
                "DEBUG: Alignment stats - ASR pending: \(stats.pendingAsrCount), Speaker pending: \(stats.pendingSpeakerCount), Total aligned: \(stats.alignedCount)"
            )
        }
    }

    /// Update speaker transcripts with a new diarized result
    private func updateSpeakerTranscripts(with result: DiarizedTranscriptionResult) async {
        let speakerId = result.speakerId
        let text = String(result.attributedText.characters)

        // DEBUG: Print transcript update
        if config.enableDebug {
            print("DEBUG: Updating transcript for Speaker \(speakerId): '\(text)' (final: \(result.isFinal))")
        }

        if result.isFinal {
            // Update finalized transcript
            if speakerTranscripts[speakerId] == nil {
                speakerTranscripts[speakerId] = ""
            }
            speakerTranscripts[speakerId]! += " " + text

            // Clear volatile for this speaker
            volatileTranscripts.removeValue(forKey: speakerId)

            // Update speaker stats
            await updateSpeakerStats(
                speakerId: speakerId,
                duration: result.audioTimeRange.duration.seconds,
                confidence: result.transcriptionConfidence
            )

            if config.enableDebug {
                print(
                    "DEBUG: Updated final transcript for Speaker \(speakerId), total length: \(speakerTranscripts[speakerId]?.count ?? 0)"
                )
            }
        } else {
            // Update volatile transcript
            volatileTranscripts[speakerId] = text
        }
    }

    /// Update statistics for a speaker
    private func updateSpeakerStats(
        speakerId: String,
        duration: TimeInterval,
        confidence: Float
    ) async {
        let currentTime = Date()

        if var stats = speakerStats[speakerId] {
            // Update existing stats
            stats = SpeakerSessionStats(
                speakerId: speakerId,
                totalSpeakingTime: stats.totalSpeakingTime + duration,
                segmentCount: stats.segmentCount + 1,
                averageSegmentDuration: (stats.averageSegmentDuration * Double(stats.segmentCount) + duration)
                    / Double(stats.segmentCount + 1),
                averageConfidence: (stats.averageConfidence * Float(stats.segmentCount) + confidence)
                    / Float(stats.segmentCount + 1),
                lastActivity: currentTime
            )
            speakerStats[speakerId] = stats
        } else {
            // Create new stats
            let newStats = SpeakerSessionStats(
                speakerId: speakerId,
                totalSpeakingTime: duration,
                segmentCount: 1,
                averageSegmentDuration: duration,
                averageConfidence: confidence,
                lastActivity: currentTime
            )
            speakerStats[speakerId] = newStats
        }
    }

    /// Emit a complete snapshot of the current state
    private func emitSnapshot() async {
        let finalizedBySpeaker = speakerTranscripts.compactMapValues { text in
            text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? nil : AttributedString(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        let volatileBySpeaker = volatileTranscripts.compactMapValues { text in
            text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? nil : AttributedString(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        // Create combined transcript in chronological order
        // For now, we'll use a simple speaker-based ordering
        let combinedText =
            speakerTranscripts
            .sorted(by: { $0.key < $1.key })
            .compactMap { (speakerId, text) in
                let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
                return cleanText.isEmpty ? nil : "Speaker \(speakerId): \(cleanText)"
            }
            .joined(separator: "\n\n")

        let snapshot = DiarizedTranscriptSnapshot(
            finalizedBySpeaker: finalizedBySpeaker,
            volatileBySpeaker: volatileBySpeaker,
            combinedTranscript: AttributedString(combinedText),
            activeSpeakers: Set(volatileTranscripts.keys),
            lastUpdated: Date()
        )

        snapshotsContinuation?.yield(snapshot)
    }
}
