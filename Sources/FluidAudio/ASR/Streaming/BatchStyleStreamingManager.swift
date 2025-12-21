import AVFoundation
import Foundation
import OSLog

/// A streaming ASR manager that uses batch-style processing for reliability.
///
/// Unlike the standard `StreamingAsrManager` which preserves decoder state across chunks,
/// this manager processes each chunk independently (like batch transcription) and uses
/// timestamp-based token merging to stitch results together. This approach is more
/// reliable because:
/// 1. No decoder state corruption across chunks
/// 2. Uses proven ChunkProcessor merge algorithm (contiguous pairs → LCS → midpoint)
/// 3. Each chunk is self-contained and reproducible
///
/// Trade-off: Higher latency (must wait for chunk to fill), but much more reliable output.
public actor BatchStyleStreamingManager {
    private let logger = AppLogger(category: "BatchStyleStreaming")
    private let audioConverter: AudioConverter = AudioConverter()
    private let config: BatchStyleStreamingConfig

    // Audio buffer with absolute indexing
    private var sampleBuffer: [Float] = []
    private var totalSamplesReceived: Int = 0

    // Chunk tracking
    private var processedChunks: [[TokenWindow]] = []
    private var lastProcessedSampleIndex: Int = 0

    // ASR components
    private var asrManager: AsrManager?
    private var audioSource: AudioSource = .microphone

    // Input stream
    private let inputSequence: AsyncStream<AVAudioPCMBuffer>
    private let inputBuilder: AsyncStream<AVAudioPCMBuffer>.Continuation

    // Output stream
    private var updateContinuation: AsyncStream<BatchStyleTranscriptionUpdate>.Continuation?

    // Processing task
    private var processingTask: Task<Void, Error>?

    // Metrics
    private var startTime: Date?

    // Token window type (matches ChunkProcessor)
    private typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float)

    /// Initialize the batch-style streaming manager
    public init(config: BatchStyleStreamingConfig = .default) {
        self.config = config

        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        self.inputSequence = stream
        self.inputBuilder = continuation

        logger.info(
            "Initialized BatchStyleStreamingManager with chunk=\(config.chunkSeconds)s overlap=\(config.overlapSeconds)s"
        )
    }

    /// Start the streaming engine with automatic model download
    public func start(source: AudioSource = .microphone) async throws {
        logger.info("Starting BatchStyleStreamingManager...")

        let models = try await AsrModels.downloadAndLoad()
        try await start(models: models, source: source)
    }

    /// Start with pre-loaded models
    public func start(models: AsrModels, source: AudioSource = .microphone) async throws {
        logger.info("Starting BatchStyleStreamingManager with pre-loaded models...")

        self.audioSource = source

        // Initialize ASR manager
        asrManager = AsrManager(config: ASRConfig(sampleRate: 16000, tdtConfig: TdtConfig()))
        try await asrManager?.initialize(models: models)

        // Reset state
        sampleBuffer.removeAll()
        totalSamplesReceived = 0
        processedChunks.removeAll()
        lastProcessedSampleIndex = 0
        startTime = Date()

        // Start processing task
        processingTask = Task {
            logger.info("Processing task started, waiting for audio...")

            for await pcmBuffer in self.inputSequence {
                do {
                    let samples = try audioConverter.resampleBuffer(pcmBuffer)
                    await self.appendSamplesAndProcess(samples)
                } catch {
                    logger.error("Audio conversion error: \(error.localizedDescription)")
                }
            }

            // Stream ended - process any remaining audio
            await self.processRemainingAudio()

            logger.info("Processing task completed")
        }

        logger.info("BatchStyleStreamingManager started successfully")
    }

    /// Stream audio data for transcription
    public func streamAudio(_ buffer: AVAudioPCMBuffer) {
        inputBuilder.yield(buffer)
    }

    /// Get async stream of transcription updates
    public var transcriptionUpdates: AsyncStream<BatchStyleTranscriptionUpdate> {
        AsyncStream { continuation in
            self.updateContinuation = continuation

            continuation.onTermination = { @Sendable _ in
                Task { [weak self] in
                    await self?.clearUpdateContinuation()
                }
            }
        }
    }

    /// Finish streaming and get final transcription
    public func finish() async throws -> String {
        logger.info("Finishing BatchStyleStreamingManager...")

        inputBuilder.finish()

        do {
            try await processingTask?.value
        } catch {
            logger.error("Processing task failed: \(error)")
            throw error
        }

        // Merge all chunks for final result
        let finalText = await getFinalTranscription()
        logger.info("Final transcription: \(finalText.count) characters")
        return finalText
    }

    /// Cancel streaming
    public func cancel() async {
        inputBuilder.finish()
        processingTask?.cancel()
        updateContinuation?.finish()
        logger.info("BatchStyleStreamingManager cancelled")
    }

    // MARK: - Private Methods

    private func clearUpdateContinuation() {
        updateContinuation = nil
    }

    /// Append samples and process chunks when ready
    private func appendSamplesAndProcess(_ samples: [Float]) async {
        sampleBuffer.append(contentsOf: samples)
        totalSamplesReceived += samples.count

        // Check if we have enough for a new chunk
        let chunkSamples = config.chunkSamples
        let overlapSamples = config.overlapSamples
        let strideSamples = chunkSamples - overlapSamples

        // Calculate how much new audio we have since last processed
        let availableNewSamples = sampleBuffer.count - (lastProcessedSampleIndex > 0 ? overlapSamples : 0)

        // Process chunks as they become available
        while availableNewSamples >= chunkSamples ||
              (sampleBuffer.count >= config.minChunkSamples && availableNewSamples >= config.minChunkSamples) {

            let chunkStart = max(0, sampleBuffer.count - availableNewSamples - overlapSamples)
            let chunkEnd = min(sampleBuffer.count, chunkStart + chunkSamples)

            guard chunkEnd > chunkStart else { break }

            let chunkSamplesSlice = Array(sampleBuffer[chunkStart..<chunkEnd])
            let absoluteStartSample = totalSamplesReceived - sampleBuffer.count + chunkStart

            await processChunk(chunkSamplesSlice, absoluteStartSample: absoluteStartSample)

            // Update tracking
            lastProcessedSampleIndex = chunkEnd

            // Trim buffer but keep overlap for next chunk
            let keepFromIndex = max(0, chunkEnd - overlapSamples)
            if keepFromIndex > 0 {
                sampleBuffer.removeFirst(keepFromIndex)
                lastProcessedSampleIndex -= keepFromIndex
            }

            break // Process one chunk at a time for now
        }
    }

    /// Process remaining audio at end of stream
    private func processRemainingAudio() async {
        guard !sampleBuffer.isEmpty else { return }

        // Process whatever is left, even if it's a partial chunk
        let absoluteStartSample = totalSamplesReceived - sampleBuffer.count
        await processChunk(sampleBuffer, absoluteStartSample: absoluteStartSample, isLastChunk: true)
    }

    /// Process a single chunk using batch-style (stateless) transcription
    private func processChunk(_ samples: [Float], absoluteStartSample: Int, isLastChunk: Bool = false) async {
        guard let asrManager = asrManager else { return }
        guard samples.count >= 16000 else {
            logger.debug("Skipping chunk with insufficient samples: \(samples.count)")
            return
        }

        let chunkStartTime = Date()

        do {
            // Create a fresh decoder state for this chunk (stateless, like batch)
            var decoderState = TdtDecoderState.make()

            // Pad audio to model input size
            let paddedSamples = asrManager.padAudioIfNeeded(samples, targetLength: 240_000)
            let actualFrameCount = ASRConstants.calculateEncoderFrames(from: samples.count)
            let globalFrameOffset = absoluteStartSample / ASRConstants.samplesPerEncoderFrame

            // Run inference
            let (hypothesis, _) = try await asrManager.executeMLInferenceWithTimings(
                paddedSamples,
                originalLength: samples.count,
                actualAudioFrames: actualFrameCount,
                decoderState: &decoderState,
                contextFrameAdjustment: 0,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset
            )

            // Convert to TokenWindow format
            let tokenWindows: [TokenWindow] = zip(
                zip(hypothesis.ySequence, hypothesis.timestamps),
                hypothesis.tokenConfidences
            ).map { (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1) }

            // Store chunk result
            processedChunks.append(tokenWindows)

            let processingTime = Date().timeIntervalSince(chunkStartTime)
            let chunkDuration = Double(samples.count) / 16000.0

            logger.debug(
                "Processed chunk \(self.processedChunks.count): \(tokenWindows.count) tokens, " +
                "\(String(format: "%.2f", chunkDuration))s audio in \(String(format: "%.3f", processingTime))s"
            )

            // Emit update
            await emitUpdate(isLastChunk: isLastChunk)

        } catch {
            logger.error("Chunk processing failed: \(error.localizedDescription)")
        }
    }

    /// Merge all processed chunks and emit update
    private func emitUpdate(isLastChunk: Bool) async {
        guard let asrManager = asrManager else { return }

        let mergedTokens = mergeAllChunks()

        // Convert tokens to text
        let result = asrManager.processTranscriptionResult(
            tokenIds: mergedTokens.map { $0.token },
            timestamps: mergedTokens.map { $0.timestamp },
            confidences: mergedTokens.map { $0.confidence },
            encoderSequenceLength: 0,
            audioSamples: [],
            processingTime: 0
        )

        let update = BatchStyleTranscriptionUpdate(
            text: result.text,
            isFinal: isLastChunk,
            confidence: result.confidence,
            timestamp: Date(),
            chunkCount: processedChunks.count,
            tokenTimings: result.tokenTimings ?? []
        )

        updateContinuation?.yield(update)

        if isLastChunk {
            updateContinuation?.finish()
        }
    }

    /// Merge all chunks using ChunkProcessor's algorithm
    private func mergeAllChunks() -> [TokenWindow] {
        guard var merged = processedChunks.first else { return [] }

        for chunk in processedChunks.dropFirst() {
            merged = mergeChunks(merged, chunk)
        }

        // Sort by timestamp
        if merged.count > 1 {
            merged.sort { $0.timestamp < $1.timestamp }
        }

        return merged
    }

    /// Get final transcription text
    private func getFinalTranscription() async -> String {
        guard let asrManager = asrManager else { return "" }

        let mergedTokens = mergeAllChunks()
        guard !mergedTokens.isEmpty else { return "" }

        let result = asrManager.processTranscriptionResult(
            tokenIds: mergedTokens.map { $0.token },
            timestamps: mergedTokens.map { $0.timestamp },
            confidences: mergedTokens.map { $0.confidence },
            encoderSequenceLength: 0,
            audioSamples: [],
            processingTime: 0
        )

        return result.text
    }

    // MARK: - Chunk Merging (adapted from ChunkProcessor)

    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }

    private let sampleRate: Int = 16000

    private func mergeChunks(_ left: [TokenWindow], _ right: [TokenWindow]) -> [TokenWindow] {
        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let frameDuration = Double(ASRConstants.samplesPerEncoderFrame) / Double(sampleRate)
        let overlapDuration = config.overlapSeconds
        let halfOverlapWindow = overlapDuration / 2

        func startTime(of token: TokenWindow) -> Double {
            Double(token.timestamp) * frameDuration
        }

        func endTime(of token: TokenWindow) -> Double {
            startTime(of: token) + frameDuration
        }

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        // No overlap - just concatenate
        if leftEndTime <= rightStartTime {
            return left + right
        }

        // Build overlap regions
        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            return mergeByMidpoint(
                left: left, right: right,
                leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration
            )
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        // Try contiguous pair matching first
        let contiguousPairs = findBestContiguousPairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right
            )
        }

        // Fall back to LCS
        let lcsPairs = findLongestCommonSubsequencePairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        guard !lcsPairs.isEmpty else {
            return mergeByMidpoint(
                left: left, right: right,
                leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration
            )
        }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right
        )
    }

    private func findBestContiguousPairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        var best: [(Int, Int)] = []

        for i in 0..<overlapLeft.count {
            for j in 0..<overlapRight.count {
                let leftToken = overlapLeft[i]
                let rightToken = overlapRight[j]

                if tokensMatch(leftToken, rightToken, tolerance: tolerance) {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j

                    while k < overlapLeft.count && l < overlapRight.count {
                        let nextLeft = overlapLeft[k]
                        let nextRight = overlapRight[l]

                        if tokensMatch(nextLeft, nextRight, tolerance: tolerance) {
                            current.append((k, l))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    private func findLongestCommonSubsequencePairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        let leftCount = overlapLeft.count
        let rightCount = overlapRight.count

        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)

        for i in 1...leftCount {
            for j in 1...rightCount {
                if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = leftCount
        var j = rightCount

        while i > 0 && j > 0 {
            if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return pairs.reversed()
    }

    private func tokensMatch(_ left: IndexedToken, _ right: IndexedToken, tolerance: Double) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow]
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            let rightIndex = rightIndices[idx]

            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
    }

    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        let trimmedLeft = left.filter { Double($0.timestamp) * frameDuration <= cutoff }
        let trimmedRight = right.filter { Double($0.timestamp) * frameDuration >= cutoff }
        return trimmedLeft + trimmedRight
    }
}

// MARK: - Configuration

/// Configuration for BatchStyleStreamingManager
public struct BatchStyleStreamingConfig: Sendable {
    /// Chunk size in seconds (should be close to but under 15s model limit)
    public let chunkSeconds: TimeInterval

    /// Overlap between chunks in seconds (for merge algorithm)
    public let overlapSeconds: TimeInterval

    /// Minimum chunk size to process (for final partial chunks)
    public let minChunkSeconds: TimeInterval

    /// Default configuration optimized for reliability
    public static let `default` = BatchStyleStreamingConfig(
        chunkSeconds: 14.0,        // Under 15s limit, leaves room for padding
        overlapSeconds: 2.0,       // Match ChunkProcessor
        minChunkSeconds: 2.0       // Don't process very short chunks
    )

    /// Low-latency configuration with smaller chunks
    public static let lowLatency = BatchStyleStreamingConfig(
        chunkSeconds: 10.0,        // Faster updates
        overlapSeconds: 2.0,
        minChunkSeconds: 2.0
    )

    public init(
        chunkSeconds: TimeInterval = 14.0,
        overlapSeconds: TimeInterval = 2.0,
        minChunkSeconds: TimeInterval = 2.0
    ) {
        self.chunkSeconds = chunkSeconds
        self.overlapSeconds = overlapSeconds
        self.minChunkSeconds = minChunkSeconds
    }

    // Sample counts at 16kHz
    var chunkSamples: Int { Int(chunkSeconds * 16000) }
    var overlapSamples: Int { Int(overlapSeconds * 16000) }
    var minChunkSamples: Int { Int(minChunkSeconds * 16000) }
}

// MARK: - Output Types

/// Transcription update from batch-style streaming
public struct BatchStyleTranscriptionUpdate: Sendable {
    /// Current transcription text (merged from all chunks so far)
    public let text: String

    /// Whether this is the final update (stream ended)
    public let isFinal: Bool

    /// Average confidence score
    public let confidence: Float

    /// Timestamp of this update
    public let timestamp: Date

    /// Number of chunks processed so far
    public let chunkCount: Int

    /// Token-level timing information
    public let tokenTimings: [TokenTiming]

    public init(
        text: String,
        isFinal: Bool,
        confidence: Float,
        timestamp: Date,
        chunkCount: Int,
        tokenTimings: [TokenTiming]
    ) {
        self.text = text
        self.isFinal = isFinal
        self.confidence = confidence
        self.timestamp = timestamp
        self.chunkCount = chunkCount
        self.tokenTimings = tokenTimings
    }
}
