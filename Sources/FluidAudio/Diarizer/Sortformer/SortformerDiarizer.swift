import Accelerate
import CoreML
import Foundation
import OSLog

/// Streaming speaker diarization using NVIDIA's Sortformer model.
///
/// Sortformer provides end-to-end streaming diarization with 4 fixed speaker slots,
/// achieving ~11% DER on DI-HARD III in real-time.
///
/// Usage:
/// ```swift
/// let diarizer = SortformerDiarizer()
/// try await diarizer.initialize(preprocessorPath: url1, mainModelPath: url2)
///
/// // Streaming mode
/// for audioChunk in audioStream {
///     if let result = try diarizer.processSamples(audioChunk) {
///         // Handle speaker probabilities
///     }
/// }
///
/// // Or complete file
/// let result = try diarizer.processComplete(audioSamples)
/// ```
public final class SortformerDiarizer: @unchecked Sendable {
    /// Accumulated results
    public private(set) var timeline: SortformerTimeline
    
    /// Check if diarizer is ready for processing.
    public var isAvailable: Bool {
        models != nil
    }
    
    /// Streaming state
    public private(set) var state: SortformerStreamingState
    
    /// Number of frames processed
    public private(set) var numFramesProcessed: Int = 0

    /// Configuration
    public let config: SortformerConfig
    
    private let logger = AppLogger(category: "SortformerDiarizer")
    private let modules: SortformerModules

    private var models: SortformerModels?

    // Native mel spectrogram (used when useNativePreprocessing is enabled)
    private lazy var melSpectrogram: NeMoMelSpectrogram = NeMoMelSpectrogram()

    // Audio buffering
    private var audioBuffer: [Float] = []
    private var lastAudioSample: Float = 0
    
    // Feature buffering
    internal var featureBuffer: [Float] = []

    // Chunk tracking
    private var startFeat: Int = 0  // Current position in mel feature stream
    private var diarizerChunkIndex: Int = 0

    // MARK: - Initialization

    public init(config: SortformerConfig = .default, postProcessingConfig: SortformerPostProcessingConfig = .default) {
        self.config = config
        self.modules = SortformerModules(config: config)
        self.state = modules.initStreamingState()
        self.timeline = SortformerTimeline(config: postProcessingConfig)
    }

    /// Initialize with CoreML models (combined pipeline mode).
    ///
    /// - Parameters:
    ///   - mainModelPath: Path to Sortformer.mlpackage
    public func initialize(
        mainModelPath: URL
    ) async throws {
        logger.info("Initializing Sortformer diarizer (combined pipeline mode)")

        let loadedModels = try await SortformerModels.load(
            config: config,
            mainModelPath: mainModelPath
        )

        self.models = loadedModels
        self.state = modules.initStreamingState()
        self.lastAudioSample = 0
        
        // Reset buffers
        resetBuffers()
        logger.info("Sortformer initialized in \(String(format: "%.2f", loadedModels.compilationDuration))s")
    }
    
    /// Initialize with pre-loaded models.
    public func initialize(models: SortformerModels) {
        self.models = models
        self.state = modules.initStreamingState()
        self.lastAudioSample = 0
        resetBuffers()
        logger.info("Sortformer initialized with pre-loaded models")
    }

    /// Reset all internal state for a new audio stream.
    public func reset() {
        state = modules.initStreamingState()
        lastAudioSample = 0
        resetBuffers()
        logger.debug("Sortformer state reset")
    }

    private func resetBuffers() {
        audioBuffer = []
        featureBuffer = []
        lastAudioSample = 0
        startFeat = 0
        diarizerChunkIndex = 0
        timeline.reset()
        
        featureBuffer.reserveCapacity((config.chunkMelFrames + config.coreFrames) * config.melFeatures)
    }

    /// Cleanup resources.
    public func cleanup() {
        models = nil
        state.cleanup()
        resetBuffers()
        logger.info("Sortformer resources cleaned up")
    }

    // MARK: - Streaming Processing

    /// Add audio samples to the processing buffer.
    ///
    /// - Parameter samples: Audio samples (16kHz mono)
    public func addAudio<C: Collection>(_ samples: C) where C.Element == Float {
        audioBuffer.append(contentsOf: samples)
        preprocessAudioToFeatures()
    }

    /// Process buffered audio and return any new results.
    ///
    /// Call this after adding audio with `addAudio()`.
    ///
    /// - Returns: New chunk results if enough audio was processed, nil otherwise
    public func process() throws -> SortformerChunkResult? {
        guard let models = models else {
            throw SortformerError.notInitialized
        }

        var newPredictions: [Float] = []
        var newTentativePredictions: [Float] = []
        var newFrameCount = 0
        var newTentativeFrameCount = 0

        // Step 1: Run preprocessor on available audio
        while let (chunkFeatures, chunkLengths) = getNextChunkFeatures() {
            let output = try models.runMainModel(
                chunk: chunkFeatures,
                chunkLength: chunkLengths,
                spkcache: state.spkcache,
                spkcacheLength: state.spkcacheLength,
                fifo: state.fifo,
                fifoLength: state.fifoLength,
                config: config
            )
            
            // Raw predictions are already probabilities (model applies sigmoid internally)
            // DO NOT apply sigmoid again
            let probabilities = output.predictions
            
            // Trim embeddings to actual length
            let embLength = output.chunkLength
            let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.preEncoderDims))
            
            // Update state with correct context values
            let updateResult = try modules.streamingUpdate(
                state: &state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: diarizerChunkIndex > 0 ? config.chunkLeftContext : 0,
                rightContext: config.chunkRightContext
            )
            
            // Accumulate confirmed results
            newPredictions.append(contentsOf: updateResult.confirmed)
            newTentativePredictions = updateResult.tentative
            newFrameCount += updateResult.confirmed.count / config.numSpeakers
            newTentativeFrameCount = updateResult.tentative.count / config.numSpeakers
            
            diarizerChunkIndex += 1
        }

        // Return new results if any
        if newPredictions.count > 0 {
            let chunk = SortformerChunkResult(
                startFrame: numFramesProcessed,
                speakerPredictions: newPredictions,
                frameCount: newFrameCount,
                tentativePredictions: newTentativePredictions,
                tentativeFrameCount: newTentativeFrameCount
            )
            
            numFramesProcessed += newFrameCount
            timeline.addChunk(chunk)
            
            return chunk
        }

        return nil
    }

    /// Process a chunk of audio in one call.
    ///
    /// Convenience method that combines `addAudio()` and `process()`.
    ///
    /// - Parameter samples: Audio samples (16kHz mono)
    /// - Returns: New chunk results if enough audio was processed
    public func processSamples(_ samples: [Float]) throws -> SortformerChunkResult? {
        addAudio(samples)
        return try process()
    }

    // MARK: - Complete File Processing

    /// Progress callback type: (processedSamples, totalSamples, chunksProcessed)
    public typealias ProgressCallback = (Int, Int, Int) -> Void

    /// Process complete audio file.
    ///
    /// - Parameters:
    ///   - samples: Complete audio samples (16kHz mono)
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Complete diarization result
    public func processComplete(
        _ samples: [Float],
        progressCallback: ProgressCallback? = nil
    ) throws -> SortformerTimeline {
        guard let models = models else {
            throw SortformerError.notInitialized
        }

        // Reset for fresh processing
        reset()

        var featureProvider = SortformerFeatureLoader(config: self.config, audio: samples)
        var chunksProcessed = 0
        var predictions: [Float] = []
        var lastTentative: [Float] = []

        let coreFrames = config.chunkLen * config.subsamplingFactor  // 48 mel frames core

        while let (chunkFeatures, chunkLength, leftOffset, rightOffset) = featureProvider.next() {
            // Run main model
            let output = try models.runMainModel(
                chunk: chunkFeatures,
                chunkLength: chunkLength,
                spkcache: state.spkcache,
                spkcacheLength: state.spkcacheLength,
                fifo: state.fifo,
                fifoLength: state.fifoLength,
                config: config
            )

            let probabilities = output.predictions

            // Trim embeddings to actual length
            let embLength = output.chunkLength
            let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.preEncoderDims))

            // Compute left/right context for prediction extraction
            let leftContext = (leftOffset + config.subsamplingFactor / 2) / config.subsamplingFactor
            let rightContext = (rightOffset + config.subsamplingFactor - 1) / config.subsamplingFactor
            print("chunk \(chunksProcessed+1)/\(featureProvider.numChunks), leftOffset: \(leftOffset), chunkLength: \(chunkLength), rightOffset: \(rightOffset), outputChunkLength: \(output.chunkLength)")

            // Debug first 5 chunks - capture state BEFORE update
            let debugSpkcacheLen = state.spkcacheLength
            let debugFifoLen = state.fifoLength

            // Update state
            let updateResult = try modules.streamingUpdate(
                state: &state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: leftContext,
                rightContext: rightContext
            )
            
            // Accumulate confirmed results (tentative not needed for batch processing)
            predictions.append(contentsOf: updateResult.confirmed)
            lastTentative = updateResult.tentative
            chunksProcessed += 1
            diarizerChunkIndex += 1

            // Progress callback
            // processedFrames is in mel frames (after subsampling)
            // Each mel frame corresponds to melStride samples
            let processedMelFrames = diarizerChunkIndex * coreFrames
            let progress = min(processedMelFrames * config.melStride, samples.count)
            progressCallback?(progress, samples.count, chunksProcessed)
        }
        predictions.append(contentsOf: lastTentative)

        // Save updated state
        numFramesProcessed = predictions.count / config.numSpeakers

        if config.debugMode {
            print(
                "[DEBUG] Phase 2 complete: diarizerChunks=\(diarizerChunkIndex), totalProbs=\(predictions.count), totalFrames=\(numFramesProcessed)"
            )
            fflush(stdout)
        }

        timeline = SortformerTimeline(
            allPredictions: predictions,
            config: timeline.config,
            isComplete: true
        )
        
        return timeline
    }

    // MARK: - Helpers
    
    /// Preprocess audio into mel features and append to feature buffer.
    private func preprocessAudioToFeatures() {
        guard !audioBuffer.isEmpty else { return }
        
        let hopLength = config.melStride
        let winLength = config.melWindow
        
        // Track how many frames we've already extracted
        let existingFrames = featureBuffer.count / config.melFeatures
        
        // Compute mel on all accumulated audio
        let (mel, melLength, _) = melSpectrogram.computeFlatTransposed(
            audio: audioBuffer,
            lastAudioSample: lastAudioSample
        )
        
        guard melLength > existingFrames else { return }
        
        // Only append NEW frames
        let startIdx = existingFrames * config.melFeatures
        let endIdx = melLength * config.melFeatures
        featureBuffer.append(contentsOf: mel[startIdx..<endIdx])
        
        // Remove consumed audio samples to prevent unbounded growth
        // Keep enough audio for the last frame's right context (for correct finalization)
        // Frame i corresponds to audio starting at: i * hopLength - padLength (with center padding)
        // So to recompute frame existingFrames, we need audio from: existingFrames * hopLength - padLength
        // But we want to keep audio that affects frames we've already extracted (for preemphasis continuity)
        // Safe approach: remove audio that's fully processed (before the last extracted frame's window)
        let lastFrameAudioStart = max(0, (melLength - 1) * hopLength)
        let samplesToKeep = audioBuffer.count - lastFrameAudioStart + winLength
        let samplesToRemove = max(0, audioBuffer.count - samplesToKeep)
        
        if samplesToRemove > 0 {
            lastAudioSample = audioBuffer[samplesToRemove - 1]
            audioBuffer.removeFirst(samplesToRemove)
        }
    }
    
    internal func getNextChunkFeatures() -> (mel: [Float], melLength: Int)? {
        // Mirror batch SortformerFeatureProvider logic
        let featLength = featureBuffer.count / config.melFeatures
        let coreFrames = config.chunkLen * config.subsamplingFactor  // 48
        let leftContextFrames = config.chunkLeftContext * config.subsamplingFactor  // 8
        let rightContextFrames = config.chunkRightContext * config.subsamplingFactor  // 56
        
        // Calculate how far we can go
        let endFeat = min(startFeat + coreFrames, featLength)
        
        // Need at least some core frames
        guard endFeat > startFeat else { return nil }
        
        // Calculate available context
        let leftOffset = min(leftContextFrames, startFeat)
        let rightOffset = min(rightContextFrames, featLength - endFeat)
        
        // Extract chunk with context
        let chunkStartFrame = startFeat - leftOffset
        let chunkEndFrame = endFeat + rightOffset
        let chunkStartIndex = chunkStartFrame * config.melFeatures
        let chunkEndIndex = chunkEndFrame * config.melFeatures
        
        guard chunkEndIndex <= featureBuffer.count else { return nil }
        
        let mel = Array(featureBuffer[chunkStartIndex..<chunkEndIndex])
        let chunkLength = chunkEndFrame - chunkStartFrame
        
        // Advance position
        startFeat = endFeat
        
        // Remove consumed frames from buffer (frames before our new startFeat - leftContext)
        let newBufferStart = max(0, startFeat - leftContextFrames)
        let framesToRemove = newBufferStart
        if framesToRemove > 0 {
            featureBuffer.removeFirst(framesToRemove * config.melFeatures)
            startFeat -= framesToRemove
        }
        
        return (mel, chunkLength)
    }
}
