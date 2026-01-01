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
///     if let result = try diarizer.processChunk(audioChunk) {
///         // Handle speaker probabilities
///     }
/// }
///
/// // Or complete file
/// let result = try diarizer.processComplete(audioSamples)
/// ```
public final class SortformerDiarizer: @unchecked Sendable {
    private let logger = AppLogger(category: "SortformerDiarizer")
    private let config: SortformerConfig
    private let modules: SortformerModules

    private var models: SortformerModels?
    private var state: SortformerStreamingState

    // Native mel spectrogram (used when useNativePreprocessing is enabled)
    private lazy var melSpectrogram: NeMoMelSpectrogram = NeMoMelSpectrogram()

    // Audio buffering
    private var audioBuffer: [Float] = []
    private var lastAudioSample: Float = 0
    
    // Feature buffering
    private var featureBuffer: [Float] = []

    // Chunk tracking
    private var preprocessorChunkIndex: Int = 0
    private var diarizerChunkIndex: Int = 0

    // Accumulated results
    private var allProbabilities: [Float] = []
    private var totalFramesProcessed: Int = 0

    // MARK: - Initialization

    public init(config: SortformerConfig = .gradientDescent) {
        self.config = config
        self.modules = SortformerModules(config: config)
        self.state = modules.initStreamingState()
    }

    /// Check if diarizer is ready for processing.
    public var isAvailable: Bool {
        models != nil
    }

    /// Initialize with CoreML models (combined pipeline mode).
    ///
    /// - Parameters:
    ///   - preprocessorPath: Path to SortformerPreprocessor.mlpackage
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
        preprocessorChunkIndex = 0
        diarizerChunkIndex = 0
        allProbabilities = []
        totalFramesProcessed = 0
        
        audioBuffer.reserveCapacity(config.preprocessorAudioSamples + config.audioHopSamples)
        featureBuffer.reserveCapacity((config.chunkMelFrames + config.chunkLen * config.subsamplingFactor) * config.melFeatures)
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

        var newProbabilities: [Float]?
        var newTentativeProbabilities: [Float]?
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

            // Debug: print raw prediction stats
            if config.debugMode && diarizerChunkIndex == 0 {
                let rawPreds = output.predictions
                let rawMin = rawPreds.min() ?? 0
                let rawMax = rawPreds.max() ?? 0
                let rawMean = rawPreds.reduce(0, +) / Float(rawPreds.count)
                print("[DEBUG] Raw preds: count=\(rawPreds.count), min=\(rawMin), max=\(rawMax), mean=\(rawMean)")
                print("[DEBUG] First 16 raw preds: \(Array(rawPreds.prefix(16)))")
                fflush(stdout)
            }

            // Raw predictions are already probabilities (model applies sigmoid internally)
            // DO NOT apply sigmoid again
            let probabilities = output.predictions

            // Trim embeddings to actual length
            let embLength = output.chunkLength
            let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.fcDModel))

            // Update state with correct context values
            let updateResult = try modules.streamingUpdate(
                state: &state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: diarizerChunkIndex > 0 ? config.chunkLeftContext : 0,
                rightContext: config.chunkRightContext
            )

            // Accumulate confirmed results
            allProbabilities.append(contentsOf: updateResult.confirmed)
            newProbabilities = updateResult.confirmed
            newTentativeProbabilities = updateResult.tentative
            newFrameCount = updateResult.confirmed.count / config.numSpeakers
            newTentativeFrameCount = updateResult.tentative.count / config.numSpeakers

            if config.debugMode && diarizerChunkIndex < 5 {
                print(
                    "[DEBUG] Diarizer chunk \(diarizerChunkIndex): confirmed=\(updateResult.confirmed.count), tentative=\(updateResult.tentative.count), totalProbs=\(allProbabilities.count)"
                )
                fflush(stdout)
            }

            diarizerChunkIndex += 1
        }

        // Save updated state
        totalFramesProcessed = allProbabilities.count / config.numSpeakers

        // Return new results if any
        if let probs = newProbabilities {
            let startTime = Float(totalFramesProcessed - newFrameCount) * config.frameDurationSeconds
            return SortformerChunkResult(
                probabilities: probs,
                frameCount: newFrameCount,
                startTimeSeconds: startTime,
                tentativeProbabilities: newTentativeProbabilities ?? [],
                tentativeFrameCount: newTentativeFrameCount
            )
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
    ) throws -> SortformerResult {
        guard let models = models else {
            throw SortformerError.notInitialized
        }

        // Reset for fresh processing
        reset()

        var featureProvider = SortformerStreamingFeatureProvider(config: self.config, audio: samples)
        
        var chunksProcessed = 0

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
            let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.fcDModel))

            if config.debugMode && diarizerChunkIndex < 1 {
                print(
                    "[DEBUG] Model output: predictions=\(probabilities.count), embLength=\(embLength), actualChunkFrames=\(chunkLength)"
                )
                // Check predictions at different offsets
                // Model input: padded_spkcache(188) + padded_fifo(40) + chunk(14) = 242 frames
                for testOffset in [0, 14, 188] {
                    let maxFrames = probabilities.count / 4 - testOffset
                    if maxFrames > 0 {
                        print("[DEBUG] Testing offset \(testOffset):")
                        for frame in 0..<min(3, maxFrames) {
                            let idx = (testOffset + frame) * 4
                            if idx + 3 < probabilities.count {
                                let vals = (0..<4).map { String(format: "%.4f", probabilities[idx + $0]) }.joined(
                                    separator: ", ")
                                print("[DEBUG]   Frame \(frame): [\(vals)]")
                            }
                        }
                    }
                }
                fflush(stdout)
            }

            // Compute left/right context for prediction extraction
            let leftContext = (leftOffset + config.subsamplingFactor / 2) / config.subsamplingFactor
            let rightContext = (rightOffset - 1) / config.subsamplingFactor + 1

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

            // Debug first 5 chunks - format to match Python output for comparison
            if config.debugMode && diarizerChunkIndex < 5 {
                print(
                    "[Swift] Chunk \(diarizerChunkIndex): lc=\(leftContext), rc=\(rightContext), spkcache=\(debugSpkcacheLen), fifo=\(debugFifoLen)"
                )

                let actualFrames = updateResult.confirmed.count / config.numSpeakers
                let chunkMin = updateResult.confirmed.min() ?? 0
                let chunkMax = updateResult.confirmed.max() ?? 0
                print(
                    "         chunk_probs shape: [\(actualFrames), \(config.numSpeakers)], min=\(String(format: "%.4f", chunkMin)), max=\(String(format: "%.4f", chunkMax))"
                )

                for frame in 0..<min(7, actualFrames) {
                    let frameStart = frame * config.numSpeakers
                    var vals: [String] = []
                    for spk in 0..<config.numSpeakers {
                        if frameStart + spk < updateResult.confirmed.count {
                            vals.append(String(format: "%.4f", updateResult.confirmed[frameStart + spk]))
                        }
                    }
                    print("         Frame \(frame): [\(vals.joined(separator: ", "))]")
                }
                print("")
                fflush(stdout)
            }

            // Accumulate confirmed results (tentative not needed for batch processing)
            allProbabilities.append(contentsOf: updateResult.confirmed)
            chunksProcessed += 1
            diarizerChunkIndex += 1

            // Progress callback
            // processedFrames is in mel frames (after subsampling)
            // Each mel frame corresponds to melStride samples
            let processedMelFrames = diarizerChunkIndex * coreFrames
            let progress = min(processedMelFrames * config.melStride, samples.count)
            progressCallback?(progress, samples.count, chunksProcessed)
        }

        // Save updated state
        totalFramesProcessed = allProbabilities.count / config.numSpeakers

        if config.debugMode {
            print(
                "[DEBUG] Phase 2 complete: diarizerChunks=\(diarizerChunkIndex), totalProbs=\(allProbabilities.count), totalFrames=\(totalFramesProcessed)"
            )
            fflush(stdout)
        }

        return SortformerResult(
            allProbabilities: allProbabilities,
            totalFrames: totalFramesProcessed,
            frameDurationSeconds: config.frameDurationSeconds
        )
    }

    // MARK: - Accessors

    /// Get all accumulated probabilities so far.
    public func getAllProbabilities() -> [Float] {
        return allProbabilities
    }

    /// Get total frames processed.
    public func getTotalFrames() -> Int {
        return totalFramesProcessed
    }

    /// Get current streaming state (for debugging).
    public func getState() -> SortformerStreamingState? {
        return state
    }

    /// Get configuration.
    public func getConfig() -> SortformerConfig {
        return config
    }
    
    // MARK: - Helpers
    
    /// Preprocess audio into mel features and append to feature buffer.
    /// Only processes audio when there's enough to produce new mel frames.
    private func preprocessAudioToFeatures() {
        // Minimum audio samples needed to produce at least one mel frame
        // Formula: window_size + (num_frames - 1) * stride
        // For 1 frame: just window_size = 400 samples
        // But mel spectrogram has overlap, so we need to think in terms of hop size
        
        // For first chunk, we need enough audio for full chunkMelFrames
        // For subsequent chunks, we only need one chunk's worth of new audio
        // But we always preprocess using the same sliding window approach
        
        // Process audio when we have enough for the preprocessor
        let numAudioSamples: Int
        let audioHopSamples: Int
        
        if preprocessorChunkIndex > 0 {
            numAudioSamples = config.chunkAudioSamples
            audioHopSamples = config.chunkHopSamples
        } else {
            // First chunk: compute full 112 frames, hop audio by full amount.
            // This positions the next preprocessing at mel frame 112, which when appended
            // to the 64 remaining frames in featureBuffer creates the next full chunk.
            numAudioSamples = config.preprocessorAudioSamples
            audioHopSamples = config.audioHopSamples
        }
        
        while audioBuffer.count >= numAudioSamples {
            let audioSamples = Array(audioBuffer.prefix(numAudioSamples))
            let (mel, melLength, _) = melSpectrogram.computeFlatTransposed(
                audio: audioSamples, 
                lastAudioSample: lastAudioSample
            )
            
            guard melLength > 0 else {
                break
            }
            
            featureBuffer.append(contentsOf: mel)
            
            lastAudioSample = audioSamples[audioHopSamples - 1]
            audioBuffer.removeFirst(audioHopSamples)
            preprocessorChunkIndex += 1
        }
    }
    
    private func getNextChunkFeatures() -> (mel: [Float], melLength: Int)? {
        let chunkMelFrames = config.chunkMelFrames
        let requiredFeats = chunkMelFrames * config.melFeatures
        
        // Need enough features for a full chunk (core + left context + right context)
        guard featureBuffer.count >= requiredFeats else {
            return nil
        }
        
        // Extract features for current chunk
        let mel = Array(featureBuffer.prefix(requiredFeats))
        
        // Only remove core frames from buffer, keeping context for next chunk
        // This allows left/right context to overlap between chunks
        let coreFrames = config.coreFrames
        let melHop = coreFrames * config.melFeatures
        featureBuffer.removeFirst(melHop)
        
        return (mel, chunkMelFrames)
    }
}
