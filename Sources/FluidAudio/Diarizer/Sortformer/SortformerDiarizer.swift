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
    private var state: SortformerStreamingState?

    // Native mel spectrogram (used when useNativePreprocessing is enabled)
    private lazy var melSpectrogram: NeMoMelSpectrogram = NeMoMelSpectrogram()

    // Audio buffering
    private var audioBuffer: [Float] = []

    // Feature buffering
    private var featureBuffer: [Float] = []

    // Chunk tracking
    private var preprocessorChunkIndex: Int = 0
    private var diarizerChunkIndex: Int = 0

    // Accumulated results
    private var allProbabilities: [Float] = []
    private var totalFramesProcessed: Int = 0

    // MARK: - Initialization

    public init(config: SortformerConfig = .default) {
        self.config = config
        self.modules = SortformerModules(config: config)
    }

    /// Check if diarizer is ready for processing.
    public var isAvailable: Bool {
        models != nil && state != nil
    }

    /// Initialize with CoreML models (combined pipeline mode).
    ///
    /// - Parameters:
    ///   - preprocessorPath: Path to SortformerPreprocessor.mlpackage
    ///   - mainModelPath: Path to Sortformer.mlpackage
    public func initialize(
        preprocessorPath: URL,
        mainModelPath: URL
    ) async throws {
        logger.info("Initializing Sortformer diarizer (combined pipeline mode)")

        let loadedModels = try await SortformerModels.load(
            config: config,
            preprocessorPath: preprocessorPath,
            mainModelPath: mainModelPath
        )

        self.models = loadedModels
        self.state = modules.initStreamingState()

        // Reset buffers
        resetBuffers()

        logger.info("Sortformer initialized in \(String(format: "%.2f", loadedModels.compilationDuration))s")
    }
    
    /// Initialize with pre-loaded models.
    public func initialize(models: SortformerModels) {
        self.models = models
        self.state = modules.initStreamingState()
        resetBuffers()
        logger.info("Sortformer initialized with pre-loaded models")
    }

    /// Reset all internal state for a new audio stream.
    public func reset() {
        state = modules.initStreamingState()
        resetBuffers()
        logger.debug("Sortformer state reset")
    }

    private func resetBuffers() {
        audioBuffer = []
        featureBuffer = []
        preprocessorChunkIndex = 0
        diarizerChunkIndex = 0
        allProbabilities = []
        totalFramesProcessed = 0
    }

    /// Cleanup resources.
    public func cleanup() {
        models = nil
        state = nil
        resetBuffers()
        logger.info("Sortformer resources cleaned up")
    }

    // MARK: - Streaming Processing

    /// Add audio samples to the processing buffer.
    ///
    /// - Parameter samples: Audio samples (16kHz mono)
    public func addAudio(_ samples: [Float]) {
        audioBuffer.append(contentsOf: samples)
    }

    /// Add audio samples from any collection.
    public func addAudio<C: Collection>(_ samples: C) where C.Element == Float {
        audioBuffer.append(contentsOf: samples)
    }

    /// Process buffered audio and return any new results.
    ///
    /// Call this after adding audio with `addAudio()`.
    ///
    /// - Returns: New chunk results if enough audio was processed, nil otherwise
    public func process() throws -> SortformerChunkResult? {
        guard let models = models, var state = state else {
            throw SortformerError.notInitialized
        }

        var newProbabilities: [Float]?
        var newTentativeProbabilities: [Float]?
        var newFrameCount = 0
        var newTentativeFrameCount = 0

        // Step 1: Run preprocessor on available audio
        while let (chunkFeatures, chunkLengths) = preprocessStreaming() {
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
            let embLength = output.chunkEmbeddingLength
            let chunkEmbs = Array(output.chunkEmbeddings.prefix(embLength * config.fcDModel))

            // Update state with correct context values
            let updateResult = try modules.streamingUpdate(
                state: &state,
                chunk: chunkEmbs,
                preds: probabilities,
                leftContext: config.chunkLeftContext,
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
        self.state = state
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
        guard var state = state else {
            throw SortformerError.notInitialized
        }

        let coreFrames = config.chunkLen * config.subsamplingFactor  // 48 mel frames core

        while let (_, chunkFeatures, chunkLength, leftOffset, rightOffset) = featureProvider.next() {
            // Prepare state for model
            // Use actual state lengths (0 is valid for empty state - matches Python/NeMo)
            let modelSpkcacheLen = state.spkcacheLength
            let modelFifoLen = state.fifoLength

            // Ensure spkcache has exactly config.spkcacheLen frames
            var paddedSpkcache = state.spkcache
            let requiredSpkcacheSize = config.spkcacheLen * config.fcDModel
            if paddedSpkcache.count < requiredSpkcacheSize {
                paddedSpkcache.append(
                    contentsOf: [Float](repeating: 0.0, count: requiredSpkcacheSize - paddedSpkcache.count))
            }

            // Ensure fifo has exactly config.fifoLen frames
            var paddedFifo = state.fifo
            let requiredFifoSize = config.fifoLen * config.fcDModel
            if paddedFifo.count < requiredFifoSize {
                paddedFifo.append(contentsOf: [Float](repeating: 0.0, count: requiredFifoSize - paddedFifo.count))
            }

            // Run main model
            let output = try models.runMainModel(
                chunk: chunkFeatures,
                chunkLength: chunkLength,
                spkcache: paddedSpkcache,
                spkcacheLength: modelSpkcacheLen,
                fifo: paddedFifo,
                fifoLength: modelFifoLen,
                config: config
            )

            let probabilities = output.predictions

            // Trim embeddings to actual length
            let embLength = output.chunkEmbeddingLength
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
            let rightContext = (rightOffset + config.subsamplingFactor - 1) / config.subsamplingFactor + 1

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
        self.state = state
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
    
    private func preprocessStreaming() -> (mel: [Float], melLength: Int)? {
        let numAudioSamples = config.preprocessorAudioSamples
        let audioHopSamples = config.audioHopSamples
        
        guard audioBuffer.count >= numAudioSamples else {
            return nil
        }
        
        let audioSamples = Array(audioBuffer.prefix(numAudioSamples))
        let (mel, melLength, _) = melSpectrogram.computeFlatTransposed(audio: audioSamples)
        
        guard melLength > 0 else {
            return nil
        }
        
        audioBuffer.removeFirst(audioHopSamples)
        
        return (mel, melLength)
    }
}
