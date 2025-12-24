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
    private var featuresProcessed: Int = 0

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
            preprocessorPath: preprocessorPath,
            mainModelPath: mainModelPath
        )

        self.models = loadedModels
        self.state = modules.initStreamingState()

        // Reset buffers
        resetBuffers()

        logger.info("Sortformer initialized in \(String(format: "%.2f", loadedModels.compilationDuration))s")
    }

    /// Initialize with separate CoreML models (matches Python behavior).
    ///
    /// This mode uses separate PreEncoder + Head models instead of a combined pipeline,
    /// which avoids potential embedding corruption issues during CoreML pipeline merging.
    ///
    /// - Parameters:
    ///   - preprocessorPath: Path to Pipeline_Preprocessor.mlpackage
    ///   - preEncoderPath: Path to Pipeline_PreEncoder.mlpackage
    ///   - headPath: Path to Pipeline_Head_Fixed.mlpackage
    public func initializeSeparate(
        preprocessorPath: URL,
        preEncoderPath: URL,
        headPath: URL
    ) async throws {
        logger.info("Initializing Sortformer diarizer (separate models mode)")

        let loadedModels = try await SortformerModels.loadSeparate(
            preprocessorPath: preprocessorPath,
            preEncoderPath: preEncoderPath,
            headPath: headPath
        )

        self.models = loadedModels
        self.state = modules.initStreamingState()

        // Reset buffers
        resetBuffers()

        logger.info(
            "Sortformer initialized in \(String(format: "%.2f", loadedModels.compilationDuration))s (separate models)")
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
        featuresProcessed = 0
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
        var newFrameCount = 0

        let audioCountBefore = audioBuffer.count

        // Step 1: Run preprocessor on available audio
        while audioBuffer.count >= config.preprocessorAudioSamples {
            let audioChunk = Array(audioBuffer.prefix(config.preprocessorAudioSamples))

            let (features, featureLength) = try models.runPreprocessor(
                audioSamples: audioChunk,
                config: config
            )
            if config.debugMode {
                print(
                    "[DEBUG] Preprocessor output: \(features.count) floats, featureLength=\(featureLength), expected chunkFrames=\(config.chunkFrames)"
                )
                fflush(stdout)
            }

            // Handle feature overlap
            let validFeatures: [Float]
            if preprocessorChunkIndex == 0 {
                validFeatures = Array(features.prefix(featureLength * config.melFeatures))
            } else {
                let skipFrames = config.overlapFrames * config.melFeatures
                validFeatures = Array(
                    features.dropFirst(skipFrames).prefix((featureLength - config.overlapFrames) * config.melFeatures))
            }

            featureBuffer.append(contentsOf: validFeatures)
            audioBuffer.removeFirst(config.audioHopSamples)
            preprocessorChunkIndex += 1
        }

        if config.debugMode {
            print(
                "[DEBUG] After Step 1: featureBuffer=\(featureBuffer.count) floats (\(featureBuffer.count/config.melFeatures) frames), audioBuffer=\(audioBuffer.count) samples, preprocessorIdx=\(preprocessorChunkIndex)"
            )
            fflush(stdout)
        }

        // Step 2: Run diarization on available features
        // Match Python's streaming_feat_loader exactly:
        // - Each chunk covers core_frames (48) + context frames
        // - left_offset = min(left_context * subsampling, current_position)
        // - right_offset = min(right_context * subsampling, remaining_frames)
        let totalFeatureFrames = featureBuffer.count / config.melFeatures
        let chunkFrames = config.chunkFrames  // 112 mel frames for NVIDIA
        let coreFrames = config.chunkLen * config.subsamplingFactor  // 48 mel frames core
        let leftContextFrames = config.chunkLeftContext * config.subsamplingFactor  // 8 frames
        let rightContextFrames = config.chunkRightContext * config.subsamplingFactor  // 56 frames

        if config.debugMode && diarizerChunkIndex < 3 {
            print(
                "[DEBUG] Feature buffer: \(featureBuffer.count) floats = \(totalFeatureFrames) frames, need \(chunkFrames) frames, core=\(coreFrames), diarizerIdx=\(diarizerChunkIndex)"
            )
            fflush(stdout)
        }

        while true {
            // Python's streaming_feat_loader logic:
            // stt_feat = current start position (advances by core_frames each iteration)
            // end_feat = stt_feat + core_frames
            // left_offset = min(left_context_frames, stt_feat)
            // right_offset = min(right_context_frames, total_frames - end_feat)
            // chunk = features[stt_feat - left_offset : end_feat + right_offset]
            let sttFeat = diarizerChunkIndex * coreFrames
            let endFeat = min(sttFeat + coreFrames, totalFeatureFrames)

            // Check if we have enough frames to process
            if endFeat > totalFeatureFrames {
                break
            }

            let leftOffset = min(leftContextFrames, sttFeat)
            let rightOffset = min(rightContextFrames, totalFeatureFrames - endFeat)

            let chunkStart = sttFeat - leftOffset
            let chunkEnd = endFeat + rightOffset
            let actualChunkFrames = chunkEnd - chunkStart

            // Need at least some frames to process
            if actualChunkFrames <= 0 {
                break
            }

            // Extract features and pad if needed
            let featStart = chunkStart * config.melFeatures
            let featEnd = chunkEnd * config.melFeatures
            var chunkFeatures = Array(featureBuffer[featStart..<featEnd])

            // Pad to chunkFrames if needed (first chunk may have fewer frames due to missing left context)
            if actualChunkFrames < chunkFrames {
                let padCount = (chunkFrames - actualChunkFrames) * config.melFeatures
                chunkFeatures.append(contentsOf: [Float](repeating: 0.0, count: padCount))
            }

            // Features are already in [T, D] format from preprocessor extraction
            // Just copy directly
            let transposedChunk = chunkFeatures

            // Prepare state for model
            // Use actual state lengths (0 is valid for empty state - matches Python/NeMo)
            let modelSpkcacheLen = state.spkcacheLength
            let modelFifoLen = state.fifoLength

            if config.debugMode {
                print("[DEBUG] State lengths: spkcache=\(modelSpkcacheLen), fifo=\(modelFifoLen)")
                fflush(stdout)
            }

            // Ensure spkcache has exactly config.spkcacheLen frames (padded with zeros)
            var paddedSpkcache = state.spkcache
            let requiredSpkcacheSize = config.spkcacheLen * config.fcDModel
            if paddedSpkcache.count < requiredSpkcacheSize {
                paddedSpkcache.append(
                    contentsOf: [Float](repeating: 0.0, count: requiredSpkcacheSize - paddedSpkcache.count))
            }

            // Ensure fifo has exactly config.fifoLen frames (padded with zeros)
            var paddedFifo = state.fifo
            let requiredFifoSize = config.fifoLen * config.fcDModel
            if paddedFifo.count < requiredFifoSize {
                paddedFifo.append(contentsOf: [Float](repeating: 0.0, count: requiredFifoSize - paddedFifo.count))
            }

            // Run main model
            // Pass the actual valid chunk length (like Python's feat_lengths)
            if config.debugMode {
                print(
                    "[DEBUG] Main model input: chunk=\(transposedChunk.count) floats (actualFrames=\(actualChunkFrames)), spkcache=\(paddedSpkcache.count) (len=\(modelSpkcacheLen)), fifo=\(paddedFifo.count) (len=\(modelFifoLen))"
                )
                print(
                    "[DEBUG] Expected: chunk=\(config.chunkFrames * config.melFeatures), spkcache=\(config.spkcacheLen * config.fcDModel), fifo=\(config.fifoLen * config.fcDModel)"
                )
                print("[DEBUG] Calling main model...")
                fflush(stdout)
            }
            let output = try models.runMainModel(
                chunk: transposedChunk,
                chunkLength: actualChunkFrames,
                spkcache: paddedSpkcache,
                spkcacheLength: modelSpkcacheLen,
                fifo: paddedFifo,
                fifoLength: modelFifoLen,
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

            // Compute left/right context for prediction extraction
            // These are in encoder frames (after subsampling), not mel frames
            // Python: lc = round(left_offset / subsampling_factor)
            //         rc = ceil(right_offset / subsampling_factor)
            let leftContext = (leftOffset + config.subsamplingFactor / 2) / config.subsamplingFactor  // round
            let rightContext = (rightOffset + config.subsamplingFactor - 1) / config.subsamplingFactor  // ceil

            // Update state with correct context values
            let chunkPreds = modules.streamingUpdate(
                state: &state,
                chunkEmbeddings: chunkEmbs,
                predictions: probabilities,
                leftContext: leftContext,
                rightContext: rightContext,
                modelSpkcacheLen: modelSpkcacheLen,
                modelFifoLen: modelFifoLen
            )

            // Accumulate results
            allProbabilities.append(contentsOf: chunkPreds)
            newProbabilities = chunkPreds
            newFrameCount = chunkPreds.count / config.numSpeakers

            if config.debugMode && diarizerChunkIndex < 5 {
                print(
                    "[DEBUG] Diarizer chunk \(diarizerChunkIndex): chunkPreds.count=\(chunkPreds.count), totalProbs=\(allProbabilities.count)"
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
                startTimeSeconds: startTime
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
    public func processChunk(_ samples: [Float]) throws -> SortformerChunkResult? {
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

        let totalSamples = samples.count

        // PHASE 1: Preprocess ALL audio first (matches Python's approach)
        // Python processes entire audio through NeMo's preprocessor before diarization
        if config.debugMode {
            print(
                "[DEBUG] Phase 1: Preprocessing all audio (\(totalSamples) samples) using \(config.useNativePreprocessing ? "native Swift" : "CoreML")"
            )
            fflush(stdout)
        }

        if config.useNativePreprocessing {
            // Use native Swift mel spectrogram (matches NeMo's full-audio preprocessing)
            let (melFlat, melLength, _) = melSpectrogram.computeFlat(audio: samples)

            // Convert from [nMels, T] row-major to [T, nMels] for feature buffer
            // melFlat is [nMels * numFrames] in row-major (mel 0 all frames, mel 1 all frames, ...)
            // featureBuffer needs [T * nMels] (frame 0 all mels, frame 1 all mels, ...)
            let nMels = config.melFeatures
            featureBuffer = [Float](repeating: 0, count: melLength * nMels)
            for frameIdx in 0..<melLength {
                for melIdx in 0..<nMels {
                    featureBuffer[frameIdx * nMels + melIdx] = melFlat[melIdx * melLength + frameIdx]
                }
            }
            preprocessorChunkIndex = 1  // Treat as single chunk
        } else {
            // Use CoreML preprocessor (chunked approach)
            var audioOffset = 0
            while audioOffset + config.preprocessorAudioSamples <= totalSamples {
                let audioChunk = Array(samples[audioOffset..<(audioOffset + config.preprocessorAudioSamples)])

                let (features, featureLength) = try models.runPreprocessor(
                    audioSamples: audioChunk,
                    config: config
                )

                // Handle feature overlap
                let validFeatures: [Float]
                if preprocessorChunkIndex == 0 {
                    validFeatures = Array(features.prefix(featureLength * config.melFeatures))
                } else {
                    let skipFrames = config.overlapFrames * config.melFeatures
                    validFeatures = Array(
                        features.dropFirst(skipFrames).prefix(
                            (featureLength - config.overlapFrames) * config.melFeatures))
                }

                featureBuffer.append(contentsOf: validFeatures)
                audioOffset += config.audioHopSamples
                preprocessorChunkIndex += 1
            }

            // Handle remaining audio with padding
            if audioOffset < totalSamples {
                var lastChunk = Array(samples[audioOffset...])
                let padCount = config.preprocessorAudioSamples - lastChunk.count
                lastChunk.append(contentsOf: [Float](repeating: 0.0, count: padCount))

                let (features, featureLength) = try models.runPreprocessor(
                    audioSamples: lastChunk,
                    config: config
                )

                let validFeatures: [Float]
                if preprocessorChunkIndex == 0 {
                    validFeatures = Array(features.prefix(featureLength * config.melFeatures))
                } else {
                    let skipFrames = config.overlapFrames * config.melFeatures
                    validFeatures = Array(
                        features.dropFirst(skipFrames).prefix(
                            (featureLength - config.overlapFrames) * config.melFeatures))
                }

                featureBuffer.append(contentsOf: validFeatures)
                preprocessorChunkIndex += 1
            }
        }

        let totalFeatureFrames = featureBuffer.count / config.melFeatures

        if config.debugMode {
            print(
                "[DEBUG] Phase 1 complete: \(featureBuffer.count) floats = \(totalFeatureFrames) mel frames from \(preprocessorChunkIndex) chunks"
            )
            // Print feature statistics
            let fMin = featureBuffer.min() ?? 0
            let fMax = featureBuffer.max() ?? 0
            let fMean = featureBuffer.reduce(0, +) / Float(featureBuffer.count)
            print("[DEBUG] Feature stats: min=\(fMin), max=\(fMax), mean=\(fMean)")
            fflush(stdout)
        }

        // PHASE 2: Run diarization on all features
        if config.debugMode {
            print("[DEBUG] Phase 2: Running diarization on \(totalFeatureFrames) mel frames")
            fflush(stdout)
        }

        var chunksProcessed = 0
        guard var state = state else {
            throw SortformerError.notInitialized
        }

        let chunkFrames = config.chunkFrames  // 112 mel frames for NVIDIA
        let coreFrames = config.chunkLen * config.subsamplingFactor  // 48 mel frames core
        let leftContextFrames = config.chunkLeftContext * config.subsamplingFactor  // 8 frames
        let rightContextFrames = config.chunkRightContext * config.subsamplingFactor  // 56 frames

        while true {
            // Python's streaming_feat_loader logic
            let sttFeat = diarizerChunkIndex * coreFrames
            let endFeat = min(sttFeat + coreFrames, totalFeatureFrames)

            if endFeat > totalFeatureFrames {
                break
            }

            let leftOffset = min(leftContextFrames, sttFeat)
            let rightOffset = min(rightContextFrames, totalFeatureFrames - endFeat)

            let chunkStart = sttFeat - leftOffset
            let chunkEnd = endFeat + rightOffset
            let actualChunkFrames = chunkEnd - chunkStart

            if actualChunkFrames <= 0 {
                break
            }

            // Extract features and pad if needed
            let featStart = chunkStart * config.melFeatures
            let featEnd = chunkEnd * config.melFeatures
            var chunkFeatures = Array(featureBuffer[featStart..<featEnd])

            // Pad to chunkFrames if needed
            if actualChunkFrames < chunkFrames {
                let padCount = (chunkFrames - actualChunkFrames) * config.melFeatures
                chunkFeatures.append(contentsOf: [Float](repeating: 0.0, count: padCount))
            }

            let transposedChunk = chunkFeatures

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
            // Pass the actual valid chunk length (like Python's feat_lengths), not the padded size.
            // Python computes: feat_lengths = clamp(feat_seq_length - stt_feat + left_offset, 0, chunk_shape)
            // For chunk 0: feat_lengths = clamp(104935 - 0 + 0, 0, 104) = 104
            let output = try models.runMainModel(
                chunk: transposedChunk,
                chunkLength: actualChunkFrames,
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
                    "[DEBUG] Model output: predictions=\(probabilities.count), embLength=\(embLength), actualChunkFrames=\(actualChunkFrames)"
                )
                // Check predictions at different offsets
                // Model input: padded_spkcache(188) + padded_fifo(188) + chunk(14) = 390 frames
                for testOffset in [0, 14, 188, 376] {
                    print("[DEBUG] Testing offset \(testOffset):")
                    for frame in 0..<min(3, (probabilities.count / 4) - testOffset) {
                        let idx = (testOffset + frame) * 4
                        if idx + 3 < probabilities.count {
                            let vals = (0..<4).map { String(format: "%.4f", probabilities[idx + $0]) }.joined(
                                separator: ", ")
                            print("[DEBUG]   Frame \(frame): [\(vals)]")
                        }
                    }
                }
                fflush(stdout)
            }

            // Compute left/right context for prediction extraction
            let leftContext = (leftOffset + config.subsamplingFactor / 2) / config.subsamplingFactor
            let rightContext = (rightOffset + config.subsamplingFactor - 1) / config.subsamplingFactor

            // Debug first 5 chunks - capture state BEFORE update
            let debugSpkcacheLen = state.spkcacheLength
            let debugFifoLen = state.fifoLength

            // Update state
            let chunkPreds = modules.streamingUpdate(
                state: &state,
                chunkEmbeddings: chunkEmbs,
                predictions: probabilities,
                leftContext: leftContext,
                rightContext: rightContext,
                modelSpkcacheLen: modelSpkcacheLen,
                modelFifoLen: modelFifoLen
            )

            // Debug first 5 chunks - format to match Python output for comparison
            if config.debugMode && diarizerChunkIndex < 5 {
                print(
                    "[Swift] Chunk \(diarizerChunkIndex): lc=\(leftContext), rc=\(rightContext), spkcache=\(debugSpkcacheLen), fifo=\(debugFifoLen)"
                )

                let actualFrames = chunkPreds.count / config.numSpeakers
                let chunkMin = chunkPreds.min() ?? 0
                let chunkMax = chunkPreds.max() ?? 0
                print(
                    "         chunk_probs shape: [\(actualFrames), \(config.numSpeakers)], min=\(String(format: "%.4f", chunkMin)), max=\(String(format: "%.4f", chunkMax))"
                )

                for frame in 0..<min(7, actualFrames) {
                    let frameStart = frame * config.numSpeakers
                    var vals: [String] = []
                    for spk in 0..<config.numSpeakers {
                        if frameStart + spk < chunkPreds.count {
                            vals.append(String(format: "%.4f", chunkPreds[frameStart + spk]))
                        }
                    }
                    print("         Frame \(frame): [\(vals.joined(separator: ", "))]")
                }
                print("")
                fflush(stdout)
            }

            // Accumulate results
            allProbabilities.append(contentsOf: chunkPreds)
            chunksProcessed += 1
            diarizerChunkIndex += 1

            // Progress callback
            // processedFrames is in mel frames (after subsampling)
            // Each mel frame corresponds to melStride samples
            let processedMelFrames = diarizerChunkIndex * coreFrames
            let progress = min(processedMelFrames * config.melStride, totalSamples)
            progressCallback?(progress, totalSamples, chunksProcessed)
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
}
