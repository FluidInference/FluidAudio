import AVFoundation
import Accelerate
import CoreML
import Foundation
import OSLog

/// VAD Manager using the trained Silero VAD model
///
/// **Beta Status**: This VAD implementation is currently in beta.
/// While it performs well in testing environments,
/// it has not been extensively tested in production environments.
/// Use with caution in production applications.
///
@available(macOS 13.0, iOS 16.0, *)
public actor VadManager {

    private let logger = AppLogger(category: "VadManager")
    public let config: VadConfig
    private let audioConverter: AudioConverter = AudioConverter()

    /// Model expects exactly 512 samples (32ms at 16kHz)
    public static let chunkSize = 512
    public static let sampleRate = 16000

    private var vadModel: MLModel?
    private let audioProcessor: VadAudioProcessor

    // Reusable buffer to avoid allocations
    private var reuseBuffer: MLMultiArray?

    public var isAvailable: Bool {
        return vadModel != nil
    }

    // MARK: - Main processing API

    /// Process an entire audio source from a file URL.
    /// Automatically converts the audio to 16kHz mono Float32 and processes in 512-sample chunks.
    /// - Parameter url: Audio file URL
    /// - Returns: Array of per-chunk VAD results
    public func process(_ url: URL) async throws -> [VadResult] {
        let samples = try audioConverter.resampleAudioFile(url)
        return try await processAudioFile(samples)
    }

    /// Process an entire in-memory audio buffer.
    /// Automatically converts the buffer to 16kHz mono Float32 and processes in 512-sample chunks.
    /// - Parameter audioBuffer: Source buffer in any format
    /// - Returns: Array of per-chunk VAD results
    public func process(_ audioBuffer: AVAudioPCMBuffer) async throws -> [VadResult] {
        let samples = try audioConverter.resampleBuffer(audioBuffer)
        return try await processAudioFile(samples)
    }

    /// Process raw 16kHz mono samples.
    /// Processes audio in 512-sample chunks (32ms at 16kHz).
    /// - Parameter samples: Audio samples (must be 16kHz, mono)
    /// - Returns: Array of per-chunk VAD results
    public func process(_ samples: [Float]) async throws -> [VadResult] {
        return try await processAudioFile(samples)
    }

    /// Initialize with configuration
    public init(config: VadConfig = .default) async throws {
        self.config = config
        self.audioProcessor = VadAudioProcessor(config: config)

        let startTime = Date()

        // Load the unified model
        try await loadUnifiedModel()

        let totalInitTime = Date().timeIntervalSince(startTime)
        logger.info("VAD system initialized in \(String(format: "%.2f", totalInitTime))s")
    }

    /// Initialize with pre-loaded model
    public init(config: VadConfig = .default, vadModel: MLModel) {
        self.config = config
        self.audioProcessor = VadAudioProcessor(config: config)
        self.vadModel = vadModel
        logger.info("VAD initialized with provided model")
    }

    /// Initialize from directory
    public init(config: VadConfig = .default, modelDirectory: URL) async throws {
        self.config = config
        self.audioProcessor = VadAudioProcessor(config: config)

        let startTime = Date()
        try await loadUnifiedModel(from: modelDirectory)

        let totalInitTime = Date().timeIntervalSince(startTime)
        logger.info("VAD system initialized in \(String(format: "%.2f", totalInitTime))s")
    }

    private func loadUnifiedModel(from directory: URL? = nil) async throws {
        let baseDirectory = directory ?? getDefaultBaseDirectory()

        // Use DownloadUtils to load the model (handles downloading if needed)
        let models = try await DownloadUtils.loadModels(
            .vad,
            modelNames: Array(ModelNames.VAD.requiredModels),
            directory: baseDirectory.appendingPathComponent("Models"),
            computeUnits: config.computeUnits
        )

        // Get the VAD model
        guard let vadModel = models[ModelNames.VAD.sileroVadFile] else {
            logger.error("Failed to load VAD model from downloaded models")
            throw VadError.modelLoadingFailed
        }

        self.vadModel = vadModel
        logger.info("VAD model loaded successfully")
    }

    private func getDefaultBaseDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent("FluidAudio", isDirectory: true)
    }

    public func processChunk(_ audioChunk: [Float]) async throws -> VadResult {
        guard let vadModel = vadModel else {
            throw VadError.notInitialized
        }

        let processingStartTime = Date()

        // Ensure chunk is correct size
        var processedChunk = audioChunk
        if processedChunk.count != Self.chunkSize {
            if processedChunk.count < Self.chunkSize {
                let paddingSize = Self.chunkSize - processedChunk.count
                // Use repeat-last padding instead of zeros to avoid energy distortion
                let lastSample = processedChunk.last ?? 0.0
                processedChunk.append(contentsOf: Array(repeating: lastSample, count: paddingSize))
            } else {
                processedChunk = Array(processedChunk.prefix(Self.chunkSize))
            }
        }

        // No normalization - preserve original amplitude information for VAD

        // Process through unified model
        let rawProbability = try await processUnifiedModel(processedChunk, model: vadModel)

        // Apply audio processing (smoothing, SNR, etc.)
        let (smoothedProbability, _, _) = audioProcessor.processRawProbability(
            rawProbability,
            audioChunk: processedChunk
        )

        let isVoiceActive = smoothedProbability >= config.threshold
        let processingTime = Date().timeIntervalSince(processingStartTime)

        return VadResult(
            probability: smoothedProbability,
            isVoiceActive: isVoiceActive,
            processingTime: processingTime
        )
    }

    private func processUnifiedModel(_ audioChunk: [Float], model: MLModel) async throws -> Float {
        // Actor already provides thread safety, can run directly
        do {
            // Reuse buffer if available, create new one otherwise
            let audioArray: MLMultiArray
            if let buffer = reuseBuffer {
                audioArray = buffer
            } else {
                audioArray = try MLMultiArray(
                    shape: [1, Self.chunkSize] as [NSNumber],
                    dataType: .float32
                )
                reuseBuffer = audioArray
            }

            // Copy audio chunk to buffer
            for i: Int in 0..<audioChunk.count {
                audioArray[[0, i] as [NSNumber]] = NSNumber(value: audioChunk[i])
            }

            // Create input provider
            let input = try MLDictionaryFeatureProvider(dictionary: ["audio_chunk": audioArray])

            // Run prediction
            let output = try model.prediction(from: input)

            // Get probability output
            guard let vadProbability = output.featureValue(for: "vad_probability")?.multiArrayValue else {
                logger.error("No vad_probability output found")
                throw VadError.modelProcessingFailed("No VAD probability output")
            }

            // Extract probability value
            let probability = Float(truncating: vadProbability[0])
            return probability

        } catch {
            logger.error("Model processing failed: \(error)")
            throw VadError.modelProcessingFailed(error.localizedDescription)
        }
    }

    /// Process multiple chunks in batch for improved performance
    public func processBatch(_ audioChunks: [[Float]]) async throws -> [VadResult] {
        guard let vadModel = vadModel else {
            throw VadError.notInitialized
        }

        guard !audioChunks.isEmpty else {
            return []
        }

        let processingStartTime = Date()

        return try autoreleasepool {
            // Process all chunks in a single batch prediction
            let batchSize = audioChunks.count
            var batchInputs: [MLFeatureProvider] = []

            // Prepare batch inputs with optimized memory management
            for audioChunk in audioChunks {
                try autoreleasepool {
                    // Pad/truncate each chunk
                    var processedChunk = audioChunk
                    if processedChunk.count != Self.chunkSize {
                        if processedChunk.count < Self.chunkSize {
                            let paddingSize = Self.chunkSize - processedChunk.count
                            let lastSample = processedChunk.last ?? 0.0
                            processedChunk.append(contentsOf: Array(repeating: lastSample, count: paddingSize))
                        } else {
                            processedChunk = Array(processedChunk.prefix(Self.chunkSize))
                        }
                    }

                    // Use ANE-optimized array creation for better performance and memory efficiency
                    let audioArray = try ANEMemoryOptimizer.shared.createAlignedArray(
                        shape: [1, Self.chunkSize] as [NSNumber],
                        dataType: .float32
                    )

                    // Use optimized copy operation
                    ANEMemoryOptimizer.shared.optimizedCopy(
                        from: processedChunk,
                        to: audioArray,
                        offset: 0
                    )

                    batchInputs.append(try MLDictionaryFeatureProvider(dictionary: ["audio_chunk": audioArray]))
                }
            }

            // Create batch provider and run prediction
            let batchProvider = MLArrayBatchProvider(array: batchInputs)
            let batchOutput = try vadModel.predictions(from: batchProvider, options: MLPredictionOptions())

            // Process results
            var results: [VadResult] = []

            for i in 0..<batchSize {
                let output = batchOutput.features(at: i)

                guard let vadProbability = output.featureValue(for: "vad_probability")?.multiArrayValue else {
                    logger.error("No vad_probability output found for batch index \(i)")
                    throw VadError.modelProcessingFailed("No VAD probability output")
                }

                let rawProbability = Float(truncating: vadProbability[0])

                // Apply audio processing (smoothing, SNR, etc.) per chunk
                let (smoothedProbability, _, _) = audioProcessor.processRawProbability(
                    rawProbability,
                    audioChunk: audioChunks[i]
                )

                let isVoiceActive = smoothedProbability >= config.threshold

                results.append(
                    VadResult(
                        probability: smoothedProbability,
                        isVoiceActive: isVoiceActive,
                        processingTime: Date().timeIntervalSince(processingStartTime) / Double(batchSize)
                    ))
            }

            return results
        }
    }

    /// Process an entire audio file using adaptive batch processing for optimal performance
    private func processAudioFile(_ audioData: [Float]) async throws -> [VadResult] {
        // Split audio into chunks
        var audioChunks: [[Float]] = []
        for i in stride(from: 0, to: audioData.count, by: Self.chunkSize) {
            let endIndex = min(i + Self.chunkSize, audioData.count)
            let chunk = Array(audioData[i..<endIndex])
            audioChunks.append(chunk)
        }

        // Use smaller batch sizes to prevent ANE memory exhaustion
        // Process in batches of maximum 50 chunks to avoid memory issues
        let maxBatchSize = 50
        var allResults: [VadResult] = []

        for batchStart in stride(from: 0, to: audioChunks.count, by: maxBatchSize) {
            let batchEnd = min(batchStart + maxBatchSize, audioChunks.count)
            let batchChunks = Array(audioChunks[batchStart..<batchEnd])

            // Process batch with memory cleanup
            let batchResults = try await processBatchWithCleanup(batchChunks)
            allResults.append(contentsOf: batchResults)

            // Force cleanup between batches to prevent ANE memory buildup
            if batchEnd < audioChunks.count {
                ANEMemoryOptimizer.shared.clearBufferPool()
            }
        }

        return allResults
    }

    /// Process batch with automatic cleanup on failure
    private func processBatchWithCleanup(_ audioChunks: [[Float]]) async throws -> [VadResult] {
        do {
            return try await processBatch(audioChunks)
        } catch {
            // If batch processing fails (likely due to memory), fall back to individual processing
            logger.warning("Batch processing failed, falling back to individual chunk processing: \(error)")
            ANEMemoryOptimizer.shared.clearBufferPool()

            var results: [VadResult] = []
            for chunk in audioChunks {
                let result = try await processChunk(chunk)
                results.append(result)
            }
            return results
        }
    }

    /// Get current configuration
    public var currentConfig: VadConfig {
        return config
    }

    // MARK: - Performance Benchmarking

    /// Performance statistics for VAD processing
    public struct VadPerformanceStats: Sendable {
        public let totalProcessingTime: TimeInterval
        public let averageChunkTime: TimeInterval
        public let chunksProcessed: Int
        public let realTimeFactorX: Double  // How many times faster than real-time
        public let cacheStats: (hits: Int, misses: Int, hitRatio: Double)?

        public init(
            totalProcessingTime: TimeInterval,
            averageChunkTime: TimeInterval,
            chunksProcessed: Int,
            realTimeFactorX: Double,
            cacheStats: (hits: Int, misses: Int, hitRatio: Double)? = nil
        ) {
            self.totalProcessingTime = totalProcessingTime
            self.averageChunkTime = averageChunkTime
            self.chunksProcessed = chunksProcessed
            self.realTimeFactorX = realTimeFactorX
            self.cacheStats = cacheStats
        }
    }
}
