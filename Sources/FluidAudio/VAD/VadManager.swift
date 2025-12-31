import AVFoundation
import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

/// VAD Manager using the trained Silero VAD model
///
/// **Beta Status**: This VAD implementation is currently in beta.
/// While it performs well in testing environments,
/// it has not been extensively tested in production environments.
/// Use with caution in production applications.
///
public actor VadManager {

    private let logger = AppLogger(category: "VadManager")
    public let config: VadConfig
    private let audioConverter: AudioConverter = AudioConverter()
    private let memoryOptimizer: ANEMemoryOptimizer = ANEMemoryOptimizer()

    /// Model expects 4096 new samples (256ms at 16kHz) plus 64-sample context (total 4160)
    public static let chunkSize = 4096
    private static let contextSize = VadState.contextLength
    private static let stateSize = 128
    private static let modelInputSize = chunkSize + contextSize
    public static let sampleRate = 16000

    private var vadModel: MLModel?

    public var isAvailable: Bool {
        return vadModel != nil
    }

    // MARK: - Main processing API

    /// Process an entire audio source from a file URL.
    /// Automatically converts the audio to 16kHz mono Float32 and processes in 4096-sample chunks (256ms).
    /// ```swift
    /// let manager = try await VadManager()
    /// let results = try await manager.process(audioURL)
    /// ```
    /// - Parameter url: Audio file URL
    /// - Returns: Array of per-chunk VAD results
    public func process(_ url: URL) async throws -> [VadResult] {
        let samples = try audioConverter.resampleAudioFile(url)
        return try await processAudioSamples(samples)
    }

    /// Process an entire in-memory audio buffer.
    /// Automatically converts the buffer to 16kHz mono Float32 and processes in 4096-sample chunks (256ms).
    /// ```swift
    /// let buffer: AVAudioPCMBuffer = ...
    /// let manager = try await VadManager()
    /// let results = try await manager.process(buffer)
    /// ```
    /// - Parameter audioBuffer: Source buffer in any format
    /// - Returns: Array of per-chunk VAD results
    public func process(_ audioBuffer: AVAudioPCMBuffer) async throws -> [VadResult] {
        let samples = try audioConverter.resampleBuffer(audioBuffer)
        return try await processAudioSamples(samples)
    }

    /// Process raw 16kHz mono samples.
    /// Processes audio in 4096-sample chunks (256ms at 16kHz).
    /// ```swift
    /// let samples = try AudioConverter().resampleAudioFile(audioURL)
    /// let manager = try await VadManager()
    /// let results = try await manager.process(samples)
    /// let rtfx = (Double(samples.count) / Double(VadManager.sampleRate)) /
    ///     results.reduce(0) { $0 + $1.processingTime }
    /// ```
    /// - Parameter samples: Audio samples (must be 16kHz, mono)
    /// - Returns: Array of per-chunk VAD results
    public func process(_ samples: [Float]) async throws -> [VadResult] {
        return try await processAudioSamples(samples)
    }

    /// Initialize with configuration
    public init(config: VadConfig = .default) async throws {
        self.config = config

        let startTime = Date()

        // Load the unified model
        try await loadUnifiedModel()

        let totalInitTime = Date().timeIntervalSince(startTime)
        logger.info("VAD system initialized in \(String(format: "%.2f", totalInitTime))s")
    }

    /// Internal initializer for logic-only use (e.g., unit tests) that avoids model loading.
    /// This allows calling segmentation helpers without performing any model I/O.
    internal init(skipModelLoading: Bool, config: VadConfig = .default) {
        self.config = config
        self.vadModel = nil
        logger.info("VAD initialized in logic-only mode (no model loaded)")
    }

    /// Initialize with pre-loaded model
    public init(config: VadConfig = .default, vadModel: MLModel) {
        self.config = config
        self.vadModel = vadModel
        logger.info("VAD initialized with provided model")
    }

    /// Initialize from directory
    public init(config: VadConfig = .default, modelDirectory: URL) async throws {
        self.config = config

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

    /// Check if audio chunk is completely silent (all zeros or below threshold)
    private func isSilentAudio(_ audioChunk: [Float]) -> Bool {
        let silenceThreshold: Float = 1e-10
        return audioChunk.allSatisfy { abs($0) <= silenceThreshold }
    }

    internal func processChunk(_ audioChunk: [Float], inputState: VadState? = nil) async throws -> VadResult {
        guard let loadedModel = vadModel else {
            throw VadError.notInitialized
        }

        let processingStartTime = Date()

        // Use input state or create initial state
        let currentState = inputState ?? VadState.initial()
        // Ensure chunk is correct size (4096 samples of new audio)
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

        let nextContext = Array(processedChunk.suffix(Self.contextSize))
        // No normalization - preserve original amplitude information for VAD

        // Process through unified model
        let (rawProbability, newHiddenState, newCellState) = try await processUnifiedModel(
            processedChunk,
            inputState: currentState,
            model: loadedModel
        )
        let outputState = VadState(
            hiddenState: newHiddenState,
            cellState: newCellState,
            context: nextContext
        )
        let processingTime = Date().timeIntervalSince(processingStartTime)

        return VadResult(
            probability: rawProbability,
            isVoiceActive: rawProbability >= config.defaultThreshold,
            processingTime: processingTime,
            outputState: outputState
        )
    }

    private func processUnifiedModel(
        _ audioChunk: [Float],
        inputState: VadState,
        model: MLModel
    ) async throws -> (Float, [Float], [Float]) {
        do {
            let result: (Float, [Float], [Float]) = try autoreleasepool {
                // Reuse ANE-aligned buffers to avoid surface churn between invocations
                let audioArray = try memoryOptimizer.getPooledBuffer(
                    key: "vad_audio_input",
                    shape: [1, NSNumber(value: Self.modelInputSize)],
                    dataType: .float32
                )
                let hiddenStateArray = try memoryOptimizer.getPooledBuffer(
                    key: "vad_hidden_state",
                    shape: [1, NSNumber(value: Self.stateSize)],
                    dataType: .float32
                )
                let cellStateArray = try memoryOptimizer.getPooledBuffer(
                    key: "vad_cell_state",
                    shape: [1, NSNumber(value: Self.stateSize)],
                    dataType: .float32
                )

                // Clear and populate audio input (context followed by current chunk)
                let audioPointer = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
                vDSP_vclr(audioPointer, 1, vDSP_Length(Self.modelInputSize))
                memoryOptimizer.optimizedCopy(
                    from: inputState.context.prefix(Self.contextSize),
                    to: audioArray
                )
                memoryOptimizer.optimizedCopy(
                    from: audioChunk.prefix(Self.chunkSize),
                    to: audioArray,
                    offset: Self.contextSize
                )

                // Clear and populate recurrent state inputs
                let hiddenPointer = hiddenStateArray.dataPointer.assumingMemoryBound(to: Float.self)
                vDSP_vclr(hiddenPointer, 1, vDSP_Length(Self.stateSize))
                memoryOptimizer.optimizedCopy(
                    from: inputState.hiddenState.prefix(Self.stateSize),
                    to: hiddenStateArray
                )

                let cellPointer = cellStateArray.dataPointer.assumingMemoryBound(to: Float.self)
                vDSP_vclr(cellPointer, 1, vDSP_Length(Self.stateSize))
                memoryOptimizer.optimizedCopy(
                    from: inputState.cellState.prefix(Self.stateSize),
                    to: cellStateArray
                )

                // Create input provider with all required inputs
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "audio_input": audioArray,
                    "hidden_state": hiddenStateArray,
                    "cell_state": cellStateArray,
                ])

                // Run prediction
                let output = try model.prediction(from: input)
                // Extract outputs using flexible name matching (model outputs may include suffixes)
                guard
                    let vadOutputArray = featureValue(
                        in: output,
                        matchingSubstrings: ["vad_output"]
                    )
                else {
                    logger.error("No vad output found")
                    throw VadError.modelProcessingFailed("No VAD output")
                }

                guard
                    let newHiddenStateArray = featureValue(
                        in: output,
                        matchingSubstrings: ["new_hidden_state"]
                    )
                else {
                    logger.error("No new hidden state output found")
                    throw VadError.modelProcessingFailed("No new hidden state output")
                }

                guard
                    let newCellStateArray = featureValue(
                        in: output,
                        matchingSubstrings: ["new_cell_state"]
                    )
                else {
                    logger.error("No new cell state output found")
                    throw VadError.modelProcessingFailed("No new cell state output")
                }

                // Extract probability value (flatten the 1x1x1 output)
                let vadOutputPointer = vadOutputArray.dataPointer.assumingMemoryBound(to: Float.self)
                let probability = vadOutputPointer[0]

                // Convert output states back to arrays
                let newHiddenPointer = newHiddenStateArray.dataPointer.assumingMemoryBound(to: Float.self)
                let newHiddenState = Array(
                    UnsafeBufferPointer(
                        start: newHiddenPointer,
                        count: Self.stateSize
                    )
                )

                let newCellPointer = newCellStateArray.dataPointer.assumingMemoryBound(to: Float.self)
                let newCellState = Array(
                    UnsafeBufferPointer(
                        start: newCellPointer,
                        count: Self.stateSize
                    )
                )
                return (probability, newHiddenState, newCellState)
            }

            return result

        } catch {
            logger.error("Model processing failed: \(error)")
            throw VadError.modelProcessingFailed(error.localizedDescription)
        }
    }

    private func featureValue(
        in provider: MLFeatureProvider,
        matchingSubstrings substrings: [String]
    ) -> MLMultiArray? {
        for name in substrings {
            if let value = provider.featureValue(for: name)?.multiArrayValue {
                return value
            }
        }
        for substring in substrings {
            let target = substring.lowercased()
            if let name = provider.featureNames.first(where: { $0.lowercased().contains(target) }) {
                if let value = provider.featureValue(for: name)?.multiArrayValue {
                    return value
                }
            }
        }
        return nil
    }

    /// Process audio samples using adaptive batch processing for optimal performance
    internal func processAudioSamples(_ audioData: [Float]) async throws -> [VadResult] {

        // Split audio into chunks of chunkSize (4096 samples) as model is optimized for this size
        var audioChunks: [[Float]] = []
        for i in stride(from: 0, to: audioData.count, by: Self.chunkSize) {
            let endIndex = min(i + Self.chunkSize, audioData.count)
            let chunk = Array(audioData[i..<endIndex])
            audioChunks.append(chunk)
        }

        guard !audioChunks.isEmpty else {
            return []
        }

        var results: [VadResult] = []
        var currentState = VadState.initial()

        for (chunk) in audioChunks {
            let result = try await processChunk(chunk, inputState: currentState)
            results.append(result)
            currentState = result.outputState
        }

        return results
    }

    /// Get current configuration
    public var currentConfig: VadConfig {
        return config
    }
}
