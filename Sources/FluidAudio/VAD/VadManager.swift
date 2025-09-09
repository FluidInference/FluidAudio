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
@available(macOS 13.0, iOS 16.0, *)
public actor VadManager {

    private let logger = AppLogger(category: "VadManager")
    public let config: VadConfig

    /// Required model names for VAD
    public static let requiredModelNames: Set<String> = ["silero_vad.mlmodelc"]
    private var vadModel: MLModel?
    private let audioProcessor: VadAudioProcessor

    public var isAvailable: Bool {
        return vadModel != nil
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
            modelNames: ["silero_vad.mlmodelc"],
            directory: baseDirectory.appendingPathComponent("Models"),
            computeUnits: config.computeUnits
        )

        // Get the VAD model
        guard let vadModel = models["silero_vad.mlmodelc"] else {
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
        let chunkSize = 512  // Fixed chunk size
        if processedChunk.count != chunkSize {
            if processedChunk.count < chunkSize {
                let paddingSize = chunkSize - processedChunk.count
                if config.debugMode {
                    logger.debug("Padding audio chunk with \(paddingSize) zeros")
                }
                processedChunk.append(contentsOf: Array(repeating: 0.0, count: paddingSize))
            } else {
                if config.debugMode {
                    logger.debug("Truncating audio chunk from \(processedChunk.count) to 512")
                }
                processedChunk = Array(processedChunk.prefix(512))
            }
        }

        // Normalize audio to [-1, 1] range (matching librosa.load behavior used in training)
        // Find the maximum absolute value in the chunk
        let maxAbsValue = processedChunk.map { abs($0) }.max() ?? 1.0

        // Only normalize if there's actual audio content (not silence)
        if maxAbsValue > 0.0001 {
            // Scale to [-1, 1] range
            processedChunk = processedChunk.map { $0 / maxAbsValue }

            if config.debugMode {
                logger.debug("Normalized audio chunk: max amplitude was \(maxAbsValue), scaled to [-1, 1]")
            }
        }

        // Process through unified model
        let rawProbability = try await processUnifiedModel(processedChunk, model: vadModel)

        // Apply audio processing (smoothing, SNR, etc.)
        let (smoothedProbability, snrValue, _) = audioProcessor.processRawProbability(
            rawProbability,
            audioChunk: processedChunk
        )

        let isVoiceActive = smoothedProbability >= config.threshold
        let processingTime = Date().timeIntervalSince(processingStartTime)

        if config.debugMode {
            let snrString = snrValue.map { String(format: "%.1f", $0) } ?? "N/A"
            let rawStr = String(format: "%.3f", rawProbability)
            let smoothStr = String(format: "%.3f", smoothedProbability)
            let threshStr = String(format: "%.3f", config.threshold)
            let timeStr = String(format: "%.3f", processingTime)

            logger.debug(
                "VAD: raw=\(rawStr), smoothed=\(smoothStr), threshold=\(threshStr), SNR=\(snrString)dB, active=\(isVoiceActive), time=\(timeStr)s"
            )
        }

        return VadResult(
            probability: smoothedProbability,
            isVoiceActive: isVoiceActive,
            processingTime: processingTime
        )
    }

    private func processUnifiedModel(_ audioChunk: [Float], model: MLModel) async throws -> Float {
        // Actor already provides thread safety, can run directly
        do {
            // Create input array
            let audioArray = try MLMultiArray(
                shape: [1, 512],
                dataType: .float32
            )

            for i in 0..<audioChunk.count {
                audioArray[i] = NSNumber(value: audioChunk[i])
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

    /// Process multiple chunks in batch
    public func processBatch(_ audioChunks: [[Float]]) async throws -> [VadResult] {
        var results: [VadResult] = []

        for chunk in audioChunks {
            let result = try await processChunk(chunk)
            results.append(result)
        }

        return results
    }

    /// Process an entire audio file
    public func processAudioFile(_ audioData: [Float]) async throws -> [VadResult] {
        var results: [VadResult] = []
        let chunkSize = 512

        // Process in chunks
        for i in stride(from: 0, to: audioData.count, by: chunkSize) {
            let endIndex = min(i + chunkSize, audioData.count)
            let chunk = Array(audioData[i..<endIndex])

            let result = try await processChunk(chunk)
            results.append(result)
        }

        return results
    }

    /// Get current configuration
    public var currentConfig: VadConfig {
        return config
    }
}
