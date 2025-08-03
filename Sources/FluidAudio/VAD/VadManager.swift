import Accelerate
import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public actor VadManager {

    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "VAD")
    private let config: VadConfig

    private var stftModel: MLModel?
    private var encoderModel: MLModel?
    private var rnnModel: MLModel?

    private var hState: MLMultiArray?
    private var cState: MLMultiArray?
    private var featureBuffer: [MLMultiArray] = []

    private var audioProcessor: VadAudioProcessor
    private let modelProcessor: VadModelProcessor

    public init(config: VadConfig = .default) {
        self.config = config
        self.audioProcessor = VadAudioProcessor(config: config)
        self.modelProcessor = VadModelProcessor(config: config)
    }

    public init(config: VadConfig = .default, stft: MLModel, encoder: MLModel, rnn: MLModel) {
        self.config = config
        self.audioProcessor = VadAudioProcessor(config: config)
        self.modelProcessor = VadModelProcessor(config: config)
        self.stftModel = stft
        self.encoderModel = encoder
        self.rnnModel = rnn
        logger.info("VadManager initialized with provided models")
    }

    public var isAvailable: Bool {
        return stftModel != nil && encoderModel != nil && rnnModel != nil
    }

    public func initialize(from directory: URL? = nil) async throws {
        let startTime = Date()
        logger.info("Initializing VAD system with CoreML")

        do {
            try await loadCoreMLModelsWithRecovery(from: directory)
        } catch {
            logger.error("Failed to initialize VAD: \(error)")
            throw error
        }

        resetState()

        let totalInitTime = Date().timeIntervalSince(startTime)
        logger.info(
            "VAD system initialized successfully in \(String(format: "%.2f", totalInitTime))s"
        )
    }

    private func loadCoreMLModelsWithRecovery(from directory: URL? = nil) async throws {
        let baseDirectory = directory ?? getDefaultBaseDirectory()

        let modelNames = [
            "silero_stft.mlmodelc",
            "silero_encoder.mlmodelc",
            "silero_rnn_decoder.mlmodelc",
        ]

        let models = try await DownloadUtils.loadModels(
            .vad,
            modelNames: modelNames,
            directory: baseDirectory,
            computeUnits: config.computeUnits
        )

        guard let stftModel = models["silero_stft.mlmodelc"],
            let encoderModel = models["silero_encoder.mlmodelc"],
            let rnnModel = models["silero_rnn_decoder.mlmodelc"]
        else {
            logger.error("Failed to load all required VAD models")
            throw VadError.modelLoadingFailed
        }

        self.stftModel = stftModel
        self.encoderModel = encoderModel
        self.rnnModel = rnnModel

        logger.info("VAD models loaded successfully")
    }

    private func getDefaultBaseDirectory() -> URL {
        if let customDirectory = config.modelCacheDirectory {
            return customDirectory
        }

        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent("FluidAudio", isDirectory: true)
    }

    public func resetState() {
        do {
            self.hState = try MLMultiArray(shape: [1, 1, 128], dataType: .float32)
            self.cState = try MLMultiArray(shape: [1, 1, 128], dataType: .float32)

            if let hState = self.hState, let cState = self.cState {
                for i in 0..<hState.count {
                    hState[i] = 0.0
                }
                for i in 0..<cState.count {
                    cState[i] = 0.0
                }
            }

            self.featureBuffer.removeAll()

        } catch {
            logger.error("Failed to reset CoreML VAD state: \(error.localizedDescription)")
        }

        audioProcessor.reset()

        if config.debugMode {
            logger.debug("VAD state reset successfully for CoreML")
        }
    }

    public func processChunk(_ audioChunk: [Float]) async throws -> VadResult {
        guard isAvailable else {
            throw VadError.notInitialized
        }

        let processingStartTime = Date()

        let rawProbability = try await processCoreMLChunk(audioChunk)

        let (smoothedProbability, snrValue, spectralFeatures) = audioProcessor.processRawProbability(
            rawProbability,
            audioChunk: audioChunk
        )

        let isVoiceActive = smoothedProbability >= config.threshold

        let processingTime = Date().timeIntervalSince(processingStartTime)

        if config.debugMode {
            let snrString = snrValue.map { String(format: "%.1f", $0) } ?? "N/A"
            let debugMessage =
                "VAD processing (CoreML): raw=\(String(format: "%.3f", rawProbability)),  smoothed=\(String(format: "%.3f", smoothedProbability)), threshold=\(String(format: "%.3f", config.threshold)), snr=\(snrString)dB, active=\(isVoiceActive), time=\(String(format: "%.3f", processingTime))s"
            logger.debug("\(debugMessage)")
        }

        return VadResult(
            probability: smoothedProbability,
            isVoiceActive: isVoiceActive,
            processingTime: processingTime,
            snrValue: snrValue,
            spectralFeatures: spectralFeatures
        )
    }

    private func processCoreMLChunk(_ audioChunk: [Float]) async throws -> Float {
        var processedChunk = audioChunk
        if processedChunk.count != config.chunkSize {
            if processedChunk.count < config.chunkSize {
                let paddingSize = config.chunkSize - processedChunk.count
                if config.debugMode {
                    logger.debug(
                        "Padding audio chunk with \(paddingSize) zeros (original: \(processedChunk.count) samples)")
                }
                processedChunk.append(contentsOf: Array(repeating: 0.0, count: paddingSize))
            } else {
                if config.debugMode {
                    logger.debug(
                        "Truncating audio chunk from \(processedChunk.count) to \(self.config.chunkSize) samples")
                }
                processedChunk = Array(processedChunk.prefix(config.chunkSize))
            }
        }

        guard let stftModel = self.stftModel,
            let encoderModel = self.encoderModel,
            let rnnModel = self.rnnModel
        else {
            throw VadError.notInitialized
        }

        let stftFeatures = try modelProcessor.processSTFT(processedChunk, stftModel: stftModel)

        manageTemporalContext(stftFeatures)

        let concatenatedFeatures = try modelProcessor.concatenateTemporalFeatures(featureBuffer)
        let encoderFeatures = try modelProcessor.processEncoder(concatenatedFeatures, encoderModel: encoderModel)
        let processedFeatures = try modelProcessor.prepareEncoderFeaturesForRNN(encoderFeatures)

        guard let hState = self.hState, let cState = self.cState else {
            logger.warning("RNN states not initialized, initializing to zero")
            resetState()
            guard let hState = self.hState, let cState = self.cState else {
                throw VadError.modelProcessingFailed("Failed to initialize RNN states")
            }
            let (rnnFeatures, newHState, newCState) = try modelProcessor.processRNN(
                processedFeatures,
                hState: hState,
                cState: cState,
                rnnModel: rnnModel
            )
            self.hState = newHState
            self.cState = newCState
            return audioProcessor.calculateVadProbability(from: rnnFeatures)
        }

        let (rnnFeatures, newHState, newCState) = try modelProcessor.processRNN(
            processedFeatures,
            hState: hState,
            cState: cState,
            rnnModel: rnnModel
        )

        self.hState = newHState
        self.cState = newCState

        return audioProcessor.calculateVadProbability(from: rnnFeatures)
    }

    private func manageTemporalContext(_ stftFeatures: MLMultiArray) {
        featureBuffer.append(stftFeatures)

        if featureBuffer.count > 4 {
            featureBuffer = Array(featureBuffer.suffix(4))
        }

        while featureBuffer.count < 4 {
            do {
                let zeroFeatures = try MLMultiArray(shape: stftFeatures.shape, dataType: .float32)
                featureBuffer.insert(zeroFeatures, at: 0)
            } catch {
                logger.error("Failed to create zero features for temporal context: \(error)")
                break
            }
        }
    }

    public func processAudioFile(_ audioData: [Float]) async throws -> [VadResult] {
        guard isAvailable else {
            throw VadError.notInitialized
        }

        resetState()

        var results: [VadResult] = []
        let chunkSize = config.chunkSize

        for chunkStart in stride(from: 0, to: audioData.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, audioData.count)
            let chunk = Array(audioData[chunkStart..<chunkEnd])

            let result = try await processChunk(chunk)
            results.append(result)
        }
        return results
    }

    public func cleanup() {
        stftModel = nil
        encoderModel = nil
        rnnModel = nil
        hState = nil
        cState = nil
        featureBuffer.removeAll()

        audioProcessor.reset()

        logger.info("VAD resources cleaned up")
    }
}
