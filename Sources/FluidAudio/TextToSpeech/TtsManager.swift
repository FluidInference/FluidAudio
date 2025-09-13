import AVFoundation
import CoreML
import Foundation
import OSLog

@available(macOS 13.0, *)
public final class TtSManager {

    private let logger = Logger(subsystem: "com.fluidaudio.tts", category: "TtSManager")

    private var kokoroModel: MLModel?
    private var ttsModels: TtsModels?

    private lazy var predictionOptions: MLPredictionOptions = {
        TtsModels.optimizedPredictionOptions()
    }()

    public init() {}

    public var isAvailable: Bool {
        kokoroModel != nil
    }

    public func initialize(models: TtsModels) async throws {
        logger.info("Initializing TtSManager with provided models")

        self.ttsModels = models
        self.kokoroModel = models.kokoro

        logger.info("TtSManager initialized successfully")
    }

    public func initialize() async throws {
        logger.info("Initializing TtSManager with downloaded models")

        let models = try await TtsModels.download()
        try await initialize(models: models)
    }

    public func synthesize(
        text: String,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0
    ) async throws -> Data {
        guard let model = kokoroModel else {
            throw TTSError.modelNotFound("Kokoro model not initialized")
        }

        logger.info("Synthesizing text: \"\(text)\" with speed: \(voiceSpeed)")

        let phonemeIds = try tokenizeText(text)
        let input = try createModelInput(
            phonemeIds: phonemeIds,
            voiceSpeed: voiceSpeed,
            speakerId: speakerId
        )

        let output = try await Task {
            try model.prediction(from: input, options: predictionOptions)
        }.value

        guard let audioArray = output.featureValue(for: "audio")?.multiArrayValue else {
            throw TTSError.processingFailed("Failed to get audio output from model")
        }

        let audioData = try processAudioOutput(audioArray)

        logger.info("Successfully synthesized \(audioData.count) bytes of audio")
        return audioData
    }

    public func synthesizeToFile(
        text: String,
        outputURL: URL,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0
    ) async throws {
        let audioData = try await synthesize(
            text: text,
            voiceSpeed: voiceSpeed,
            speakerId: speakerId
        )

        try audioData.write(to: outputURL)
        logger.info("Saved synthesized audio to: \(outputURL.path)")
    }

    private func tokenizeText(_ text: String) throws -> [Int32] {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !cleanText.isEmpty else {
            throw TTSError.processingFailed("Input text is empty")
        }

        let basicTokens = cleanText.unicodeScalars.map { scalar in
            Int32(scalar.value)
        }

        return basicTokens
    }

    private func createModelInput(
        phonemeIds: [Int32],
        voiceSpeed: Float,
        speakerId: Int
    ) throws -> MLFeatureProvider {
        let phonemeShape = [1, phonemeIds.count] as [NSNumber]
        guard let phonemeArray = try? MLMultiArray(shape: phonemeShape, dataType: .int32) else {
            throw TTSError.processingFailed("Failed to create phoneme array")
        }

        for (index, id) in phonemeIds.enumerated() {
            phonemeArray[index] = NSNumber(value: id)
        }

        let speedShape = [1] as [NSNumber]
        guard let speedArray = try? MLMultiArray(shape: speedShape, dataType: .float32) else {
            throw TTSError.processingFailed("Failed to create speed array")
        }
        speedArray[0] = NSNumber(value: voiceSpeed)

        let speakerShape = [1] as [NSNumber]
        guard let speakerArray = try? MLMultiArray(shape: speakerShape, dataType: .int32) else {
            throw TTSError.processingFailed("Failed to create speaker array")
        }
        speakerArray[0] = NSNumber(value: speakerId)

        var features: [String: MLFeatureValue] = [:]
        features["text"] = MLFeatureValue(multiArray: phonemeArray)
        features["speed"] = MLFeatureValue(multiArray: speedArray)
        features["speaker_id"] = MLFeatureValue(multiArray: speakerArray)

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private func processAudioOutput(_ audioArray: MLMultiArray) throws -> Data {
        let sampleRate: Double = 24000
        let numSamples = audioArray.shape[1].intValue

        guard numSamples > 0 else {
            throw TTSError.processingFailed("No audio samples generated")
        }

        let audioFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!

        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: audioFormat,
                frameCapacity: AVAudioFrameCount(numSamples)
            )
        else {
            throw TTSError.processingFailed("Failed to create audio buffer")
        }

        buffer.frameLength = AVAudioFrameCount(numSamples)

        let channelData = buffer.floatChannelData![0]
        let dataPointer = audioArray.dataPointer.bindMemory(
            to: Float.self,
            capacity: numSamples
        )

        for i in 0..<numSamples {
            channelData[i] = dataPointer[i]
        }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]

        guard let format = AVAudioFormat(settings: settings) else {
            throw TTSError.processingFailed("Failed to create audio format")
        }

        let converter = AVAudioConverter(from: audioFormat, to: format)!

        guard
            let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: buffer.frameLength
            )
        else {
            throw TTSError.processingFailed("Failed to create output buffer")
        }

        try converter.convert(to: outputBuffer, from: buffer)

        return Data(
            bytes: outputBuffer.floatChannelData![0],
            count: Int(outputBuffer.frameLength) * MemoryLayout<Float>.size
        )
    }

    public func cleanup() {
        kokoroModel = nil
        ttsModels = nil
        logger.info("TtSManager cleaned up")
    }
}
