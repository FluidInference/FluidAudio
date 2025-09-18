import Foundation
import OSLog

@available(macOS 13.0, *)
public final class TtSManager {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "TtSManager")
    private let outputSampleRate: Double = 24_000.0

    private var isInitialized = false

    public init() {}

    public var isAvailable: Bool {
        isInitialized && KokoroModel.isModelInitialized
    }

    public func initialize(models: TtsModels) async throws {
        logger.info("Initializing TtSManager with provided models")
        try await KokoroModel.ensureRequiredFiles()
        KokoroModel.useExternalModel(models.kokoro)
        isInitialized = true
        logger.info("TtSManager initialized successfully")
    }

    public func initialize() async throws {
        logger.info("Initializing TtSManager with Kokoro resources")
        try await KokoroModel.ensureRequiredFiles()
        if !KokoroModel.isModelInitialized {
            try await KokoroModel.loadModel()
        }
        isInitialized = true
        logger.info("TtSManager initialized successfully")
    }

    public func synthesize(
        text: String,
        voiceSpeed: Float = 1.0,
        speakerId: Int = 0
    ) async throws -> Data {
        guard voiceSpeed > 0 else {
            throw TTSError.processingFailed("voiceSpeed must be positive")
        }
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else {
            throw TTSError.processingFailed("Input text is empty")
        }
        guard isAvailable else {
            throw TTSError.modelNotFound("Kokoro model not initialized")
        }

        logger.info(
            "Synthesizing text: \"\(cleanText)\" speed=\(voiceSpeed) speaker=\(speakerId)"
        )

        let baseSamples = try await KokoroModel.synthesizeSamples(text: text)
        let adjustedSamples = adjustSamples(baseSamples, speed: voiceSpeed)
        let audioData = try AudioWAV.data(from: adjustedSamples, sampleRate: outputSampleRate)

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

    public func cleanup() {
        isInitialized = false
        logger.info("TtSManager cleaned up")
    }

    private func adjustSamples(_ samples: [Float], speed: Float) -> [Float] {
        guard !samples.isEmpty else { return samples }
        if abs(speed - 1.0) < 0.001 { return samples }

        let scale = Double(1.0 / speed)
        let newCount = max(1, Int(round(Double(samples.count) * scale)))
        if newCount == samples.count { return samples }

        var output = [Float](repeating: 0, count: newCount)
        let lastIndex = samples.count - 1
        for index in 0..<newCount {
            let sourcePosition = Double(index) / scale
            let lowerIndex = min(Int(floor(sourcePosition)), lastIndex)
            let upperIndex = min(lowerIndex + 1, lastIndex)
            let fraction = Float(sourcePosition - Double(lowerIndex))
            let lowerValue = samples[lowerIndex]
            let upperValue = samples[upperIndex]
            output[index] = lowerValue + (upperValue - lowerValue) * fraction
        }

        return output
    }
}
