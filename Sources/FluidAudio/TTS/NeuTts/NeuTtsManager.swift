import Foundation

/// Public API for NeuTTS-2E synthesis (emotional English TTS, 24 kHz).
///
/// Requires macOS 15 / iOS 18: the LM decode step keeps its KV cache in
/// CoreML `MLState` buffers.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
///
/// ```swift
/// let manager = NeuTtsManager()
/// try await manager.initialize()
/// let audio = try await manager.synthesize(
///     text: "I can't believe it's finally here!", speaker: "emily", emotion: "happy")
/// ```
@available(macOS 15.0, iOS 18.0, *)
public actor NeuTtsManager {

    private static let logger = AppLogger(category: "NeuTtsManager")

    public struct Audio: Sendable {
        public let samples: [Float]
        public let sampleRate: Int
    }

    private var models: NeuTtsModels?

    public init() {}

    /// Download (if needed) and load the three CoreML models + tokenizer.
    public func initialize(progressHandler: ProgressHandler? = nil) async throws {
        guard models == nil else { return }
        models = try await NeuTtsModels.load(progressHandler: progressHandler)
        Self.logger.info("NeuTTS-2E models ready")
    }

    /// Synthesize `text` with the given fixed speaker and emotion.
    ///
    /// - Parameters:
    ///   - speaker: one of `NeuTtsConstants.speakers`
    ///   - emotion: one of `NeuTtsConstants.emotions`
    ///   - seed: sampling seed; equal seeds reproduce equal audio
    public func synthesize(
        text: String,
        speaker: String = NeuTtsConstants.defaultSpeaker,
        emotion: String = NeuTtsConstants.defaultEmotion,
        temperature: Float = NeuTtsConstants.temperature,
        topK: Int = NeuTtsConstants.topK,
        seed: UInt64 = UInt64.random(in: 0..<UInt64.max)
    ) async throws -> Audio {
        if models == nil { try await initialize() }
        guard let models else {
            throw NeuTtsPrompt.PromptError.missingSpecialToken("models unavailable")
        }

        let synthesizer = NeuTtsSynthesizer(models: models)
        let result = try await synthesizer.synthesize(
            text: text, speaker: speaker, emotion: emotion,
            temperature: temperature, topK: topK, seed: seed)

        let duration = Double(result.samples.count) / Double(NeuTtsConstants.sampleRate)
        let msPerToken = 1000.0 * result.decodeSeconds / Double(max(result.decodedTokens, 1))
        Self.logger.info(
            "Synthesized \(String(format: "%.2f", duration))s "
                + "(\(result.codeCount) codes): prefill \(Int(result.prefillSeconds * 1000))ms, "
                + "decode \(String(format: "%.1f", msPerToken))ms/token, "
                + "codec \(Int(result.codecSeconds * 1000))ms")
        return Audio(samples: result.samples, sampleRate: NeuTtsConstants.sampleRate)
    }
}
