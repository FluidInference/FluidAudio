import Foundation

/// Public API for Chatterbox Multilingual synthesis (ResembleAI, 24 kHz).
///
/// Requires macOS 15 / iOS 18: the T3 decode step keeps its KV cache in
/// CoreML `MLState` buffers.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
///
/// ```swift
/// let manager = ChatterboxManager()
/// try await manager.initialize()
/// let audio = try await manager.synthesize(
///     text: "Der schnelle braune Fuchs springt über den faulen Hund.",
///     language: "de")
/// ```
@available(macOS 15.0, iOS 18.0, *)
public actor ChatterboxManager {

    private static let logger = AppLogger(category: "ChatterboxManager")

    public struct Audio: Sendable {
        public let samples: [Float]
        public let sampleRate: Int
    }

    private var models: ChatterboxModels?

    public init() {}

    /// Download (if needed) and load the four CoreML models + tables + tokenizer.
    public func initialize(progressHandler: ProgressHandler? = nil) async throws {
        guard models == nil else { return }
        models = try await ChatterboxModels.load(progressHandler: progressHandler)
        Self.logger.info("Chatterbox Multilingual models ready")
    }

    /// Synthesize `text` in the given language with the built-in voice.
    ///
    /// - Parameters:
    ///   - language: lowercase ISO code; see
    ///     `ChatterboxConstants.supportedLanguages`
    ///   - seed: sampling seed; equal seeds reproduce equal audio
    public func synthesize(
        text: String,
        language: String = ChatterboxConstants.defaultLanguage,
        cfgWeight: Float = ChatterboxConstants.cfgWeight,
        temperature: Float = ChatterboxConstants.temperature,
        repetitionPenalty: Float = ChatterboxConstants.repetitionPenalty,
        minP: Float = ChatterboxConstants.minP,
        topP: Float = ChatterboxConstants.topP,
        seed: UInt64 = UInt64.random(in: 0..<UInt64.max)
    ) async throws -> Audio {
        if models == nil { try await initialize() }
        guard let models else {
            throw ChatterboxError.processingFailed("models unavailable")
        }

        let synthesizer = ChatterboxSynthesizer(models: models)
        let result = try await synthesizer.synthesize(
            text: text, language: language,
            cfgWeight: cfgWeight, temperature: temperature,
            repetitionPenalty: repetitionPenalty, minP: minP, topP: topP,
            seed: seed)

        let duration = Double(result.samples.count) / Double(ChatterboxConstants.sampleRate)
        let msPerToken = 1000.0 * result.decodeSeconds / Double(max(result.decodedTokens, 1))
        Self.logger.info(
            "Synthesized \(String(format: "%.2f", duration))s "
                + "(\(result.speechTokens) tokens): prefill \(Int(result.prefillSeconds * 1000))ms, "
                + "decode \(String(format: "%.1f", msPerToken))ms/token, "
                + "flow \(Int(result.flowSeconds * 1000))ms, "
                + "vocoder \(Int(result.vocoderSeconds * 1000))ms")
        return Audio(samples: result.samples, sampleRate: ChatterboxConstants.sampleRate)
    }
}
