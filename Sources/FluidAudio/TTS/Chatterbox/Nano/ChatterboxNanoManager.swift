import Foundation

/// Public API for Chatterbox Nano synthesis (ResembleAI, 110M, English,
/// 24 kHz, paralinguistic tags).
///
/// Requires macOS 15 / iOS 18: the T3 decode step keeps its KV cache in
/// CoreML `MLState` buffers.
///
/// Per-call budgets are set by the static model shapes *minus the voice's
/// own footprint* (#924): with the built-in voice, text is capped at 135
/// BPE tokens (~500–550 characters) and generated audio at ≈9.9 s
/// (`.standard`) or ≈29.9 s (`.extended`, an extra ~280 MB download).
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
///
/// ```swift
/// let manager = ChatterboxNanoManager()          // or (outputCapacity: .extended)
/// try await manager.initialize()
/// let audio = try await manager.synthesize(
///     text: "Well that went better than expected [chuckle], see you tomorrow.")
/// ```
@available(macOS 15.0, iOS 18.0, *)
public actor ChatterboxNanoManager {

    private static let logger = AppLogger(category: "ChatterboxNanoManager")

    public struct Audio: Sendable {
        public let samples: [Float]
        public let sampleRate: Int
    }

    private var models: ChatterboxNanoModels?
    private let outputCapacity: ChatterboxNanoOutputCapacity

    /// - Parameter outputCapacity: which S3Gen bucket pair to load —
    ///   `.standard` (≈9.9 s of generated audio per call with the built-in
    ///   voice) or `.extended` (≈29.9 s; separate ~280 MB download, roughly
    ///   double the flow/vocoder latency per call).
    public init(outputCapacity: ChatterboxNanoOutputCapacity = .standard) {
        self.outputCapacity = outputCapacity
    }

    /// Download (if needed) and load the four CoreML models + tables + tokenizer.
    public func initialize(progressHandler: ProgressHandler? = nil) async throws {
        guard models == nil else { return }
        models = try await ChatterboxNanoModels.load(
            capacity: outputCapacity, progressHandler: progressHandler)
        Self.logger.info("Chatterbox Nano models ready (\(self.outputCapacity.rawValue) capacity)")
    }

    /// Synthesize English `text` with the built-in voice. Paralinguistic
    /// tags (`[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, …) are part of the
    /// vocabulary and can be embedded directly in the text.
    ///
    /// - Parameter seed: sampling seed; equal seeds reproduce equal audio
    public func synthesize(
        text: String,
        temperature: Float = ChatterboxNanoConstants.temperature,
        topK: Int = ChatterboxNanoConstants.topK,
        topP: Float = ChatterboxNanoConstants.topP,
        repetitionPenalty: Float = ChatterboxNanoConstants.repetitionPenalty,
        seed: UInt64 = UInt64.random(in: 0..<UInt64.max)
    ) async throws -> Audio {
        if models == nil { try await initialize() }
        guard let models else {
            throw ChatterboxError.processingFailed("models unavailable")
        }

        let synthesizer = ChatterboxNanoSynthesizer(models: models)
        let result = try await synthesizer.synthesize(
            text: text,
            temperature: temperature, topK: topK, topP: topP,
            repetitionPenalty: repetitionPenalty,
            seed: seed)

        let duration = Double(result.samples.count) / Double(ChatterboxNanoConstants.sampleRate)
        let msPerToken = 1000.0 * result.decodeSeconds / Double(max(result.decodedTokens, 1))
        Self.logger.info(
            "Synthesized \(String(format: "%.2f", duration))s "
                + "(\(result.speechTokens) tokens): prefill \(Int(result.prefillSeconds * 1000))ms, "
                + "decode \(String(format: "%.1f", msPerToken))ms/token, "
                + "flow \(Int(result.flowSeconds * 1000))ms, "
                + "vocoder \(Int(result.vocoderSeconds * 1000))ms")
        return Audio(samples: result.samples, sampleRate: ChatterboxNanoConstants.sampleRate)
    }
}
