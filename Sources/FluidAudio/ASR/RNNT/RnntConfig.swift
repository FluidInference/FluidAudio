/// Configuration for RNNT (Recurrent Neural Network Transducer) decoding
/// Used with Parakeet Realtime EOU 120M model
public struct RnntConfig: Sendable {
    /// Maximum symbols to emit per encoder frame before forcing advancement
    public let maxSymbolsPerStep: Int

    /// Blank token ID (silence/no-emission marker)
    public let blankId: Int

    /// End-of-utterance token ID
    public let eouTokenId: Int

    /// End-of-block token ID
    public let eobTokenId: Int

    /// Vocabulary size (excluding blank)
    public let vocabSize: Int

    /// Maximum tokens to process per chunk (prevents runaway decoding)
    public let maxTokensPerChunk: Int

    /// Consecutive blank limit before terminating decoding
    public let consecutiveBlankLimit: Int

    /// Encoder hidden size for this model
    public let encoderHiddenSize: Int

    /// Decoder hidden size for this model
    public let decoderHiddenSize: Int

    /// Default configuration for Parakeet Realtime EOU 120M
    public static let parakeetEOU = RnntConfig(
        maxSymbolsPerStep: 10,
        blankId: ModelNames.ASREOU.blankId,
        eouTokenId: ModelNames.ASREOU.eouTokenId,
        eobTokenId: ModelNames.ASREOU.eobTokenId,
        vocabSize: ModelNames.ASREOU.vocabSize,
        maxTokensPerChunk: 150,
        consecutiveBlankLimit: 5,
        encoderHiddenSize: 512,
        decoderHiddenSize: 640
    )

    public init(
        maxSymbolsPerStep: Int = 10,
        blankId: Int = 1026,
        eouTokenId: Int = 1024,
        eobTokenId: Int = 1025,
        vocabSize: Int = 1026,
        maxTokensPerChunk: Int = 150,
        consecutiveBlankLimit: Int = 5,
        encoderHiddenSize: Int = 512,
        decoderHiddenSize: Int = 640
    ) {
        self.maxSymbolsPerStep = maxSymbolsPerStep
        self.blankId = blankId
        self.eouTokenId = eouTokenId
        self.eobTokenId = eobTokenId
        self.vocabSize = vocabSize
        self.maxTokensPerChunk = maxTokensPerChunk
        self.consecutiveBlankLimit = consecutiveBlankLimit
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
    }
}
