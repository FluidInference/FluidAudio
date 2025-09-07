/// Hypothesis for TDT beam search decoding
struct TdtHypothesis: Sendable {
    var score: Float = 0.0
    var ySequence: [Int] = []
    var decState: TdtDecoderState?
    var timestamps: [Int] = []
    var tokenDurations: [Int] = []
    var tokenConfidences: [Float] = []
    /// Last non-blank token decoded in this hypothesis.
    /// Used to initialize the decoder for the next chunk, maintaining context across chunk boundaries.
    var lastToken: Int?

    /// Initialize with a decoder state
    init(decState: TdtDecoderState) {
        self.decState = decState
    }
}
