/// Hypothesis for TDT beam search decoding
/// Note: Not Sendable because TdtDecoderState contains CoreML MLMultiArray which isn't Sendable.
struct TdtHypothesis {
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

    // MARK: - Helper Properties and Methods

    /// Check if hypothesis has no tokens
    var isEmpty: Bool { ySequence.isEmpty }

    /// Get number of tokens in this hypothesis
    var tokenCount: Int { ySequence.count }

    /// Check if hypothesis has any tokens
    var hasTokens: Bool { !ySequence.isEmpty }

    /// Get the last token if it exists
    var computedLastToken: Int? { ySequence.last }

    /// Get the maximum timestamp value, or 0 if no timestamps
    var maxTimestamp: Int { timestamps.max() ?? 0 }

    /// Return tuple for backward compatibility with existing APIs
    var destructured: (tokens: [Int], timestamps: [Int], confidences: [Float]) {
        (ySequence, timestamps, tokenConfidences)
    }
}
