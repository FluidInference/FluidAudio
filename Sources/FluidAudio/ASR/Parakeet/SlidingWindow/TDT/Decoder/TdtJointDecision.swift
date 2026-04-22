/// Joint model decision for a single encoder/decoder step.
///
/// Represents the output of the TDT joint network which combines encoder and decoder features
/// to predict the next token, its probability, and how many audio frames to skip.
internal struct TdtJointDecision {
    /// Predicted token ID from vocabulary
    let token: Int

    /// Softmax probability for this token
    let probability: Float

    /// Duration bin index (maps to number of encoder frames to skip)
    let durationBin: Int
}
