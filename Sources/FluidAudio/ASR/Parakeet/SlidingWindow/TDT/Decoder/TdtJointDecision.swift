/// Joint model decision for a single encoder/decoder step.
///
/// Represents the output of the TDT joint network which combines encoder and decoder features
/// to predict the next token, its probability, and how many audio frames to skip.
internal struct TdtJointDecision: Sendable {
    /// Predicted token ID from vocabulary
    let token: Int

    /// Full-vocab softmax probability for `token`, clamped to [0, 1].
    /// See `topKLogits` — the two scales are not directly comparable.
    let probability: Float

    /// Duration bin index (maps to number of encoder frames to skip)
    let durationBin: Int

    /// Top-K candidate token IDs (optional, only present in JointDecisionv3).
    /// Parallel to `topKLogits` — index `i` in both arrays refers to the same
    /// candidate. The init enforces equal lengths via `assert`.
    let topKIds: [Int]?

    /// Top-K candidate logits (optional, only present in JointDecisionv3).
    ///
    /// NOTE: These are **raw logits**, not probabilities. A softmax over these
    /// K values is not equal to `probability`: `probability` is the full-vocab
    /// argmax softmax (denominator ≈ vocab-size), whereas `softmax(topKLogits)`
    /// has denominator of only K terms and is systematically larger. Consumers
    /// that want a comparable probability should go through
    /// `TokenLanguageFilter.filterTopK`, which returns the top-K softmax
    /// explicitly.
    let topKLogits: [Float]?

    /// Explicit initializer with default `nil` for the top-K fields so callers
    /// that don't care about language-aware script filtering (non-v3 joint
    /// outputs, most tests) can omit them.
    ///
    /// Why not stored-property defaults (`let topKIds: [Int]? = nil`)? Swift
    /// excludes stored `let` properties with default values from the
    /// synthesized memberwise initializer — a `let` with a default is treated
    /// as a compile-time-initialized constant, not a parameter. That would
    /// force `TdtModelInference` (which does pass top-K data) to either mutate
    /// the fields post-init or write its own initializer. Keeping the explicit
    /// init keeps a single uniform construction path.
    init(
        token: Int,
        probability: Float,
        durationBin: Int,
        topKIds: [Int]? = nil,
        topKLogits: [Float]? = nil
    ) {
        // Debug-only invariant: the two arrays are parallel indexes into the
        // same top-K candidate list, so they must match in length. CoreML
        // output shapes guarantee this today; the assert catches schema drift.
        assert(
            topKIds?.count == topKLogits?.count,
            "topKIds and topKLogits must have the same length (got \(topKIds?.count ?? -1) vs \(topKLogits?.count ?? -1))"
        )
        self.token = token
        self.probability = probability
        self.durationBin = durationBin
        self.topKIds = topKIds
        self.topKLogits = topKLogits
    }
}
