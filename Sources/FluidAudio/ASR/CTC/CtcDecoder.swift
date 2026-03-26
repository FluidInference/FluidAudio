import CoreML
import Foundation

// MARK: - CTC Greedy Decode

/// Greedy CTC decode: argmax per timestep, collapse repeated tokens, remove blanks.
///
/// Works with `[[Float]]` log-probabilities (the format produced by `CtcKeywordSpotter`).
///
/// - Parameters:
///   - logProbs: Per-frame log-probabilities, shape [T][V].
///   - vocabulary: Token vocabulary mapping token ID → string.
///   - blankId: CTC blank token index (default 1024).
/// - Returns: Decoded text string with SentencePiece markers replaced by spaces.
public func ctcGreedyDecode(
    logProbs: [[Float]],
    vocabulary: [Int: String],
    blankId: Int = 1024
) -> String {
    var ids: [Int] = []
    var prev = -1
    for frame in logProbs {
        guard !frame.isEmpty else { continue }
        var bestIdx = 0
        var bestVal = frame[0]
        for v in 1..<frame.count {
            if frame[v] > bestVal {
                bestVal = frame[v]
                bestIdx = v
            }
        }
        if bestIdx != blankId && bestIdx != prev { ids.append(bestIdx) }
        prev = bestIdx
    }
    return decodeCtcTokenIds(ids, vocabulary: vocabulary)
}

/// Greedy CTC decode from an MLMultiArray of shape [1, T, V].
///
/// - Parameters:
///   - logProbs: MLMultiArray of shape [1, T, V] containing log-probabilities.
///   - vocabulary: Token vocabulary mapping token ID → string.
///   - blankId: CTC blank token index (default 1024).
/// - Returns: Decoded text string.
public func ctcGreedyDecode(
    logProbs: MLMultiArray,
    vocabulary: [Int: String],
    blankId: Int = 1024
) -> String {
    let timeSteps = logProbs.shape[1].intValue
    let vocabSize = logProbs.shape[2].intValue
    let ptr = logProbs.dataPointer.assumingMemoryBound(to: Float32.self)
    var ids: [Int] = []
    var prev = -1
    for t in 0..<timeSteps {
        let base = t * vocabSize
        var bestVal: Float = -.infinity
        var bestIdx = 0
        for v in 0..<vocabSize {
            let x = ptr[base + v]
            if x > bestVal {
                bestVal = x
                bestIdx = v
            }
        }
        if bestIdx != blankId && bestIdx != prev { ids.append(bestIdx) }
        prev = bestIdx
    }
    return decodeCtcTokenIds(ids, vocabulary: vocabulary)
}

// MARK: - CTC Beam Search

/// A single hypothesis in the CTC beam search.
public struct CtcBeam {
    public var prefix: [Int]
    public var pBlank: Float
    public var pNonBlank: Float
    public var lmScore: Float
    public var wordPieces: [String]
    public var prevWord: String?

    public var totalAcoustic: Float { logAddExp(pBlank, pNonBlank) }
    public var total: Float { totalAcoustic + lmScore }
    public var lastToken: Int? { prefix.last }

    public init(
        prefix: [Int], pBlank: Float, pNonBlank: Float,
        lmScore: Float, wordPieces: [String], prevWord: String?
    ) {
        self.prefix = prefix
        self.pBlank = pBlank
        self.pNonBlank = pNonBlank
        self.lmScore = lmScore
        self.wordPieces = wordPieces
        self.prevWord = prevWord
    }
}

/// CTC prefix beam search with optional ARPA language model rescoring.
///
/// Uses corrected repeat-token handling (Graves 2006 fix):
/// - Non-blank path continues: `p_nb(l) += p_nb(l) * P(c)`
/// - Blank path creates new token: `p_nb(l+c) += p_b(l) * P(c)`
///
/// Word-level LM scores are applied at SentencePiece word boundaries (`▁` prefix).
///
/// - Parameters:
///   - logProbs: Per-frame log-probabilities, shape [T][V].
///   - vocabulary: Token vocabulary mapping token ID → string.
///   - lm: Optional ARPA language model for rescoring.
///   - beamWidth: Number of hypotheses to maintain (default 100).
///   - lmWeight: LM score scaling factor (alpha, default 0.3).
///   - wordBonus: Per-word bonus in nats (beta, default 0.0).
///   - blankId: CTC blank token index (default 1024).
///   - tokenCandidates: Number of top tokens to consider per frame (default 40).
/// - Returns: Decoded text string.
public func ctcBeamSearch(
    logProbs: [[Float]],
    vocabulary: [Int: String],
    lm: ARPALanguageModel? = nil,
    beamWidth: Int = 100,
    lmWeight: Float = 0.3,
    wordBonus: Float = 0.0,
    blankId: Int = 1024,
    tokenCandidates: Int = 40
) -> String {
    guard !logProbs.isEmpty else { return "" }
    let vocabSize = logProbs[0].count
    guard vocabSize > 0 else { return "" }

    var beams: [[Int]: CtcBeam] = [
        []: CtcBeam(
            prefix: [], pBlank: 0.0, pNonBlank: -.infinity,
            lmScore: 0.0, wordPieces: [], prevWord: nil)
    ]

    for frame in logProbs {
        let blankLp = (blankId >= 0 && blankId < frame.count) ? frame[blankId] : -.infinity

        // Find top token candidates (excluding blank)
        let topTokens = (0..<vocabSize)
            .filter { $0 != blankId }
            .sorted { frame[$0] > frame[$1] }
            .prefix(tokenCandidates)

        var newBeams: [[Int]: CtcBeam] = [:]

        func merge(_ beam: CtcBeam) {
            let k = beam.prefix
            if var existing = newBeams[k] {
                existing.pBlank = logAddExp(existing.pBlank, beam.pBlank)
                existing.pNonBlank = logAddExp(existing.pNonBlank, beam.pNonBlank)
                newBeams[k] = existing
            } else {
                newBeams[k] = beam
            }
        }

        for (_, beam) in beams {
            let prevTotal = beam.totalAcoustic

            // Blank extension
            var blankBeam = beam
            blankBeam.pBlank = prevTotal + blankLp
            blankBeam.pNonBlank = -.infinity
            merge(blankBeam)

            for v in topTokens {
                let tokenLp = frame[v]
                let isRepeat = (beam.lastToken == v)
                let piece = vocabulary[v] ?? ""

                // Word tracking + LM delta
                var newWordPieces = beam.wordPieces
                var newPrevWord = beam.prevWord
                var lmDelta: Float = 0.0
                if let lm = lm, piece.hasPrefix("▁") {
                    let completedWord = newWordPieces.joined()
                    let hasCompletedWord = !completedWord.isEmpty
                    lmDelta =
                        hasCompletedWord
                        ? lmWeight * lm.score(word: completedWord, prev: newPrevWord) + wordBonus
                        : 0.0
                    newPrevWord = hasCompletedWord ? completedWord : newPrevWord
                    let stripped = String(piece.dropFirst())
                    newWordPieces = stripped.isEmpty ? [] : [stripped]
                } else if lm != nil {
                    newWordPieces.append(piece)
                }

                if isRepeat {
                    // Corrected repeat handling
                    var sameBeam = beam
                    sameBeam.pBlank = -.infinity
                    sameBeam.pNonBlank = beam.pNonBlank + tokenLp
                    merge(sameBeam)

                    merge(
                        CtcBeam(
                            prefix: beam.prefix + [v],
                            pBlank: -.infinity,
                            pNonBlank: beam.pBlank + tokenLp,
                            lmScore: beam.lmScore + lmDelta,
                            wordPieces: newWordPieces,
                            prevWord: newPrevWord
                        ))
                } else {
                    merge(
                        CtcBeam(
                            prefix: beam.prefix + [v],
                            pBlank: -.infinity,
                            pNonBlank: prevTotal + tokenLp,
                            lmScore: beam.lmScore + lmDelta,
                            wordPieces: newWordPieces,
                            prevWord: newPrevWord
                        ))
                }
            }
        }

        // Prune to beam width
        let sorted = newBeams.values.sorted { $0.total > $1.total }
        beams = [:]
        for beam in sorted.prefix(beamWidth) {
            beams[beam.prefix] = beam
        }
    }

    // Finalize: score trailing partial word
    let finalBeams = beams.values.map { beam -> CtcBeam in
        guard let lm = lm else { return beam }
        let lastWord = beam.wordPieces.joined()
        guard !lastWord.isEmpty else { return beam }
        var b = beam
        b.lmScore += lmWeight * lm.score(word: lastWord, prev: beam.prevWord) + wordBonus
        return b
    }
    guard let best = finalBeams.max(by: { $0.total < $1.total }) else { return "" }
    return decodeCtcTokenIds(best.prefix, vocabulary: vocabulary)
}

/// CTC beam search from an MLMultiArray of shape [1, T, V].
///
/// Convenience overload that extracts per-frame log-probabilities from the MLMultiArray
/// and delegates to the `[[Float]]` version.
public func ctcBeamSearch(
    logProbs: MLMultiArray,
    vocabulary: [Int: String],
    lm: ARPALanguageModel? = nil,
    beamWidth: Int = 100,
    lmWeight: Float = 0.3,
    wordBonus: Float = 0.0,
    blankId: Int = 1024,
    tokenCandidates: Int = 40
) -> String {
    let timeSteps = logProbs.shape[1].intValue
    let vocabSize = logProbs.shape[2].intValue
    let ptr = logProbs.dataPointer.assumingMemoryBound(to: Float32.self)

    var frames: [[Float]] = []
    frames.reserveCapacity(timeSteps)
    for t in 0..<timeSteps {
        let base = t * vocabSize
        var frame = [Float](repeating: 0, count: vocabSize)
        for v in 0..<vocabSize {
            frame[v] = ptr[base + v]
        }
        frames.append(frame)
    }

    return ctcBeamSearch(
        logProbs: frames, vocabulary: vocabulary, lm: lm,
        beamWidth: beamWidth, lmWeight: lmWeight, wordBonus: wordBonus,
        blankId: blankId, tokenCandidates: tokenCandidates
    )
}

// MARK: - Helpers

/// Numerically stable log-space addition: `log(exp(a) + exp(b))`.
public func logAddExp(_ a: Float, _ b: Float) -> Float {
    if a == -.infinity { return b }
    if b == -.infinity { return a }
    let m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
}

/// Decode a sequence of token IDs using a vocabulary mapping.
///
/// Replaces SentencePiece `▁` markers with spaces and trims whitespace.
public func decodeCtcTokenIds(_ ids: [Int], vocabulary: [Int: String]) -> String {
    ids.compactMap { vocabulary[$0] }
        .joined()
        .replacingOccurrences(of: "▁", with: " ")
        .trimmingCharacters(in: .whitespaces)
}
