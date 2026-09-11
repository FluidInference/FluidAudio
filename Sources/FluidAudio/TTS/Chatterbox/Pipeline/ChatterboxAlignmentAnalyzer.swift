import Foundation

/// Port of upstream `AlignmentStreamAnalyzer` (multilingual T3 sampling),
/// consuming the alignment-head attention rows the CoreML T3 models emit as
/// their `align_attn` output instead of hooking transformer internals.
/// Mirrors mobius `models/tts/chatterbox/coreml/verify/analyzer_port.py`.
///
/// The T3 prefill emits the rows for both trailing BOS queries
/// (`[3, 2, ctx]`); each decode step emits one row (`[3, ctx]`). Rows are
/// averaged over the three aligned heads, sliced to the text-token span, and
/// accumulated into an alignment matrix that drives EOS suppression and
/// forced-EOS hallucination bailouts.
struct ChatterboxAlignmentAnalyzer {

    /// Text-token span `[start, end)` within the prefill context.
    private let textStart: Int
    private let textEnd: Int
    private let eosIndex: Int
    /// Accumulated alignment rows, each of length `textEnd - textStart`.
    private var alignment: [[Float]] = []
    private var currFramePos = 0
    private var textPosition = 0
    private var started = false
    private var complete = false
    private var completedAt: Int?
    private var generatedTokens: [Int] = []

    init(textStart: Int, textEnd: Int, eosIndex: Int) {
        self.textStart = textStart
        self.textEnd = textEnd
        self.eosIndex = eosIndex
    }

    /// Apply one analyzer step to CFG-combined `logits`.
    ///
    /// - Parameters:
    ///   - alignRows: head-averaged attention rows over the live context,
    ///     one per query (`[queries][ctx]`; 2 queries from prefill, 1 per step).
    ///   - nextToken: the previously sampled token (repetition tracking).
    mutating func step(logits: inout [Float], alignRows: [[Float]], nextToken: Int?) {
        let span = textEnd - textStart
        // Monotonic masking: zero positions past currFramePos+1.
        var chunk: [[Float]] = []
        chunk.reserveCapacity(alignRows.count)
        for row in alignRows {
            var slice = Array(row[textStart..<min(textEnd, row.count)])
            if slice.count < span { slice += [Float](repeating: 0, count: span - slice.count) }
            if currFramePos + 1 < span {
                for i in (currFramePos + 1)..<span { slice[i] = 0 }
            }
            chunk.append(slice)
        }
        alignment.append(contentsOf: chunk)

        let history = alignment
        let historyCount = history.count

        // Current text position = argmax of the newest row.
        let lastRow = chunk[chunk.count - 1]
        var curTextPosn = 0
        var best = -Float.infinity
        for (i, v) in lastRow.enumerated() where v > best {
            best = v
            curTextPosn = i
        }
        let delta = curTextPosn - textPosition
        let discontinuity = !(delta > -4 && delta < 7)
        if !discontinuity { textPosition = curTextPosn }

        // False-start detection over the last 2 rows / first 4 columns.
        if !started {
            var tail2Max: Float = 0
            for row in history.suffix(2) {
                for v in row.suffix(2) { tail2Max = max(tail2Max, v) }
            }
            var head4Max: Float = 0
            for row in history {
                for v in row.prefix(4) { head4Max = max(head4Max, v) }
            }
            let falseStart = tail2Max > 0.1 || head4Max < 0.5
            started = !falseStart
        }

        complete = complete || textPosition >= span - 3
        if complete && completedAt == nil { completedAt = historyCount }

        var longTail = false
        var alignmentRepetition = false
        if complete, let completedAt {
            // Sum of the last-3-column activations since completion.
            var colSums = [Float](repeating: 0, count: 3)
            var repetitionSum: Float = 0
            for row in history[completedAt...] {
                for (j, v) in row.suffix(3).enumerated() { colSums[j] += v }
                if span > 5 {
                    var rowMax: Float = 0
                    for v in row.prefix(span - 5) { rowMax = max(rowMax, v) }
                    repetitionSum += rowMax
                }
            }
            longTail = (colSums.max() ?? 0) >= 5
            alignmentRepetition = repetitionSum > 5
        }

        if let nextToken {
            generatedTokens.append(nextToken)
            if generatedTokens.count > 8 {
                generatedTokens.removeFirst(generatedTokens.count - 8)
            }
        }
        let tokenRepetition =
            generatedTokens.count >= 3
            && generatedTokens[generatedTokens.count - 1] == generatedTokens[generatedTokens.count - 2]

        // Suppress premature EOS while the alignment is mid-text.
        if curTextPosn < span - 3 && span > 5 {
            logits[eosIndex] = -32768
        }

        // Force EOS on hallucination signals.
        if longTail || alignmentRepetition || tokenRepetition {
            for i in 0..<logits.count { logits[i] = -32768 }
            logits[eosIndex] = 32768
        }

        // Upstream increments by 1 per step even though the first chunk
        // contributes two rows — replicate exactly.
        currFramePos += 1
    }
}
