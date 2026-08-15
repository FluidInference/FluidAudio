import Foundation

/// Pure host-side math for the LuxTTS pipeline: ratio-based duration
/// estimation, average-token-duration expansion indices, anchor-Euler
/// solver timesteps and state update. All functions mirror the upstream
/// ZipVoice implementations exactly (fixture-pinned).
public enum LuxTtsSolver {

    /// Total feature length: prompt frames + ratio-estimated generated frames.
    /// Mirrors `prompt_len + ceil(prompt_len / prompt_tokens * text_tokens / speed)`.
    public static func featuresLength(
        promptFrames: Int, promptTokenCount: Int, textTokenCount: Int, speed: Double
    ) -> Int {
        let generated =
            Double(promptFrames) / Double(promptTokenCount) * Double(textTokenCount) / speed
        return promptFrames + Int(generated.rounded(.up))
    }

    /// Solver timesteps: `t_shift * u / (1 + (t_shift - 1) * u)` over
    /// `linspace(0, 1, numSteps + 1)` (upstream `get_time_steps`).
    public static func timeSteps(numSteps: Int, tShift: Double) -> [Double] {
        (0...numSteps).map { i in
            let u = Double(i) / Double(numSteps)
            return tShift * u / (1.0 + (tShift - 1.0) * u)
        }
    }

    /// Per-frame token index for the duration expansion. Mirrors upstream
    /// `prepare_avg_tokens_durations` + `get_tokens_index`: every token gets
    /// `featuresLength / tokensCount` frames and the remainder frames map to
    /// index `tokensCount` — the appended pad-slot embedding row.
    public static func tokensIndex(tokensCount: Int, featuresLength: Int) throws -> [Int] {
        let avg = featuresLength / tokensCount
        // avg < 1 collapses every real token to zero frames: the whole
        // sequence maps to the pad slot and synthesis is silent. Fail loudly
        // instead of emitting garbage.
        guard avg >= 1 else {
            throw LuxTtsError.degenerateDuration(
                featuresLength: featuresLength, tokensCount: tokensCount)
        }
        var index = [Int](repeating: tokensCount, count: featuresLength)
        var frame = 0
        for token in 0..<tokensCount {
            for _ in 0..<avg {
                index[frame] = token
                frame += 1
            }
        }
        return index
    }

    /// One anchor-Euler update:
    /// `x1p = x + (1 - t)·v`, `x0p = x - t·v`, then
    /// `x ← (1 - tNext)·x0p + tNext·x1p` (or `x1p` on the last step).
    public static func anchorEulerUpdate(
        x: [Double], v: [Double], tCur: Double, tNext: Double, isLast: Bool
    ) -> [Double] {
        precondition(x.count == v.count)
        return zip(x, v).map { xi, vi in
            let x1p = xi + (1.0 - tCur) * vi
            if isLast { return x1p }
            let x0p = xi - tCur * vi
            return (1.0 - tNext) * x0p + tNext * x1p
        }
    }
}
