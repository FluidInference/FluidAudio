import Accelerate
import Foundation

/// Host-side long-utterance helpers for LuxTTS.
enum LuxTtsContinuation {

    private static let analysisWindowsPerSecond = 100
    /// Activity shorter than this is a click or breath, not speech. Every
    /// vocoder pass opens with a ~20 ms transient followed by 150–300 ms of
    /// silence; anchoring on sustained speech keeps both trim and pause
    /// detection from latching onto it.
    private static let sustainedSpeechSeconds = 0.05
    private static let onsetPrerollSeconds = 0.03
    private static let tailPostrollSeconds = 0.03

    /// Runs of sustained speech, in `windowSize`-sample analysis windows.
    struct SpeechRuns {
        let runs: [Range<Int>]
        let windowSize: Int
    }

    /// Whether the whole text can go through one pass unchanged: at most
    /// `continuationSpanTokens`, and within the token bucket, the 1024-frame
    /// graph, and the largest vocoder bucket.
    static func fitsSinglePass(
        textTokenCount: Int, promptFrames: Int, promptTokenCount: Int, speed: Double
    ) -> Bool {
        guard textTokenCount <= LuxTtsConstants.continuationSpanTokens,
            promptTokenCount + textTokenCount + 1 <= LuxTtsConstants.maxTokens
        else { return false }
        guard promptFrames > 0, promptTokenCount > 0, speed > 0 else { return true }
        let featuresLength = LuxTtsSolver.featuresLength(
            promptFrames: promptFrames,
            promptTokenCount: promptTokenCount,
            textTokenCount: textTokenCount,
            speed: speed)
        return featuresLength <= LuxTtsConstants.maxFrames
            && featuresLength - promptFrames <= (LuxTtsConstants.vocoderBuckets.max() ?? 0)
    }

    /// Largest span (in target tokens) that fits the token bucket next to
    /// this prompt's transcript and generates at most
    /// `continuationSpanFrameBudget` frames for its frames-per-token ratio
    /// at `speed`, capped at `continuationSpanTokens`.
    static func maxSpanTokens(promptFrames: Int, promptTokenCount: Int, speed: Double) -> Int {
        let cap = min(
            LuxTtsConstants.continuationSpanTokens,
            LuxTtsConstants.maxTokens - 1 - promptTokenCount)
        guard promptFrames > 0, promptTokenCount > 0, speed > 0 else { return max(1, cap) }
        let framesPerToken = Double(promptFrames) / Double(promptTokenCount)
        let budgetTokens = Int(
            (Double(LuxTtsConstants.continuationSpanFrameBudget) * speed / framesPerToken)
                .rounded(.down))
        return max(1, min(cap, budgetTokens))
    }

    /// Split a token sequence into balanced spans, preferring word/pause
    /// boundaries nearest each ideal split point.
    static func chunks(
        tokenIds: [Int], maxTokens: Int, boundaryTokenIds: Set<Int>
    ) -> [[Int]] {
        precondition(maxTokens > 0, "maxTokens must be positive")
        guard tokenIds.count > maxTokens else { return tokenIds.isEmpty ? [] : [tokenIds] }

        let chunkCount = (tokenIds.count + maxTokens - 1) / maxTokens
        var chunks: [[Int]] = []
        chunks.reserveCapacity(chunkCount)

        var start = 0
        for chunkIndex in 0..<(chunkCount - 1) {
            let remainingChunks = chunkCount - chunkIndex
            let remainingTokens = tokenIds.count - start
            let idealLength = Int(
                (Double(remainingTokens) / Double(remainingChunks)).rounded())
            let idealEnd = start + idealLength
            let minimumEnd = max(
                start + 1, tokenIds.count - (remainingChunks - 1) * maxTokens)
            let maxEnd = min(start + maxTokens, tokenIds.count - (remainingChunks - 1))
            let end = nearestBoundaryEnd(
                in: tokenIds,
                minimumEnd: minimumEnd,
                idealEnd: idealEnd,
                maxEnd: maxEnd,
                searchRadius: max(1, maxTokens / 4),
                boundaryTokenIds: boundaryTokenIds)

            chunks.append(Array(tokenIds[start..<end]))
            start = end
        }
        chunks.append(Array(tokenIds[start...]))
        return chunks
    }

    /// Pauses the text itself calls for: pause punctuation inside the span,
    /// ignoring trailing boundary tokens (their silence falls after speech).
    static func expectedPauseCount(
        in tokenIds: [Int], pauseTokenIds: Set<Int>, boundaryTokenIds: Set<Int>
    ) -> Int {
        var end = tokenIds.count
        while end > 0, boundaryTokenIds.contains(tokenIds[end - 1]) { end -= 1 }
        return tokenIds[..<end].reduce(0) { $0 + (pauseTokenIds.contains($1) ? 1 : 0) }
    }

    /// Pause boundaries the English G2P renders without a token: ellipses
    /// and quotes/brackets (`LuxTtsG2p` treats them as clause breaks).
    /// Apostrophes are left out — they are almost always contractions.
    static func textPauseAllowance(_ text: String) -> Int {
        let ellipses = text.components(separatedBy: "...").count - 1
        let silentBreaks = text.reduce(0) { count, character in
            count + ("…\"()[]«»".contains(character) ? 1 : 0)
        }
        return ellipses + silentBreaks
    }

    /// Whether the span's last spoken token is pause punctuation (trailing
    /// spaces ignored).
    static func endsWithPausePunctuation(
        _ tokenIds: [Int], pauseTokenIds: Set<Int>, boundaryTokenIds: Set<Int>
    ) -> Bool {
        let spaceTokenIds = boundaryTokenIds.subtracting(pauseTokenIds)
        guard let last = tokenIds.last(where: { !spaceTokenIds.contains($0) }) else { return false }
        return pauseTokenIds.contains(last)
    }

    /// Runs of at least `sustainedSpeechSeconds` above `pauseFloorDb`
    /// relative to the clip's peak.
    static func speechRuns(_ samples: [Float], sampleRate: Int) -> SpeechRuns {
        guard sampleRate > 0, !samples.isEmpty else { return SpeechRuns(runs: [], windowSize: 1) }
        let windowSize = max(1, sampleRate / analysisWindowsPerSecond)
        let windowCount = samples.count / windowSize
        guard windowCount > 0 else { return SpeechRuns(runs: [], windowSize: windowSize) }

        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        guard peak > 0 else { return SpeechRuns(runs: [], windowSize: windowSize) }
        let floorMeanSquare = peak * peak * powf(10, LuxTtsConstants.pauseFloorDb / 10)
        let minimumWindows = max(
            1, Int((sustainedSpeechSeconds * Double(sampleRate) / Double(windowSize)).rounded(.up)))

        var runs: [Range<Int>] = []
        var runStart: Int?
        samples.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            for window in 0...windowCount {
                var active = false
                if window < windowCount {
                    var meanSquare: Float = 0
                    vDSP_measqv(base + window * windowSize, 1, &meanSquare, vDSP_Length(windowSize))
                    active = meanSquare > floorMeanSquare
                }
                if active {
                    if runStart == nil { runStart = window }
                } else if let start = runStart {
                    if window - start >= minimumWindows { runs.append(start..<window) }
                    runStart = nil
                }
            }
        }
        return SpeechRuns(runs: runs, windowSize: windowSize)
    }

    /// Gaps of at least `pauseMinimumSeconds` between consecutive runs of
    /// sustained speech. Leading and trailing padding never count.
    static func innerPauseCount(_ speech: SpeechRuns, sampleRate: Int) -> Int {
        guard speech.runs.count > 1 else { return 0 }
        let minimumWindows = max(
            1,
            Int(
                (LuxTtsConstants.pauseMinimumSeconds * Double(sampleRate) / Double(speech.windowSize))
                    .rounded(.up)))
        return zip(speech.runs, speech.runs.dropFirst()).reduce(0) { count, pair in
            count + (pair.1.lowerBound - pair.0.upperBound >= minimumWindows ? 1 : 0)
        }
    }

    static func innerPauseCount(_ samples: [Float], sampleRate: Int) -> Int {
        innerPauseCount(speechRuns(samples, sampleRate: sampleRate), sampleRate: sampleRate)
    }

    /// Sample range to keep from a span: the vocoder's onset padding is cut
    /// (leaving a short preroll so unvoiced consonants are not clipped) and
    /// the tail padding likewise (leaving a short postroll). Clips without
    /// sustained speech are kept whole.
    static func speechSlice(
        _ speech: SpeechRuns, sampleCount: Int, sampleRate: Int, trimLeading: Bool, trimTrailing: Bool
    ) -> Range<Int> {
        guard let first = speech.runs.first, let last = speech.runs.last else { return 0..<sampleCount }
        var start = 0
        var end = sampleCount
        if trimLeading {
            let preroll = Int(onsetPrerollSeconds * Double(sampleRate))
            start = max(0, first.lowerBound * speech.windowSize - preroll)
        }
        if trimTrailing {
            let postroll = Int(tailPostrollSeconds * Double(sampleRate))
            end = min(sampleCount, last.upperBound * speech.windowSize + postroll)
        }
        return start..<max(start, end)
    }

    static func trimmingLeadingPadding(_ samples: [Float], sampleRate: Int) -> [Float] {
        let slice = speechSlice(
            speechRuns(samples, sampleRate: sampleRate), sampleCount: samples.count,
            sampleRate: sampleRate, trimLeading: true, trimTrailing: false)
        return Array(samples[slice])
    }

    static func trimmingTrailingPadding(_ samples: [Float], sampleRate: Int) -> [Float] {
        let slice = speechSlice(
            speechRuns(samples, sampleRate: sampleRate), sampleCount: samples.count,
            sampleRate: sampleRate, trimLeading: false, trimTrailing: true)
        return Array(samples[slice])
    }

    /// Join two mono clips with a linear crossfade.
    static func appendWithCrossfade(
        _ next: ArraySlice<Float>, to output: inout [Float], crossfadeSamples: Int
    ) {
        guard !output.isEmpty else {
            output = Array(next)
            return
        }
        guard !next.isEmpty else { return }

        let overlap = min(crossfadeSamples, output.count, next.count)
        guard overlap > 0 else {
            output.append(contentsOf: next)
            return
        }

        let outputStart = output.count - overlap
        let nextStart = next.startIndex
        if overlap == 1 {
            output[outputStart] = (output[outputStart] + next[nextStart]) * 0.5
        } else {
            for index in 0..<overlap {
                let nextWeight = Float(index) / Float(overlap - 1)
                output[outputStart + index] =
                    output[outputStart + index] * (1 - nextWeight)
                    + next[nextStart + index] * nextWeight
            }
        }
        output.append(contentsOf: next.dropFirst(overlap))
    }

    static func appendWithCrossfade(
        _ next: [Float], to output: inout [Float], crossfadeSamples: Int
    ) {
        appendWithCrossfade(next[...], to: &output, crossfadeSamples: crossfadeSamples)
    }

    private static func nearestBoundaryEnd(
        in tokenIds: [Int],
        minimumEnd: Int,
        idealEnd: Int,
        maxEnd: Int,
        searchRadius: Int,
        boundaryTokenIds: Set<Int>
    ) -> Int {
        var bestEnd: Int?
        var bestDistance = Int.max
        let searchStart = max(minimumEnd, idealEnd - searchRadius)
        let searchEnd = min(maxEnd, idealEnd + searchRadius)
        guard searchStart <= searchEnd else {
            return min(max(idealEnd, minimumEnd), maxEnd)
        }

        for end in searchStart...searchEnd {
            guard boundaryTokenIds.contains(tokenIds[end - 1]) else { continue }
            let distance = abs(end - idealEnd)
            if distance < bestDistance {
                bestEnd = end
                bestDistance = distance
            }
        }
        return bestEnd ?? min(max(idealEnd, minimumEnd), maxEnd)
    }
}
