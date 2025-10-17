import Foundation

struct StreamingWindow {
    let samples: [Float]
    let startSample: Int
    let centerStartSample: Int
    let centerSampleCount: Int
    let isFinalChunk: Bool
}

/// Maintains a rolling audio buffer and emits chunked windows with left/right context padding for streaming ASR.
struct StreamingWindowProcessor {
    /// Minimum chunk duration for partial windows during streaming onset (seconds).
    /// Lower values reduce first-token latency but may impact accuracy.
    private static let minimumPartialChunkSeconds: Double = 3.0

    /// Minimum reserved capacity for the streaming buffer (samples).
    /// Matches the VAD chunk size (256 ms at 16 kHz) to keep reallocations off the hot path.
    private static let bufferPreallocationFloor: Int = VadManager.chunkSize

    /// Caller-provided chunk, context, stabilizer, and VAD configuration.
    private let config: StreamingAsrConfig
    /// Circular buffer that owns the streaming samples currently in-flight.
    private var sampleBuffer: CircularBuffer<Float>
    /// Absolute sample index of `sampleBuffer[0]`; advances as we drop consumed audio.
    private var bufferStartIndex: Int = 0
    /// Absolute sample index where the next window's center region should begin.
    private var nextWindowCenterStart: Int = 0
    /// Logical silence appended to preserve right context when no real samples are available yet.
    private var virtualTrailingSilence: Int = 0
    /// Silence duration observed before the first real samples arrive; released once audio starts.
    private var pendingLeadingSilence: Int = 0
    /// Partial chunk floor (samples) used when we emit early windows during VAD onset.
    private let partialChunkMinimum: Int

    /// Prepares the processor with the desired chunk/context sizes and computes partial-window thresholds.
    init(config: StreamingAsrConfig) {
        self.config = config
        let requiredCapacity = config.chunkSamples + config.leftContextSamples + config.rightContextSamples
        // Preallocate enough space for the largest window plus context to keep buffer mutations cheap.
        let initialCapacity = max(requiredCapacity, Self.bufferPreallocationFloor)
        self.sampleBuffer = CircularBuffer(initialCapacity: initialCapacity)
        let minimumSamples = Int(Self.minimumPartialChunkSeconds * Double(ASRConstants.sampleRate))
        // Clamp partial windows to a safe range: ≥3s (or half chunk) but never exceeding the full chunk size.
        self.partialChunkMinimum = min(config.chunkSamples, max(minimumSamples, config.chunkSamples / 2))
    }

    mutating func reset() {
        sampleBuffer.removeAll(keepingCapacity: false)
        bufferStartIndex = 0
        nextWindowCenterStart = 0
        virtualTrailingSilence = 0
        pendingLeadingSilence = 0
    }

    mutating func append(
        _ samples: [Float],
        allowPartialChunk: Bool = false,
        minimumCenterSamples: Int? = nil
    ) -> [StreamingWindow] {
        if sampleBuffer.isEmpty && !samples.isEmpty {
            bufferStartIndex += pendingLeadingSilence
            pendingLeadingSilence = 0
        }
        sampleBuffer.append(contentsOf: samples)
        let minimum = minimumCenterSamples ?? (allowPartialChunk ? partialChunkMinimum : config.chunkSamples)
        return drainWindows(
            allowPartialChunk: allowPartialChunk,
            minimumCenterSamples: minimum,
            markFinalChunk: false
        )
    }

    mutating func flushRemaining() -> [StreamingWindow] {
        if virtualTrailingSilence < config.rightContextSamples {
            virtualTrailingSilence = config.rightContextSamples
        }
        return drainWindows(
            allowPartialChunk: true,
            minimumCenterSamples: 1,
            markFinalChunk: true
        )
    }

    mutating func advanceBySilence(_ samples: Int) {
        guard samples > 0 else { return }
        if sampleBuffer.isEmpty && virtualTrailingSilence == 0 {
            pendingLeadingSilence += samples
        } else {
            virtualTrailingSilence += samples
        }
    }

    private mutating func drainWindows(
        allowPartialChunk: Bool,
        minimumCenterSamples: Int,
        markFinalChunk: Bool
    ) -> [StreamingWindow] {
        var windows: [StreamingWindow] = []
        let chunk = config.chunkSamples
        let right = config.rightContextSamples
        let left = config.leftContextSamples
        while true {
            let availableSamples = sampleBuffer.count
            let currentAbsEnd = bufferStartIndex + availableSamples + virtualTrailingSilence
            let availableAhead = currentAbsEnd - nextWindowCenterStart
            if allowPartialChunk {
                if availableAhead < minimumCenterSamples { break }
            } else if currentAbsEnd < (nextWindowCenterStart + chunk + right) {
                break
            }

            if allowPartialChunk && availableAhead <= 0 { break }

            let effectiveChunk = allowPartialChunk ? min(chunk, availableAhead) : chunk

            let leftStartAbs = max(0, nextWindowCenterStart - left)
            let rightEndAbs =
                allowPartialChunk
                ? nextWindowCenterStart + effectiveChunk
                : nextWindowCenterStart + chunk + right

            let startIndex = max(leftStartAbs - bufferStartIndex, 0)
            let rawEndIndex = rightEndAbs - bufferStartIndex
            // Clamp to `startIndex` to avoid negative spans while right-context silence is still virtual.
            let endIndex =
                allowPartialChunk
                ? max(rawEndIndex, startIndex)
                : max(rawEndIndex, startIndex)

            if startIndex < 0 || endIndex < startIndex { break }

            let clampedEnd = min(endIndex, availableSamples)
            var windowSamples: [Float] = []

            let leftPadCount = max(0, bufferStartIndex - leftStartAbs)
            if leftPadCount > 0 {
                windowSamples.append(contentsOf: [Float](repeating: 0.0, count: leftPadCount))
            }

            if startIndex < clampedEnd {
                let segment = sampleBuffer.copyRange(startIndex..<clampedEnd)
                windowSamples.append(contentsOf: segment)
            }

            let actualEndAbs = bufferStartIndex + clampedEnd
            let desiredEndAbs = rightEndAbs
            let availableTrailingSilence = virtualTrailingSilence

            var rightPadCount = max(0, desiredEndAbs - actualEndAbs)
            if rightPadCount > availableTrailingSilence {
                rightPadCount = availableTrailingSilence
            }
            if rightPadCount > 0 {
                windowSamples.append(contentsOf: [Float](repeating: 0.0, count: rightPadCount))
                virtualTrailingSilence -= rightPadCount
            }

            if startIndex < clampedEnd || !windowSamples.isEmpty {
                windows.append(
                    StreamingWindow(
                        samples: windowSamples,
                        startSample: leftStartAbs,
                        centerStartSample: nextWindowCenterStart,
                        centerSampleCount: effectiveChunk,
                        isFinalChunk: false
                    )
                )
            } else {
                break
            }

            nextWindowCenterStart += allowPartialChunk ? effectiveChunk : chunk

            let trimToAbs = max(0, nextWindowCenterStart - left)
            let dropCount = max(0, trimToAbs - bufferStartIndex)
            if dropCount > 0 && dropCount <= availableSamples {
                bufferStartIndex += dropCount
                sampleBuffer.dropFirst(dropCount)
            }
        }

        if markFinalChunk, let lastIndex = windows.indices.last {
            for index in windows.indices {
                windows[index] = StreamingWindow(
                    samples: windows[index].samples,
                    startSample: windows[index].startSample,
                    centerStartSample: windows[index].centerStartSample,
                    centerSampleCount: windows[index].centerSampleCount,
                    isFinalChunk: index == lastIndex
                )
            }
        }

        return windows
    }

    func hasBufferedAudio() -> Bool {
        !sampleBuffer.isEmpty
    }

}
