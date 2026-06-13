import Foundation

/// Pure window/frame bookkeeping for unified chunked streaming.
///
/// Mirrors NeMo's `StreamingBatchedAudioBuffer` inference loop: the first step
/// waits for chunk+right samples (initial latency), subsequent steps for chunk
/// samples. Each step encodes the last `windowSamples` ending at the consumed
/// position and decodes every not-yet-decoded encoder frame while holding back
/// the right context (re-encoded with more future audio next step).
///
/// Streaming-only — the offline batch path uses `UnifiedBatchLayout` instead.
struct UnifiedStreamingWindower {
    let config: UnifiedConfig

    /// Global samples fed to the encoder so far.
    private(set) var consumedSamples: Int = 0
    /// Global encoder frames decoded so far.
    private(set) var decodedFrames: Int = 0
    /// Whether the final window (holdback 0) has been emitted. Termination
    /// must not depend on re-deriving the encoder's exact length formula —
    /// the final flush is emitted at most once.
    private(set) var finalFlushEmitted: Bool = false

    struct WindowPlan {
        /// Global sample range to place in the encoder window (zero-padded to windowSamples).
        let bufferStart: Int
        let bufferEnd: Int
        /// Global encoder frame index of the window start.
        let bufferStartFrame: Int
        /// Encoder frames withheld from decoding (right context; 0 on the final window).
        let holdbackFrames: Int
    }

    init(config: UnifiedConfig) {
        self.config = config
    }

    /// Plans the next encoder window, or returns nil when not enough audio has
    /// accumulated (or, with `isFinal`, no audio remains).
    mutating func nextWindow(totalSamples: Int, isFinal: Bool) -> WindowPlan? {
        guard !finalFlushEmitted else { return nil }
        let feed = consumedSamples == 0 ? config.chunkSamples + config.rightSamples : config.chunkSamples
        let newConsumed: Int
        if consumedSamples + feed <= totalSamples {
            newConsumed = consumedSamples + feed
        } else if isFinal && totalSamples > consumedSamples {
            newConsumed = totalSamples
        } else if isFinal && totalSamples > 0 && consumedSamples == totalSamples {
            // Stream ended exactly on a chunk boundary: no new audio to feed,
            // but the right context held back by the last window still needs
            // decoding. Re-encode the final window with holdback 0.
            newConsumed = totalSamples
        } else {
            return nil
        }

        let isLast = isFinal && newConsumed >= totalSamples
        if isLast { finalFlushEmitted = true }
        var bufferStart = max(0, newConsumed - config.windowSamples)
        // Frame-align upward so the buffer never exceeds the window.
        bufferStart += (config.frameSamples - bufferStart % config.frameSamples) % config.frameSamples
        consumedSamples = newConsumed

        return WindowPlan(
            bufferStart: bufferStart,
            bufferEnd: newConsumed,
            bufferStartFrame: bufferStart / config.frameSamples,
            holdbackFrames: isLast ? 0 : config.rightFrames
        )
    }

    /// Local encoder-frame range to decode for this window, given the
    /// encoder's reported valid length. Advances the global decode position.
    mutating func decodeRange(encoderLength: Int, plan: WindowPlan) -> Range<Int>? {
        let localStart = decodedFrames - plan.bufferStartFrame
        let localEnd = encoderLength - plan.holdbackFrames
        guard localEnd > localStart, localStart >= 0 else { return nil }
        decodedFrames += localEnd - localStart
        return localStart..<localEnd
    }

    mutating func reset() {
        consumedSamples = 0
        decodedFrames = 0
        finalFlushEmitted = false
    }
}
