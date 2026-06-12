import Foundation

/// Configuration for Parakeet Unified 0.6B chunked-attention streaming.
///
/// The unified model streams by re-running its stateless encoder over a
/// `[left | chunk | right]` audio window whose chunked attention mask was
/// baked in at CoreML conversion time. Context sizes are expressed in 80 ms
/// encoder frames (1280 samples @ 16 kHz), matching the conversion pipeline
/// in mobius `models/stt/parakeet-unified-en-0.6b/coreml`.
public struct UnifiedStreamingConfig: Sendable {
    /// Left context in encoder frames (history visible to each chunk).
    public let leftFrames: Int
    /// Chunk size in encoder frames (new audio decoded per step).
    public let chunkFrames: Int
    /// Right context in encoder frames (look-ahead; adds latency).
    public let rightFrames: Int

    public let sampleRate: Int
    /// Samples per encoder frame (80 ms @ 16 kHz).
    public let frameSamples: Int
    public let melFeatures: Int
    public let decoderLayers: Int
    public let decoderHidden: Int
    public let blankIdx: Int
    public let maxSymbolsPerFrame: Int

    /// Default export: [70, 13, 13] = 5.6 s left, 1.04 s chunk, 1.04 s right
    /// (2.08 s theoretical latency — the model card's best-WER streaming mode).
    public init(
        leftFrames: Int = 70,
        chunkFrames: Int = 13,
        rightFrames: Int = 13,
        sampleRate: Int = 16000,
        frameSamples: Int = 1280,
        melFeatures: Int = 128,
        decoderLayers: Int = 2,
        decoderHidden: Int = 640,
        blankIdx: Int = 1024,
        maxSymbolsPerFrame: Int = 10
    ) {
        self.leftFrames = leftFrames
        self.chunkFrames = chunkFrames
        self.rightFrames = rightFrames
        self.sampleRate = sampleRate
        self.frameSamples = frameSamples
        self.melFeatures = melFeatures
        self.decoderLayers = decoderLayers
        self.decoderHidden = decoderHidden
        self.blankIdx = blankIdx
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
    }

    /// Total encoder window in samples (left + chunk + right).
    public var windowSamples: Int { (leftFrames + chunkFrames + rightFrames) * frameSamples }
    public var chunkSamples: Int { chunkFrames * frameSamples }
    public var rightSamples: Int { rightFrames * frameSamples }
    /// Theoretical latency in milliseconds (chunk + right context).
    public var latencyMs: Int { (chunkFrames + rightFrames) * frameSamples * 1000 / sampleRate }
    /// Suffix used in streaming encoder file names (e.g. "70_13_13").
    public var contextSuffix: String { "\(leftFrames)_\(chunkFrames)_\(rightFrames)" }
}

/// Pure window/frame bookkeeping for unified chunked streaming.
///
/// Mirrors NeMo's `StreamingBatchedAudioBuffer` inference loop: the first step
/// waits for chunk+right samples (initial latency), subsequent steps for chunk
/// samples. Each step encodes the last `windowSamples` ending at the consumed
/// position and decodes every not-yet-decoded encoder frame while holding back
/// the right context (re-encoded with more future audio next step).
struct UnifiedStreamingWindower {
    let config: UnifiedStreamingConfig

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

    init(config: UnifiedStreamingConfig) {
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
