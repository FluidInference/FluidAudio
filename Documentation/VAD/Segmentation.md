# Configuration fields

Configuration for turning raw VAD probabilities into stable speech segments.

This struct applies rules for minimum durations, thresholds, and hysteresis to avoid jittery cuts and to produce clean, ASR-ready segments.

```swift
public struct VadSegmentationConfig: Sendable {
    /// Minimum length of detected speech to keep as a segment (default: 0.15s).
    /// Prevents clicks or coughs from being treated as speech.
    public var minSpeechDuration: TimeInterval

    /// Minimum silence required to end a segment (default: 0.75s).
    /// Prevents early cut-offs when a speaker pauses briefly.
    public var minSilenceDuration: TimeInterval

    /// Maximum length of a single speech segment (default: 14s).
    /// Segments longer than this will be forcibly split to match ASR model limits.
    public var maxSpeechDuration: TimeInterval

    /// Extra padding (before and after) each detected speech segment (default: 0.1s).
    /// Keeps context around words so they aren’t clipped.
    public var speechPadding: TimeInterval

    /// Probability threshold below which audio is treated as silence (default: 0.3).
    /// Lower = stricter silence detection, higher = more tolerant.
    public var silenceThresholdForSplit: Float

    /// Explicit override for the *exit* hysteresis threshold (default: nil).
    /// If not set, the system computes it automatically from the base threshold minus `negativeThresholdOffset`.
    public var negativeThreshold: Float?

    /// How far below the base threshold the *exit* threshold should be (default: 0.15).
    /// Example: if entry = 0.5, exit = 0.35. Prevents rapid flipping on noisy inputs.
    public var negativeThresholdOffset: Float

    /// Minimum silence enforced when splitting a max-length segment (default: 0.098s).
    /// Ensures forced splits don’t land mid-phoneme.
    public var minSilenceAtMaxSpeech: TimeInterval

    /// If true, try to split at the longest silence near the max duration cutoff.
    /// Produces cleaner segment boundaries compared to a hard cut.
    public var useMaxPossibleSilenceAtMaxSpeech: Bool

    public static let `default` = VadSegmentationConfig()

    public init(
        minSpeechDuration: TimeInterval = 0.15,
        minSilenceDuration: TimeInterval = 0.75,
        maxSpeechDuration: TimeInterval = 14.0,
        speechPadding: TimeInterval = 0.1,
        silenceThresholdForSplit: Float = 0.3,
        negativeThreshold: Float? = nil,
        negativeThresholdOffset: Float = 0.15,
        minSilenceAtMaxSpeech: TimeInterval = 0.098,
        useMaxPossibleSilenceAtMaxSpeech: Bool = true
    ) {
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.maxSpeechDuration = maxSpeechDuration
        self.speechPadding = speechPadding
        self.silenceThresholdForSplit = silenceThresholdForSplit
        self.negativeThreshold = negativeThreshold
        self.negativeThresholdOffset = negativeThresholdOffset
        self.minSilenceAtMaxSpeech = minSilenceAtMaxSpeech
        self.useMaxPossibleSilenceAtMaxSpeech = useMaxPossibleSilenceAtMaxSpeech
    }

    /// Computes the working negative threshold for hysteresis:
    /// - If `negativeThreshold` is set, that value is used.
    /// - Otherwise, it is computed as (baseThreshold – negativeThresholdOffset).
    /// - This creates a "sticky zone" between thresholds:
    ///   - Enter speech when prob > baseThreshold
    ///   - Exit speech when prob < negativeThreshold
    ///   - Stay in current state in between
    public func effectiveNegativeThreshold(baseThreshold: Float) -> Float {
        if let override = negativeThreshold {
            return override
        }
        return max(baseThreshold - negativeThresholdOffset, 0.01)
    }
}
```

The entry threshold for hysteresis defaults to `VadConfig.defaultThreshold`, set when you construct a
`VadManager`. If you provide a `negativeThreshold`, the streaming helpers derive an entry threshold
by adding `negativeThresholdOffset` (clamped to 1.0), allowing per-request tuning without rebuilding
the manager. To change the baseline entry threshold globally, create the manager with a different
`defaultThreshold`.
