import Foundation

/// Per-stage wall-clock timings (milliseconds) for one synthesis call.
public struct KokoroAneStageTimings: Sendable, Equatable {
    public var albert: Double = 0
    public var postAlbert: Double = 0
    public var alignment: Double = 0
    public var prosody: Double = 0
    public var noise: Double = 0
    public var vocoder: Double = 0
    public var tail: Double = 0

    /// Sum of all stages, in milliseconds.
    public var totalMs: Double {
        albert + postAlbert + alignment + prosody + noise + vocoder + tail
    }

    public init() {}

    /// Accumulate another call's per-stage timings into this one — used when
    /// a long prompt is synthesized in several chunks (issue #712).
    mutating func add(_ other: KokoroAneStageTimings) {
        albert += other.albert
        postAlbert += other.postAlbert
        alignment += other.alignment
        prosody += other.prosody
        noise += other.noise
        vocoder += other.vocoder
        tail += other.tail
    }
}

/// Detailed result of a `KokoroAneManager.synthesizeDetailed` call.
public struct KokoroAneSynthesisResult: Sendable {
    /// 24 kHz mono fp32 PCM samples (raw, not WAV-wrapped).
    public let samples: [Float]
    /// Sample rate (24,000 Hz for the laishere chain).
    public let sampleRate: Int
    /// `T_enc` — phoneme tokens including BOS/EOS.
    public let encoderTokens: Int
    /// `T_a` — acoustic frames produced by PostAlbert / Alignment.
    public let acousticFrames: Int
    /// Token ids passed to the Kokoro chain, including BOS/EOS.
    ///
    /// Indices align one-to-one with ``predictedDurations``.
    public let inputIds: [Int32]
    /// PostAlbert `pred_dur`: acoustic-frame counts for each input token.
    ///
    /// Kokoro uses these exact durations to build the alignment consumed by
    /// the downstream prosody/vocoder stages. Exposing them lets callers
    /// derive token/word timestamps without re-aligning the synthesized audio.
    public let predictedDurations: [Int32]
    /// Text after written-form normalization, exactly as handed to the G2P
    /// frontend (`"$45"` → `"forty five dollars"`). Lets callers align their
    /// display words to the spoken words behind ``inputIds`` /
    /// ``predictedDurations`` without re-running normalization (issue #943).
    ///
    /// `nil` when the caller bypassed G2P via
    /// `synthesizeFromPhonemesDetailed`. Equals the input when nothing was
    /// rewritten or the input was passed through as pre-computed phonemes.
    public let normalizedText: String?
    /// Phoneme string fed to the chain — what ``inputIds`` encodes.
    public let phonemes: String
    /// Per-stage timings.
    public let timings: KokoroAneStageTimings

    /// Convenience: audio duration in seconds.
    public var durationSeconds: Double {
        Double(samples.count) / Double(sampleRate)
    }

    public init(
        samples: [Float],
        sampleRate: Int,
        encoderTokens: Int,
        acousticFrames: Int,
        timings: KokoroAneStageTimings,
        inputIds: [Int32] = [],
        predictedDurations: [Int32] = [],
        normalizedText: String? = nil,
        phonemes: String = ""
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.encoderTokens = encoderTokens
        self.acousticFrames = acousticFrames
        self.inputIds = inputIds
        self.predictedDurations = predictedDurations
        self.normalizedText = normalizedText
        self.phonemes = phonemes
        self.timings = timings
    }

    /// Copy with the frontend provenance attached. The chain stage only sees
    /// token ids; the manager knows the text and phonemes that produced them.
    func attachingFrontend(normalizedText: String?, phonemes: String) -> KokoroAneSynthesisResult {
        KokoroAneSynthesisResult(
            samples: samples,
            sampleRate: sampleRate,
            encoderTokens: encoderTokens,
            acousticFrames: acousticFrames,
            timings: timings,
            inputIds: inputIds,
            predictedDurations: predictedDurations,
            normalizedText: normalizedText,
            phonemes: phonemes
        )
    }
}

/// One of the 7 stages in the laishere chain.
public enum KokoroAneStage: String, CaseIterable, Sendable {
    case albert
    case postAlbert
    case alignment
    case prosody
    case noise
    case vocoder
    case tail

    /// `.mlmodelc` filename on disk and on HuggingFace.
    public var bundleName: String {
        switch self {
        case .albert: return "KokoroAlbert.mlmodelc"
        case .postAlbert: return "KokoroPostAlbert.mlmodelc"
        case .alignment: return "KokoroAlignment.mlmodelc"
        case .prosody: return "KokoroProsody.mlmodelc"
        case .noise: return "KokoroNoise_v2.mlmodelc"  // v2: atan2 phase-correction (HF-noise fix)
        case .vocoder: return "KokoroVocoder.mlmodelc"
        case .tail: return "KokoroTail_v2.mlmodelc"  // v2: COLA-normalized iSTFT (level fix, #852)
        }
    }
}
