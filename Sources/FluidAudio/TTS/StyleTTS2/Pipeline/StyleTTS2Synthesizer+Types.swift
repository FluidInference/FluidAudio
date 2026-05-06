import Foundation

/// One of the 7 stages in the StyleTTS2-ANE chain.
///
/// The Stage enum mirrors `KokoroAneStage` 1:1 *except* `.diffusionStep`
/// replaces Kokoro's single-shot prosody graph (StyleTTS2's defining
/// feature is the diffusion sampler) and `.tail` is folded into
/// `.vocoder` because StyleTTS2's HiFi-GAN is iSTFT-free.
public enum StyleTTS2Stage: String, CaseIterable, Sendable {
    case plBert
    case postBert
    case alignment
    case diffusionStep
    case prosody
    case noise
    case vocoder

    /// `.mlmodelc` filename on disk and on HuggingFace
    /// (`FluidInference/StyleTTS-2-coreml/ANE/`).
    public var bundleName: String {
        switch self {
        case .plBert: return "styletts2_ane_plbert.mlmodelc"
        case .postBert: return "styletts2_ane_postbert.mlmodelc"
        case .alignment: return "styletts2_ane_alignment.mlmodelc"
        case .diffusionStep: return "styletts2_ane_diffusion_step.mlmodelc"
        case .prosody: return "styletts2_ane_prosody.mlmodelc"
        case .noise: return "styletts2_ane_noise.mlmodelc"
        case .vocoder: return "styletts2_ane_vocoder.mlmodelc"
        }
    }
}

/// Per-stage wall-clock timings (milliseconds) for one synthesis call.
///
/// `diffusionStep` is the cumulative wall-clock across all 11 invocations
/// (5 ADPM2 midpoint steps × 2 + 1 final).
public struct StyleTTS2StageTimings: Sendable, Equatable {
    public var plBert: Double = 0
    public var postBert: Double = 0
    public var alignment: Double = 0
    public var diffusionStep: Double = 0
    public var prosody: Double = 0
    public var noise: Double = 0
    public var vocoder: Double = 0

    /// Sum of all stages, in milliseconds.
    public var totalMs: Double {
        plBert + postBert + alignment + diffusionStep + prosody + noise + vocoder
    }

    public init() {}
}

/// Detailed result of a `StyleTTS2Manager.synthesizeDetailed` call.
public struct StyleTTS2SynthesisResult: Sendable {
    /// 24 kHz mono fp32 PCM samples (raw, not WAV-wrapped).
    public let samples: [Float]
    /// Sample rate (24,000 Hz for the LibriTTS HiFi-GAN).
    public let sampleRate: Int
    /// `T_tok` — phoneme tokens fed to PLBERT.
    public let encoderTokens: Int
    /// `T_a` — acoustic frames (= sum of predicted durations).
    public let acousticFrames: Int
    /// Per-stage wall-clock timings.
    public let timings: StyleTTS2StageTimings

    public var durationSeconds: Double {
        Double(samples.count) / Double(sampleRate)
    }

    public init(
        samples: [Float],
        sampleRate: Int,
        encoderTokens: Int,
        acousticFrames: Int,
        timings: StyleTTS2StageTimings
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.encoderTokens = encoderTokens
        self.acousticFrames = acousticFrames
        self.timings = timings
    }
}
