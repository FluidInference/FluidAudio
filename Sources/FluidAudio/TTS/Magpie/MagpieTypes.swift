import Foundation

/// Supported Magpie TTS languages.
///
/// Japanese (`ja`) is intentionally omitted in this Swift port; it requires OpenJTalk
/// (a static C++ lib) and the OpenJTalk MeCab dictionary (~102 MB), both deferred to a
/// follow-up PR.
public enum MagpieLanguage: String, Sendable, CaseIterable, Hashable {
    case english = "en"
    case spanish = "es"
    case german = "de"
    case french = "fr"
    case italian = "it"
    case vietnamese = "vi"
    case mandarin = "zh"
    case hindi = "hi"
}

/// Built-in Magpie speakers (index 0–4). Voice quality varies — see model card.
public enum MagpieSpeaker: Int, Sendable, CaseIterable {
    case john = 0
    case sofia = 1
    case aria = 2
    case jason = 3
    case leo = 4

    public var displayName: String {
        switch self {
        case .john: return "John"
        case .sofia: return "Sofia"
        case .aria: return "Aria"
        case .jason: return "Jason"
        case .leo: return "Leo"
        }
    }
}

/// Tuning knobs for a single synthesis call.
public struct MagpieSynthesisOptions: Sendable {
    public var temperature: Float
    public var topK: Int
    public var maxSteps: Int
    public var minFrames: Int
    public var cfgScale: Float
    public var seed: UInt64?
    public var peakNormalize: Bool
    /// When `true`, `|...|` regions in the input text are tokenized directly as IPA
    /// (space-separated IPA characters) and the rest of the text flows through the
    /// normal language tokenizer / G2P. When `false`, `|` is treated as a literal
    /// character. Always on by default — matches the Magpie model card guidance.
    public var allowIpaOverride: Bool

    public init(
        temperature: Float = MagpieConstants.defaultTemperature,
        topK: Int = MagpieConstants.defaultTopK,
        maxSteps: Int = MagpieConstants.maxSteps,
        minFrames: Int = MagpieConstants.minFrames,
        cfgScale: Float = MagpieConstants.defaultCfgScale,
        seed: UInt64? = nil,
        peakNormalize: Bool = true,
        allowIpaOverride: Bool = true
    ) {
        self.temperature = temperature
        self.topK = topK
        self.maxSteps = maxSteps
        self.minFrames = minFrames
        self.cfgScale = cfgScale
        self.seed = seed
        self.peakNormalize = peakNormalize
        self.allowIpaOverride = allowIpaOverride
    }

    public static let `default` = MagpieSynthesisOptions()
}

/// Pre-tokenized phoneme input, bypassing every text-frontend stage (normalization,
/// G2P, `|`-override lexing). Use when you want full control over pronunciation, or
/// when importing token ids from an external phonemizer.
///
/// Expected format: raw token ids from the language's `*_token2id.json` map, each in
/// `[0, vocab)`. The frontend will pad/truncate to `MagpieConstants.maxTextLength`
/// and build the encoder mask automatically.
public struct MagpiePhonemeTokens: Sendable {
    public let tokenIds: [Int32]
    public let language: MagpieLanguage

    public init(tokenIds: [Int32], language: MagpieLanguage) {
        self.tokenIds = tokenIds
        self.language = language
    }
}

/// Per-stage wallclock timings for a synthesis call (seconds).
public struct MagpieSynthesisTimings: Sendable {
    public let textEncoderSeconds: Double
    public let prefillSeconds: Double
    public let arLoopSeconds: Double
    public let decoderStepSeconds: Double
    public let samplerSeconds: Double
    public let nanocodecSeconds: Double

    public init(
        textEncoderSeconds: Double, prefillSeconds: Double, arLoopSeconds: Double,
        decoderStepSeconds: Double, samplerSeconds: Double, nanocodecSeconds: Double
    ) {
        self.textEncoderSeconds = textEncoderSeconds
        self.prefillSeconds = prefillSeconds
        self.arLoopSeconds = arLoopSeconds
        self.decoderStepSeconds = decoderStepSeconds
        self.samplerSeconds = samplerSeconds
        self.nanocodecSeconds = nanocodecSeconds
    }
}

/// One incremental output of a streaming synthesis call.
///
/// Audio chunks arrive in sequence (`sequenceIndex` strictly increasing). They
/// can be played gaplessly: each non-final chunk's `samples` already includes
/// the trailing silence the chunker assigned for natural prosody breaks at
/// sentence/clause boundaries, plus a brief edge-fade at both ends to mask
/// any boundary discontinuity.
public struct MagpieAudioChunk: Sendable {
    /// fp32 PCM samples in [-1, 1], mono.
    public let samples: [Float]
    /// Always `MagpieConstants.audioSampleRate` (22050 Hz).
    public let sampleRate: Int
    /// 0-based index into the chunk sequence for this synthesis call.
    public let sequenceIndex: Int
    /// `true` for the last chunk of the call (after which the stream finishes).
    public let isFinal: Bool
    /// Source text for this chunk — useful for rolling captions.
    public let text: String
    /// Number of codec frames (pre-NanoCodec) that produced this chunk.
    public let codeCount: Int
    /// Whether the AR loop ended on EOS (vs hitting `maxSteps`) for this chunk.
    public let finishedOnEos: Bool

    public init(
        samples: [Float], sampleRate: Int, sequenceIndex: Int, isFinal: Bool,
        text: String, codeCount: Int, finishedOnEos: Bool
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.sequenceIndex = sequenceIndex
        self.isFinal = isFinal
        self.text = text
        self.codeCount = codeCount
        self.finishedOnEos = finishedOnEos
    }

    public var durationSeconds: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate)
    }
}

/// Result of a synthesis call.
public struct MagpieSynthesisResult: Sendable {
    /// 32-bit float PCM samples in [-1, 1], mono.
    public let samples: [Float]
    /// Always `MagpieConstants.audioSampleRate` (22050 Hz) for Magpie.
    public let sampleRate: Int
    /// Number of codec frames generated (before NanoCodec expansion).
    public let codeCount: Int
    /// Whether generation stopped because an EOS token was emitted (vs hitting `maxSteps`).
    public let finishedOnEos: Bool
    /// Per-stage timings.
    public let timings: MagpieSynthesisTimings

    public var durationSeconds: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate)
    }

    public init(
        samples: [Float], sampleRate: Int, codeCount: Int, finishedOnEos: Bool,
        timings: MagpieSynthesisTimings
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.codeCount = codeCount
        self.finishedOnEos = finishedOnEos
        self.timings = timings
    }
}
