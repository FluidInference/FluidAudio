import Foundation

/// Result of a CosyVoice3 synthesis call.
public struct CosyVoice3SynthesisResult: Sendable {
    /// Raw 24 kHz fp32 PCM samples.
    public let samples: [Float]
    /// Sample rate (always 24_000).
    public let sampleRate: Int
    /// Number of speech tokens the LLM actually generated.
    public let generatedTokenCount: Int
    /// Decoded speech token ids (useful for debugging + round-trip).
    public let decodedTokens: [Int32]
}

/// Options controlling a CosyVoice3 parity / synthesis call.
public struct CosyVoice3ParityOptions: Sendable {
    /// Maximum number of new tokens to generate (capped by `flowTotalTokens - N_prompt`).
    public let maxNewTokens: Int?
    /// Sampler seed (for the fallback multinomial path; parity replay overrides this).
    public let seed: UInt64
    /// When true, disables sampling and replays the fixture's `decodedTokens`.
    public let replayDecodedTokens: Bool

    public init(
        maxNewTokens: Int? = nil,
        seed: UInt64 = 42,
        replayDecodedTokens: Bool = true
    ) {
        self.maxNewTokens = maxNewTokens
        self.seed = seed
        self.replayDecodedTokens = replayDecodedTokens
    }
}

/// Options for the Phase 2 text-driven synthesis path.
///
/// Thin wrapper around `CosyVoice3ParityOptions` that omits the parity-only
/// `replayDecodedTokens` flag (text mode always samples).
public struct CosyVoice3SynthesisOptions: Sendable {
    /// Maximum number of new speech tokens the LLM may generate (capped by
    /// `flowTotalTokens - N_prompt` at runtime).
    public let maxNewTokens: Int?
    /// Sampler seed for the top-p/top-k + multinomial fallback path.
    public let seed: UInt64

    public init(maxNewTokens: Int? = nil, seed: UInt64 = 42) {
        self.maxNewTokens = maxNewTokens
        self.seed = seed
    }
}
