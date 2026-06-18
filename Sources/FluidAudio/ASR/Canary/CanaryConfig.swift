@preconcurrency import CoreML
import Foundation

/// Canary encoder/decoder weight precision.
///
/// `int4` (per-block-32 symmetric) runs on the Neural Engine and is the
/// smallest build (~573 MB) — but int4 weight payloads require iOS18 / macOS 15.
/// `fp16` is the iOS17 parity fallback (exact match to PyTorch). `int8`
/// (per-channel) decodes correctly only on CPU — it crashes the GPU/ANE MPSGraph
/// backend — so it is a CPU/size-only option.
public enum CanaryPrecision: String, Sendable, CaseIterable {
    case int4
    case fp16
    case int8

    var encoderName: String {
        switch self {
        case .int4: return ModelNames.Canary.encoderInt4
        case .fp16: return ModelNames.Canary.encoder
        case .int8: return ModelNames.Canary.encoderInt8
        }
    }

    var decoderName: String {
        switch self {
        case .int4: return ModelNames.Canary.decoderInt4
        case .fp16: return ModelNames.Canary.decoder
        case .int8: return ModelNames.Canary.decoderInt8
        }
    }

    /// int8 only decodes correctly on CPU; int4/fp16 run on the Neural Engine.
    var computeUnits: MLComputeUnits {
        self == .int8 ? .cpuOnly : .cpuAndNeuralEngine
    }
}

/// Fixed-shape contract for the canary-1b-v2 CoreML pipeline (15 s window).
public enum CanaryConfig {
    public static let sampleRate = 16000
    /// 15 s window — the preprocessor input is fixed at this sample count.
    public static let maxSamples = 240_000
    /// Overlap between adjacent windows when chunking audio longer than 15 s.
    /// 3 s (~19 tokens) gives the seam LCS-merge enough shared context to align
    /// reliably while wasting little recompute. Hop = maxSamples − this.
    public static let chunkOverlapSeconds = 3.0
    public static let chunkOverlapSamples = 48_000
    public static let melDim = 128
    public static let melFrames = 1501
    public static let encoderHidden = 1024
    public static let encoderFrames = 188
    /// Decoder is exported at a fixed `[1, maxDecoderSteps]`. 128 covers a 15 s
    /// window (max observed ~108 tokens incl. prompt) and is ~1.5× faster than 256.
    /// `CanaryManager` reads the real length from the loaded model, so this is just
    /// the contract/fallback value.
    public static let maxDecoderSteps = 128
    public static let vocabSize = 16384

    // Special token ids (the model's real decoder ids — see vocab.json).
    public static let eosId = 3  // <|endoftext|>
    public static let padId = 2  // <pad>
    public static let bosId = 4  // <|startoftranscript|>

    /// canary2 prompt for English transcribe + punctuation/capitalization:
    /// ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|en|> <|en|>
    /// <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    public static let promptEnTranscribePnc: [Int32] = [16053, 7, 4, 16, 64, 64, 5, 9, 11, 13]
}
