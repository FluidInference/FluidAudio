import Foundation

// MARK: - Configuration

/// Configuration for Nemotron 3 Diarization streaming inference (8-speaker streaming Sortformer).
///
/// Mirrors NeMo `SortformerModules` parameters for `nvidia/Nemotron-3-Diarization-preview`.
/// Latency = (chunkLen + chunkRightContext) * 80 ms.
///
/// - Important: The preview checkpoint is under an NVIDIA evaluation license. Converted CoreML
///   models are loaded from a local directory only — there is no HuggingFace download path.
public struct Nemotron3Config: Sendable {

    // MARK: Architecture (fixed by the checkpoint)

    public let numSpeakers: Int = 8
    public let preEncoderDims: Int = 512
    public let subsamplingFactor: Int = 8
    public let melFeatures: Int = 128
    public let sampleRate: Int = 16000

    /// High-resolution output upsample factor (80 ms encoder frame -> 10 ms output frames).
    public let upsampleFactor: Int = 8

    // MARK: Streaming parameters (must match the converted model's fixed shapes)

    public var chunkLen: Int
    public var chunkLeftContext: Int
    public var chunkRightContext: Int
    public var fifoLen: Int
    public var spkcacheLen: Int
    public var spkcacheUpdatePeriod: Int

    // MARK: Compression constants (NeMo model_config.yaml)

    public var silenceThreshold: Float = 0.2
    public var predScoreThreshold: Float = 0.25
    public var scoresBoostLatest: Float = 0.05
    public var strongBoostRate: Float = 0.75
    public var weakBoostRate: Float = 1.5
    public var minPosScoresRate: Float = 0.5
    public var spkcacheSilFramesPerSpk: Int = 1
    public let maxIndex: Int = 99999

    public var debugMode: Bool = false

    /// Model file name inside the models directory, e.g. `Nemotron3Diarizer_low.mlmodelc`.
    public var modelFileName: String

    /// Split-graph mode: the model contains only the pure-fp transformer+head
    /// (inputs `packed`/`attn_bias`/`output_mask`); feature stacking, the 1024->512
    /// projection, state packing, and mask construction run host-side. Requires
    /// `pre_encode_proj_t.bin` next to the model. Runs 100% ANE-resident and is not
    /// subject to the monolithic graph's chunk-length ANECCompile cliff.
    public var splitGraph: Bool = false

    // MARK: Derived

    /// Mel frames the CoreML `chunk` input expects: (lc + chunk + rc) * 8.
    public var chunkMelFrames: Int {
        (chunkLeftContext + chunkLen + chunkRightContext) * subsamplingFactor
    }

    /// Encoder frames of the chunk region (physical capacity incl. contexts).
    public var chunkEncFrames: Int {
        chunkLeftContext + chunkLen + chunkRightContext
    }

    /// Packed sequence length of the model output: spkcache + fifo + chunk regions.
    public var packedFrames: Int {
        spkcacheLen + fifoLen + chunkEncFrames
    }

    /// Output frame duration for high-resolution predictions (10 ms).
    public var outputFrameSeconds: Float { 0.01 }

    // MARK: Presets (model card recommended profiles)

    /// 30.4 s input-buffer latency, offline-style quality; highest-throughput batch profile.
    public static let offline = Nemotron3Config(
        chunkLen: 340, chunkRightContext: 40, fifoLen: 40, spkcacheUpdatePeriod: 300,
        modelFileName: "Nemotron3Diarizer_offline.mlmodelc")

    /// 1.04 s latency streaming.
    public static let low = Nemotron3Config(
        chunkLen: 9, chunkRightContext: 4, fifoLen: 264, spkcacheUpdatePeriod: 222,
        modelFileName: "Nemotron3Diarizer_low.mlmodelc")

    /// 0.64 s latency streaming.
    public static let veryLow = Nemotron3Config(
        chunkLen: 6, chunkRightContext: 2, fifoLen: 264, spkcacheUpdatePeriod: 222,
        modelFileName: "Nemotron3Diarizer_verylow.mlmodelc")

    /// 0.32 s latency streaming.
    public static let ultraLow = Nemotron3Config(
        chunkLen: 3, chunkRightContext: 1, fifoLen: 264, spkcacheUpdatePeriod: 222,
        modelFileName: "Nemotron3Diarizer_ultra.mlmodelc")

    /// 1.04 s latency with a 40-frame FIFO: packed sequence 317 vs 541 frames —
    /// substantially faster per call on ANE than `low` at a small quality cost.
    /// Not a model-card profile.
    public static let fast = Nemotron3Config(
        chunkLen: 9, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
        modelFileName: "Nemotron3Diarizer_fast.mlmodelc")

    /// 4.16 s latency, 48-frame chunk: amortizes the static spkcache+FIFO cost per call for
    /// high-throughput batch/near-live use.
    public static let efficient = Nemotron3Config(
        chunkLen: 48, chunkRightContext: 4, fifoLen: 264, spkcacheUpdatePeriod: 222,
        modelFileName: "Nemotron3Diarizer_efficient.mlmodelc")

    /// 2.24 s latency, 1.92 s audio per call at `fast`-class per-call cost —
    /// bigger chunks recover the small-FIFO quality penalty.
    public static let fast24 = Nemotron3Config(
        chunkLen: 24, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
        modelFileName: "Nemotron3Diarizer_fast24.mlmodelc")

    /// 2.88 s latency, 2.56 s audio per call — matches the card-standard `low`
    /// profile's quality at a fraction of its per-call ANE cost. Recommended default
    /// when latency up to ~3 s is acceptable.
    public static let fast32 = Nemotron3Config(
        chunkLen: 32, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
        modelFileName: "Nemotron3Diarizer_fast32.mlmodelc")

    /// 10.56 s latency, 10.24 s audio per call; largest monolithic chunk that still
    /// compiles for ANE (192 fails ANECCompile). Best quality of the streaming preset
    /// lineup. High-throughput near-live tier; for pure GPU batch prefer `.offline`.
    public static let fast128 = Nemotron3Config(
        chunkLen: 128, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
        modelFileName: "Nemotron3Diarizer_fast128.mlmodelc")

    public init(
        chunkLen: Int,
        chunkLeftContext: Int = 0,
        chunkRightContext: Int,
        fifoLen: Int,
        spkcacheLen: Int = 264,
        spkcacheUpdatePeriod: Int,
        modelFileName: String,
        splitGraph: Bool = false
    ) {
        self.chunkLen = chunkLen
        self.chunkLeftContext = chunkLeftContext
        self.chunkRightContext = chunkRightContext
        self.fifoLen = fifoLen
        self.spkcacheLen = spkcacheLen
        self.spkcacheUpdatePeriod = spkcacheUpdatePeriod
        self.modelFileName = modelFileName
        self.splitGraph = splitGraph
    }

    public static func preset(named name: String) -> Nemotron3Config? {
        // "<preset>-int8" selects the int8-quantized model file with identical parameters.
        if name.hasSuffix("-int8"), var base = preset(named: String(name.dropLast(5))) {
            base.modelFileName = base.modelFileName.replacingOccurrences(
                of: ".mlmodelc", with: "_int8.mlmodelc")
            return base
        }
        // "<name>-split" selects a split-graph model (see `splitGraph`); the underlying
        // model files use the sweep naming (s32 = fast32's shape).
        switch name {
        case "offline": return .offline
        case "low": return .low
        case "verylow": return .veryLow
        case "ultra": return .ultraLow
        case "fast": return .fast
        case "fast24": return .fast24
        case "fast32": return .fast32
        case "fast128": return .fast128
        case "efficient": return .efficient
        case "fast32-split":
            return Nemotron3Config(
                chunkLen: 32, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
                modelFileName: "Nemotron3Diarizer_s32_split.mlmodelc", splitGraph: true)
        case "fast32-split-w8a8":
            return Nemotron3Config(
                chunkLen: 32, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
                modelFileName: "Nemotron3Diarizer_s32_split_w8a8.mlmodelc", splitGraph: true)
        case "c128-split":
            return Nemotron3Config(
                chunkLen: 128, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
                modelFileName: "Nemotron3Diarizer_c128_split.mlmodelc", splitGraph: true)
        case "c128-split-w8a8":
            return Nemotron3Config(
                chunkLen: 128, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
                modelFileName: "Nemotron3Diarizer_c128_split_w8a8.mlmodelc", splitGraph: true)
        case "c192-split":
            return Nemotron3Config(
                chunkLen: 192, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
                modelFileName: "Nemotron3Diarizer_c192_split.mlmodelc", splitGraph: true)
        case "c256-split":
            return Nemotron3Config(
                chunkLen: 256, chunkRightContext: 4, fifoLen: 40, spkcacheUpdatePeriod: 40,
                modelFileName: "Nemotron3Diarizer_c256_split.mlmodelc", splitGraph: true)
        default: return nil
        }
    }
}

// MARK: - Streaming State

/// Fixed-capacity streaming state, mirroring NeMo's async `StreamingSortformerState` at batch 1.
///
/// `spkcache`/`fifo` are always full physical capacity (zero-padded past the valid length),
/// matching the CoreML model's fixed input shapes.
public struct Nemotron3StreamingState: Sendable {
    /// [spkcacheLen, 512] flattened, valid frames left-packed.
    public var spkcache: [Float]
    public var spkcacheLength: Int
    /// [spkcacheLen, 8] flattened. Meaningful only from the first compression onward.
    public var spkcachePreds: [Float]
    public var spkcacheCompressed: Bool

    /// [fifoLen, 512] flattened, valid frames left-packed.
    public var fifo: [Float]
    public var fifoLength: Int
    /// [fifoLen, 8] flattened.
    public var fifoPreds: [Float]

    public init(config: Nemotron3Config) {
        let d = config.preEncoderDims
        let s = config.numSpeakers
        self.spkcache = [Float](repeating: 0, count: config.spkcacheLen * d)
        self.spkcachePreds = [Float](repeating: 0, count: config.spkcacheLen * s)
        self.spkcacheLength = 0
        self.spkcacheCompressed = false
        self.fifo = [Float](repeating: 0, count: config.fifoLen * d)
        self.fifoPreds = [Float](repeating: 0, count: config.fifoLen * s)
        self.fifoLength = 0
    }
}

// MARK: - Results

/// Per-chunk streaming result at 10 ms resolution.
public struct Nemotron3ChunkResult: Sendable {
    /// Speaker activity probabilities for this chunk's core frames, [frames * 8] flattened,
    /// 10 ms per frame.
    public let probabilities: [Float]
    public let frameCount: Int
    public let numSpeakers: Int
}

/// A contiguous speech segment attributed to one speaker slot (arrival-ordered).
public struct Nemotron3Segment: Sendable {
    public let speakerIndex: Int
    public let startSeconds: Float
    public let endSeconds: Float
}

// MARK: - Errors

public enum Nemotron3Error: Error, LocalizedError {
    case modelLoadFailed(String)
    case inferenceFailed(String)
    case invalidState(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let m): return "Failed to load Nemotron 3 diarization model: \(m)"
        case .inferenceFailed(let m): return "Nemotron 3 diarization inference failed: \(m)"
        case .invalidState(let m): return "Invalid Nemotron 3 diarization state: \(m)"
        }
    }
}
