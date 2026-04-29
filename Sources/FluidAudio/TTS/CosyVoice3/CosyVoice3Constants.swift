import Foundation

/// Central constants for the CosyVoice3 (Mandarin) CoreML pipeline.
///
/// Shipping config (frozen):
/// - LLM-Prefill-T256-M768-fp16           (cpuAndNeuralEngine)
/// - LLM-Decode-M768-fp16                 (cpuAndGPU — see note)
/// - Flow-N250-fp16                       (cpuAndGPU — an ANE-port
///   BC1S rewrite was attempted and reverted: the converted graph ran
///   ~3× faster but numerically broken (mel dynamic range collapsed
///   from [-12.5, +5.2] to [-10.1, -0.8], MAE 2.58 vs fp32 reference,
///   yielding HiFT audio at ~40× lower peak amplitude → unintelligible
///   to ASR). See `coreml/TRIALS_AND_ERRORS.md` "Flow ANE port" for
///   the full journey, including the residual 77-op CPU island in
///   `input_embed.conv_pos_embed` (`Conv1d(1024,1024,k=31)+Mish`)
///   that three rewrite attempts couldn't move — ANEF rejects the
///   conv footprint regardless of group count.)
/// - HiFT-T500-fp16                       (cpuAndGPU — pinned off
///   ANE because the `.cpuAndNeuralEngine` planner left at least one
///   op on the BNNS CPU path, which tripped a hard async-dispatch
///   watchdog mid-corpus on long phrases:
///   `E5RT: Submit Async failed ... HiFT-T500-fp16_main__Op104_BnnsCpuInference
///   has timed out`. GPU placement is deterministic and avoids the
///   ANE+BNNS mixed-compute pathology.)
///
/// Decode runs **stateless** with an external KV cache: prefill emits
/// `kv_k` / `kv_v` of shape `[24, 1, 2, 768, 64]` fp32, and decode
/// accepts the same tensors as inputs and returns `kv_k_out` / `kv_v_out`
/// at the same shape/dtype. The cache is round-tripped once per step
/// (≈18 MB total). ANE still rejects this graph (`MILCompilerForANE
/// ANECCompile() FAILED` on the rotary + sliced SDPA), so decode is
/// pinned to `.cpuAndGPU`. The library floor is macOS 14 / iOS 17 — no
/// MLState dependency.
public enum CosyVoice3Constants {

    // MARK: - LLM shapes
    public static let prefillLength = 256
    public static let kvMaxLength = 768
    public static let embedDim = 896
    public static let numLayers = 24
    public static let kvHeads = 2
    public static let headDim = 64

    // MARK: - Flow / HiFT shapes
    public static let flowTotalTokens = 250
    public static let tokenMelRatio = 2
    public static let hiftMaxFrames = 500
    public static let hiftSamplesPerFrame = 480
    public static let sampleRate = 24_000
    public static let melBins = 80
    public static let speakerEmbeddingDim = 192

    // MARK: - Speech token vocab
    public static let speechVocab = 6_761
    public static let speechTokenSize = 6_561
    public static let sosId: Int32 = 6_561
    public static let eosId: Int32 = 6_562
    public static let taskId: Int32 = 6_563
    /// Any token id in this range is treated as a stop signal.
    public static let stopRange: ClosedRange<Int32> = 6_561...6_760

    // MARK: - Sampler
    public static let topP: Float = 0.8
    public static let topK: Int = 25
    public static let rasWindow: Int = 10
    public static let rasTauR: Float = 0.1

    // MARK: - Cache layout
    /// Subdirectory under the shared `~/.cache/fluidaudio/` (or iOS Caches) dir
    /// where every TTS backend stores its HF-mirrored models.
    public static let defaultModelsSubdirectory = "Models"

    // MARK: - Files (local build dir layout)
    public enum Files {
        public static let llmPrefill = "LLM-Prefill-T256-M768-fp16.mlpackage"
        public static let llmPrefillSubdir = "llm-fp16"
        public static let llmDecode = "LLM-Decode-M768-fp16.mlpackage"
        public static let llmDecodeSubdir = "llm-fp16-decode"
        public static let flow = "Flow-N250-fp16.mlpackage"
        public static let flowSubdir = "flow-fp16-n250"
        public static let hift = "HiFT-T500-fp16.mlpackage"
        public static let hiftSubdir = "hift-fp16-t500"
        public static let speechEmbeddings = "speech_embedding-fp16.safetensors"
        public static let speechEmbeddingsSubdir = "embeddings"
    }
}
