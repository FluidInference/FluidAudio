import Foundation

/// Constants for the PocketTTS flow-matching language model TTS backend.
public enum PocketTtsConstants {

    // MARK: - Audio

    public static let audioSampleRate: Int = 24_000
    /// Each generation step produces one frame of audio: 1920 samples = 80ms at 24kHz.
    public static let samplesPerFrame: Int = 1_920

    // MARK: - Model dimensions

    /// Audio code dimensionality — output of flow_decoder, input to mimi_decoder.
    public static let latentDim: Int = 32
    /// Transformer hidden state size — shared by flowlm_step output and flow_decoder input.
    public static let transformerDim: Int = 1024
    /// SentencePiece vocabulary size for text tokenization.
    public static let vocabSize: Int = 4001
    /// Embedding dimension for voice and text tokens (matches transformerDim).
    public static let embeddingDim: Int = 1024

    // MARK: - Generation parameters

    /// Number of Euler integration steps in flow_decoder (noise → audio code).
    ///
    /// Informational: the LSD Euler step count BAKED INTO `flow_decoder_fused`
    /// at conversion (`convert_flow_decoder_fused.py --num-steps`). NOT read at
    /// runtime — `flowDecode` calls the fused model which runs this many steps
    /// internally. The shipped v2.1 packs were converted with 8. To change it,
    /// re-convert the fused model and update this value; editing it alone has no
    /// runtime effect.
    public static let numLsdSteps: Int = 8

    /// Fixed conditioning-block length compiled into `cond_prefill.mlpackage`
    /// (`convert_cond_prefill.py --t-max`). The host pads the real voice+text
    /// block to this length and passes the true count as `valid_len`; if a
    /// block exceeds this, the prefill falls back to per-token `cond_step`.
    public static let condPrefillMaxTokens: Int = 256
    /// Controls randomness in flow_decoder: scales initial noise by sqrt(temperature).
    public static let temperature: Float = 0.7
    /// flowlm_step EOS logit threshold — above this means the model is done speaking.
    public static let eosThreshold: Float = -4.0
    public static let shortTextPadFrames: Int = 3
    public static let longTextExtraFrames: Int = 1
    public static let extraFramesAfterDetection: Int = 2
    public static let shortTextWordThreshold: Int = 5
    /// Max text tokens per chunk — keeps total KV cache usage under kvCacheMaxLen.
    public static let maxTokensPerChunk: Int = 50

    // MARK: - KV cache

    /// Max KV cache positions: voice (~125) + text (≤50) + generated frames.
    public static let kvCacheMaxLen: Int = 512

    // MARK: - Voice

    public static let defaultVoice: String = "alba"
    /// Default voice prompt length in frames. Cloned voices may differ (up to 250).
    public static let voicePromptLength: Int = 125

    // MARK: - Repository

    public static let defaultModelsSubdirectory: String = "Models"
}

/// Supported PocketTTS language packs (matches upstream
/// `kyutai/pocket-tts/languages/<id>/` folder names exactly).
///
/// All packs live under `v2/<id>/` on `FluidInference/pocket-tts-coreml`.
public enum PocketTtsLanguage: String, Sendable, CaseIterable {
    case english
    case french24L = "french_24l"
    case german
    case german24L = "german_24l"
    case italian
    case italian24L = "italian_24l"
    case portuguese
    case portuguese24L = "portuguese_24l"
    case spanish
    case spanish24L = "spanish_24l"

    /// Number of transformer layers in this language pack (6 or 24).
    public var transformerLayers: Int {
        switch self {
        case .english, .german, .italian, .portuguese, .spanish:
            return 6
        case .french24L, .german24L, .italian24L, .portuguese24L, .spanish24L:
            return 24
        }
    }

    /// HF subdirectory under the pocket-tts-coreml repo root.
    public var repoSubdirectory: String {
        // v2.1 = optimized re-conversion of v2 (fused flow decoder on ANE,
        // one-shot cond prefill, fp16 flowlm). Same weights as v2.
        "v2.1/\(rawValue)"
    }
}

/// Precision variant of the PocketTTS CoreML models.
///
/// Both variants live in the upstream `FluidInference/pocket-tts-coreml` repo
/// under `v2/<lang>/`. They share `cond_step`, `flow_decoder`, `mimi_decoder`,
/// and `constants_bin/`; only the FlowLM transformer differs — `flowlm_step`
/// (default) vs `flowlm_stepv2` (int8 weight quantization on the FlowLM
/// transformer's attention + FFN linear layers, following the recipe from
/// kyutai-labs/pocket-tts#147).
///
/// `fp16` is the upstream default — fp16 weights on disk, ~290 MB for the
/// FlowLM transformer alone (~767 MB total for the language pack). The
/// `PocketTtsModelStore` uses `.cpuAndGPU` compute units so CoreML upcasts
/// activations to fp32 at runtime, matching the Python reference
/// implementation's numerical fidelity.
///
/// `int8` replaces only `flowlm_step`'s attention + FFN linear weights with
/// int8 (per-channel scale tensors stay fp16); `cond_step`, `flow_decoder`,
/// and `mimi_decoder` keep the default fp16 weights to preserve the
/// autoregressive feedback loop's numerical fidelity. Saves ~217 MB on
/// disk for English (~767 MB → ~550 MB), no measurable WER regression in
/// upstream's evaluation.
///
/// ## Why per-submodel quantization isn't exposed yet
///
/// Upstream's `experiment/pocket-tts-int8` branch prototyped a richer API
/// (`PocketTtsQuantization` with per-submodel `PocketTtsModelPrecision`),
/// alongside this internal A/B quality data on int8 quantization
/// (English 6L, prompt-relative):
///
/// | Submodel        | speaker_sim | Pearson | Notes                                       |
/// |-----------------|-------------|---------|---------------------------------------------|
/// | cond_step       | 0.984       | 0.94    | safe                                        |
/// | flowlm_step     | 0.989       | 0.94    | safe (this is what we ship)                 |
/// | flow_decoder    | 0.981       | 0.78    | risky — LSD 8-step loop compounds error     |
/// | mimi_decoder    | 0.998       | 1.00    | transparent (per upstream's measurements)   |
///
/// Their measurements disagree with the Kyutai recipe on `mimi_decoder`:
/// Kyutai keeps it at the default precision because of the autoregressive
/// feedback loop, while upstream's A/B reports it transparent. The
/// conservative reading is to trust both — keep mimi at the default
/// precision until either source revises.
///
/// We don't expose the richer API here because the per-submodel int8 files
/// (`cond_step_int8.mlmodelc`, etc.) are **not published on HuggingFace**
/// today. Only `flowlm_stepv2.mlmodelc` is published. Adding a per-submodel
/// API would let callers request configurations that 404 at download time.
/// If upstream publishes per-submodel int8 artifacts, this enum can grow
/// into the experiment branch's `PocketTtsQuantization` shape mechanically.
public enum PocketTtsPrecision: Sendable, Hashable {
    case fp16
    case int8
}

/// Which transformer-model formulation (and compute-unit policy) to load.
///
/// The v2.1 packs ship rank-5 KV-cache graphs that the ANE compiler rejects
/// outright (`ANECCompile FAILED`), so they run on GPU. The rank-4
/// scatter-free rewrites (mobius Trials 19/20) are ANE-eligible:
/// `flowlm_step_ane` plans 100% ANE and `cond_prefill_ane` 92% under
/// `.cpuAndNeuralEngine`, bit-identical math to the rank-5 graphs.
///
/// On M-series Macs the GPU is slightly faster per call (flowlm 3.04 ms GPU
/// vs 3.68 ms ANE), so `.gpu` stays the default. `.ane` removes the GPU
/// from the synthesis path entirely — the per-frame loop becomes
/// ANE (flowlm) → ANE (flow decoder) → CPU (mimi) — for power, engine
/// parallelism with `useCrossEnginePipeline`, and iOS placement.
public enum PocketTtsModelPlacement: String, Sendable, Hashable {
    /// v2.1 rank-5 models, GPU-placed (default).
    case gpu
    /// Rank-4 ANE-eligible models (`flowlm_step_ane` + `cond_prefill_ane`).
    /// Requires the `*_ane.mlmodelc` files in the language pack (fp16 only;
    /// `precision` is ignored for the FlowLM when this is selected).
    case ane
    /// MLState multifunction pipeline (mobius Trial 23): ONE
    /// `pocket_state.mlmodelc` package exposing `write_state` / `prefill` /
    /// `generate` functions over a shared 12-buffer fp16 KV state that stays
    /// resident on the ANE — the 24-tensor cache I/O of `.gpu`/`.ane` is
    /// bypassed entirely, and the per-frame `generate` call fuses
    /// flowlm_step + flow_decoder into one dispatch.
    ///
    /// Requires macOS 15+/iOS 18+ at RUNTIME (both `MLState` and
    /// multifunction models); `loadIfNeeded` throws on older OSes. fp16 only
    /// (`precision` is ignored). Python-measured −35.8%/utterance on the
    /// flowlm+flow share vs the `.ane` IO pipeline (mobius TRIALS.md,
    /// Trial 23); mimi decode is unchanged.
    case aneState = "ane-state"
}
