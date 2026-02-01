# PocketTTS: Lightweight On-Device Text-to-Speech

## Overview

PocketTTS is a lightweight, high-quality text-to-speech backend designed for on-device inference on Apple platforms. It uses a flow-matching language model architecture to generate natural-sounding speech autoregressively at 24kHz. Each generation step produces an 80ms audio frame (1920 samples).

## Pipeline

```
Text Input
  |
  v
SentencePiece tokenizer --> token IDs --> embedding table --> [T, 1024]
Voice safetensors --> [125, 1024]
  |
  v
COND_STEP (prefill, ~141 calls)
  Processes voice tokens (0..124), then text tokens
  Fills KV caches for the transformer
  |
  v  KV caches + positions
FLOWLM_STEP (autoregressive, 20-50 calls per chunk)
  Generates transformer_out [1, 1024] + EOS logit
  Stops when EOS logit > -4.0
  |
  v  transformer_out
FLOW_DECODER (8 Euler steps per generation step)
  Denoises random latent via iterative velocity field
  Produces latent [32]
  |
  v  latent --> denormalize --> quantize [512]
MIMI_DECODER (streaming, 1 call per generation step)
  Converts quantized latent to audio frame [1920 samples]
  23 streaming state tensors maintained across frames
  |
  v
Audio frames --> concatenate --> WAV (24kHz, 16-bit mono)
```

## Model Architecture

### cond_step (KV Cache Prefill)

Processes one conditioning token (voice or text) and updates the transformer KV cache.

**Inputs:**
- `conditioning` [1, 1, 1024] — single embedded token
- `cache0`-`cache5` [2, 1, 512, 16, 64] — KV cache per layer
- `position0`-`position5` [1] — write position per layer

**Outputs:**
- Updated KV caches and positions

**Architecture:**
- 6 transformer layers, each:
  - LayerNorm(1024)
  - RoPE streaming attention (16 heads, 64 head dim, circular buffer of 512 positions)
  - Residual connection
  - LayerNorm(1024)
  - FFN: Linear(1024, 4096, GELU) -> Linear(4096, 1024)
  - Residual connection
- Final LayerNorm(1024)

**Prefill order is critical:** voice tokens first (positions 0..124), then text tokens. Reversing order degrades KV cache cosine similarity to 0.6-0.9.

---

### flowlm_step (Autoregressive Generation)

Generates one transformer hidden state from the cached conditioning and previous latent.

**Inputs:**
- `sequence` [1, 1, 32] — previous latent (NaN for first step = BOS)
- `bos_emb` [32] — BOS embedding constant
- `cache0`-`cache5` + `position0`-`position5` — from cond_step prefill

**Outputs:**
- `transformer_out` [1, 1, 1024] — hidden state (CoreML key: `"input"`)
- `eos_logit` [1] — stop signal (CoreML key: `"var_2582"`)
- Updated KV caches and positions

**Architecture:**
- Linear(32, 1024) input projection
- Same 6-layer transformer as cond_step (shared architecture)
- LayerNorm(1024) output norm
- Linear(1024, 1) EOS head

**EOS detection:** When `eos_logit > -4.0`, generation continues for a few extra frames (3 for short text, 1 for long text) then stops.

---

### flow_decoder (Lagrangian Self-Distillation)

Converts transformer output to a 32-dimensional audio latent via 8 iterative denoising steps (Euler integration from t=0 to t=1).

**Inputs:**
- `transformer_out` [1, 1024] — from flowlm_step (squeezed)
- `latent` [1, 32] — starts as `randn * sqrt(temperature)`
- `s` [1, 1] — start time
- `t` [1, 1] — end time

**Outputs:**
- `velocity` [1, 32] — flow direction

**Architecture:**
- 2 TimestepEmbedders (sinusoidal frequencies -> MLP with RMSNorm)
- Linear(32, flow_dim) input projection
- Linear(1024, flow_dim) condition embedding
- 2-4 AdaLN residual blocks (LayerNorm + MLP + SiLU + adaptive modulation)
- Linear(flow_dim, 32) output

**Euler integration loop (called 8 times per generation step):**
```
dt = 1/8 = 0.125
latent starts as randn(32) * sqrt(temperature)
for step in 0..<8:
    velocity = flow_decoder(transformer_out, latent, s=step*dt, t=(step+1)*dt)
    latent += velocity * dt
```

**Stateless** — no caches, no feedback tensors.

---

### mimi_decoder (Streaming Audio Synthesis)

Converts a quantized latent to 1920 audio samples (80ms at 24kHz). Maintains 23 streaming state tensors across frames.

**Inputs:**
- `latent` [1, 512, 1] — quantized latent
- 23 state tensors (attention caches, convolution history, partial buffers)

**Outputs:**
- `audio` [1, 1, 1920] — audio frame (CoreML key: `"var_1445"`)
- 23 updated state tensors

**Architecture:**
```
[1, 512, 1]
  |
  v  ConvTranspose 16x upsampling
[1, 512, 32]
  |
  v  2x streaming attention (8 heads, 256 context, RoPE)
[1, 512, 32]
  |
  v  Conv0 [512->256] + ResBlock
  v  ConvTranspose [256->256]     \
  v  ResBlock [256->128]           |  progressive
  v  ConvTranspose [128->128]      |  upsampling
  v  ResBlock [128->64]            |
  v  ConvTranspose [64->64]       /
  v  Final Conv [64->1]
[1, 1, 1920]
```

**Streaming state tensors (23 after stripping 3 zero-length):**

| Category | Tensors | Purpose |
|----------|---------|---------|
| Attention caches | `attn0_cache` [2,1,8,256,64], `attn1_cache` [2,1,8,256,64] | KV cache for 2 transformer layers |
| Attention offsets | `attn0_offset`, `attn0_end_offset`, `attn1_offset`, `attn1_end_offset` | Circular buffer position tracking |
| Upsampling | `upsample_partial` [1,512,16] | Partial ConvTranspose buffer |
| Convolution history | `conv0_prev`, `res*_conv*_prev`, `conv_final_prev` | Causal conv input history |
| First-frame flags | `conv0_first`, `res*_conv*_first`, `conv_final_first` | Boundary condition handling |
| ConvTranspose partials | `convtr0_partial`, `convtr1_partial`, `convtr2_partial` | Overlap-add buffers |

**Mimi state is continuous across text chunks** — only KV caches are reset per chunk.

---

## Post-Processing

After generation, the latent undergoes two transforms before entering Mimi:

1. **Denormalization:** `latent_denorm = latent * emb_std + emb_mean` (element-wise, 32 dims)
2. **Quantization:** `quantized = matmul(latent_denorm, quantizer_weight.T)` ([32] -> [512])

Audio post-processing (optional):
- De-essing: -3dB reduction of high-frequency sibilance

No peak normalization — output preserves natural levels to match MLX reference.

## Text Chunking

Long text is split into chunks of <=50 tokens to stay within the KV cache capacity (512 positions, minus ~125 voice tokens and ~25 overhead).

Splitting priority:
1. Sentence boundaries (`.!?`)
2. Clause boundaries (`,;:`)
3. Word boundaries (fallback)

Each chunk gets its own KV cache prefill, but Mimi streaming state is continuous across chunks for seamless audio.

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| audioSampleRate | 24,000 Hz | Output sample rate |
| samplesPerFrame | 1,920 | Samples per 80ms frame |
| latentDim | 32 | Flow latent dimension |
| transformerDim | 1,024 | Transformer hidden dimension |
| quantizerOutDim | 512 | Mimi input dimension |
| vocabSize | 4,001 | SentencePiece vocabulary |
| embeddingDim | 1,024 | Token embedding dimension |
| kvCacheLayers | 6 | Transformer layers |
| kvCacheMaxLen | 512 | Circular buffer positions |
| numLsdSteps | 8 | Flow decoder Euler steps |
| temperature | 0.7 | Generation temperature |
| eosThreshold | -4.0 | EOS detection threshold |
| voicePromptLength | 125 | Voice conditioning tokens |

## Model Summary

| Model | Params | Size | Calls/utterance | Stateful |
|-------|--------|------|-----------------|----------|
| cond_step | ~50M | ~200MB | ~141 (prefill) | KV cache |
| flowlm_step | ~50M | ~200MB | 20-50 | KV cache |
| flow_decoder | ~50M | ~190MB | 160-400 (8x/step) | No |
| mimi_decoder | ~5M | ~11MB | 20-50 | 23 tensors |
| **Total** | **~155M** | **~601MB** | | |

## CoreML Compute Units

All models use `.cpuAndGPU` to avoid ANE float16 precision loss. The ANE processes in native float16 which causes audible beeping artifacts in the Mimi decoder's streaming state feedback loop — small quantization errors compound across frames in overlap-add buffers and attention caches. CPU/GPU compute in float32 matches the Python reference.

## Known Issues

See `IOS_COREML_ISSUES.md` in the mobius repo for the full list. Key issues:
- **ANE beeping (resolved):** Float16 precision loss in Mimi state feedback. Fixed with `.cpuAndGPU`.
- **Zero-length tensors (resolved):** 3 zero-length state tensors crash CoreML Espresso. Fixed by stripping from model.
- **Voice-dependent duration drift (open):** Some voices produce different durations on CoreML vs MLX (up to +/-2s) due to EOS detection sensitivity to floating-point differences between backends.

## Available Voices

Voice conditioning data is stored as `<voice>_audio_prompt.bin` (125 * 1024 float32 values):
- `alba` (default)
- `azelma`
- `cosette`
- `javert`

## Usage

```swift
import FluidAudioTTS

let manager = PocketTtsManager()
try await manager.initialize()

// Simple synthesis
let audioData = try await manager.synthesize(text: "Hello, world!")

// With options
let audioData = try await manager.synthesize(
    text: "Hello, world!",
    voice: "cosette",
    temperature: 0.7,
    deEss: true
)

// Write directly to file
try await manager.synthesizeToFile(
    text: "Hello, world!",
    outputURL: URL(fileURLWithPath: "/tmp/output.wav")
)
```

## How It Works (Conceptual)

### Text In, Audio Out

PocketTTS takes raw text and a voice identity, then generates audio 80ms at a time:

1. **Tokenize** — SentencePiece splits text into subword tokens (no phonemizer or espeak needed)
2. **Embed** — Each token ID is an index into a precomputed table. Row 42 = a 1024-float vector representing that token. Just an array lookup, no neural network.
3. **Prefill** — Voice tokens (125) and text tokens are fed one-by-one into the transformer, filling the KV cache. This cache stores pre-computed key/value pairs so the transformer doesn't have to reprocess all previous tokens every step. After ~141 calls, the cache holds the full context: "what voice to use" and "what text to say."
4. **Generate** — Each frame starts with 32 random floats (noise). The pipeline shapes them into audio matching the voice and continuing from previous frames:
   - **flowlm_step** asks "what should this frame sound like?" by attending to the KV cache (voice + text + all prior frames). Also outputs an EOS signal for when to stop.
   - **flow_decoder** (x8 steps) iteratively refines the random noise into a clean 32-dim latent, guided by the transformer's hidden state. Each step predicts "which direction to push" and nudges the latent.
   - **denormalize + quantize** — two simple math ops (scale + matrix multiply) to convert [32] → [512] in the format Mimi expects.
   - **mimi_decoder** synthesizes 1920 audio samples from the quantized code. Its 23 streaming state tensors carry convolution history across frames for seamless audio.
5. **Output** — Frames concatenate into a WAV file (24kHz mono)

The randomness is the starting point, not the output. Two runs with different seeds produce slightly different but equally valid pronunciations — like how a person never says the same sentence exactly the same way twice.

### How It Differs From Kokoro

Both follow the same high-level pattern: voice conditioning + text → flow matching → audio codec → waveform. The key difference is where the autoregressive loop lives.

**Kokoro** generates the entire audio representation in one pass (all frames at once). The flow matching plans everything, then a vocoder (Vocos) converts the full spectrogram to audio. It's plan-everything-then-speak.

**PocketTTS** generates one frame at a time. The transformer (flowlm_step) is autoregressive — each frame is conditioned on all previous frames via the KV cache. The flow matching runs *inside* each step to refine that single frame. It's think-one-step-speak-one-step.

| | Kokoro | PocketTTS |
|---|---|---|
| Text input | Phonemes (IPA via espeak) | Raw text (SentencePiece) |
| Voice conditioning | Style embedding vector | 125 audio prompt tokens |
| Generation | All frames at once | Frame-by-frame autoregressive |
| Flow matching target | Mel spectrogram | 32-dim latent per frame |
| Audio synthesis | Vocos vocoder | Mimi streaming codec |
| Latency to first audio | Must wait for full plan | ~80ms after prefill |

PocketTTS is not "Kokoro but streaming." The generation strategy is fundamentally different — Kokoro parallelizes across time (fast total, but must wait), PocketTTS is sequential across time (slower total, but audio starts immediately).

### What PocketTTS Does and Doesn't Do

**Does:** Text-to-speech with 4 pre-encoded voice identities (alba, azelma, cosette, javert). No phonemizer dependency. Streaming frame-by-frame generation suitable for real-time playback.

**Doesn't:** Runtime voice cloning. The voices are pre-encoded audio prompts shipped with the model. Adding a new voice requires encoding reference audio with the Mimi encoder (not included in the CoreML pipeline). The Moshi architecture supports bidirectional dialogue, but PocketTTS only uses the generation half.

### Research Lineage

PocketTTS is based on the **Kyutai Moshi** architecture — a real-time bidirectional speech dialogue model. It adapts the Moshi generation pipeline and Mimi neural codec for TTS. This is distinct from the Kokoro/StyleTTS/Matcha-TTS lineage which uses mel spectrograms and vocoders.

## License

The original PocketTTS weights by [Kyutai](https://huggingface.co/kyutai/pocket-tts) are licensed under **CC-BY-4.0**. The CoreML conversion at [FluidInference/pocket-tts-coreml](https://huggingface.co/FluidInference/pocket-tts-coreml) inherits this license. Attribution to Kyutai is required.

Voice cloning weights are gated separately by Kyutai and are **not included** in the CoreML package — only the 4 pre-encoded voices (alba, azelma, cosette, javert) are shipped.

## References

- **PocketTTS:** [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
- **Flow Matching:** Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (ICLR 2023)
- **Mimi Codec:** Defossez et al., [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037) (2024)
- **Conversion Scripts:** [FluidInference/mobius](https://github.com/FluidInference/mobius) `models/tts/pocket_tts/`
- **CoreML Models:** [FluidInference/pocket-tts-coreml](https://huggingface.co/FluidInference/pocket-tts-coreml)
