# FluidAudio Architecture

---

## 1. Cross-Cutting Patterns

These patterns appear in every module. Understanding them once explains
most of the code shape you'll see across ASR, TTS, VAD, and Diarization.

### 1.1 Actor-Based Model Stores

Every component that owns CoreML models exposes them through an `actor`
(or, occasionally, a class that delegates mutable state to an inner actor).

- ASR: `public actor AsrManager` — `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/AsrManager.swift:6`
- ASR: `public actor Qwen3AsrManager` — `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrManager.swift:21`
- TTS: `public actor PocketTtsModelStore` — `Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsModelStore.swift:12`
- TTS: `public actor MultilingualG2PModel` — `Sources/FluidAudio/TTS/G2P/MultilingualG2PModel.swift:11`
- VAD: `public actor VadManager` — `Sources/FluidAudio/VAD/VadManager.swift:14`
- Diarization: `actor SpeakerManager` — `Sources/FluidAudio/Diarizer/Clustering/SpeakerManager.swift:13`

Why actors and not `DispatchQueue` + locks?

1. CoreML's `MLModel` load and prediction APIs are async but **not
   reentrant** concurrent calls can corrupt internal scratch buffers.
   Actor isolation serializes access without manual locking.
2. We never use `@unchecked Sendable` (see project guidelines). Actors are
   the canonical Swift 6-safe way to share mutable model state.
3. Real bug we hit: a previous `DispatchQueue` implementation in the
   diarizer triggered libmalloc heap corruption via concurrent
   copy-on-write `[Float]` mutation. Moving `SpeakerManager` to an actor
   eliminated it entirely.

The class-with-inner-actor variant (`DiarizerManager`,
`OfflineDiarizerManager`) is used when the public API needs to be a class
for ergonomic reasons but only a slice of state actually mutates. The
mutating slice goes into an actor (`SpeakerManager`); the immutable models
sit on the class as `nonisolated(unsafe)` read-only references.

### 1.2 AsyncStream / AsyncThrowingStream for Long-Running Work

Streaming inference uses Swift concurrency streams instead of delegate
protocols or callbacks.

- ASR sliding-window updates: `AsyncStream<SlidingWindowTranscriptionUpdate>`
  (non-throwing — chunk failure resets state, doesn't fatal-fail the run)
  — `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/SlidingWindowAsrManager.swift:217`
- ASR progress: `AsyncThrowingStream<Double, Error>` (throwing — long-form
  transcription must surface errors)
- TTS streaming synthesis (PocketTTS, Magpie):
  `AsyncThrowingStream<AudioFrame, Error>`
- Offline diarization: `AsyncThrowingStream<SegmentationChunk, Error>`
  drives concurrent segmentation/embedding stages
- VAD: deliberately **not** an AsyncStream — see §4.2.

Throwing vs non-throwing is a real semantic choice: throwing if a
mid-stream failure should terminate the whole run, non-throwing if the
stream represents incremental updates that survive transient errors.

### 1.3 CoreML Compute Unit Selection

`MLComputeUnits` is set per model — never globally. The choice is driven
by **measured precision**, not preference.

| Model | Compute Units | Why |
|-------|---------------|-----|
| Parakeet TDT (ASR) | `.cpuAndNeuralEngine` | TDT blank-skipping is FP32-tolerant; ANE wins on throughput |
| Kokoro TTS (single-graph) | `.all` (iOS 26+: `.cpuAndGPU`) | iOS 26 ANE compiler regressed; `.cpuAndGPU` is the workaround |
| KokoroAne (7-stage split) | Per-stage tuning | Albert/Alignment/Vocoder on ANE; Prosody/Noise/Tail on `.all` |
| PocketTTS (4 models) | `.cpuAndGPU` (forced) | Mimi decoder's streaming state feedback loop is FP16-sensitive; ANE introduces audible artifacts |
| CosyVoice3 Flow | `.cpuAndGPU` (forced) | Fused `layer_norm` produces NaNs on ANE (CoreMLTools limitation) |
| Magpie | `.cpuAndNeuralEngine` | Throughput priority; experimental |
| VAD (Silero) | `.cpuAndNeuralEngine` (default) | LSTM is small; ANE eliminates GPU contention |
| Diarization | `.all` (CI: `.cpuAndNeuralEngine`) | Segmentation + embedding both ANE-friendly |

The TTS group is the most heterogeneous because TTS models vary the most
in numerical sensitivity. ASR and VAD converge on ANE; only TTS routinely
forces CPU+GPU for precision.

Defaults live in `Sources/FluidAudio/Shared/MLModelConfigurationUtils.swift`.

### 1.4 Pure Swift vs CoreML

The dividing line is **per-element vs tensor-throughput**.

Pure Swift:
- Tokenizers (SentencePiece, BPE, byte-level) — branchy state machines,
  not amenable to GPU/ANE parallelism
- G2P preprocessing (text normalization, IPA conversion outside the model)
- SSML parsing
- Audio post-processing (de-essing biquad, padding, fade-in/out)
- ODE / Euler integration glue between CoreML calls (PocketTTS,
  CosyVoice3 Flow)
- Embedding lookups (`text_embed_table.bin` flat-file index — Qwen3 also
  does this to eliminate one CoreML graph)
- Clustering thresholds and state machines (VAD hysteresis, online
  diarizer cosine assignment)

CoreML:
- Acoustic encoders, language-model decoders, vocoders, neural codecs
- Segmentation (frame-level multi-class) and embedding (256-d vector)
  models
- Anything that's a transformer block, conv stack, or LSTM with
  meaningful FLOPs

The `FastClusterWrapper` C++17 binding is the one exception: clustering
linkage isn't compute-heavy in the CoreML sense, but the existing
`fastcluster` library is ~10× faster than a from-scratch Swift port, so
we wrap it.

---

## 2. ASR

ASR is the most internally diverse module because it hosts three model
families with three different decoders. The shape comes from one core
constraint: **each model family ships with its own streaming story**, and
we expose them through compatible-but-not-identical entry points instead
of forcing a single abstraction.

### 2.1 Three Families, Three Decoders

| Family | Decoder | Frontend | Streaming model |
|--------|---------|----------|-----------------|
| Parakeet TDT (`ASR/Parakeet/`) | Token-and-Duration Transducer | `Preprocessor.mlmodelc` (mel + CMVN inside CoreML) | Stateless encoder + sliding window |
| Qwen3 (`ASR/Qwen3/`) | Qwen2 LM, autoregressive | Whisper mel spectrogram (Swift) | Single-shot |
| Cohere (`ASR/Cohere/`) | Encoder-decoder + repetition penalty | FilterbankFeatures-compatible mel | Single-shot |

TDT (Parakeet) deserves its own note. RNN-T predicts a token per frame;
TDT additionally predicts a **duration** ("skip the next N frames"). The
decoder loop hits a fast inner branch on blank/silence and skips LSTM
updates entirely. This is the source of the 2-3× ANE speedup vs vanilla
RNN-T.
Reference: `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/Decoder/TdtDecoderV3.swift:1`.

### 2.2 Sliding Window, Not Stateful Streaming

Parakeet is a **non-streaming** encoder it expects bounded chunks.
Real-time use is built on top via overlapping windows
(`SlidingWindowAsrManager`):

- 15.0s chunk, 10.0s left context, 2.0s right context (default)
- 11.0s chunk for the latency-optimized variant
- Two-tier transcript state: `volatileTranscript` (unconfirmed, may
  change as more audio arrives) vs `confirmedTranscript` (locked at 0.85
  confidence + ≥10s context)

Files: `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/SlidingWindowAsrManager.swift:678`.

Why this shape instead of a stateful streaming encoder:

1. The CoreML conversion of stateful encoders (with cache-aware
   attention) is materially harder than converting an offline encoder.
2. Sliding window gets us pseudo-streaming for free. The cost is some
   compute redundancy in the overlap; the win is no encoder-side cache
   bookkeeping.
3. Other model families (Nemotron EOU, true streaming) plug into the
   same package via the streaming protocol in `ASR/Parakeet/Streaming/`
   when they're available.

### 2.3 Why Chunking is 14.88 s, Not 15.0 s

The Parakeet CoreML encoder is traced with a fixed input of `240_000`
samples (`ASRConstants.maxModelSamples`, 15.0 s at 16 kHz). That's the
hard cap; the *actual* per-chunk audio length is shorter, because two
margins have to come out of the 15 s budget before the chunk is rounded
to the encoder's frame stride.

Walking the math in `ChunkProcessor.chunkSamples`
(`Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/ChunkProcessor.swift:33`):

```
maxModelSamples       = 240_000             // encoder hard cap (15.0 s)
melContextSamples     =   1_280             // 80 ms = 1 encoder frame
melHopSize            =     160             // 10 ms STFT hop
samplesPerEncoderFrame=   1_280             // encoder frame stride

maxActualChunk = maxModelSamples - melContextSamples   = 238_720
raw            = maxActualChunk  - melHopSize          = 238_560
chunkSamples   = floor(raw / 1_280) * 1_280            = 238_080
                                                       → 14.88 s
```

Each subtraction exists for a distinct reason:

1. **`−melContextSamples` (1 280, 80 ms).** Every non-first chunk gets
   80 ms of left context **prepended** before being passed to the
   encoder. The mel-spectrogram STFT window and the FastConformer
   encoder's depthwise convolutions need lookback; without it, the
   leading frames produce features that cause all-blank predictions
   (`ChunkProcessor.swift:26-29`). That 80 ms has to fit *inside* the
   240 000-sample budget, so it's reserved out of `chunkSamples` up
   front: `chunkSamples + melContextSamples ≤ maxModelSamples`.
2. **`−melHopSize` (160, 10 ms).** One STFT hop of slack on the right
   edge so the trailing mel frame doesn't clip a partial window.
3. **`floor(... / 1280) * 1280`.** Round the result down to a whole
   encoder frame. The encoder's 8× subsampling means non-frame-aligned
   inputs would leave a partial trailing frame, complicating decoder
   bookkeeping for no benefit.

The 2.0 s overlap (`overlapSeconds`) is computed against `chunkSamples`
and capped at half of it, again rounded down to an encoder frame
(`ChunkProcessor.swift:40`).

> **Stale figure: 14.96 s.** Several docs and a comment on
> `ChunkProcessor.swift:22` reference a chunk size of `~14.96 s`
> (= 187 × 1 280 = 239 360 samples). That was the value *before*
> `melContextSamples` was reserved out of the 15 s budget: the old
> formula was `floor((240_000 − 160) / 1280) * 1280 = 239_360`.
> Today the same formula yields 238 080 (14.88 s) because the
> 80 ms context reservation now lives inside the 15 s envelope. The
> stale references are tracked in `CLAUDE.md:175`,
> `Documentation/API.md:279`, and
> `Documentation/ASR/TDT-CTC-110M.md:113, 154, 298`.

### 2.4 Qwen3 Eliminates an Embedding CoreML Graph

`Qwen3AsrManager` does not load a separate embedding model. Token IDs
index a flat-file embedding table from Swift, then feed the decoder.
Same trick as PocketTTS uses for `text_embed_table.bin`. Eliminating one
CoreML graph removes one async load and one ANE warm-up.
File: `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrManager.swift:8`.

---

## 3. TTS

TTS is structurally **the most diverse** module. There is **no unified
`TtsBackend` protocol** each model has its own manager, its own
pipeline shape, and its own compute-unit profile.

### 3.1 Why No Unified TTS Backend

We tried. Each TTS family has fundamentally different I/O contracts:

- **Kokoro** is one CoreML graph, one prediction call.
- **KokoroAne** is seven graphs, one per stage, all called sequentially.
- **PocketTTS** is four graphs, one of which (cond_step) runs once and
  three of which (flowlm_step + flow_decoder + mimi_decoder) run inside
  a per-frame loop, with one of those (flow_decoder) inside a fixed
  8-iteration inner loop.
- **CosyVoice3** is a 5-stage diffusion pipeline (LLM prefill → decode →
  flow → vocoder).
- **Magpie** is a 4-graph autoregressive pipeline plus a Swift-side
  1-layer "local transformer" to sample 8 codebook tokens per frame.
- **StyleTTS2** is a 4-stage diffusion pipeline (text predictor →
  ADPM2 diffusion → F0/energy → HiFi-GAN decoder).

A protocol that accepted all of them would either expose a useless
lowest-common-denominator (`func synth(text:) -> [Float]`) or be a fat
union of every quirk. We pay the price of duplication to keep each
manager honest about its own pipeline.

### 3.2 The PocketTTS Reference Pipeline

PocketTTS is the canonical "multi-stage autoregressive TTS with neural
codec" implementation in this repo. The architecture is documented in
detail in `Documentation/TTS/PocketTTS.md` and PocketTTS-specific files,
but the high-level shape is worth understanding because other models
echo pieces of it:

```
text
  ↓ SentencePiece tokenizer (Swift)
token IDs
  ↓ text_embed_table.bin lookup (Swift)
[N × 1024] embeddings
  ↓ cond_step.mlmodelc (CoreML, ONE call — KV cache prefill)
KV cache [layers × 2 × 1 × 512 × 16 × 64]
  ↓ outer loop until EOS (each iteration = one 80 ms audio frame):
  │   flowlm_step.mlmodelc (transformer step + EOS classifier)
  │   ↓
  │   inner loop ×8 (Euler ODE integration):
  │     flow_decoder.mlmodelc (velocity prediction)
  │   ↓
  │   mimi_decoder.mlmodelc (32-d latent → 1920 audio samples + 23 streaming states)
  ↓
PCM frames (24 kHz)
```

The two key shape choices:

1. **cond_step is split out** because the prefill is a different graph
   than the per-step generator. Prefill processes all input embeddings
   in one shot and strips the LM/EOS heads (we don't need them yet).
   Generation processes one token at a time and keeps the heads.
2. **Frame count is emergent**. The outer loop is `for _ in 0..<maxGenLen
   { ... break }` syntactically but semantically a `while`. EOS is
   decided per-frame by `flowlm_step`'s classifier. `maxGenLen` is a
   safety cap, not a target.

### 3.3 Kokoro vs KokoroAne — the ANE Split

These are the same model, packaged two ways:

- **Kokoro** ships as one CoreML graph. Easy to load, runs on `.all` or
  `.cpuAndGPU`, supports multi-voice and arbitrarily long text via
  Swift-side chunking.
- **KokoroAne** splits the graph into 7 stages. ANE prefers smaller
  ops; split models stay resident on ANE between calls and bypass the
  graph-too-big compilation issues. Trade: single voice (`af_heart`),
  no chunking (≤512 IPA tokens per call), no custom lexicon.

This is a recurring tension across TTS: ANE is fastest **when it works**,
but compiler regressions and FP16 precision issues mean we ship CPU+GPU
fallbacks for anything with streaming state feedback or fused norms.
Files: `Sources/FluidAudio/TTS/KokoroAne/Pipeline/KokoroAneModelStore.swift:9`.

### 3.4 G2P and SSML Are Pure Swift

`MultilingualG2PModel` (CharsiuG2P ByT5) wraps a CoreML model, but the
surrounding G2P plumbing is pure Swift. SSML parsing (`SSMLProcessor`,
`SSMLTagParser`, `SayAsInterpreter`) is entirely Swift — no model
involvement. These pieces are shared across TTS models that opt into
phoneme input.

---

## 4. VAD

VAD is the smallest module by line count and arguably the most carefully
shaped, because the whole point is **low latency with stateful LSTM**.

### 4.1 Single-Model, Stateful LSTM

Silero VAD ships as one CoreML model that encapsulates the entire
network. The interesting structure is the **state contract**:

- Input per call: 4096 new samples + 64 samples of context (256 ms +
  4 ms at 16 kHz)
- Hidden state: 128-d
- Cell state: 128-d
- Output: one probability + new (hidden, cell, context)

The 64-sample context is deliberate — it's the minimum LSTM
"carry-forward" needed without keeping the full audio history.
File: `Sources/FluidAudio/VAD/VadManager.swift:21`.

### 4.2 Why VAD Doesn't Use AsyncStream

Every other long-running component uses `AsyncStream`. VAD doesn't.
Instead it exposes a synchronous-feeling
`processStreamingChunk(_:state:)` that takes a `VadStreamState` and
returns `(VadStreamResult, VadStreamState)`.

Why: the caller already owns the audio source (a CAF file, a microphone
tap, an ASR pipeline), and VAD is most often used as a **filter** in a
larger stream the caller is already managing. Wrapping VAD in its own
AsyncStream would force the caller to bridge two streams. The state-in,
state-out shape composes cleanly inside whatever loop the caller has.
Files: `Sources/FluidAudio/VAD/VadManager+Streaming.swift:11`.

### 4.3 Hysteresis State Machine, Not a Bare Threshold

Speech segmentation uses a small state machine
(`VadManager+SpeechSegmentation.swift`):

- Enter speech when `prob ≥ threshold`
- Exit speech when `prob < threshold − negativeOffset` **and** silence
  duration ≥ `minSilenceDuration`
- Optional max-length splitting at silence gaps if a segment exceeds
  `maxSpeechDuration` (default 14 s — sized for ASR chunk boundaries)
- Pre/post-padding (default 0.1 s) to avoid clipping context words

The defaults are tuned for downstream ASR, not for raw VAD recall. A
caller doing VAD-only analysis (e.g., voice typing UX) typically tightens
the thresholds.

---

## 5. Diarization

Diarization ships as **two separate managers** for one reason: real-time
speaker tracking and offline best-accuracy diarization want fundamentally
different clustering algorithms.

### 5.1 Online vs Offline

| | Online (`DiarizerManager`) | Offline (`OfflineDiarizerManager`) |
|---|---|---|
| Use case | Live meeting, growing transcript | Post-hoc transcript with speaker labels |
| Clustering | Cosine distance + threshold (assigns to existing speaker or creates new) | VBx (Variational Bayes) with AHC warm start |
| Latency | Per chunk | Full file, then output |
| DER | ~14 % (clean) | **17.7 %** (AMI-SDM) |
| Streaming | Per-chunk callback | `AsyncThrowingStream` between segmentation/embedding/clustering |
| Concurrency | Class + actor `SpeakerManager` | Class + `nonisolated(unsafe)` immutable models |

Files:
- `Sources/FluidAudio/Diarizer/Core/DiarizerManager.swift:6`
- `Sources/FluidAudio/Diarizer/Offline/Core/OfflineDiarizerManager.swift:7`

The online path is intentionally **not a clustering algorithm**. It's a
nearest-centroid assignment with hysteresis on the embedding update. New
speakers are only created if the segment is ≥ 1.0 s — short fragments
are dropped to avoid speaker explosion.
File: `Sources/FluidAudio/Diarizer/Clustering/SpeakerManager.swift:128`.

The offline path is the full Bayesian pipeline:
1. **Segmentation** (pyannote 3.1 / community-1, CoreML) → 7-class
   powerset over 3-speaker combinations, frame stride ≈ 16.98 ms
2. **Embedding** (WeSpeaker v2, ECAPA-TDNN variant, CoreML) →
   256-d L2-normalized vectors, masked to non-overlapping speech frames
3. **AHC** (FastClusterWrapper / fastcluster, C++17) → dendrogram for
   warm start
4. **VBx EM** (BUT Speech@FIT VBx, Swift) → soft assignments + ELBO
   convergence

Files: `Sources/FluidAudio/Diarizer/Offline/Clustering/VBxClustering.swift:6`,
`Sources/FluidAudio/Diarizer/Offline/Clustering/AHCClustering.swift`.

### 5.2 Why C++ for Linkage

`FastClusterWrapper` exists because:
- `fastcluster` (Müllner) is a battle-tested ~10× faster centroid linkage
  implementation than a naive port.
- The function is called once per file in the offline path; the Swift
  bridge is thin (feature matrix in, dendrogram out).
- Swift's array-of-arrays performance for this kind of triangular
  distance computation is much worse than C's flat-buffer math.

C++17 is set package-wide via `cxxLanguageStandard: .cxx17` in
`Package.swift` purely for this binding.

### 5.3 No Built-in VAD-Gated Diarization

Diarization accepts full audio. It does not call VAD internally. Callers
who want VAD-gated diarization compose the two modules explicitly. Reason:
the segmentation model already does its own implicit silence handling
(empty powerset frames), and VAD-gating would change recall
characteristics in ways the diarizer's clustering wasn't tuned for.

---

## 6. What's Shared vs What's Per-Module

### Shared (`Sources/FluidAudio/Shared/`)

- `AudioConverter` — sample-rate conversion to 16 kHz / 24 kHz, mono
  downmix. Used by VAD, Diarization, ASR.
- `AudioMelSpectrogram` — log-mel for ASR (not used by VAD/Diarization
  because their CoreML models do their own feature extraction).
- `ANEMemoryOptimizer` — page-aligned `MLMultiArray` allocations. Used
  by anything that hits ANE in a hot loop (TDT decoder, VAD, segmentation,
  embedding).
- `AppLogger` — category-based `os.Logger` wrapper. One logger per
  component.
- `MLModelConfigurationUtils` — default `MLComputeUnits` selection,
  CI overrides.
- `DownloadUtils` — HuggingFace fetch + caching + token resolution.

### Per-Module

Anything **specific to a model family** lives under that module:
tokenizers, decoders, frame schedulers, voice loaders, codec wrappers,
G2P, post-processors. We pull things into `Shared/` only when at least
two unrelated modules need them.

This is why, for example, `text_embed_table.bin` parsing lives in
`TTS/PocketTTS/Assets/` even though Qwen3 ASR has its own version of
the same trick — they have different on-disk layouts and different
header conventions.

---

## 7. Map of Manager Entry Points

If you're tracing through the repo, these are the canonical entry
points:

| Module | Manager | File |
|---|---|---|
| ASR (Parakeet TDT) | `AsrManager`, `SlidingWindowAsrManager` | `ASR/Parakeet/SlidingWindow/TDT/AsrManager.swift:6` |
| ASR (Qwen3) | `Qwen3AsrManager` | `ASR/Qwen3/Qwen3AsrManager.swift:21` |
| ASR (Cohere) | (via `CoherePipeline`) | `ASR/Cohere/CoherePipeline.swift:1` |
| TTS (Kokoro) | `KokoroTtsManager` | `TTS/Kokoro/KokoroTtsManager.swift:38` |
| TTS (KokoroAne) | `KokoroAneModelStore` | `TTS/KokoroAne/Pipeline/KokoroAneModelStore.swift:4` |
| TTS (PocketTTS) | `PocketTtsModelStore`, `PocketTtsSynthesizer` | `TTS/PocketTTS/Pipeline/PocketTtsModelStore.swift:12` |
| TTS (CosyVoice3) | `CosyVoice3TtsManager` | `TTS/CosyVoice3/CosyVoice3TtsManager.swift:1` |
| TTS (Magpie) | `MagpieTtsManager` | `TTS/Magpie/MagpieTtsManager.swift:1` |
| TTS (StyleTTS2) | `StyleTTS2Manager` | `TTS/StyleTTS2/StyleTTS2Manager.swift:1` |
| TTS (G2P) | `MultilingualG2PModel.shared` | `TTS/G2P/MultilingualG2PModel.swift:11` |
| VAD | `VadManager` | `VAD/VadManager.swift:14` |
| Diarization (online) | `DiarizerManager` | `Diarizer/Core/DiarizerManager.swift:6` |
| Diarization (offline) | `OfflineDiarizerManager` | `Diarizer/Offline/Core/OfflineDiarizerManager.swift:7` |

---

## 8. Recurring Design Choices, in One Sentence

- **Actors for mutable model state** — Swift 6 concurrency safety without
  `@unchecked Sendable`.
- **Per-module error enums** — every domain owns its vocabulary.
- **Lazy HuggingFace downloads** — no model in the binary, ever.
- **AsyncStream where the caller needs incremental output, plain
  state-in/state-out where the caller already has a stream loop.**
- **CoreML for tensor throughput, Swift for branchy state machines and
  per-element work, C++ for one specific clustering hot path.**
- **Per-model compute units, set by measured precision** — never a
  global `.all`.
- **No premature abstractions across model families** — TTS especially
  refuses a unified backend protocol because the pipelines genuinely
  differ.
