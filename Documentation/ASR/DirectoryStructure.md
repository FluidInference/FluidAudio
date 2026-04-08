# ASR Directory Structure

This document explains the organization of the `Sources/FluidAudio/ASR/` directory after the restructuring in PR #440.

## Old Structure

```
ASR/
├── ANEOptimizer.swift
├── AsrManager.swift
├── AsrModels.swift
├── AsrManager+Transcription.swift
├── AsrTypes.swift
├── AudioBuffer.swift
├── ChunkProcessor.swift
├── MLArrayCache.swift
├── PerformanceMetrics.swift
├── ProgressEmitter.swift
├── CTC/
│   ├── ARPALanguageModel.swift
│   └── CtcDecoder.swift
├── CustomVocabulary/
│   ├── BKTree/
│   ├── Rescorer/
│   └── WordSpotting/
├── TDT/
│   ├── BlasIndex.swift
│   ├── TdtConfig.swift
│   ├── TdtDecoderV2.swift
│   ├── TdtDecoderV3.swift
│   └── ...
├── Streaming/
│   ├── StreamingAsrManager.swift
│   ├── StreamingAsrSession.swift
│   ├── StreamingEouAsrManager.swift
│   ├── StreamingNemotronAsrManager.swift
│   ├── NemotronChunkSize.swift
│   ├── NemotronPipeline.swift
│   ├── NemotronStreamingConfig.swift
│   ├── RnntDecoder.swift
│   └── Tokenizer.swift
└── Qwen3/
    ├── Qwen3AsrManager.swift
    ├── Qwen3AsrConfig.swift
    └── ...
```

### Problems

1. **Parakeet files at the ASR root.** Files like `AsrManager.swift`, `ChunkProcessor.swift`, and `AudioBuffer.swift` are Parakeet-specific but sit at the `ASR/` root as if they are shared infrastructure. Qwen3 has its own manager and models and shares none of this code.

2. **`Streaming/` conflates two different things.** `StreamingAsrManager` uses an offline encoder with overlapping sliding-window chunks — it is not true streaming. It lived alongside `StreamingEouAsrManager` and `StreamingNemotronAsrManager`, which are actual cache-aware streaming engines. The naming made it unclear which was which.

3. **`CTC/` and `CustomVocabulary/` at the top level.** These are only used by the sliding-window pipeline, not by true streaming or Qwen3. Their placement suggested they were shared ASR utilities.

4. **`TDT/` naming.** "TDT" refers to the Token-and-Duration Transducer algorithm, but the directory contains general decoder logic (hypothesis management, BLAS indexing, frame views) that isn't specific to TDT as a concept.

## New Structure

```
ASR/
├── Parakeet/
│   ├── ParakeetLanguageModels.swift
│   ├── AsrTypes.swift
│   ├── AudioBuffer.swift
│   │
│   ├── SlidingWindow/
│   │   ├── SlidingWindowAsrManager.swift
│   │   ├── SlidingWindowAsrSession.swift
│   │   │
│   │   ├── TDT/                         ← All TDT batch processing
│   │   │   ├── AsrManager.swift         (multilingual, internal engine)
│   │   │   ├── AsrManager+Pipeline.swift
│   │   │   ├── AsrManager+TokenProcessing.swift
│   │   │   ├── AsrManager+Transcription.swift
│   │   │   ├── AsrModels.swift
│   │   │   ├── ChunkProcessor.swift
│   │   │   ├── TdtJaManager.swift       (Japanese)
│   │   │   ├── TdtJaModels.swift
│   │   │   └── Decoder/                 (TDT infrastructure)
│   │   │       ├── BlasIndex.swift
│   │   │       ├── EncoderFrameView.swift
│   │   │       ├── TdtConfig.swift
│   │   │       ├── TdtDecoderState.swift
│   │   │       ├── TdtDecoderV2.swift
│   │   │       ├── TdtDecoderV3.swift
│   │   │       ├── TdtHypothesis.swift
│   │   │       ├── TdtModelInference.swift
│   │   │       ├── TdtJointDecision.swift
│   │   │       ├── TdtJointInputProvider.swift
│   │   │       ├── TdtDurationMapping.swift
│   │   │       └── TdtFrameNavigation.swift
│   │   │
│   │   ├── CTC/                         ← All CTC batch + language variants
│   │   │   ├── ARPALanguageModel.swift
│   │   │   ├── CtcDecoder.swift
│   │   │   ├── CtcJaManager.swift       (Japanese)
│   │   │   ├── CtcJaModels.swift
│   │   │   ├── CtcZhCnManager.swift     (Chinese)
│   │   │   └── CtcZhCnModels.swift
│   │   │
│   │   └── CustomVocabulary/
│   │       ├── BKTree/
│   │       ├── Rescorer/
│   │       └── WordSpotting/
│   │
│   ├── Streaming/                       (true streaming engines)
│   │   ├── StreamingAsrManager.swift
│   │   ├── ParakeetModelVariant.swift
│   │   ├── RnntDecoder.swift
│   │   ├── Tokenizer.swift
│   │   ├── EOU/
│   │   │   └── StreamingEouAsrManager.swift
│   │   └── Nemotron/
│   │       ├── NemotronChunkSize.swift
│   │       ├── StreamingNemotronAsrManager.swift
│   │       ├── StreamingNemotronAsrManager+Pipeline.swift
│   │       └── NemotronStreamingConfig.swift
│   │
│   └── TokenDeduplication/
│       ├── SequenceMatch.swift
│       └── SequenceMatcher.swift
│
└── Qwen3/
    ├── Qwen3AsrConfig.swift
    ├── Qwen3AsrManager.swift
    ├── Qwen3AsrModels.swift
    ├── Qwen3RoPE.swift
    ├── Qwen3StreamingManager.swift
    └── WhisperMelSpectrogram.swift
```

### Algorithm-based organization: TDT/ vs CTC/

Batch processing managers are now grouped by decoding algorithm within `SlidingWindow/`:

**TDT (Token-and-Duration Transducer):**
- `AsrManager` - Multilingual batch engine (en, es, fr, de, etc.)
- `TdtJaManager` - Japanese-specific batch processing
- `Decoder/` - Shared TDT infrastructure (beam search, BLAS indexing, hypothesis management)

**CTC (Connectionist Temporal Classification):**
- `CtcDecoder` - Greedy CTC decoding with optional LM
- `CtcJaManager` - Japanese CTC batch processing
- `CtcZhCnManager` - Chinese CTC batch processing
- `ARPALanguageModel` - ARPA LM support for CTC decoding

Both algorithm families use the sliding-window approach (large overlapping chunks with offline encoder), distinguishing them from the true streaming engines in `Streaming/`.

## What Changed and Why

### Model family split: `Parakeet/` vs `Qwen3/`

Parakeet and Qwen3 share zero code. Parakeet uses a FastConformer encoder with TDT decoding. Qwen3 uses a Whisper-style encoder-decoder transformer. Grouping by model family makes ownership obvious — if you are working on Parakeet, everything you need is under `Parakeet/`.

### `TDT/` renamed to `Decoder/`

The directory contains decoder infrastructure (hypothesis beam management, BLAS-accelerated indexing, encoder frame views) that is broader than the TDT algorithm itself. `Decoder/` is a more accurate name for what lives there.

### `Streaming/` split into `SlidingWindow/` and `Streaming/`

The old `Streaming/` directory mixed two architecturally different approaches:

| | SlidingWindow | Streaming |
|---|---|---|
| **Encoder** | Offline (full-context) | Cache-aware (incremental) |
| **Processing** | Overlapping chunks with merge | Sequential chunks, no overlap |
| **Latency** | ~15s chunks, seconds of delay | 80ms–1280ms chunks, real-time |
| **State** | Stateless per chunk, decoder resets | Stateful encoder cache across chunks |

`SlidingWindowAsrManager` (formerly `StreamingAsrManager`) processes audio in large overlapping windows using an offline encoder. The name "streaming" was misleading — it streams audio *in*, but the encoder sees each chunk in isolation.

`StreamingEouAsrManager` and `StreamingNemotronAsrManager` are true streaming engines with cache-aware encoders that maintain state across chunks.

### Renames

| Old | New | Reason |
|---|---|---|
| `StreamingAsrManager` | `SlidingWindowAsrManager` | Uses sliding windows, not true streaming |
| `StreamingAsrSession` | `SlidingWindowAsrSession` | Companion to the manager above |
| `TDT/` | `Decoder/` | Contains general decoder logic, not just TDT-specific code |

### `CTC/` and `CustomVocabulary/` moved under `SlidingWindow/`

These features are only used by the sliding-window pipeline. CTC decoding runs on the offline encoder's output. Custom vocabulary boosting operates on TDT hypotheses from sliding-window chunks. Neither applies to the true streaming engines.

### New files in `Streaming/`

| File | Purpose |
|---|---|
| `StreamingAsrManager.swift` | Actor protocol defining the interface for true streaming engines (`loadModels`, `appendAudio`, `processBufferedAudio`, `finish`, `reset`) |
| `ParakeetModelVariant.swift` | Enum cataloguing all available streaming variants (EOU 160ms/320ms/1280ms, Nemotron 560ms/1120ms) with their repos, chunk sizes, and `createManager()` factory method |

### `Streaming/` subdivided into `EOU/` and `Nemotron/`

EOU (End-of-Utterance) and Nemotron are different streaming architectures. EOU is a 120M parameter model with utterance boundary detection. Nemotron is a 0.6B parameter FastConformer RNNT. Shared streaming infrastructure (`RnntDecoder`, `Tokenizer`, the protocol and factory) stays at the `Streaming/` root.

## Why a Protocol Instead of a Base Class

EOU and Nemotron follow the same high-level flow (buffer audio, process chunks, accumulate tokens, decode transcript), which raises the question of whether they should share a base class. They should not.

### Swift actors cannot inherit from other actors

Both engines are `actor` types for thread safety. Swift has no actor inheritance — an actor cannot subclass another actor. Using a base class would mean dropping to a regular `class`, losing actor isolation, and reintroducing manual locking. That defeats the purpose.

### The shared code is trivial

What looks duplicated between the two engines:

| Code | Lines | Notes |
|---|---|---|
| `appendAudio(_:)` | 2 | Resample + append to buffer |
| `getPartialTranscript()` | 2 | Decode accumulated token IDs |
| `finish()` | ~10 | Pad remaining audio, process final chunk, decode — but calls different internal methods and has different guards |
| `processBufferedAudio()` | ~6 | Same while-loop shape but different chunk size sources and different shift logic (EOU overlaps, Nemotron does not) |

Total genuinely identical code: ~4 lines across `appendAudio` and `getPartialTranscript`. Not worth an abstraction.

### The internals are fundamentally different

| | EOU (120M) | Nemotron (0.6B) |
|---|---|---|
| **Mel spectrogram** | Native Swift (`AudioMelSpectrogram`) | CoreML preprocessor model |
| **Encoder** | Loopback encoder with 4 cache tensors (`preCache`, `cacheLastChannel`, `cacheLastTime`, `cacheLastChannelLen`) | Int8 encoder with 3 cache tensors (`cacheChannel`, `cacheTime`, `cacheLen`) + separate mel cache |
| **Cache shapes** | Hardcoded (e.g. `[17, 1, 70, 512]`) | Config-driven from `metadata.json` |
| **Chunk shifting** | Overlapping — shifts by `shiftSamples` (e.g. 50% overlap for 160ms) | Non-overlapping — shifts by full `chunkSamples` |
| **Extra features** | EOU detection with debounce timer | None |
| **State reset** | 6 cache arrays + EOU debounce state + RNNT decoder | 5 cache arrays + LSTM states from config shapes |

A base class would have almost no real implementation to share — just abstract methods everywhere. The `StreamingAsrManager` protocol is the right tool: it defines the contract without pretending these engines share internals.

## CLI and Test Mirrors

The CLI commands and tests follow the same structure:

```
Sources/FluidAudioCLI/Commands/ASR/
├── Parakeet/
│   ├── SlidingWindow/    (transcribe, benchmarks, multi-stream)
│   └── Streaming/        (EOU command, Nemotron transcribe/benchmark)
└── Qwen3/                (Qwen3 transcribe, benchmark)

Tests/FluidAudioTests/ASR/
├── Parakeet/
│   ├── SlidingWindow/
│   │   ├── TDT/          (AsrManager, ChunkProcessor, TdtJa, Decoder tests)
│   │   ├── CTC/          (CtcJa, CtcZhCn tests)
│   │   └── CustomVocabulary/ (BKTree, Rescorer, WordSpotting tests)
│   └── Streaming/        (EOU, Nemotron, engine protocol tests)
└── Qwen3/                (config, RoPE tests)
```
