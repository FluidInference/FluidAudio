# Long Transcription

This document explains the Parakeet TDT long-form batch path and the quality
issues that are easy to miss when testing only short clips.

## Overview

Parakeet TDT Core ML models accept a fixed encoder window of 240,000 samples
(15 seconds at 16 kHz). Longer files are split into overlapping chunks, decoded,
and merged back into one transcript.

Most long-transcription regressions happen at chunk seams. A short benchmark can
look healthy while a longer recording still loses words, repeats fragments, or
drifts into the wrong language after several boundaries. FLEURS is useful as a
multilingual smoke test, but most FLEURS samples are short read-speech clips, so
they do not exercise repeated seam behavior very much.

## Failure Modes

When reviewing long-form ASR output, check the transcript for:

- boundary word drops, especially short function words or one-word clauses
- duplicated words or partial BPE fragments around overlaps
- missing clauses or full sentences after a boundary
- wrong-language insertions in otherwise single-language audio
- wrong-script bursts on multilingual v3 audio
- sentence breaks or punctuation that move far enough to change readability
- real mixed-language switches being removed or delayed

Aggregate WER can hide these problems. A transcript with a good average score
may still be unusable if a seam drops a sentence or inserts a wrong-language
phrase at the wrong point.

## Current Paths

| Path | Enabled by | Scope | Purpose |
|---|---|---|---|
| Default mel-context | `ASRConfig.melChunkContext = true` | Batch TDT long audio | Preserves the existing 80 ms left-context behavior for non-first chunks. |
| v3 no-mel | `ASRConfig.melChunkContext = false`, CLI `--no-mel-context` | Parakeet TDT v3 batch long audio | Avoids the v3 multilingual drift introduced by prepending mel context at chunk boundaries. |
| v3 dual-decode arbitration | `melChunkContext = false` plus `ASRConfig.dualDecodeArbitration = true`, CLI `--no-mel-context --dual-decode-arbitration` | Parakeet TDT v3 no-mel batch long audio | Opt-in quality mode for files where one boundary strategy is clearly safer than another. |
| Parallel chunk workers | `ASRConfig.parallelChunkConcurrency` (default `4`, clamped to `>= 1`) | Stateless chunked batch TDT (all of the above) | Decodes independent chunks concurrently across a worker pool of cloned `AsrManager` instances. |

The dual-decode path probes the first few non-first chunks with three strategies:

- silence-aligned boundaries without warmup
- silence-aligned boundaries with a hidden short real-audio warmup prefix
- regular fixed-stride boundaries without warmup

After the probe, the whole file commits to one strategy. That keeps the overlap
merger from stitching together adjacent chunks decoded under different boundary
rules, which was one source of mid-word artifacts and clause loss.

The choice is based on decoder confidence, emitted-token counts, and agreement
between probe paths. It is meant to decide between chunking strategies, not to
rewrite transcript text.

## Why This Helps

The original long-form path used an 80 ms mel-context prepend so non-first
chunks had stable leading encoder frames. That helps avoid blank-boundary
failures, but on multilingual v3 audio it can also change the first-frame
distribution enough that the decoder starts a chunk from the wrong prior. The
visible symptom is usually not random noise; it is a plausible-looking phrase in
the wrong language near a seam.

The no-mel v3 path removes that context shift and prefers acoustic boundaries
near low-energy regions. The arbitration mode adds a short probe for cases where
different boundary strategies preserve different content. Committing globally to
one strategy favors consistency over per-chunk switching.

## Parallel Chunk Processing

Long files are split into independent chunks that share no decoder state across
seams, so chunk decoding parallelizes cleanly. `ChunkProcessor` runs a worker
pool of cloned `AsrManager` instances and merges results in chunk-emission
order, preserving the same overlap merge logic used by the single-worker path.

| Field | Default | Notes |
|---|---|---|
| `ASRConfig.parallelChunkConcurrency` | `4` | Number of chunks decoded concurrently. Clamped to `max(1, …)`. Applies only to stateless chunked transcription paths (long-form batch TDT). |

How it works:

- `ChunkProcessor.process(using:)` reads `manager.parallelChunkConcurrency` and
  builds a worker pool via `manager.makeWorkerClone()`. Each clone reuses the
  already-loaded encoder/decoder/joint Core ML models, so no model
  re-initialization happens per worker.
- Chunks are dispatched with `ThrowingTaskGroup`. The dispatch loop reuses an
  `availableWorkers` index list so the number of in-flight tasks never exceeds
  `parallelChunkConcurrency` (backpressure).
- Each task constructs a fresh `TdtDecoderState` (stateless per-chunk
  decoding), runs `transcribeChunk` against its assigned worker, and returns a
  `TaskResult { index, tokens, workerIndex }`. Results are gathered into a
  pre-sized `chunkOutputs` array indexed by chunk order, then merged exactly
  as the serial path did.
- Streaming and real-time paths
  (`StreamingAsrManager`, `SlidingWindowAsrManager`) are unaffected: they
  remain single-decoder and cache-aware, since they depend on persistent
  decoder/encoder state across windows.

Notes for tuning:

- Default `4` was selected from device-matrix testing; benchmarks on Apple M3
  with a 1-hour file show roughly 2.2–2.8× wall-clock speedup over the serial
  path across Parakeet v2/v3 variants, with about 19–31 MiB extra resident
  memory for the additional worker clones.
- Setting `parallelChunkConcurrency = 1` is the closest configuration to the
  pre-parallel behavior and is useful for A/B-ing transcripts against older
  output. It does not bypass `ChunkProcessor`; the worker pool collapses to a
  single worker that reuses the calling `AsrManager`.
- Word timings and per-chunk decoding are unchanged by the parallel path —
  the parallelization is in chunk dispatch, not in decoder behavior, and
  transcripts and timings remain identical to the serial version for the same
  inputs.

## Validation Strategy

A long-transcription change should be checked with a fixed matrix, not only with
one successful clip. The matrix should include:

- issue-specific canaries that previously reproduced boundary drops or drift
- long single-language recordings with source text
- long multilingual recordings across several languages
- intentional mixed-language recordings where the real switch must remain
- short public benchmarks such as FLEURS to catch broad multilingual regressions

For each fixture, keep the transcript and compare it against the source text or
the best known baseline. The review should answer concrete questions:

- Did any word or clause disappear?
- Did the seam introduce a wrong-language phrase?
- Did a mixed-language switch remain at the right place?
- Did overlap merging duplicate or truncate words?
- Did punctuation move enough to make the sentence boundary wrong?

When adding a new fixture, record the language, approximate duration, reference
source, and the specific failure it is meant to catch. This makes future changes
auditable instead of relying on memory of why a clip was added.

## Relevant Code

- `Sources/FluidAudio/ASR/Parakeet/AsrTypes.swift`
  - `ASRConfig.melChunkContext`
  - `ASRConfig.dualDecodeArbitration`
  - `ASRConfig.parallelChunkConcurrency`
- `Sources/FluidAudio/ASR/Parakeet/AsrManager.swift`
  - `parallelChunkConcurrency` actor-isolated accessor
  - `makeWorkerClone()` factory used to populate the chunk worker pool
- `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/AsrManager+Transcription.swift`
  - routes long audio through `ChunkProcessor`
- `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/ChunkProcessor.swift`
  - chunk sizing, boundary search, regular long-form chunk loop, overlap merge,
    worker-pool construction (`makeWorkerPool`), and the static
    `transcribeChunk(...)` task body used by the parallel dispatch loop
- `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/DualDecodeArbitration.swift`
  - opt-in v3/no-mel arbitration path
- `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/Decoder/TdtDecoderV3.swift`
  - token emission gates and decoder state behavior
- `Sources/FluidAudioCLI/Commands/ASR/Parakeet/SlidingWindow/TranscribeCommand.swift`
  - CLI flags for local reproduction

## Focused Tests

Unit tests catch chunking and decoder invariants, but they do not replace a
source-backed transcript matrix for long-form quality.

Useful focused checks:

```bash
swift test --filter ChunkProcessorTests
swift test --filter TdtRefactoredComponentsTests
swift test --filter TdtDecoderV2Tests
swift test --filter ASRConfigTests   # covers parallelChunkConcurrency default, clamping, override
```
