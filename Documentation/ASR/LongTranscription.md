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

## Chunk Geometry

The numbers below come from `ASRConstants` and `ChunkProcessor`. They are not
configurable at runtime — they are derived from the encoder's frame rate and
the Core ML window size — so they are the same on every device.

| Quantity | Value | Source |
|---|---|---|
| Sample rate | 16,000 Hz | `ASRConstants.sampleRate` |
| Encoder window | 240,000 samples (≈ 15.00 s) | `ASRConstants.maxModelSamples` |
| Encoder frame | 1,280 samples (80 ms) | `ASRConstants.samplesPerEncoderFrame` |
| Mel hop | 160 samples (10 ms) | `ASRConstants.melHopSize` |
| Visible chunk | ≈ 14.96 s, frame-aligned | `ChunkProcessor.chunkSamples(...)` |
| Overlap target | 2.0 s, frame-aligned, capped at `chunkSamples / 2` | `ChunkProcessor.overlapSeconds` |
| Stride | `chunkSamples − overlap`, frame-aligned | `ChunkProcessor.strideSamples(...)` |
| Minimum seam overlap | 6 encoder frames (480 ms) | `silenceAlignedChunkStarts` |

`chunkSamples` is the *visible* window decoded into transcript tokens; it is
slightly smaller than `maxModelSamples` to reserve room for either an 80 ms
mel-context prepend (default path) or a 0–7 frame acoustic warmup prefix
(no-mel paths). Visible windows are always whole numbers of encoder frames, so
chunk timestamps land on frame boundaries.

## Failure Modes

When reviewing long-form ASR output, check the transcript for:

- boundary word drops, especially short function words or one-word clauses
- duplicated words or partial BPE fragments around overlaps
- glued words (two words joined without a space) or hybrid words stitched
  from both windows' segmentation of the same seam word — `"worksks"`,
  `"automoti"`, mid-word punctuation like `"ye,ah"` (issue #683)
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

## Boundary Search

`ChunkProcessor` picks the start sample of each non-first chunk by one of two
strategies, selected from `melChunkContext` and `modelVersion`:

| Mode | When | Behavior |
|---|---|---|
| `regularChunkStarts` | Default mel-context path, or non-v3 models | Fixed stride: `start_i = i × strideSamples`. Cheap, predictable, but ignores acoustic content at the seam. |
| `silenceAlignedChunkStarts` | `melChunkContext = false` *and* model version is v3 | Searches for a low-energy frame near the target boundary and starts the chunk there. |

The silence-aligned search is two-tier:

1. **Silence pass.** Within `±4 s` of the target frame, score each candidate
   frame by mean-square energy in a `±80 ms` window (`boundaryEnergyScore`).
   A candidate is accepted as "near silence" if its score is at most
   `0.05 × medianScore` over the window. Adaptive thresholds let the search
   tolerate noisy recordings without re-tuning a hardcoded absolute energy.
2. **Valley fallback.** If no near-silence candidate is found, repeat the same
   search within `±0.5 s` and accept the best candidate if it scores below
   `0.35 × medianScore`. This catches inter-word valleys when the audio has
   no real silence near the target.
3. **Speech-tail check.** If using a silence boundary would force the *next*
   forced boundary (`candidate + chunkSamples − minOverlap`) into speech,
   fall back to the original target start. This prevents pulling a boundary
   too early when doing so would dump a speech-only tail onto the following
   chunk.

The search keeps at least 6 encoder frames (480 ms) of overlap with the
previous chunk so the overlap merger always has at least a handful of
candidate tokens to align on.

## Warmup Prefix vs Mel Context

Non-first chunks always include some samples from before the visible window,
but the two mechanisms behave differently:

- **Mel context** (`melContextSamples`, 1 encoder frame / 80 ms). Prepended to
  the chunk so the FastConformer encoder's depthwise convolutions have stable
  left context for the first emitted frame. The decoder is told to *skip*
  those leading frames via `contextSamples`; they do not produce tokens.
  Enabled when `ASRConfig.melChunkContext = true` (the default).
- **Warmup prefix** (`warmupPrefixSamples`, 0–7 encoder frames). Real audio
  from before the chunk start, decoded normally from frame 0; emitted tokens
  are suppressed up to the chunk start via `emitTokensAfterFrame`. Used only
  by the v3 no-mel arbitration probe (path B); the default v3 no-mel path
  keeps `warmupPrefixFrames = 0`.

`shouldUseWarmupPrefix` further gates the warmup decision when a silence
boundary is available: if there is at least 200 ms of stable quiet audio in
the 500 ms lookahead from the boundary (`rms < 0.003`), warmup is skipped —
the encoder will see real silence anyway, so there is no language-prior
drift to warm out of.

## Why This Helps

The original long-form path (PR #264) used an 80 ms mel-context prepend so
non-first chunks had stable leading encoder frames. That helps avoid
blank-boundary failures on long English audio (where the encoder otherwise
emits nothing for the first few frames of a chunk).

Issue #594 surfaced a second, opposite failure on `parakeet-tdt-0.6b-v3-coreml`
multilingual audio: the 80 ms prepend can shift the encoder's first-frame
distribution enough that the SOS-primed TDT decoder drifts back to v3's
English prior. The visible symptom is usually not random noise; it is a
plausible-looking phrase in the *wrong* language near a seam (commonly
French → English, also observed Spanish → English).

The earlier attempt to fix that — persisting decoder state across chunks and
extending the audio prefix to 2.0 s of real audio (commit `eb9c19f7`) — was
correct in isolation but incompatible with parallel chunk decoding
(`fcd80f10`, PR #507): every chunk needs to start from a fresh
`TdtDecoderState` for the worker pool to be independent. The shipped fix
keeps decoding stateless per chunk and instead removes the shifting prepend
on v3, with the no-mel path preferring silence-aligned boundaries so the
encoder sees a natural acoustic onset rather than a discontinuity. The
arbitration mode adds a short probe for cases where different boundary
strategies preserve different content; committing globally to one strategy
favors consistency over per-chunk switching.

A third interacting issue from #594 — the decoder occasionally entering a
BLANK-trap after a sentence-final token — was masked by the per-chunk SOS
reset and re-surfaced briefly under persistent state. With stateless
per-chunk decoding restored, this trap is again not reachable through
chunked transcription; long-form streaming paths still guard against it
explicitly.

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

## Overlap Merge

After each chunk decodes independently, `ChunkProcessor.mergeChunks` stitches
adjacent chunks into a single token timeline. The merger never re-runs the
decoder and never invents tokens; it only chooses which side of the overlap
each token comes from. The strategy is a four-step ladder:

1. **Disjoint shortcut.** If the left chunk's last token ends before the
   right chunk's first token begins, concatenate without merging.
2. **Contiguous time-tolerant match.** Tokens in the overlap region are
   compared with a tolerance of `overlapSeconds / 2`. `SequenceMatcher`
   finds the longest contiguous run where the same token ID appears in both
   chunks within the tolerance window; if that run is at least half of the
   overlap, the merger splices both halves around it.
3. **LCS fallback.** If no good contiguous run exists, fall back to a
   longest-common-subsequence match over the same overlap window with the
   same tolerance, then splice using each LCS pair.
4. **Midpoint fallback.** If LCS also returns nothing, split at the midpoint
   of the overlap (`mergeByMidpoint`): keep left-chunk tokens before the
   midpoint and right-chunk tokens after it.

The matcher uses *token ID + frame-time tolerance* rather than text — so it
cannot collapse two different words that happen to share a substring, and it
is robust to small per-chunk timestamp jitter. The contiguous-match path
preserves order strictly; LCS is only entered when adjacent chunks disagree
enough that a contiguous run would be dishonest.

### Word-Boundary Splice Repair

Matching alone does not make the *splice points* safe: the two windows
tokenize the same seam audio independently, and the right window often
segments its first word differently (frequently without the SentencePiece
`▁` word-start prefix, since for that decoder the word is utterance-initial).
Splicing a continuation piece from the right stream directly after a left
token decodes as a glued or hybrid word — `"work" + "ks"` → `"worksks"`
(issue #683).

To prevent this, `mergeChunks` derives a set of *splice-safe* token IDs from
the model vocabulary once per merge pass: pieces with a `▁`/space prefix or
punctuation-only pieces. Splices are then repaired at word granularity:

- **`mergeUsingMatches`** — when the post-match tail of the right stream
  starts with a continuation piece, the merger adopts the right window's
  segmentation of the entire seam word (the left chunk is typically the one
  cut mid-word). Only when the right stream itself begins mid-word — the
  construction that produced glued words — does the left window keep the
  seam word, with the right stream resuming at its next word-initial piece.
- **`mergeByMidpoint`** — the time cutoff is adjusted so the left stream
  finishes the word it started, and orphaned continuation pieces (whose
  word-initial piece was trimmed away) are dropped from the right stream's
  head.

The repair inspects only the tokenizer's own word-boundary marker, never
transcript text, so it is language-agnostic. Without a vocabulary the merge
is byte-for-byte unchanged.

## Streaming Threshold for Large Files

`ASRConfig` also exposes two knobs that are not about chunk boundary quality
but about memory pressure on very long files:

| Field | Default | Notes |
|---|---|---|
| `ASRConfig.streamingEnabled` | `true` | When `true`, files larger than `streamingThreshold` are read incrementally from disk by the chunked path instead of being loaded entirely into memory. |
| `ASRConfig.streamingThreshold` | `480_000` samples (≈ 30 s at 16 kHz) | Threshold above which `streamingEnabled` actually kicks in. Below this, the file is held in a single `[Float]` buffer. |

This pair affects which `AudioSampleSource` `ChunkProcessor` is constructed
with; it does not change chunk geometry or boundary search. For files
significantly longer than the threshold (an hour of audio is ≈ 57.6 M
samples) the streaming path is the difference between a few hundred MiB of
peak resident memory and a few hundred KiB. Both knobs are orthogonal to
`parallelChunkConcurrency` — worker pool size is bounded independently — but
the worker pool's clones each hold their own short-lived decoder/encoder
buffers, so for the most memory-constrained environments setting
`parallelChunkConcurrency = 1` and leaving streaming enabled is the lowest
high-water-mark configuration.

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

### Seam Canary (required for boundary or merge changes)

Any change to `ChunkProcessor`'s boundary search or overlap merge must be
A/B'd against the baseline build on a seam-dense fixture before merging.
Short-clip WER is not a substitute: the issue #683 fix repaired six seam
artifacts in one hour of earnings audio while remaining exactly WER-neutral
on FLEURS — a merge bug can be invisible to the aggregate metric and still
make transcripts unusable.

The procedure:

1. Transcribe a long fixture with both builds. The cached
   `earnings22-1h/earnings22_top4_1h.wav` (~1 h, ~277 seams) is the standard
   choice; `swift run fluidaudiocli transcribe <file>` on each build.
2. Word-diff the two transcripts (`diff <(tr ' ' '\n' < a.txt) <(tr ' ' '\n'
   < b.txt)`). Inspect every diff individually — each one must be
   explainable as an intended repair. Do not accept a net count.
3. Scan both outputs for mechanical seam artifacts that need no reference
   transcript: mid-word punctuation (`grep -oE '[a-z],[a-z]+'`), and any
   diff hunk that concatenates two previously separate words.

### Invariants Must Be Executable

The history of this path is a caution: the overlap merge ladder shipped in
PR #177 (2025-11) and its first token-stream tests arrived with PR #604
(2026-05); issue #683's glued-word class survived that entire gap. The
failure mode was even listed in this document while the Overlap Merge
section asserted the matcher was safe — the claim was about the wrong
layer, and nothing executable connected the two.

The rules that follow from that:

- A safety claim about the merge belongs in a test, not (only) in this
  document. If a sentence here says the merger "cannot" do something,
  there must be a unit test that fails when it does.
- Merge unit tests must drive realistic token streams *with* a splice-safe
  vocabulary set (`mergeTokenWindowsForTesting(left:right:spliceSafeTokenIds:)`).
  Bare integer token IDs are structurally blind to word-level artifacts —
  that blindness is exactly why #683 was untestable before PR #688.
- Every seam-affecting invariant the code enforces today is listed here;
  keep this list in sync when adding one:
  - a splice from the right window never starts mid-word
    (`spliceSafeTokenIds` gating, issue #683)
  - the midpoint cutoff never strands a word's continuation pieces on the
    wrong side of the seam (`mergeByMidpoint`, issue #683)
  - chunk starts are always frame-aligned (`chunkLayout`)
  - at least 6 encoder frames of seam overlap are preserved
    (`silenceAlignedChunkStarts`)

## Relevant Code

- `Sources/FluidAudio/Shared/ASRConstants.swift`
  - `maxModelSamples`, `samplesPerEncoderFrame`, `melHopSize`,
    `secondsPerEncoderFrame` — fixed encoder geometry
- `Sources/FluidAudio/ASR/Parakeet/AsrTypes.swift`
  - `ASRConfig.melChunkContext`
  - `ASRConfig.dualDecodeArbitration`
  - `ASRConfig.parallelChunkConcurrency`
  - `ASRConfig.streamingEnabled` / `ASRConfig.streamingThreshold`
- `Sources/FluidAudio/ASR/Parakeet/AsrManager.swift`
  - `parallelChunkConcurrency` actor-isolated accessor
  - `makeWorkerClone()` factory used to populate the chunk worker pool
- `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/AsrManager+Transcription.swift`
  - routes long audio through `ChunkProcessor`
- `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/ChunkProcessor.swift`
  - `chunkLayout(...)` and `chunkSamples(...)` — frame-aligned chunk sizing
  - `regularChunkStarts(...)` / `silenceAlignedChunkStarts(...)` /
    `bestBoundaryCandidate(...)` — boundary search
  - `shouldUseWarmupPrefix(...)` / `wouldCompressSpeechTail(...)` — warmup
    gating
  - `mergeChunks(...)` / `mergeUsingMatches(...)` / `mergeByMidpoint(...)` —
    overlap merge ladder (contiguous → LCS → midpoint)
  - `spliceSafeTokenIds(vocabulary:)` / `isSpliceSafePiece(...)` —
    vocabulary-derived word-boundary gating for seam splices (issue #683)
  - `makeWorkerPool(...)` and the static `transcribeChunk(...)` task body
    used by the parallel dispatch loop
- `Sources/FluidAudio/ASR/Parakeet/TokenDeduplication/SequenceMatcher.swift`
  - `findContiguousMatches` and `findLongestCommonSubsequence` used by the
    overlap merger
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

`ChunkProcessorTests` includes word-boundary splice tests that pass a
splice-safe token set. When adding merge tests, do the same: a merge test
without one exercises only the token-ID layer and cannot catch glued or
hybrid words (see "Invariants Must Be Executable" above).
