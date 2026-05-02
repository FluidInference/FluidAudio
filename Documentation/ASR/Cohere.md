# Cohere Transcribe

> **Beta**
>
> Encoder-decoder ASR using [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
> converted to CoreML. The FluidAudio integration uses a mixed-precision pipeline
> (INT8 encoder + FP16 cache-external decoder) exposed through
> `CoherePipeline`.

## Model

**CoreML models**:
- INT8 hybrid (recommended): [FluidInference/cohere-transcribe-q8-cache-external-coreml](https://huggingface.co/FluidInference/cohere-transcribe-q8-cache-external-coreml) — 1.8 GB encoder, iOS 18+
- FP16 reference: [FluidInference/cohere-transcribe-cache-external-coreml](https://huggingface.co/FluidInference/cohere-transcribe-cache-external-coreml) — 3.6 GB encoder, iOS 17+

The integration splits the encoder and decoder across precisions. Pass
`encoderDir` + `decoderDir` to `CoherePipeline.loadModels` when the two
variants live in different directories.

## Architecture

- **Encoder**: 48-layer Conformer, hidden size 1280, mel input `[1, 128, 3500]`
  (35 s at 10 ms hop), output `[1, 438, 1024]` (FP16)
- **Decoder**: 8-layer transformer with external KV cache (Parakeet pattern),
  hidden size 1024, 8 heads × 128 head-dim
- **Cache window**: 108 tokens
- **Vocabulary**: 16,384 SentencePiece tokens
- **Max audio**: 35 s per call (single chunk)

### Upstream model spec (`cohere-pytorch/config.json`)

The 35 s limit is not arbitrary — it comes straight from the upstream
config:

| Field | Value | Meaning |
|---|---|---|
| `max_audio_clip_s` | **35** | Hard cap on per-call audio length |
| `overlap_chunk_second` | 5 | Reference Python pipeline uses 5 s overlap when chunking >35 s audio |
| `pos_emb_max_len` | 5000 | Encoder positional embeddings could in theory handle 50 s, but the official preprocessor enforces 35 s |
| `preprocessor.window_size` | 0.025 | 25 ms STFT analysis window |
| `preprocessor.window_stride` | 0.01 | 10 ms hop → 100 fps |

Math chain: `35 s × 100 fps = 3500 mel frames` → CoreML encoder input
shape `[1, 128, 3500]` is baked at conversion time → Swift port mirrors
via `CohereAsrConfig.maxAudioSeconds = 35.0`.

## Supported Languages

14 languages. Cohere Transcribe uses a conditioned prompt that hard-codes the
language — pass `language:` explicitly.

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| en | English | fr | French | de | German |
| es | Spanish | it | Italian | pt | Portuguese |
| nl | Dutch | pl | Polish | el | Greek |
| ar | Arabic | ja | Japanese | zh | Chinese |
| ko | Korean | vi | Vietnamese | | |

## Usage

### CLI

```bash
# Single-precision (FP16 or INT8 in one dir)
swift run -c release fluidaudiocli cohere-transcribe audio.wav \
    --model-dir /path/to/cohere-fp16 \
    --language en

# Mixed precision (INT8 encoder + FP16 decoder)
swift run -c release fluidaudiocli cohere-transcribe audio.wav \
    --encoder-dir /path/to/q8 \
    --decoder-dir /path/to/f16 \
    --vocab-dir /path/to/f16 \
    --language en
```

### Swift API

```swift
import CoreML
import FluidAudio

let encoderDir = URL(fileURLWithPath: "/path/to/q8")
let decoderDir = URL(fileURLWithPath: "/path/to/f16")

let models = try await CoherePipeline.loadModels(
    encoderDir: encoderDir,
    decoderDir: decoderDir,
    vocabDir: decoderDir
)

let pipeline = CoherePipeline()
let result = try await pipeline.transcribe(
    audio: samples,        // 16 kHz mono Float32, up to 35 s
    models: models,
    language: .english
)
print(result.text)
```

`TranscriptionResult` also exposes `encoderSeconds`, `decoderSeconds`, and
`totalSeconds` for per-stage profiling.

## Benchmarks

### LibriSpeech test-clean (INT8 encoder + FP16 cache-external decoder)

Full split, all 2,620 utterances, single-chunk (longest utterance in
test-clean is ~35 s so nothing is skipped). Measured on Apple M2 (2022),
Tahoe 26.0 via `fluidaudiocli cohere-benchmark`.

| Subset | Samples | WER | CER | RTFx (per-file mean) | RTFx (total audio/compute) |
|---|---:|---:|---:|---:|---:|
| test-clean | 2,620 | **1.77%** | **0.60%** | 2.04× | 1.72× |

Totals: 5h 24m of audio, 3h 09m of compute, 3h 12m wall time (includes
one-time ~6 min ANE cold-start compile on file 1).

```bash
swift run -c release fluidaudiocli cohere-benchmark \
    --dataset librispeech --subset test-clean \
    --model-dir /path/to/q8 \
    --auto-download \
    --output cohere_testclean.json
```

### FLEURS (INT8 encoder + FP16 cache-external decoder)

Full FLEURS splits, one sample per file, single-chunk (no long-form stitching).
Measured on M4 Pro, 48 GB, Tahoe 26.0 via `fluidaudiocli cohere-benchmark`.

| FLEURS code | Language | Samples | WER | CER | RTFx |
|---|---|---:|---:|---:|---:|
| en_us | English | 647 | 5.63% | 3.19% | 2.49× |
| fr_fr | French | 676 | 6.22% | 3.11% | 2.21× |
| de_de | German | 862 | 5.84% | 2.83% | 1.98× |
| es_419 | Spanish (Latin America) | 908 | 4.53% | 2.40% | 1.34× |
| it_it | Italian | 865 | **4.03%** | 2.04% | **3.15×** |
| pt_br | Portuguese (Brazil) | 919 | 6.44% | 3.38% | 2.79× |
| nl_nl | Dutch | 364 | 8.07% | 4.14% | 2.04× |
| pl_pl | Polish | 758 | 7.49% | 3.23% | 1.98× |
| el_gr | Greek | 650 | 11.50% | 5.45% | 2.00× |
| ar_eg | Arabic (Egypt) | 428 | 18.46% | 6.71% | 2.06× |
| ja_jp | Japanese | 650 | 60.13%† | 6.25% | 2.23× |
| cmn_hans_cn | Chinese (Mandarin, Simplified) | 945 | 98.52%† | 12.01% | 1.85× |
| ko_kr | Korean | 382 | 16.39% | 6.67% | 1.84× |
| vi_vn | Vietnamese | 857 | 9.55% | 6.87% | 1.55× |

†Japanese and Mandarin are written without word boundaries, so WER on the raw
hypothesis is meaningless; **CER is the real accuracy metric** for those
languages. The 98.52% WER on Mandarin is tokenization artifact, not
transcription quality (CER is 12.01%).

```bash
# Single language
swift run -c release fluidaudiocli cohere-benchmark \
    --encoder-dir /path/to/q8 --decoder-dir /path/to/f16 --vocab-dir /path/to/f16 \
    --languages en_us --auto-download \
    --output cohere_en_us.json
```

### Comparison vs Cohere reference (Figure 4)

Cohere's [technical report](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)
Figure 4 reports per-language error rates **averaged across FLEURS, Common
Voice 17.0, MLS, and Wenet** — not FLEURS-only. Numbers therefore aren't
directly comparable (the averaging of easier corpora like MLS/CV17 generally
pulls Cohere's reported numbers below pure FLEURS), but the shape matches:
WER for most languages, CER for zh/ja/ko.

| Language | Metric | FluidAudio (FLEURS only, INT8+FP16) | Cohere (FLEURS+CV+MLS+Wenet avg, FP16) | Δ |
|---|:---:|---:|---:|---:|
| de | WER | 5.84% | ~3.8% | +2.0 |
| fr | WER | 6.22% | ~4.5% | +1.7 |
| it | WER | 4.03% | ~3.7% | +0.3 |
| es | WER | 4.53% | ~2.8% | +1.7 |
| pt | WER | 6.44% | ~5.9% | +0.5 |
| el | WER | 11.50% | ~8.7% | +2.8 |
| nl | WER | 8.07% | ~5.8% | +2.3 |
| pl | WER | 7.49% | ~5.3% | +2.2 |
| ar | WER | 18.46% | ~16.5% | +2.0 |
| vi | WER | 9.55% | ~8.7% | +0.9 |
| zh | CER | 12.01% | ~10.6% | +1.4 |
| ja | CER | 6.25% | ~8.3% | -2.1 |
| ko | CER | 6.67% | ~3.8% | +2.9 |

Reference numbers read from Figure 4 (`HF_model_card_per-language-avg-plot.png`)
and are approximate (chart only). English is not shown in Figure 4 (Cohere
claims a 5.42% average on the HF Open ASR leaderboard; ours is 5.63% on FLEURS
en_us).

Two sources of the ~1-3% gap on most languages:

1. **Dataset mix**: ours is FLEURS-only, Cohere's is averaged over 4 corpora.
   FLEURS tends to be the hardest of the four, so a pure-FLEURS run is
   expected to land above the average.
2. **Quantization**: FluidAudio ships INT8 encoder + FP16 cache-external
   decoder; Cohere's reported numbers are full FP16 PyTorch inference.

### Cold-start vs warm inference (isolated, single process)

The headline RTFx numbers above assume the encoder/decoder are loaded **once**
and reused across many transcribes. Spawning a fresh process per WAV pays the
ANE compile cost on every call and looks dramatically slower.

Isolated single-process bench, M2 (2022) Tahoe 26.0, INT8 encoder + FP16
cache-external decoder, `cohere-transcribe` invoked once with five WAVs sharing
one model load:

| # | Duration | Encoder | Decoder | Total | RTFx | Notes |
|--:|---:|---:|---:|---:|---:|---|
| 0 |  3.32 s | 186.33 s | 1.46 s | 187.81 s | 0.02× | **cold ANE compile** |
| 1 |  6.87 s |   2.47 s | 0.90 s |   3.41 s | 2.02× | warm |
| 2 |  8.94 s |   3.31 s | 1.21 s |   4.57 s | 1.96× | warm |
| 3 | 37.15 s |   1.51 s | 2.54 s |   4.26 s | 8.73× | warm, full 35 s window |

Takeaways:

- **Cold compile is one-time per process**, not per call. `anecompilerservice`
  re-emits the encoder MIL graph for ANE the first time a fresh process loads
  the model. ~3 min on M2 Tahoe; longer on first-ever compiles for some
  language splits (the cmn_hans_cn FLEURS run logged a ~6 min cold-start).
  Wiping `~/Library/Caches/.../com.apple.e5rt.e5bundlecache/` does not avoid
  it; a clean process always pays it.
- **Warm encoder is shape-bound, not duration-bound.** Every call processes a
  fixed 3,500-frame mel (35 s @ 10 ms hop). Short audio does not finish the
  encoder faster — warm encoder cost stays in a 1.5–3.3 s band regardless of
  input length.
- **Audio > 35 s is auto-chunked** by `transcribeLong` with a 30 s hop and 5 s
  overlap (matches upstream `overlap_chunk_second: 5`); adjacent chunk token
  streams are stitched via token-level longest-common-substring merge. The
  raw `transcribe` call still uses the strict 35 s window with
  `padOrTruncate(... fixedFrames: 3_500)` at
  `Sources/FluidAudio/ASR/Cohere/CoherePipeline.swift:250`, so picking
  `transcribeLong` is the right default for unknown-length audio.
- **Process reuse is what unlocks the documented RTFx.** The LibriSpeech
  test-clean run above (5h 24m audio, RTFx 1.72× total) absorbs the cold
  compile across 2,620 utterances; per-call warm path is closer to RTFx
  2–9× depending on how much of the 35 s window is filled. Re-spawning per
  WAV would push effective RTFx well below 0.1×.

## Notes and Caveats

- **INT8 encoder quality parity**: encoder hidden states match FP16 within
  compute noise on LibriSpeech cross-checks. WER differences FP16 ↔ INT8-encoder
  are within ±0.5%, so the INT8 hybrid is the recommended default.
- **Cache-external decoder stays FP16**: INT8 decoder quantization regresses
  quality significantly in testing and is not shipped.
- **Long-form audio**: the encoder always sees a fixed 35 s window
  (`max_audio_clip_s: 35`). Use `CoherePipeline.transcribeLong` for audio of
  unknown / arbitrary length — it slides a 35 s window with 5 s overlap
  (matching upstream `overlap_chunk_second: 5`) and stitches adjacent chunks
  via token-level longest-common-substring merge. The raw `transcribe` API is
  the single-chunk path and silently truncates input > 35 s. The CLI commands
  (`cohere-transcribe`, `cohere-benchmark`, `tts-benchmark`) all use
  `transcribeLong` and no longer skip > 35 s files.
- **Language must be specified**: no automatic language ID. Pass
  `CohereAsrConfig.Language` on every call; the wrong language produces
  degenerate output.
- **EOS token = 3** (not 151643). All bundled models use the correct value.

## Files

| Component | Description |
|---|---|
| `cohere_encoder.mlmodelc` | 48-layer Conformer encoder (INT8 or FP16) |
| `cohere_decoder_cache_external.mlmodelc` | 8-layer decoder, external KV cache |
| `vocab.json` | id → piece map (16,384 entries) |
| `tokenizer.model` | SentencePiece tokenizer |
