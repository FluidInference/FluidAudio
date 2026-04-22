# Cohere Transcribe

> **Beta**
>
> Encoder-decoder ASR using [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
> converted to CoreML. The FluidAudio integration uses a mixed-precision pipeline
> (INT8 encoder + FP16 cache-external decoder) exposed through
> `CohereFixedPipeline`.

## Model

**CoreML models**:
- INT8 hybrid (recommended): [FluidInference/cohere-transcribe-q8-cache-external-coreml](https://huggingface.co/FluidInference/cohere-transcribe-q8-cache-external-coreml) — 1.8 GB encoder, iOS 18+
- FP16 reference: [FluidInference/cohere-transcribe-cache-external-coreml](https://huggingface.co/FluidInference/cohere-transcribe-cache-external-coreml) — 3.6 GB encoder, iOS 17+

The integration splits the encoder and decoder across precisions. Pass
`encoderDir` + `decoderDir` to `CohereFixedPipeline.loadModels` when the two
variants live in different directories.

## Architecture

- **Encoder**: 48-layer Conformer, hidden size 1280, mel input `[1, 128, 3500]`
  (35 s at 10 ms hop), output `[1, 438, 1024]` (FP16)
- **Decoder**: 8-layer transformer with external KV cache (Parakeet pattern),
  hidden size 1024, 8 heads × 128 head-dim
- **Cache window**: 108 tokens
- **Vocabulary**: 16,384 SentencePiece tokens
- **Max audio**: 35 s per call (single chunk)

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
swift run -c release fluidaudiocli cohere-mixed audio.wav \
    --model-dir /path/to/cohere-fp16 \
    --language en

# Mixed precision (INT8 encoder + FP16 decoder)
swift run -c release fluidaudiocli cohere-mixed audio.wav \
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

let models = try await CohereFixedPipeline.loadModels(
    encoderDir: encoderDir,
    decoderDir: decoderDir,
    vocabDir: decoderDir
)

let pipeline = CohereFixedPipeline()
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

### FLEURS (INT8 encoder + FP16 cache-external decoder)

Full FLEURS splits, one sample per file, single-chunk (no long-form stitching).
Measured on M4 Pro, 48 GB, Tahoe 26.0 via `fluidaudiocli cohere-mixed-benchmark`.

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
# Reproduce (all 14 languages, resumable — one JSON per language)
MODEL_DIR=/path/to/cohere-mixed \
OUT_DIR=./benchmark_results/cohere \
./Scripts/run_cohere_per_lang.sh

# Single language
swift run -c release fluidaudiocli cohere-mixed-benchmark \
    --encoder-dir /path/to/q8 --decoder-dir /path/to/f16 --vocab-dir /path/to/f16 \
    --languages en_us --auto-download \
    --output cohere_en_us.json
```

### Comparison vs Cohere reference (Figure 4)

Cohere's [technical report](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)
Figure 4 reports per-language error rates **averaged across FLEURS, Common
Voice 17.0, MLS, and Wenet** — not FLEURS-only. Numbers therefore aren't
directly comparable, but the shape matches: WER for most languages, CER for
zh/ja/ko.

| Language | FluidAudio FLEURS | Cohere avg (FLEURS+CV+MLS+Wenet) | Metric |
|---|---:|---:|:---:|
| de | 5.84% | ~3.8% | WER |
| fr | 6.22% | ~4.5% | WER |
| it | 4.03% | ~3.7% | WER |
| es | 4.53% | ~2.8% | WER |
| pt | 6.44% | ~5.9% | WER |
| el | 11.50% | ~8.7% | WER |
| nl | 8.07% | ~5.8% | WER |
| pl | 7.49% | ~5.3% | WER |
| ar | 18.46% | ~16.5% | WER |
| vi | 9.55% | ~8.7% | WER |
| zh | 12.01% | ~10.6% | CER |
| ja | 6.25% | ~8.3% | CER |
| ko | 6.67% | ~3.8% | CER |

Reference numbers read from Figure 4 (`HF_model_card_per-language-avg-plot.png`)
and are approximate (chart only). FluidAudio numbers are FLEURS-only, single-
dataset, so ~1-3% higher error is expected — FLEURS field recordings are
generally harder than the multi-dataset average. English is not shown in
Figure 4.

## Notes and Caveats

- **INT8 encoder quality parity**: encoder hidden states match FP16 within
  compute noise on LibriSpeech cross-checks. WER differences FP16 ↔ INT8-encoder
  are within ±0.5%, so the INT8 hybrid is the recommended default.
- **Cache-external decoder stays FP16**: INT8 decoder quantization regresses
  quality significantly in testing and is not shipped.
- **Single-chunk only**: the CoreML pipeline processes one 35 s window per
  call. Longer audio requires external chunking (FluidAudio does not implement
  a long-form wrapper for Cohere yet).
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
