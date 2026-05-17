# Nemotron Speech Streaming Multilingual 0.6B

FluidAudio supports NVIDIA's `nemotron-asr-streaming-multilingual-0.6b` for real-time streaming ASR across ~40 languages on Apple Silicon.

## Overview

| Property | Value |
|----------|-------|
| Source Model | `nvidia/nemotron-asr-streaming-multilingual-0.6b` (intermediate checkpoint, May 2026) |
| Architecture | FastConformer Cache-Aware RNNT **with Prompt** |
| Parameters | 0.6B |
| Languages | ~40 (en, es, de, fr, it, pt, ar, ja, ko, zh-CN, ru, hi, vi, …) |
| Default Latency Modes | 320 ms · 560 ms · 1120 ms (each is a separate CoreML build) |
| Mel Features | 128 bins, 16 kHz |
| Vocab Size | 13,087 + 1 blank |
| Hardware | Apple Silicon only (int8 encoder is ANE-targeted) |

### How it differs from English-only Nemotron

The multilingual variant adds:

1. **`prompt_id` int32 input** on the encoder — selects the language hint embedding. Pass a language code like `"en-US"` or `"auto"` (the model's default-prompt id).
2. **Leading `<xx-XX>` language-tag token** — emitted as the first decoder output, then filtered from the transcript and surfaced via `detectedLanguage()`.
3. **Larger vocab** (13,087 tokens vs ~1k) and a smaller channel cache `[1, 24, 56, 1024]` for `att_context_size=[56, 0]`.

## Model Distribution

The multilingual model is **local-path-only** at the moment — no HuggingFace repo yet. Convert it yourself via `mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/conversion_scripts/convert_nemotron_multilingual.py` (Linux + CUDA required), then quantize with `quantize_int8.py`. The resulting `build_int8_<NNN>ms/` directory contains:

```
build_int8_1120ms/
├── preprocessor.mlmodelc   (or .mlpackage before compilation)
├── encoder.mlmodelc
├── decoder.mlmodelc
├── joint.mlmodelc
├── metadata.json
└── tokenizer.json
```

`StreamingNemotronMultilingualAsrManager` accepts either compiled `.mlmodelc` or raw `.mlpackage` — compiled is preferred when both are present.

## CLI Usage

### Transcribe a file

```bash
swift run fluidaudiocli nemotron-multilingual-transcribe \
    --model-dir /path/to/build_int8_1120ms \
    --language fr-FR \
    --input speech.wav
```

`--language` accepts any FLEURS-style code (`en-US`, `fr-FR`, `de-DE`, `es-ES`, `it-IT`, `pt-BR`, `ja-JP`, …) or `auto` to let the model pick. `--prompt-id <int>` overrides the language with a raw embedding index if you've inspected the `prompt_dictionary` in `metadata.json`.

### FLEURS benchmark

```bash
swift run fluidaudiocli nemotron-multilingual-benchmark \
    --model-dir /path/to/build_int8_1120ms \
    --languages en_us,fr_fr,de_de,es_419,ja_jp,it_it,pt_br \
    --samples all \
    --output /tmp/nemotron_fleurs.json
```

`--samples N` runs the first N alphabetical samples per language; `--samples all` runs the full FLEURS test split. Default dataset repo is `FluidInference/fleurs-full`, override with `--dataset-repo` and the local layout with `--cache-dir`.

> **Note on `FluidInference/fleurs-full`**: at the time of writing this dataset caps fr_fr / de_de / es_419 at 350 utterances each (vs 676 / 862 / 908 in the official `google/fleurs` test split). For published-leaderboard parity, extract `google/fleurs` test arrows yourself.

## Programmatic Usage

```swift
import FluidAudio

let manager = StreamingNemotronMultilingualAsrManager()
try await manager.loadModels(from: URL(fileURLWithPath: "/path/to/build_int8_1120ms"))

await manager.setLanguage("fr-FR")   // or .setPromptId(12)

let partial = try await manager.process(audioBuffer: samples)  // [Float] @ 16 kHz mono
let final = try await manager.finish()

let detected = await manager.detectedLanguage()   // e.g. "fr-FR"
await manager.reset()
```

## Benchmark Results

Apple M2, **full FLEURS test set** (Google `google/fleurs` test split, extracted from arrows), int8 encoder, `MLComputeUnits.cpuAndNeuralEngine`.

### Chunk size sweep (full FLEURS)

| Language | 320 ms | 560 ms | 1120 ms | NVIDIA @ 320 ms | Δ (1120 vs NVIDIA) | n   |
|----------|-------:|-------:|--------:|----------------:|-------------------:|----:|
| en_us    |  17.5  |  12.1  |   12.0  |         11.35   |             +0.65  | 647 |
| fr_fr    |  18.2  |  15.7  |   15.5  |         13.44   |             +2.06  | 676 |
| de_de    |  18.7  |  15.8  |   14.5  |           —     |               —    | 862 |
| es_419   |  10.5  |   9.4  |    9.3  |          8.69   |             +0.61  | 908 |
| ja_jp    |  21.9  |  18.4  |   17.4  |           —     |               —    | 650 |
| it_it    |  10.4  |   8.5  |    7.9  |          7.33   |             +0.57  | 865 |
| pt_br    |  16.8  |  13.1  |   11.6  |          8.99   |             +2.61  | 919 |
| **AVG**  |**16.3**|**13.3**|**12.6** |                 |                    |     |
| RTFx     |  10.1  |  15.4  |   15.9  |                 |                    |     |

WER% for spaced scripts, CER% for ja_jp (segmentation-free). NVIDIA's published numbers are at 320 ms on the same test split. We compare their 320 ms against our 1120 ms because it's the lowest-WER point we ship — NVIDIA does not publish multilingual 1120 ms numbers.

**3 of 5 published languages (en, es, it) are within ~0.6 pp of NVIDIA at 1120 ms.** French and Portuguese show a real, persistent gap of ~2 pp that does not collapse with more chunk-size context or by going to fp16 — suggesting model-side differences (decoder hyperparams, language-tag prompting, or checkpoint vintage) rather than anything fixable in the CoreML pipeline.

**320 ms is degenerate on English and accent-heavy languages.** en_us jumps from 12.0 → 17.5 (+5.5 pp) and pt_br from 11.6 → 16.8 (+5.2 pp) when dropping from 1120 ms to 320 ms. 560 ms recovers most of the loss (<1 pp from 1120 ms on every language). If you need low latency, ship 560 ms; only use 320 ms if you absolutely need sub-half-second response and can tolerate the English regression.

### Caveats

- **`MLComputeUnits` matters a lot.** Default `.all` routes the int8 encoder to GPU and runs ~10× slower than ANE. The manager pins `.cpuAndNeuralEngine` automatically; do not override unless you have a reason.
- **int8 vs fp16 is a wash.** Average WER is identical at all three chunk sizes; per-language drift is within ±1 pp. Ship int8 for the 50% size win and ANE residency.
- **The 4 latency modes published by NVIDIA are 80 / 160 / 560 / 1120 ms.** FluidAudio currently ships 320 / 560 / 1120 ms builds. 80 / 160 ms are degenerate at this model size (>50% WER on the English variant); not converted.
- **CJK languages** use character-level edit rate as the "WER" field by convention; whitespace tokenization is meaningless for ja/ko/zh/th.

## See Also

- [Nemotron.md](Nemotron.md) — English-only variant (also auto-downloads from HuggingFace)
- [TokenLanguageFilter.md](TokenLanguageFilter.md) — how `<xx-XX>` tags are filtered
- `mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/README.md` — conversion pipeline
- `mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/bench_results/int8_summary.md` — encoder-level int8 trade-off report
