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

Apple M2, FLEURS test set, int8 encoder, `MLComputeUnits.cpuAndNeuralEngine`.

### Normalizer

Scoring follows the [HF Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) convention used by NVIDIA in the Canary/Parakeet-v3 paper:

- **English** → `EnglishTextNormalizer` (whisper-normalizer 0.1.12 equivalent: contraction expansion, British→American, number folding, abbreviation expansion). Our `TextNormalizer.normalize()`.
- **Non-English Latin** (fr, de, es, it, pt, …) → `BasicTextNormalizer(remove_diacritics=False)` plus an inverse text normalization (ITN) pass: digit runs in the reference are spelled out via `NumberFormatter.spellOut` for the language's locale before WER computation. Required because the model emits "mille neuf cent soixante-seize" while FLEURS keeps "1976" in the reference. Thousands separators handled across all five Unicode space variants FLEURS actually uses (U+0020/00A0/2007/2009/202F). Our `TextNormalizer.basicNormalize(_, spellOutLocale:)`.
- **CJK** (ja, ko, zh, th) → character-level edit rate after whitespace stripping (segmentation-free). Reported in the "WER" column by community convention.

### Chunk size sweep (FLEURS test split, full data)

All three builds use `att_context_size=[56,0]` (NVIDIA's lowest-latency mode); they differ only in `chunk_mel_frames` (32 / 56 / 112 → 320 / 560 / 1120 ms processing chunks). NVIDIA's published FLEURS numbers are also at `[56,0]`, so the comparison is architecturally apples-to-apples.

| Language | 320 ms | 560 ms | 1120 ms | NVIDIA ([56,0]) | Δ (1120 vs NVIDIA) | n   |
|----------|-------:|-------:|--------:|----------------:|-------------------:|----:|
| en_us    |  17.5  |  12.1  |   12.0  |         11.35   |             +0.65  | 647 |
| fr_fr    |  16.4  |  13.9  |   13.8  |         13.44   |             +0.36  | 676 |
| de_de    |  17.8  |  14.9  |   13.6  |           —     |               —    | 862 |
| es_419   |   8.6  |   7.4  |    7.4  |          8.69   |             −1.29  | 908 |
| ja_jp    |  21.9  |  18.4  |   17.4  |           —     |               —    | 650 |
| it_it    |   9.8  |   7.9  |    7.4  |          7.33   |             +0.07  | 865 |
| pt_br    |  13.4  |  10.0  |    8.4  |          8.99   |             −0.59  | 919 |
| **AVG**  |**15.0**|**12.1**|**11.4** |                 |                    |     |
| RTFx     |   8.6  |  16.8  |   22.0  |                 |                    |     |

WER% for spaced scripts, CER% for ja_jp (segmentation-free). Full `google/fleurs` test splits (en=647, fr=676, de=862, es=908, ja=650, it=865, pt=919). The "Δ (1120 vs NVIDIA)" column compares our highest-accuracy build against NVIDIA's published number for the same `[56,0]` attention mode.

**All 5 published languages are within ~0.7 pp of NVIDIA at 1120 ms.** es-419 and pt-br actually beat the reference (−1.29 and −0.59 pp respectively); en, fr, it are +0.65 / +0.36 / +0.07. At 560 ms (the recommended low-latency build) all 5 are within ~1 pp; es-419 still beats NVIDIA by −1.29 pp.

**320 ms shows boundary effects on English and accent-heavy languages.** en_us jumps from 12.0 → 17.5 (+5.5 pp) and pt_br from 8.4 → 13.4 (+5.0 pp) when dropping from 1120 ms to 320 ms. 560 ms recovers most of the loss (<1.6 pp from 1120 ms on every language). If you need low latency, ship 560 ms; only use 320 ms if you absolutely need sub-half-second response and can tolerate the English regression.

### Caveats

- **`MLComputeUnits` matters a lot.** Default `.all` routes the int8 encoder to GPU and runs ~10× slower than ANE. The manager pins `.cpuAndNeuralEngine` automatically; do not override unless you have a reason.
- **int8 vs fp16 is a wash.** Average WER is identical at all three chunk sizes; per-language drift is within ±1 pp. Ship int8 for the 50% size win and ANE residency.
- **Two independent latency axes.** NVIDIA's published modes (`att_context_size = [56,0] / [56,3] / [56,6] / [56,13]` → ~80 / 320 / 560 / 1120 ms architectural lookahead) control right-context inside the encoder. Our `320 / 560 / 1120 ms` build labels refer to `chunk_mel_frames` (processing chunk size), not lookahead. All FluidAudio builds currently ship `[56,0]` (no lookahead).
- **CJK languages** use character-level edit rate as the "WER" field by convention; whitespace tokenization is meaningless for ja/ko/zh/th.

## See Also

- [Nemotron.md](Nemotron.md) — English-only variant (also auto-downloads from HuggingFace)
- [TokenLanguageFilter.md](TokenLanguageFilter.md) — how `<xx-XX>` tags are filtered
- `mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/README.md` — conversion pipeline
- `mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/bench_results/int8_summary.md` — encoder-level int8 trade-off report
