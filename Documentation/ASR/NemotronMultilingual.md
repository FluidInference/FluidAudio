# Nemotron Speech Streaming Multilingual 0.6B

FluidAudio supports NVIDIA's `nemotron-asr-streaming-multilingual-0.6b` for real-time streaming ASR across ~40 languages on Apple Silicon.

## Overview

| Property | Value |
|----------|-------|
| Source Model | `nvidia/nemotron-asr-streaming-multilingual-0.6b` (intermediate checkpoint, May 2026) |
| Architecture | FastConformer Cache-Aware RNNT **with Prompt** |
| Parameters | 0.6B |
| Languages | ~40 (en, es, de, fr, it, pt, ar, ja, ko, zh-CN, ru, hi, vi, …) |
| Default Latency Modes | 560 ms · 1120 ms · 2240 ms (each is a separate CoreML build) |
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

### Chunk size sweep (FLEURS full test split)

Re-measured 2026-06-28 with the **native-Swift mel front-end** (`NemotronMelExtractor`;
no CoreML preprocessor — see issue #739) over the full `google/fleurs` test splits. All
builds use `att_context_size=[56,0]`; they differ only in `chunk_mel_frames` → processing
chunk size. The shipped tiers are now **560 / 1120 / 2240 ms** (the earlier 320 ms tier was
dropped, 2240 ms added). The per-language vocab-pruned ship and the full multilingual ship
score identically (en_us @ 2240 ms = 8.72 % on both), so the table uses the full ship.

| Language | 560 ms | 1120 ms | 2240 ms | NVIDIA ([56,0]) | n   |
|----------|-------:|--------:|--------:|----------------:|----:|
| en_us    |  9.05  |   8.73  |   8.72  |         11.35   | 647 |
| fr_fr    |  9.80  |   9.44  |   9.36  |         13.44   | 676 |
| de_de    | 10.61  |  10.01  |   9.96  |           —     | 862 |
| es_419   |  4.85  |   4.75  |   4.73  |          8.69   | 908 |
| ja_jp    | 14.27  |  13.79  |  13.78  |           —     | 650 |
| it_it    |  5.40  |   5.43  |   5.39  |          7.33   | 865 |
| pt_br    |  6.38  |   6.16  |   6.19  |          8.99   | 919 |
| **AVG**  |**8.62**|**8.33** |**8.30** |                 |     |
| agg RTFx | 40.5x  | 66.0x   |  73.1x  |                 |     |

WER% for spaced scripts, CER% for ja_jp (segmentation-free, whitespace-stripped). Same
normalizer pipeline as the row above (HF Open-ASR-Leaderboard convention). Aggregate RTFx
is total audio ÷ total processing across all 7 languages, end-to-end single-stream on Apple
Silicon (machine/load-dependent — treat the relative ordering, not the absolute, as meaningful).

**Accuracy improves monotonically with chunk size** and meets-or-beats NVIDIA's published
`[56,0]` numbers on all five published languages (at 2240 ms: en −2.6, fr −4.1, es −4.0,
it −1.9, pt −2.8 pp). These numbers are ~2–4 pp better than the prior version of this table;
the gain comes from model / decode-path / normalizer updates since it was written — **not**
the Swift mel port, which is numerically parity to the removed CoreML preprocessor
(max |Δ| ≈ 9e-3 vs NeMo PyTorch, confirmed at conversion time). Cross-comparison to NVIDIA is
sensitive to normalization and should be read as indicative.

Reproduce (one run per tier):

```bash
swift run -c release fluidaudiocli nemotron-multilingual-benchmark \
    --model-dir <multilingual ship dir> \
    --languages en_us,fr_fr,de_de,es_419,ja_jp,it_it,pt_br \
    --samples all --chunk-ms <560|1120|2240> --output results.json
```

### Caveats

- **`MLComputeUnits` matters a lot.** Default `.all` routes the int8 encoder to GPU and runs ~10× slower than ANE. The manager pins `.cpuAndNeuralEngine` automatically; do not override unless you have a reason.
- **int8 vs fp16 is a wash.** Average WER is identical at all three chunk sizes; per-language drift is within ±1 pp. Ship int8 for the 50% size win and ANE residency.
- **Two independent latency axes.** NVIDIA's published modes (`att_context_size = [56,0] / [56,3] / [56,6] / [56,13]` → ~80 / 320 / 560 / 1120 ms architectural lookahead) control right-context inside the encoder. Our `560 / 1120 / 2240 ms` build labels refer to `chunk_mel_frames` (processing chunk size), not lookahead. All FluidAudio builds currently ship `[56,0]` (no lookahead).
- **CJK languages** use character-level edit rate as the "WER" field by convention; whitespace tokenization is meaningless for ja/ko/zh/th.
- **Punctuation density drops at small chunk sizes** ([#687](https://github.com/FluidInference/FluidAudio/issues/687)). On long continuous speech the 560 ms build starts punctuating normally, then commas/periods become increasingly sparse as the session continues; 1120 ms and 2240 ms retain noticeably more punctuation on the same audio, and a session reset restores it. The words themselves are unaffected (WER-neutral) — only punctuation marks thin out. Cause is model-side: shorter chunks give the encoder less right context at sentence boundaries than the published builds' `att_context_size` assumes, and greedy RNN-T decoding compounds the miss over the session. If punctuation matters for your use case, ship 1120 ms or larger, or segment long streams (e.g. reset on VAD silence).

## See Also

- [Nemotron.md](Nemotron.md) — English-only variant (also auto-downloads from HuggingFace)
- [TokenLanguageFilter.md](TokenLanguageFilter.md) — how `<xx-XX>` tags are filtered
- `mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/README.md` — conversion pipeline
- `mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/bench_results/int8_summary.md` — encoder-level int8 trade-off report
