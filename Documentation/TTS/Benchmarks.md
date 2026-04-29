# TTS Benchmarks

Quantitative comparison of FluidAudio's TTS backends. Numbers come from
`fluidaudio tts-benchmark`, which is intended to make the
*latency-vs-efficiency-vs-quality* tradeoff visible without anyone having
to re-derive it from RTFx alone.

> Status: scaffolding. The harness wires **Kokoro ANE, Kokoro,
> PocketTTS, Magpie, and CosyVoice3**. StyleTTS2 is intentionally absent
> — its synthesis path throws "not yet implemented" upstream. Numeric
> cells in the result tables are placeholders until the harness is run
> on a reference M-series machine and committed.

## Why not just RTFx?

RTFx (audio_seconds / synth_seconds) is a useful single number for batch
synthesis, but for conversational use it hides the things users actually
feel:

1. **Cold start** — first model load + ANE compile after install or
   reboot. On Apple Silicon the system's `anecompilerservice` can take
   ~20s on first invocation; subsequent loads finish in ~1s.
2. **TTFT (time-to-first-audio)** — for streaming agents the question is
   "how long until the user hears *something*", not "how long until the
   whole utterance is rendered". For one-shot backends in this slice
   `ttft_ms == synth_ms`; streaming-aware TTFT for PocketTTS / Magpie is
   a follow-up.
3. **Per-stage compute units** — Kokoro ANE / Magpie are pipelines of
   6–7 graphs. Sometimes ANE is *slower per call* but more efficient.
   The "right" compute-unit choice differs per stage.
4. **Memory footprint** — drives whether a backend is mobile-viable.
5. **Quality** — RTFx alone tells you nothing about whether the model
   pronounced "Reykjavík" or "$1,234.56" correctly. We measure WER + CER
   via Parakeet roundtrip on a fixed English corpus; non-English
   backends run with `--skip-asr` for now.

## Methodology

### Corpus

Four categories ship under `Benchmarks/tts/corpus/`:

| Name          | Phrases | Stress                                      |
|---------------|---------|---------------------------------------------|
| `prose-en`    | 20      | Harvard sentences (PB-balanced, IEEE 269)   |
| `numbers-en`  | 10      | Currencies, dates, phone numbers, time, units |
| `names-en`    | 10      | Proper nouns, acronyms, brands, jargon      |
| `prose-zh`    | 8       | Short Mandarin sentences (CosyVoice3, no WER) |

Lines beginning with `#` are comments. Custom corpora can be passed via
`--corpus-path <file.txt>`. CosyVoice3 defaults to `prose-zh`; everything
else defaults to `prose-en`.

### Metrics

Per phrase:
- `ttft_ms` — time-to-first-audio. For one-shot backends this equals
  `synth_ms`; streaming-aware TTFT for PocketTTS / Magpie is a follow-up.
- `synth_ms` — total synth wall time.
- `audio_ms` — generated audio duration.
- `rtfx` — `audio_ms / synth_ms`.
- `wer`, `cer` — via Parakeet ASR roundtrip on the rendered WAV.
- `stage_ms` — per-stage breakdown (backend-specific keys; empty for
  Kokoro / PocketTTS / CosyVoice3).
- Backend-specific extras: `encoder_tokens`, `acoustic_frames`,
  `chunk_count`, `frame_count`, `code_count`, `finished_on_eos`,
  `generated_token_count`, etc.

Aggregates:
- `cold_start_s` — `manager.initialize()` wall time. CosyVoice3 also
  includes voice-asset load.
- `first_synth_ms` — first synth call after init (still cold-ish).
- `ttft_ms_p50` / `ttft_ms_p95`.
- `warm_synth_ms_p50` / `warm_synth_ms_p95`.
- `agg_rtfx` — `Σ audio_ms / Σ synth_ms` across the corpus.
- `peak_rss_mb` — process-wide peak resident set, via
  `task_vm_info_data_t.resident_size_peak`.
- Per-category macro WER / CER.

### Compute-unit sweep

`--compute-units` selects how each stage is mapped to silicon:

| Preset         | Mapping                                              |
|----------------|------------------------------------------------------|
| `default`      | Backend's empirically-tuned per-stage layout         |
| `all-ane`      | Force every stage to `.cpuAndNeuralEngine`           |
| `cpu-and-gpu`  | Force every stage to `.cpuAndGPU`                    |
| `cpu-only`     | Force every stage to `.cpuOnly`                      |

Reporting all four for each backend exposes the latency-vs-efficiency
tradeoff. Caveats:

- **Kokoro ANE** has a per-stage compute-unit struct
  (`KokoroAneComputeUnits`) and honors all four presets natively.
- **Kokoro / Magpie / CosyVoice3** take a single `MLComputeUnits`; the
  preset maps via `TtsComputeUnitPreset.uniformUnits`.
- **PocketTTS** does not expose per-call compute-unit overrides — the
  preset is logged and ignored.

### Reproducibility

```bash
# From the package root.
swift run fluidaudio tts-benchmark \
  --backend kokoro-ane \
  --corpus prose-en \
  --voice af_heart \
  --compute-units default \
  --output-json bench.json \
  --audio-dir bench-wavs/
```

The harness writes a JSON report to `--output-json` and (optionally)
keeps WAVs under `--audio-dir`. Pass `--skip-asr` to drop the Parakeet
roundtrip. CosyVoice3 forces `--skip-asr`.

## Results

### Per-backend top-line

Reference machine TBD (intent: M-series Mac, AC-powered, Quiet thermals).
Cells below are placeholders until the harness is run and committed.

| Backend     | License     | Languages              | Footprint | Cold start | Warm synth p50 / p95 | Agg RTFx | Peak RSS | WER prose | CER prose | Notes |
|-------------|-------------|------------------------|-----------|------------|----------------------|----------|----------|-----------|-----------|-------|
| Kokoro ANE  | Apache-2.0  | en (af_heart only)     | ~330 MB   | TBD        | TBD / TBD            | TBD      | TBD      | TBD       | TBD       | per-stage CU sweep |
| Kokoro      | Apache-2.0  | en (af_heart, others untested) | ~330 MB | TBD | TBD / TBD       | TBD      | TBD      | TBD       | TBD       | single CU |
| PocketTTS   | research    | en + de + it + pt + es + fr (6L / 24L) | ~140 / ~520 MB | TBD | TBD / TBD | TBD | TBD | TBD       | TBD       | streaming-capable |
| StyleTTS2   | MIT         | en                     | TBD       | —          | —                    | —        | —        | —         | —         | synthesis stub |
| Magpie      | research    | en/es/de/fr/it/vi/zh/hi | ~1.3 GB   | TBD        | TBD / TBD            | TBD      | TBD      | TBD       | TBD       | 6-stage timings |
| CosyVoice3  | Apache-2.0  | zh (mandarin)          | TBD       | TBD        | TBD / TBD            | TBD      | TBD      | n/a       | n/a       | macOS 15+, no WER |

Em-dash (`—`) marks backends not wired to `tts-benchmark` (StyleTTS2).

### Kokoro ANE — compute-unit sweep

Placeholder. Re-run after the first reference machine measurement.

| Preset         | Cold start (s) | Warm synth p50 (ms) | Warm synth p95 (ms) | Agg RTFx | Peak RSS (MB) |
|----------------|----------------|---------------------|---------------------|----------|----------------|
| `default`      | TBD            | TBD                 | TBD                 | TBD      | TBD            |
| `all-ane`      | TBD            | TBD                 | TBD                 | TBD      | TBD            |
| `cpu-and-gpu`  | TBD            | TBD                 | TBD                 | TBD      | TBD            |
| `cpu-only`     | TBD            | TBD                 | TBD                 | TBD      | TBD            |

### Kokoro ANE — per-stage breakdown (default preset)

Placeholder. Stages map to the 7-CoreML-graph split documented in
[KokoroAne.md](KokoroAne.md).

| Stage        | Mean ms | % of total |
|--------------|---------|------------|
| `albert`     | TBD     | TBD        |
| `post_albert`| TBD     | TBD        |
| `alignment`  | TBD     | TBD        |
| `prosody`    | TBD     | TBD        |
| `noise`      | TBD     | TBD        |
| `vocoder`    | TBD     | TBD        |
| `tail`       | TBD     | TBD        |
| **total**    | TBD     | 100%       |

### Magpie — per-stage breakdown (default preset)

Placeholder. 6 stages reported in `MagpieSynthesisTimings`.

| Stage          | Mean ms | % of total |
|----------------|---------|------------|
| `text_encoder` | TBD     | TBD        |
| `prefill`      | TBD     | TBD        |
| `ar_loop`      | TBD     | TBD        |
| `decoder_step` | TBD     | TBD        |
| `sampler`      | TBD     | TBD        |
| `nanocodec`    | TBD     | TBD        |
| **total**      | TBD     | 100%       |

### Per-category quality (Kokoro ANE, default preset)

Placeholder.

| Corpus       | Phrases | Macro WER | Macro CER | Synth p50 (ms) | Synth p95 (ms) | RTFx |
|--------------|---------|-----------|-----------|-----------------|-----------------|------|
| `prose-en`   | 20      | TBD       | TBD       | TBD             | TBD             | TBD  |
| `numbers-en` | 10      | TBD       | TBD       | TBD             | TBD             | TBD  |
| `names-en`   | 10      | TBD       | TBD       | TBD             | TBD             | TBD  |

## JSON report schema

```jsonc
{
  "summary": {
    "backend": "kokoro-ane",       // or "kokoro" / "pocket-tts" / "magpie" / "cosyvoice3"
    "voice": "af_heart",            // backends with a string voice
    "speaker": "John",              // magpie only
    "language": "en",               // pocket-tts / magpie
    "corpus": "prose-en",
    "phrase_count": 20,
    "compute_units": "default",
    "cold_start_s": 19.87,
    "first_synth_ms": 412,
    "ttft_ms_p50": 73,
    "ttft_ms_p95": 119,
    "warm_synth_ms_p50": 73,
    "warm_synth_ms_p95": 119,
    "agg_rtfx": 28.4,
    "peak_rss_mb": 643,
    "asr_skipped": false
  },
  "categories": [
    { "category": "prose-en", "phrase_count": 20,
      "macro_wer": 0.012, "macro_cer": 0.004,
      "synth_ms_p50": 73, "synth_ms_p95": 119, "rtfx": 28.4 }
  ],
  "phrases": [
    {
      "index": 1, "category": "prose-en",
      "reference": "The birch canoe slid on the smooth planks.",
      "hypothesis": "the birch canoe slid on the smooth planks",
      "ttft_ms": 67, "synth_ms": 67, "audio_ms": 2010, "rtfx": 30.0,
      "wer": 0.0, "cer": 0.0, "asr_ms": 142,
      "stage_ms": {
        // Kokoro ANE: 7 stages + total
        "albert": 4, "post_albert": 2, "alignment": 5,
        "prosody": 12, "noise": 7, "vocoder": 31, "tail": 6,
        "total": 67
        // Magpie: text_encoder, prefill, ar_loop, decoder_step, sampler, nanocodec
        // Other backends: empty {}
      },
      // Backend-specific extras (any subset):
      "encoder_tokens": 28, "acoustic_frames": 81,         // kokoro-ane
      "chunk_count": 1, "wav_bytes": 96108,                // kokoro
      "frame_count": 25, "eos_step": -1,                   // pocket-tts
      "code_count": 81, "finished_on_eos": true,           // magpie
      "generated_token_count": 142, "decoded_token_count": 142, // cosyvoice3
      "wav_path": ""
    }
  ]
}
```

Numbers above are illustrative; the actual reference values are written
the first time the harness is run and committed.

## How to add or improve a backend

Each driver is ~70 lines and lives in `Sources/FluidAudioCLI/Commands/TtsBenchmarkCommand.swift`:

1. Implement `runX(...)` that mirrors the existing pattern: build the
   manager with the requested compute units, time `initialize()`, run a
   single warm-up synth, then call `runPhraseLoop` with a closure that
   returns `BackendPhraseSample`.
2. If the backend has its own per-stage compute-units struct, add an
   `init(preset: TtsComputeUnitPreset)` mirroring
   `KokoroAneComputeUnits.init(preset:)`.
3. Add the backend's case to the `Backend` enum and the `parseBackend`
   dispatch.
4. Re-run on the reference machine and replace the placeholder row in
   the **Per-backend top-line** table above.

Streaming-aware TTFT (using `synthesizeStreaming` for PocketTTS and
`synthesizeStream` for Magpie) is a follow-up — current TTFT == total
synth time for one-shot drivers.

## See also

- [TTS overview & status matrix](README.md)
- [Kokoro ANE deep dive](KokoroAne.md)
- [Kokoro deep dive](Kokoro.md)
- [PocketTTS deep dive](PocketTTS.md)
- [Magpie deep dive](Magpie.md)
- [CosyVoice3 deep dive](CosyVoice3.md)
- [Voice quality notes](voice-quality.md)
