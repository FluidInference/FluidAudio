# TTS Benchmarks

Quantitative comparison of FluidAudio's TTS backends. Numbers are produced
by `fluidaudio tts-benchmark`, which is intended to make the
*latency-vs-efficiency-vs-quality* tradeoff visible without anyone having
to re-derive it from RTFx alone.

> Status: scaffolding. Slice-1 wires only **Kokoro ANE**; PocketTTS,
> StyleTTS2, Magpie, and CosyVoice3 land in follow-ups. The Kokoro ANE row
> below uses placeholder values until the harness is run on a reference
> M-series machine and committed.

## Why not just RTFx?

RTFx (audio_seconds / synth_seconds) is a useful single number for batch
synthesis, but for conversational use it hides the things users actually
feel:

1. **Cold start** — first model load + ANE compile after install or
   reboot. On Apple Silicon the system's `anecompilerservice` can take
   ~20s on first invocation; subsequent loads finish in ~1s. RTFx says
   nothing about this.
2. **TTFT (time-to-first-audio)** — for streaming agents the question is
   "how long until the user hears *something*", not "how long until the
   whole utterance is rendered".
3. **Per-stage compute units** — Kokoro ANE / PocketTTS / StyleTTS2 are
   pipelines of 4–7 CoreML graphs. Sometimes ANE is *slower per call*
   but more efficient (lower power, lower CPU contention). The "right"
   compute-unit choice differs per stage.
4. **Memory footprint** — drives whether a backend is mobile-viable.
5. **Quality** — RTFx alone tells you nothing about whether the model
   pronounced "Reykjavík" or "$1,234.56" correctly. We measure WER + CER
   via Parakeet roundtrip on a fixed corpus.

So `tts-benchmark` reports all of the above instead of a single RTFx.

## Methodology

### Corpus

Three categories ship under `Benchmarks/tts/corpus/`:

| Name          | Phrases | Stress                                      |
|---------------|---------|---------------------------------------------|
| `prose-en`    | 20      | Harvard sentences (PB-balanced, IEEE 269)   |
| `numbers-en`  | 10      | Currencies, dates, phone numbers, time, units |
| `names-en`    | 10      | Proper nouns, acronyms, brands, jargon      |

Lines beginning with `#` are comments. Custom corpora can be passed via
`--corpus-path <file.txt>`.

### Metrics

Per phrase:
- `ttft_ms` — time-to-first-audio. For one-shot backends this equals
  `synth_ms`; streaming backends will report the timestamp of the first
  emitted PCM chunk.
- `synth_ms` — total synth wall time.
- `audio_ms` — generated audio duration.
- `rtfx` — `audio_ms / synth_ms`.
- `wer`, `cer` — via Parakeet ASR roundtrip on the rendered WAV.
- `stage_ms` — per-stage breakdown (backend-specific keys).
- `encoder_tokens`, `acoustic_frames` — Kokoro ANE only.

Aggregates:
- `cold_start_s` — `manager.initialize()` wall time.
- `first_synth_ms` — first synth call after init (still cold-ish).
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
tradeoff (e.g. ANE-only often wins power but loses tail latency on
Vocoder; CPU+GPU often wins tail latency but spikes CPU).

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
roundtrip.

## Results

### Per-backend top-line

Reference machine TBD (intent: M-series Mac, AC-powered, Quiet thermals).
Cells below are placeholders until the harness is run and the values are
committed.

| Backend     | License     | Languages              | Footprint | Cold start | Warm TTFT p50 / p95 | Agg RTFx | Peak RSS | WER prose | CER prose | Mobile-ready |
|-------------|-------------|------------------------|-----------|------------|---------------------|----------|----------|-----------|-----------|--------------|
| Kokoro ANE  | Apache-2.0  | en (af_heart)          | ~330 MB   | TBD        | TBD / TBD           | TBD      | TBD      | TBD       | TBD       | TBD          |
| Kokoro      | Apache-2.0  | en + ja + zh + …       | ~330 MB   | —          | —                   | —        | —        | —         | —         | —            |
| PocketTTS   | research    | en + de + it + pt + es + fr | ~140 / ~520 MB | — | —             | —        | —        | —         | —         | —            |
| StyleTTS2   | MIT         | en                     | TBD       | —          | —                   | —        | —        | —         | —         | —            |
| Magpie      | research    | multilingual (357M)    | ~1.3 GB   | —          | —                   | —        | —        | —         | —         | —            |
| CosyVoice3  | Apache-2.0  | multilingual (cloning) | TBD       | —          | —                   | —        | —        | —         | —         | —            |

Em-dashes (`—`) mark backends not yet wired to `tts-benchmark`.

### Kokoro ANE — compute-unit sweep

Placeholder. Re-run after the first reference machine measurement.

| Preset         | Cold start (s) | Warm TTFT p50 (ms) | Warm TTFT p95 (ms) | Agg RTFx | Peak RSS (MB) |
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
    "backend": "kokoro-ane",
    "voice": "af_heart",
    "corpus": "prose-en",
    "phrase_count": 20,
    "compute_units": "default",
    "cold_start_s": 19.87,
    "first_synth_ms": 412,
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
      "encoder_tokens": 28, "acoustic_frames": 81,
      "stage_ms": {
        "albert": 4, "post_albert": 2, "alignment": 5,
        "prosody": 12, "noise": 7, "vocoder": 31, "tail": 6,
        "total": 67
      },
      "wav_path": ""
    }
  ]
}
```

Numbers above are illustrative; the actual reference values are written
the first time the harness is run and committed.

## How to add a backend

Each backend lands in its own PR:

1. Add `init(preset: TtsComputeUnitPreset)` on the backend's compute-unit
   struct (mirroring `KokoroAneComputeUnits.init(preset:)`).
2. Add a `runX` driver in `TtsBenchmarkCommand` that returns the same
   `summary` / `categories` / `phrases` JSON shape.
3. Extend `parseBackend(_:)` and remove the `slice-1` guard.
4. Re-run on the reference machine and replace the placeholder row in
   the **Per-backend top-line** table above.

## See also

- [TTS overview & status matrix](README.md)
- [Kokoro ANE deep dive](KokoroAne.md)
- [PocketTTS deep dive](PocketTTS.md)
- [Voice quality notes](voice-quality.md)
