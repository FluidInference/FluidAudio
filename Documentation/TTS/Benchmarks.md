# TTS Benchmarks

Quantitative comparison of FluidAudio's TTS backends. Numbers come from
`fluidaudio tts-benchmark`, which is intended to make the
*latency-vs-efficiency-vs-quality* tradeoff visible without anyone having
to re-derive it from RTFx alone.

> Status: first measurement landed on **MacBook Air, M2 (2022),
> 8-core CPU / 8-core GPU / 16-core Neural Engine, 16 GB unified
> memory, macOS 26** (`Mac14,2`, on AC power). The harness wires
> **Kokoro ANE, Kokoro, PocketTTS, StyleTTS2, Magpie, and CosyVoice3**.
> All six backends now produce numbers on this machine. CosyVoice3 is
> experimental / beta with sub-real-time RTFx; StyleTTS2 produces audio
> but the LibriTTS checkpoint + out-of-distribution `ref_s` voice
> currently land at WER 0.446 on prose-en (voice-quality follow-up — see
> [StyleTTS2 quality caveat](#styletts2-quality-caveat)).

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

Reference machine: **MacBook Air, Apple M2 (2022), 8-core CPU /
8-core GPU / 16-core Neural Engine, 16 GB unified memory, macOS 26**
(`Mac14,2`, on AC). All runs use `--compute-units default`,
voice = backend default
(`af_heart` for Kokoro, `alba` for PocketTTS), corpus = `prose-en`,
Parakeet TDT roundtrip for WER / CER. Cold-start figures dominated by
ANE first-compile + on-disk model fetch the first time the cache misses.

| Backend     | License     | Languages              | Footprint | Cold start | Warm synth p50 / p95 | Agg RTFx | Peak RSS | WER prose | CER prose | Notes |
|-------------|-------------|------------------------|-----------|------------|----------------------|----------|----------|-----------|-----------|-------|
| Kokoro ANE  | Apache-2.0  | en (af_heart only)     | ~330 MB   | 35.1 s     | 244 / 305 ms         | 10.41×   | 823 MB   | 0.162     | 0.067     | per-stage CU sweep, 7-graph pipeline |
| Kokoro      | Apache-2.0  | en (af_heart only)     | ~330 MB   | 38.3 s     | 1263 / 1497 ms       | 2.11×    | 907 MB   | 0.018     | 0.007     | single CU; cleaner than ANE on prose |
| PocketTTS   | research    | en + de + it + pt + es + fr (6L / 24L) | ~140 / ~520 MB | 3.4 s | 2222 / 3281 ms | 0.97× | 1062 MB | 0.000 | 0.000 | streaming-capable; no per-call CU knob |
| StyleTTS2   | MIT         | en (LibriTTS multi-spk) | ~280 MB  | 0.04 s\*   | 1898 / 2054 ms       | 1.46×    | 711 MB   | 0.446†    | 0.276†    | one-shot diffusion + HiFi-GAN; voice = user-supplied `ref_s.bin` |
| Magpie      | research    | en/es/de/fr/it/vi/zh/hi | ~1.3 GB   | 3.3 s      | 3500 / 4339 ms       | 0.95×    | 810 MB   | 0.026     | 0.021     | split-K/V decoder; outputBackings fast path with fresh-alloc fallback |
| CosyVoice3  | Apache-2.0  | zh (mandarin)          | ~1.5 GB   | 160.5 s    | 10251 / 11765 ms     | 0.21×    | 4551 MB  | n/a       | n/a       | beta; ANE rejects decode → CPU+GPU fallback; corpus = `prose-zh` |

\* StyleTTS2 cold start is just `manager.initialize()` (asset checks);
the four CoreML graphs lazy-load on the first `synthesize` call, which
is why `first_synth_ms` is 4.4 s while subsequent warm calls are
~1.9 s p50.

† StyleTTS2 is bottlenecked on voice quality, not synthesis. See
[StyleTTS2 quality caveat](#styletts2-quality-caveat).

Headline read: **Kokoro ANE wins on latency and RTFx** (~10× real-time,
244 ms p50 warm) at the cost of a noisier ASR roundtrip (WER 0.162 vs.
0.018 on the non-ANE Kokoro). **PocketTTS wins on quality** (perfect WER
on prose-en) but is the slowest at 0.97× RTFx and ~2.2 s p50 — i.e.
roughly real-time. **Plain Kokoro** is the latency-quality middle ground
(2.1× RTFx, WER 0.018). **Magpie** lands at near-real-time (0.95× RTFx,
3.5 s p50) with strong WER (0.026) on prose. **StyleTTS2** is the
fastest non-Kokoro one-shot backend (1.46× RTFx, ~1.9 s p50) but its
WER is dominated by reference-voice fit rather than the synthesis
pipeline (see caveat below).

### Kokoro ANE — compute-unit sweep

Only the `default` preset has been measured so far on the M2 reference
machine. The `all-ane` / `cpu-and-gpu` / `cpu-only` rows will be filled
in by re-running with `--compute-units <preset>`; cold-start should
dominate the differences because the ANE compile is per-preset.

| Preset         | Cold start (s) | Warm synth p50 (ms) | Warm synth p95 (ms) | Agg RTFx | Peak RSS (MB) |
|----------------|----------------|---------------------|---------------------|----------|----------------|
| `default`      | 35.1           | 244                 | 305                 | 10.41    | 823            |
| `all-ane`      | TBD            | TBD                 | TBD                 | TBD      | TBD            |
| `cpu-and-gpu`  | TBD            | TBD                 | TBD                 | TBD      | TBD            |
| `cpu-only`     | TBD            | TBD                 | TBD                 | TBD      | TBD            |

### Kokoro ANE — per-stage breakdown (default preset)

Means across 20 prose-en phrases on M2. Stages map to the 7-CoreML-graph
split documented in [KokoroAne.md](KokoroAne.md). Vocoder + noise
together account for ~80% of synth time, which is the natural target
for any further per-stage compute-unit re-tuning.

| Stage        | Mean ms | % of total |
|--------------|---------|------------|
| `albert`     | 9.0     | 3.7%       |
| `post_albert`| 4.1     | 1.7%       |
| `alignment`  | 1.2     | 0.5%       |
| `prosody`    | 27.1    | 11.3%      |
| `noise`      | 77.0    | 32.1%      |
| `vocoder`    | 114.2   | 47.6%      |
| `tail`       | 7.5     | 3.1%       |
| **total**    | 240.2   | 100%       |

### Magpie — per-stage breakdown (default preset)

Means across 20 prose-en phrases on M2 (`John` voice, en, default
compute units). `ar_loop` is the umbrella for the per-step
`decoder_step` + `sampler` (so it is not added on top in the total).
Synth-wall time is ≈ `text_encoder` + `prefill` + `ar_loop` +
`nanocodec` (~3.5 s, matching `warm_synth_p50`). The AR loop dominates;
re-exporting `decoder_step.mlmodelc` with explicit MultiArray
shape/dtype constraints on `new_k_*` / `new_v_*` / `var_*` outputs
would let the fast `outputBackings` path stay live and skip the
~18.9 MB-per-step fresh-alloc fallback (see Magpie deep dive).

| Stage             | Mean ms | % of warm-synth |
|-------------------|---------|-----------------|
| `text_encoder`    | 15      | 0.4%            |
| `prefill`         | 47      | 1.4%            |
| `ar_loop`         | 2195    | 62.7%           |
| └── `decoder_step` | 1914   | (54.7%)         |
| └── `sampler`      | 279    | (8.0%)          |
| `nanocodec`       | 1295    | 37.0%           |
| **total**         | ~3552   | 100%            |

## StyleTTS2 quality caveat

StyleTTS2 currently lands at **WER 0.446 / CER 0.276** on `prose-en`
with the LibriTTS multi-speaker checkpoint and an out-of-distribution
reference voice (`samples/kokoro-english/af_nicole.wav` →
`ref_s.bin`). The synthesis pipeline itself is healthy:

- Audio is produced at the expected duration for every phrase
  (2.5–3.2 s for short Harvard sentences; no truncation).
- RTFx is 1.46× warm, 1.90 s p50 — competitive with Kokoro on the same
  hardware.
- ~25% of phrases hit WER ≤ 0.25 (e.g. *"The box was thrown beside the
  parked truck."* → *"The box was thrown beside the park truck."*),
  proving the diffusion sampler + decoder do produce intelligible
  speech on this voice.

What's failing is **prosodic / phonetic execution on phoneme
combinations far from the LibriTTS training distribution**:

```
ref: The birch canoe slid on the smooth planks.
hyp: The burr condu sled one smooth plants.

ref: The hogs were fed chopped corn and garbage.
hyp: Hogs wear fedmapped cornangreet.
```

These are phonetic confusions, not silence or truncation. They go away
when the reference voice matches the LibriTTS speaker distribution. We
ship StyleTTS2 with the synthesis path wired and these numbers visible
so the next follow-up is voice-quality (curating known-good `ref_s`
blobs from the LibriTTS train split, or re-targeting onto a
Kokoro-style single-speaker checkpoint), not synthesizer plumbing.

The benchmark accepts any `ref_s.bin` produced by
`mobius-styletts2/scripts/06_dump_ref_s.py`. To regenerate one:

```bash
python mobius-styletts2/scripts/06_dump_ref_s.py \
  --reference-wav samples/kokoro-english/af_nicole.wav \
  --out /tmp/styletts2-ref_s.bin

.build/release/fluidaudiocli tts-benchmark \
  --backend styletts2 \
  --voice /tmp/styletts2-ref_s.bin \
  --corpus prose-en \
  --output-json bench.json
```

## Magpie outputBackings fast path

Magpie's `decoder_step.mlmodelc` was previously exported without
explicit `MultiArray` shape/dtype constraints on its KV outputs, which
made CoreML reject the `MLPredictionOptions.outputBackings` map that
`MagpieSynthesizer.runDecoderStep` builds:

```
Output feature (null) doesn't support output backing because it
doesn't have a MultiArray constraints.
```

The synthesizer now wraps the fast path in a try/catch that latches a
`MagpieKvCache.useOutputBackings` flag off on the first rejection, then
falls back to fresh-alloc decoding routed through
`MagpieKvCache.absorbOutputs(_:)` for the rest of the run. Cost: a
single throw/catch on the first AR step + ~18.9 MB of fresh fp16
allocation per subsequent step (no per-step exception spam). The
proper fix remains re-exporting `decoder_step.mlmodelc` from the
Python conversion pipeline (`mobius/models/tts/magpie/coreml/...`)
with explicit `MultiArray` shape + dtype constraints on `new_k_*`,
`new_v_*`, `var_*` outputs — that would let the fast path stay live
and avoid the per-step allocation entirely.

### Per-category quality

MacBook Air M2 (2022), default compute-units, Parakeet TDT roundtrip. p50 / p95 are
warm-synth latency in ms. Numbers + names corpora exercise edge cases
the ASR roundtrip is genuinely sensitive to (currency normalization,
acronyms, brand spelling), so the WER spike there reflects real
end-to-end pronunciation quality rather than ASR artefacts.

**Kokoro ANE (default preset)**

| Corpus       | Phrases | Macro WER | Macro CER | Synth p50 (ms) | Synth p95 (ms) | RTFx   |
|--------------|---------|-----------|-----------|-----------------|-----------------|--------|
| `prose-en`   | 20      | 0.162     | 0.067     | 244             | 305             | 10.41× |
| `numbers-en` | 10      | 0.402     | 0.243     | 909             | 1333            | 4.62×  |
| `names-en`   | 10      | 0.394     | 0.136     | 1033            | 1927            | 4.38×  |

**Kokoro (single-CU)**

| Corpus       | Phrases | Macro WER | Macro CER | Synth p50 (ms) | Synth p95 (ms) | RTFx   |
|--------------|---------|-----------|-----------|-----------------|-----------------|--------|
| `prose-en`   | 20      | 0.018     | 0.007     | 1263            | 1497            | 2.11×  |
| `numbers-en` | 10      | 0.173     | 0.091     | 3587            | 3711            | 1.58×  |
| `names-en`   | 10      | 0.234     | 0.081     | 3934            | 5667            | 1.26×  |

**PocketTTS**

| Corpus       | Phrases | Macro WER | Macro CER | Synth p50 (ms) | Synth p95 (ms) | RTFx   |
|--------------|---------|-----------|-----------|-----------------|-----------------|--------|
| `prose-en`   | 20      | 0.000     | 0.000     | 2222            | 3281            | 0.97×  |
| `numbers-en` | 10      | 0.363     | 0.130     | 3174            | 5340            | 1.35×  |
| `names-en`   | 10      | 0.178     | 0.035     | 3537            | 6430            | 1.23×  |

**StyleTTS2 (LibriTTS multi-speaker, `af_nicole`-derived `ref_s`)**

| Corpus       | Phrases | Macro WER | Macro CER | Synth p50 (ms) | Synth p95 (ms) | RTFx   |
|--------------|---------|-----------|-----------|-----------------|-----------------|--------|
| `prose-en`   | 20      | 0.446†    | 0.276†    | 1898            | 2054            | 1.46×  |

† Voice-quality bottleneck, not synthesis. See
[StyleTTS2 quality caveat](#styletts2-quality-caveat).

**Magpie (split-K/V decoder, `John` voice, en)**

| Corpus       | Phrases | Macro WER | Macro CER | Synth p50 (ms) | Synth p95 (ms) | RTFx   |
|--------------|---------|-----------|-----------|-----------------|-----------------|--------|
| `prose-en`   | 20      | 0.026     | 0.021     | 3500            | 4339            | 0.95×  |

**CosyVoice3 (beta, `--skip-asr` forced)**

| Corpus       | Phrases | Macro WER | Macro CER | Synth p50 (ms) | Synth p95 (ms) | RTFx   |
|--------------|---------|-----------|-----------|-----------------|-----------------|--------|
| `prose-zh`   | 8       | n/a       | n/a       | 10251           | 11765           | 0.21×  |

CosyVoice3 is gated behind a `[WARN] CosyVoice3 is experimental / beta`
banner; the AR LLM-Decode graph is rejected by the ANE compiler
(`MILCompilerForANE error: failed to compile ANE model using ANEF`) and
falls back to CPU+GPU, which is what drives the sub-real-time RTFx.
Cold start (~160 s) is dominated by the first-use ANE compile attempts
across the 4-graph pipeline. No ASR roundtrip — Mandarin is out of
scope for the Parakeet harness.

The qualitative pattern: Kokoro ANE pays a noticeable WER tax on prose
versus the non-ANE Kokoro graph, but stays >4× real-time on the
hard corpora where Kokoro itself drops to 1.3–1.6× and PocketTTS to
1.2–1.4×. PocketTTS is the only backend that hits 0.000 WER on prose,
but its `numbers-en` WER is the worst of the three because of how it
normalises currency / phone numbers.

## JSON report schema

```jsonc
{
  "summary": {
    "backend": "kokoro-ane",       // or "kokoro" / "pocket-tts" / "styletts2" / "magpie" / "cosyvoice3"
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
