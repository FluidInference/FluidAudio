# Phonon-2.5 (multilingual, GPU encoder)

`AsrModelVersion.phonon25` loads the multilingual
[Fermion Research Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5)
checkpoint through the existing v3-family Core ML runtime. It is distinct from
[Phonon-2](Phonon2.md), the English-only model already supported by `.phonon2`.

## Usage

```swift
let models = try await AsrModels.downloadAndLoad(version: .phonon25)
let manager = AsrManager(config: .default, models: models)
```

```bash
swift run fluidaudiocli transcribe speech.wav --model-version phonon25
# Alias: --model-version phonon-2.5
```

The model supports 25 languages and requires macOS 15 / iOS 18 or newer for
its compressed encoder operations. Only the encoder defaults to `.cpuAndGPU`;
preprocessor remains CPU-only, decoder/joint retain the caller's configuration
(default CPU + Neural Engine). GPU execution is intended for foreground use;
iOS background execution is not established. Explicit ANE-capable encoder
requests (`.cpuAndNeuralEngine` or `.all`) fail before downloading. `.cpuOnly`
is available for diagnostics, and a CPU-only configuration (for example
background-safe apps) keeps the encoder on the CPU too. Other models' placement
policies are unchanged.

The first transcription after each launch includes the GPU encoder's first
prediction (~2.5 s on an M5 Pro, against ~0.05 s for v3 on the ANE); warm the
model with a short clip if that latency matters. Peak memory footprint while
transcribing is ~480 MB (v3: ~47 MB).

## Measured (FluidAudio, M5 Pro, macOS 27, GPU encoder)

| Data | Phonon-2.5 WER | v3 WER | Phonon-2.5 / v3 RTFx |
|---|---|---|---|
| LibriSpeech test-clean (2620) | 2.08% | 2.05% | 159x / 148x |
| LibriSpeech test-other (2939) | 3.93% | 3.87% | 150x / 143x |
| FLEURS en / es / fr / de (100 each) | 4.4 / 4.1 / 9.0 / 5.5% | 4.4 / 3.9 / 6.6 / 5.8% | |
| FLEURS it / el / lv (100 each) | 4.5 / 35.5 / 15.3% | 3.4 / 38.6 / 26.2% | |

Corpus WER with FluidAudio's text normalization. Greek output writes a medial
sigma (σ) where the final form (ς) belongs; mapping word-final σ to ς brings
Greek to ~22% WER, so the 35.5% above is mostly that spelling.

## Published artifacts and conversion

The initial component set is hosted at
[Muesli-HQ/phonon-2.5-coreml](https://huggingface.co/Muesli-HQ/phonon-2.5-coreml/tree/a042ef89966d289df295709b58ffdabb50bd2510),
pinned to revision `a042ef89966d289df295709b58ffdabb50bd2510`. It contains the
~253 MB dense LUT3 encoder, the checkpoint's own decoder/joint/vocabulary, and a
shared preprocessor (~290 MB total). It uses its own cache folder `phonon-2.5`.
Original CC BY 4.0 weight license/notices and a per-file SHA256 manifest are
retained in the artifact repository. Hosting can be migrated through a follow-up
once maintainers choose their preferred location.

Conversion and reproducible validation tools:
[FluidInference/mobius#113](https://github.com/FluidInference/mobius/pull/113).
Downstream integration: [Muesli #681](https://github.com/Muesli-HQ/muesli/pull/681).

## Validation limits and ANE follow-up

The dense encoder passed the retained numerical gate on Apple M5/macOS 26.6.2:
cosine > 0.999, NRMSE < 0.05 (observed approximately 0.999997 / 0.00149).
The separate sparse ANE conversion failed that gate (approximately 0.997446 /
0.0714); it is not distributed or selected here. Updating the probe from
FluidAudio 0.17.3 to 0.17.7 did not change those ANE outputs. This does not rule
out a future converter or platform fix; do not enable that artifact without
revalidation.

Earlier public-fixture evaluation covered 83 clips in English, French, German
and Spanish, with 72/83 normalized transcripts matching the fp32 reference.
That is a smoke sample, not full-corpus or all-language parity. Other chips,
iOS execution, remaining languages and the English-only 2.5 checkpoint have not
been validated. No throughput guarantee is claimed.

Ordinary tests cover registry, cache identity, v3-family routing, platform and
encoder-placement contracts without downloading model weights. To run real
integration, provide a dedicated managed model cache and public fixtures:

```bash
FLUIDAUDIO_PHONON25_MODELS=/path/to/models/phonon-2.5 \
FLUIDAUDIO_PHONON25_FIXTURES=/path/to/public-fixtures \
swift test --filter Phonon25Tests
```

The integration test uses the four filenames listed in `Phonon25Tests`: LibriSpeech
test-clean `1089-134686-0000` (converted to 16 kHz wav) and the google/fleurs test
clips with those ids, which are `fr_fr_0000`, `de_de_0000` and `es_419_0000` in
[FluidInference/fleurs](https://huggingface.co/datasets/FluidInference/fleurs). It tests
both exact-directory loading and the registered managed loader, with automatic
GPU placement and unchanged decoder/joint defaults. Do not point this test at
another application's active model cache, since managed cache repair can remove
mismatched entries.
