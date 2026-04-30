# Kokoro ANE (7-Stage)

ANE-resident sibling of [Kokoro](Kokoro.md). Splits the Kokoro 82M graph into 7
CoreML stages so the ANE-friendly layers (Albert / PostAlbert / Alignment /
Vocoder) stay resident on the Neural Engine while Prosody / Noise / Tail run
on CPU+GPU. Yields **3-11× RTFx** on Apple Silicon vs. the single-graph
[`KokoroTtsManager`](Kokoro.md), at the cost of single-voice / no chunker /
no custom lexicon.

Derived from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml),
used with the author's permission. Conversion lives in
[mobius/models/tts/kokoro/laishere-coreml](https://github.com/FluidInference/mobius/tree/main/models/tts/kokoro/laishere-coreml).

## When To Pick This Over `KokoroTtsManager`

|                  | `KokoroTtsManager`        | `KokoroAneManager`           |
|------------------|---------------------------|------------------------------|
| Compute          | CPU + GPU (single graph)  | 4 stages on ANE, 3 on GPU    |
| Voices           | Multi (`.json` packs)     | Single (`af_heart.bin`)      |
| Long input       | Built-in chunker          | ≤ 510 IPA phonemes / utt.    |
| Custom lexicon   | Yes (`TtsCustomLexicon`)  | No                           |
| SSML             | Yes                       | No                           |
| HF path          | `kokoro-82m-coreml/`      | `kokoro-82m-coreml/ANE/`     |

Use `KokoroAneManager` when you want the lowest latency on Apple Silicon and
can live with `af_heart` only and short inputs. Use `KokoroTtsManager` when
you need any of: multi-voice, custom pronunciations, SSML, long-form text.

## Quick Start

### CLI

```bash
swift run fluidaudiocli tts "Welcome to FluidAudio" \
  --backend kokoro-ane \
  --output ~/Desktop/demo.wav
```

First invocation downloads the 7 `.mlmodelc` bundles + `vocab.json` +
`af_heart.bin` from
[`FluidInference/kokoro-82m-coreml/ANE/`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE);
later runs reuse the cached assets.

### Swift

```swift
import FluidAudio

let manager = KokoroAneManager()
try await manager.initialize()

let audioData = try await manager.synthesize(text: "Hello from FluidAudio!")
try audioData.write(to: URL(fileURLWithPath: "/tmp/demo.wav"))
```

### Per-stage timings

```swift
let result = try await manager.synthesizeDetailed(text: "...", speed: 1.0)
print("samples: \(result.samples.count) @ \(result.sampleRate) Hz")
let t = result.timings
print("  albert=\(t.albert) postAlbert=\(t.postAlbert) alignment=\(t.alignment)")
print("  prosody=\(t.prosody) noise=\(t.noise) vocoder=\(t.vocoder) tail=\(t.tail)")
print("  total: \(t.totalMs) ms")
```

### Bypass G2P

```swift
let wav = try await manager.synthesizeFromPhonemes("həˈloʊ wɝld")
```

Useful when you've already phonemized upstream (e.g. you're streaming IPA
from a different G2P).

## Pipeline

```
text → G2P (CoreML BART) → IPA → vocab.json → token ids
                                                  │
        ┌─────────────────────────────────────────┘
        ▼
  ┌──────────┐  ┌────────────┐  ┌───────────┐
  │  Albert  │→ │ PostAlbert │→ │ Alignment │      ANE
  └──────────┘  └────────────┘  └───────────┘
                                       │
        ┌──────────────────────────────┘
        ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Prosody  │→ │  Noise   │→ │ Vocoder  │→ │   Tail   │  → 24 kHz PCM
  └──────────┘  └──────────┘  └──────────┘  └──────────┘
       all          all           ANE          all
```

| Stage        | Input               | Output                       | Compute units            |
|--------------|---------------------|------------------------------|--------------------------|
| Albert       | input_ids           | albert hidden states         | `cpuAndNeuralEngine`     |
| PostAlbert   | albert + style_s    | duration + d_en              | `cpuAndNeuralEngine`     |
| Alignment    | duration            | en (T_a frames)              | `cpuAndNeuralEngine`     |
| Prosody      | en + style_s        | F0, N (fp16)                 | `all`                    |
| Noise        | F0, N + style_timbre| har, noise (fp32)            | `all`                    |
| Vocoder      | har, noise + timbre | x_pre (fp16)                 | `cpuAndNeuralEngine`     |
| Tail         | x_pre               | 24 kHz waveform via iSTFT    | `all`                    |

Override per-stage assignment with `KokoroAneComputeUnits`:

```swift
let manager = KokoroAneManager(
    computeUnits: .cpuAndGpu  // skip ANE entirely (debugging baseline)
)
```

## Voice Pack

The single shipping voice (`af_heart.bin`) is a flat `[510, 256]` fp32
matrix. Row index = `min(max(phonemeCount - 1, 0), 509)` (utterance-length
bucket); columns split as `[0..<128]` = `style_timbre` (→ Noise + Vocoder),
`[128..<256]` = `style_s` (→ PostAlbert + Prosody).

This single-voice constraint is intrinsic to the upstream conversion — adding
voices requires re-converting `KokoroPostAlbert` / `KokoroProsody` /
`KokoroNoise` / `KokoroVocoder` against the new style embeddings.

## Limits

- **Phonemes:** ≤ 510 IPA chars per call (ALBERT context = 512 incl. BOS/EOS).
  No built-in chunker — split upstream if you need longer inputs.
- **Voices:** `af_heart` only.
- **Custom lexicon / SSML / Markdown overrides:** not supported. The pipeline
  goes `text → G2P → IPA → token ids` with no interception point.
- **Acoustic frames:** `T_a ≤ 2000` (compile-time `--max-frames` baked into
  the converted models).

## Performance

Cold load (first ever — `anecompilerservice` has to compile each stage for
ANE) is ≈ 20 s on M1; warm load is ≈ 0.3 s. Synthesis itself runs at
**3-11× RTFx** on Apple Silicon depending on utterance length. Per-stage
timing (5 s of audio, M1):

| Stage      | Time     |
|------------|----------|
| Albert     | ~5 ms    |
| PostAlbert | ~10 ms   |
| Alignment  | ~5 ms    |
| Prosody    | ~30 ms   |
| Noise      | ~80 ms   |
| Vocoder    | ~120 ms  |
| Tail       | ~50 ms   |

Vocoder dominates. Total ≈ 300 ms for 5 s audio (~16× RTFx). For
full-corpus numbers (warm-synth p50 / p95, peak RSS, WER) on the
MiniMax-English 100-phrase suite — including the longer paragraph
phrases that pull the per-corpus aggregate down to ~5.2× — see
[Benchmarks.md](Benchmarks.md).

## Source

- HuggingFace: [`FluidInference/kokoro-82m-coreml/ANE/`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE)
- Upstream PyTorch: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Conversion script: [mobius/models/tts/kokoro/laishere-coreml](https://github.com/FluidInference/mobius/tree/main/models/tts/kokoro/laishere-coreml)
- Original CoreML fork: [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml)
