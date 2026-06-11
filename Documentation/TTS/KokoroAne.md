# Kokoro ANE (7-Stage)

Splits the Kokoro 82M graph into 7 CoreML stages so the ANE-friendly layers
(Albert / PostAlbert / Alignment / Vocoder) stay resident on the Neural Engine
while Prosody / Noise / Tail run on CPU+GPU. Yields **3-11× RTFx** on Apple
Silicon.

Derived from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml),
used with the author's permission. Conversion lives in
[mobius/models/tts/kokoro/laishere-coreml](https://github.com/FluidInference/mobius/tree/main/models/tts/kokoro/laishere-coreml).

## Constraints

| Aspect           | `KokoroAneManager`                              |
|------------------|-------------------------------------------------|
| Compute          | 4 stages on ANE, 3 on GPU                       |
| Voices           | Single per variant (`af_heart` / `zf_001`)      |
| Input length     | ≤ 510 IPA / Bopomofo phonemes / utt.            |
| Custom lexicon   | No                                              |
| SSML             | No                                              |
| Languages        | English (`ANE/`) + Mandarin (`ANE-zh/`)         |

For multi-voice / SSML / long-form, use `PocketTtsSynthesizer` or
`StyleTTS2Manager` instead.

## Variants

The 7-stage chain is language-agnostic by construction (input ids, voice
slices, and per-stage I/O contracts are identical across variants). Only the
embedding vocab, HF subdirectory, voice-file layout, default voice, and the
text-to-phoneme frontend differ.

| Variant       | HF subdir   | Vocab | Default voice | Voice layout                | Frontend                                   |
|---------------|-------------|-------|---------------|-----------------------------|--------------------------------------------|
| `.english`    | `ANE/`      | 177   | `af_heart`    | flat (`<voice>.bin`)        | G2P CoreML (BART seq2seq) → IPA            |
| `.mandarin`   | `ANE-zh/`   | 171   | `zf_001`      | nested (`voices/<voice>.bin`) | Rule-based dict lookup → Bopomofo + tones |

Pick the variant on construction:

```swift
let english  = KokoroAneManager(variant: .english)   // default
let mandarin = KokoroAneManager(variant: .mandarin)
```

## Quick Start

### CLI

```bash
# English (default)
swift run fluidaudiocli tts "Welcome to FluidAudio" \
  --backend kokoro-ane \
  --output ~/Desktop/demo.wav

# Mandarin
swift run fluidaudiocli tts "你好世界，今天天气真好。" \
  --backend kokoro-ane --variant zh \
  --output ~/Desktop/demo_zh.wav
```

First invocation downloads the 7 `.mlmodelc` bundles + `vocab.json` +
default voice from
[`FluidInference/kokoro-82m-coreml/ANE/`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE)
(English) or
[`ANE-zh/`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE-zh)
(Mandarin); later runs reuse the cached assets. The Mandarin variant
additionally fetches the G2P pinyin dictionaries from
[`ANE-zh/assets/`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE-zh/assets)
on first synthesis (~10 MB, cached at `<repoDir>/g2p/`).

### Swift

```swift
import FluidAudio

// English
let english = KokoroAneManager()
try await english.initialize()
let enWav = try await english.synthesize(text: "Hello from FluidAudio!")

// Mandarin — give it Hanzi, the built-in G2P handles segmentation,
// pinyin lookup, tone sandhi, and Bopomofo encoding.
let mandarin = KokoroAneManager(variant: .mandarin)
try await mandarin.initialize()
let zhWav = try await mandarin.synthesize(text: "你好世界")
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
// English: pre-computed IPA
let enWav = try await english.synthesizeFromPhonemes("həˈloʊ wɝld")

// Mandarin: pre-computed Bopomofo + tone digits matching the
// `ANE-zh/vocab.json` token set.
let zhWav = try await mandarin.synthesizeFromPhonemes("ㄋㄧ2ㄏㄠ3")
```

Useful when you've already phonemized upstream.

## Pipeline

```
English:   text → G2P (CoreML BART) → IPA → vocab.json → token ids
Mandarin:  text → MandarinG2P (dict + sandhi) → Bopomofo → vocab.json → token ids
                                                                          │
        ┌─────────────────────────────────────────────────────────────────┘
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

Each shipping voice (`af_heart.bin` for English, `zf_001.bin` for Mandarin)
is a flat `[510, 256]` fp32 matrix. Row index = `min(max(phonemeCount - 1,
0), 509)` (utterance-length bucket); columns split as `[0..<128]` =
`style_timbre` (→ Noise + Vocoder), `[128..<256]` = `style_s` (→ PostAlbert
+ Prosody).

The English bundle stores voice packs flat at the bundle root
(`<voice>.bin`); the Mandarin bundle nests them under `voices/<voice>.bin`.
This single-voice-per-variant constraint is intrinsic to the upstream
conversion — adding voices requires re-converting `KokoroPostAlbert` /
`KokoroProsody` / `KokoroNoise` / `KokoroVocoder` against the new style
embeddings.

## Mandarin G2P

The Mandarin variant ships a self-contained, network-free Hanzi → Bopomofo
pipeline modelled on
[`misaki[zh]`](https://github.com/hexgrad/misaki/blob/main/misaki/zh_frontend.py):

1. **Punctuation normalization** — fullwidth `，。！？；：` collapse to ASCII.
2. **Forward maximum match segmentation** — greedy phrase lookup against
   ~411k phrase entries, falls back to per-character single lookup
   (~42k Hanzi).
3. **Pinyin normalization** — diacritic tone marks (`níhǎo`) → digit form
   (`ni2hao3`); ü-row collapses to `v`.
4. **Tone sandhi** — three high-impact, POS-independent rules from
   `misaki/tone_sandhi.py`: 3+3 chain (`3 3 3 → 2 2 3`), 不-promotion
   before tone 4, 一-promotion based on the next syllable's tone.
5. **Bopomofo + tone-digit encoding** — initials/finals split, sibilant
   `i`-fix (`zi/ci/si → ㄭ`, `zhi/chi/shi/ri → 十`), j/q/x + u → ü, then
   one bopomofo character per part + the tone digit.

Asset footprint (downloaded on first synthesis, cached at
`<repoDir>/g2p/`):

| File                  | Size   | Source                                  |
|-----------------------|--------|-----------------------------------------|
| `pinyin_phrases.bin`  | 9.5 MB | `kokoro-82m-coreml/ANE-zh/assets/`      |
| `pinyin_single.bin`   | 480 KB | same                                    |

What the Mandarin G2P intentionally does **not** ship: jieba HMM
fallback, POS-aware tone sandhi, neural polyphone disambiguation
(g2pW-style), erhua handling, number/date verbalization. These are all
viable upgrades — the current pipeline trades them for a zero-network
~10 MB footprint that handles short conversational text well.

## Limits

- **Phonemes:** ≤ 510 IPA / Bopomofo chars per call (ALBERT context = 512
  incl. BOS/EOS). No built-in chunker — split upstream if you need longer
  inputs.
- **Voices:** `af_heart` only (English) / `zf_001` only (Mandarin).
- **Custom lexicon / SSML / Markdown overrides:** not supported. The pipeline
  goes `text → G2P → phonemes → token ids` with no interception point.
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

- HuggingFace (English): [`FluidInference/kokoro-82m-coreml/ANE/`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE)
- HuggingFace (Mandarin): [`FluidInference/kokoro-82m-coreml/ANE-zh/`](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE-zh)
- Upstream PyTorch (English): [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Upstream PyTorch (Mandarin): [hexgrad/Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)
- Mandarin G2P reference: [hexgrad/misaki](https://github.com/hexgrad/misaki) (`zh_frontend.py`, `tone_sandhi.py`)
- Conversion script: [mobius/models/tts/kokoro/laishere-coreml](https://github.com/FluidInference/mobius/tree/main/models/tts/kokoro/laishere-coreml)
- Original CoreML fork: [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml)
