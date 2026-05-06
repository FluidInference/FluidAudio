# StyleTTS2 (7-Stage ANE)

ANE-resident StyleTTS2 backend. Splits the `StyleTTS-2-coreml`
checkpoint into 7 small CoreML stages so that 6 of them stay resident on
the Neural Engine; only `noise.mlmodelc` runs on `.all` (fp32 SineGen
phase precision). Mirrors [KokoroAne.md](KokoroAne.md) 1:1 in shape —
same per-stage split, same `StyleTTS2ComputeUnits` preset surface,
same `synthesizeDetailed` timing report.

The diffusion sampler is preserved — it is StyleTTS2's defining feature
and has no Kokoro analog. The 5-step ADPM2 Karras loop runs in Swift;
only the per-step UNet (`diffusion_step.mlmodelc`) is a graph and is
invoked **11×** per utterance (5 midpoint × 2 + 1 final).

Conversion lives in
[`mobius/models/tts/styletts2/scripts/ane`](https://github.com/FluidInference/mobius/tree/main/models/tts/styletts2/scripts/ane).

## Pipeline Shape

| Property               | `StyleTTS2Manager`                                              |
|------------------------|--------------------------------------------------------------------|
| Compute                | 6 stages on ANE, 1 on `.all` (Noise), 7 graphs total               |
| Disk footprint         | ~330 MB (int8 palettization, kmeans nbits=8)                       |
| Voices                 | LibriTTS multi-speaker via `ref_s.bin` blobs                       |
| Bucketing              | RangeDim(2..512) for stages 1-3; static T_a=2000 for stages 4-7    |
| Custom lexicon / SSML  | No                                                                 |
| Languages              | English (LibriTTS espeak-ng IPA)                                   |

`StyleTTS2Manager` is the only StyleTTS2 backend shipped by
FluidAudio. The legacy 4-graph CoreML pipeline has been retired.

## Quick Start

### CLI

```bash
# Synthesize through the 7-graph ANE backend
swift run fluidaudiocli styletts2 "Welcome to FluidAudio." \
  --voice ~/voices/ref_s.bin \
  --output ~/Desktop/demo.wav

# Run the benchmark harness against the MiniMax-English corpus
swift run fluidaudiocli tts-benchmark \
  --backend styletts2-ane \
  --corpus minimax-english \
  --voice ~/voices/ref_s.bin \
  --output-json bench.json \
  --audio-dir bench-wavs/
```

First invocation downloads the 7 `.mlmodelc` bundles from
[`FluidInference/StyleTTS-2-coreml/ANE/`](https://huggingface.co/FluidInference/StyleTTS-2-coreml/tree/main/ANE);
later runs reuse the cached assets. The English G2P CoreML assets are
also fetched from the kokoro repo on first synthesis (~10 MB, cached at
`~/.cache/fluidaudio/Models/kokoro/g2p/`) and reused if you've already
run any kokoro backend.

### Swift

```swift
import FluidAudio

let manager = StyleTTS2Manager()
try await manager.initialize()

let voiceURL = URL(fileURLWithPath: "/path/to/ref_s.bin")
let wav = try await manager.synthesize(
    text: "Hello from FluidAudio.",
    voiceStyleURL: voiceURL
)
```

### Per-stage timings

```swift
let result = try await manager.synthesizeDetailed(
    text: "...",
    voiceStyleURL: voiceURL
)
print("samples: \(result.samples.count) @ \(result.sampleRate) Hz")
let t = result.timings
print("  plBert=\(t.plBert) postBert=\(t.postBert) alignment=\(t.alignment)")
print("  diffusionStep=\(t.diffusionStep) prosody=\(t.prosody)")
print("  noise=\(t.noise) vocoder=\(t.vocoder)")
print("  total: \(t.totalMs) ms")
```

The CLI's `styletts2` command prints the same breakdown after each
synthesis.

## Pipeline

```
text → G2P (CoreML BART) → IPA → espeak-remap → vocab → token ids
                                                          │
        ┌─────────────────────────────────────────────────┘
        ▼
  ┌─────────┐  ┌──────────┐  ┌───────────┐
  │ PLBert  │→ │ PostBert │→ │ Alignment │      ANE
  └─────────┘  └──────────┘  └───────────┘
                                   │
        ┌──────────────────────────┘
        ▼
  ┌──────────────┐    (×11 invocations: 5 midpoint × 2 + 1 final)
  │ DiffusionStep│      ANE  ← ADPM2 + Karras sampler in Swift
  └──────────────┘
        │
        ▼
  ┌─────────┐  ┌─────────┐  ┌──────────┐
  │ Prosody │→ │  Noise  │→ │ Vocoder  │  → 24 kHz PCM
  └─────────┘  └─────────┘  └──────────┘
      ANE         all          ANE
```

| Stage           | Input                              | Output                       | Compute units            | Precision |
|-----------------|------------------------------------|------------------------------|--------------------------|-----------|
| `plBert`        | input_ids                          | bert hidden states           | `cpuAndNeuralEngine`     | fp16+int8 |
| `postBert`      | bert + ref_p                       | duration + d_en              | `cpuAndNeuralEngine`     | fp16+int8 |
| `alignment`     | duration                           | en (T_a frames)              | `cpuAndNeuralEngine`     | fp16+int8 |
| `diffusionStep` | xt + sigma + d_en + ref            | step UNet output             | `cpuAndNeuralEngine`     | fp16+int8 |
| `prosody`       | en + style                         | F0, N                        | `cpuAndNeuralEngine`     | fp16+int8 |
| `noise`         | F0, N                              | har, noise (fp32)            | `all`                    | fp32+int8 |
| `vocoder`       | en + har + noise + style           | 24 kHz waveform              | `cpuAndNeuralEngine`     | fp16+int8 |

Override the per-stage assignment with one of the four shipped presets
(`StyleTTS2ComputeUnits.default` / `.allAne` / `.cpuAndGpu` /
`.cpuOnly`):

```swift
let manager = StyleTTS2Manager(
    computeUnits: .cpuAndGpu  // skip ANE entirely (debugging baseline)
)
```

## Voice Format

Each voice is a flat 256-fp32 LE blob (`ref_s.bin`, 1024 bytes), split
column-wise as `[0..<128]` → acoustic / "ref" half (consumed by Prosody
+ Vocoder), `[128..<256]` → prosody half (consumed by PostBert +
DiffusionStep). This is **byte-identical** to the legacy 4-graph
`ref_s.bin` format — voice export was deliberately not re-cut as part
of this PR (the style encoders are PyTorch-only).

The repo ships no voices in the ANE folder. Dump one yourself with the
upstream
[`06_dump_ref_s.py`](https://github.com/FluidInference/mobius/blob/main/models/tts/styletts2/scripts/06_dump_ref_s.py)
helper from any LibriTTS speaker WAV.

## Limits

- **Phonemes:** ≤ 510 IPA chars per call (PLBert context = 512 incl.
  BOS/EOS). No built-in chunker — split upstream if you need longer
  inputs.
- **Acoustic frames:** `T_a ≤ 2000` (compile-time `--max-frames` baked
  into Prosody / Noise / Vocoder).
- **Voices:** any LibriTTS-compatible `ref_s.bin` (acoustic + prosody
  halves, 256 fp32 LE).
- **Custom lexicon / SSML / Markdown overrides:** not supported. The
  pipeline goes `text → G2P → phonemes → token ids` with no
  interception point. Custom IPA **is** supported via the CLI's
  `--phonemes` path (delegates to a phoneme-encoded synthesis call).
- **Streaming:** not yet — the diffusion loop renders the whole
  utterance before the vocoder can emit. Streaming is a follow-up.

## Source

- HuggingFace: [`FluidInference/StyleTTS-2-coreml/ANE/`](https://huggingface.co/FluidInference/StyleTTS-2-coreml/tree/main/ANE)
- Upstream PyTorch: [`yl4579/StyleTTS2`](https://github.com/yl4579/StyleTTS2)
- Conversion script: [`mobius/models/tts/styletts2/scripts/ane`](https://github.com/FluidInference/mobius/tree/main/models/tts/styletts2/scripts/ane)
- Sampler reuse: `Sources/FluidAudio/TTS/StyleTTS2/Pipeline/StyleTTS2Sampler.swift`
- Voice format reuse: `Sources/FluidAudio/TTS/StyleTTS2/Voice/StyleTTS2VoiceStyle.swift`
