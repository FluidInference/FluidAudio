# Chatterbox (Multilingual + Nano) Swift Inference

> **⚠️ Beta:** both Chatterbox backends are beta model conversions — API,
> model artifacts, and accuracy may change. Report issues with the seed and
> the exact text so runs can be reproduced.

CoreML ports of ResembleAI's Chatterbox TTS family. Two variants ship:

| | Chatterbox Multilingual | Chatterbox Nano |
|---|---|---|
| Backend id | `chatterbox` | `chatterbox-nano` |
| Upstream | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) | [ResembleAI/chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano) |
| Models | [FluidInference/chatterbox-multilingual-coreml](https://huggingface.co/FluidInference/chatterbox-multilingual-coreml) | [FluidInference/chatterbox-nano-coreml](https://huggingface.co/FluidInference/chatterbox-nano-coreml) |
| T3 token generator | Llama-520M, CFG batch 2, alignment-analyzer EOS control | GPT2-small 110M, batch 1 (no CFG, no analyzer) |
| Mel decoder | S3Gen flow, 10-step CFG Euler | S3Gen **meanflow**, 2 plain Euler steps (distilled) |
| Languages | 18 of the 23 upstream (zh/ja/he/ko/ru need unported text transforms) | English only |
| Extras | — | Inline paralinguistic tags: `[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, … |
| Weight footprint (fp16 on disk) | ~2.3 GB | ~711 MB |
| Wall RTFx (M5 Pro, release build) | ~1× | **5.4–6.0×** |
| Manager | `ChatterboxManager` | `ChatterboxNanoManager` |

Both require **macOS 15 / iOS 18** (the T3 decode keeps its KV cache in
CoreML `MLState`), output 24 kHz mono, and currently ship the built-in
voice only — voice cloning needs upstream encoder conversions (tracked as a
follow-up). Conversion toolkits, parity reports, and the Python reference
drivers live in mobius (`models/tts/chatterbox/coreml`, `REPORT.md` /
`REPORT-NANO.md`).

## Usage

```swift
// Multilingual (beta)
let mtl = ChatterboxManager()
try await mtl.initialize()
let de = try await mtl.synthesize(
    text: "Der schnelle braune Fuchs springt über den faulen Hund.",
    language: "de", seed: 42)

// Nano (beta) — tags go straight in the text
let nano = ChatterboxNanoManager()
try await nano.initialize()
let en = try await nano.synthesize(
    text: "Well that went better than expected [chuckle], see you tomorrow.",
    seed: 42)
```

CLI:

```bash
swift run -c release fluidaudiocli tts --backend chatterbox --lang de "Guten Morgen." --seed 42
swift run -c release fluidaudiocli tts --backend chatterbox-nano "Hi there [chuckle], one minute?" --seed 42
swift run -c release fluidaudiocli tts-benchmark --backend chatterbox-nano --corpus minimax-english
```

Equal seeds reproduce equal audio; different seeds give different prosody
takes (the T3 stage samples stochastically).

## Sampling defaults (upstream-faithful)

| | Multilingual | Nano |
|---|---|---|
| Order | rep-penalty → temperature → min-p → top-p | temperature → top-k → top-p → rep-penalty (on filtered logits) |
| Values | cfg 0.5, temp 0.8, rep 2.0, min-p 0.05, top-p 1.0 | temp 0.8, top-k 1000, top-p 0.95, rep 1.2 |

## Limits & caveats

- One-shot synthesis (no streaming): the AR decode + flow + vocoder finish
  before audio is available.
- Generation is capped by the 500-token flow bucket ≈ **10 s of audio per
  call** after the built-in voice's prompt tokens; split long text into
  sentences.
- Never force `.cpuOnly` — the Multilingual T3 packages hard-crash there
  (Nano untested; both load `.cpuAndGPU`).
- **Benchmark with `-c release`.** Debug builds spend ~12 ms/token in
  unoptimized per-step sampling sorts, tripling apparent decode time.
- Upstream watermarks generated audio with Resemble's Perth watermarker in
  its Python host; these Swift backends do not implement that stage.
