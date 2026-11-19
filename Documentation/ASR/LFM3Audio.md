# LFM3-Audio-2B

Liquid AI's [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) on device: offline speech recognition
(long-form included), LiveASR (streaming English transcription in 0.96 s chunks) and text-to-speech, all from one 2B
audio language model. `LFM3AudioManager` runs it on Core ML (macOS 15 / iOS 18+) or with the language model on
Core AI (macOS 27 / iOS 27+).

Requested in [#49](https://github.com/FluidInference/FluidAudio/issues/49#issuecomment-6501658263).

## Models

**Core ML**: [FluidInference/lfm3-audio-2b-coreml](https://huggingface.co/FluidInference/lfm3-audio-2b-coreml)
**Core AI LM**: [FluidInference/lfm3-audio-2b-coreai](https://huggingface.co/FluidInference/lfm3-audio-2b-coreai)

| Part | File | Precision | Compute unit (default) |
|---|---|---|---|
| LM backbone (2 stateful stages) | `coreml/LFM3-A-C4160-int8b32.mlmodelc`, `coreml/LFM3-B-C4160-int8b32.mlmodelc` | int8 (block 32) | GPU |
| LM head | `coreml/LFM3-Head-int8b32.mlmodelc` | int8 | GPU |
| Audio encoder (`n12` / `n96` / `n375`) | `coreml/AudioEncoder-fp16.mlmodelc` | fp16 | GPU offline, ANE for LiveASR |
| Depth decoder (`depth_in`, `depth_pass`) | `coreml/DepthDecoder-fp16.mlmodelc` | fp16 | GPU |
| Codec decoder (`f1` / `f4` / `f25`) | `coreml/CodecDecoder-fp16.mlmodelc` | fp16 | ANE |
| Host tables, tokenizer | `assets/*` | fp16 / fp32 | CPU |

ASR downloads ~2.5 GB; TTS adds ~0.4 GB (`includeSpeech: false` skips it). The Core AI backend downloads the Core AI
LM bundle (`coreai/LFM3-Audio-u8b32-f16L20-23.aimodel`, 2.16 GB: uint8 block 32, decoder layers 20-23 fp16) and,
for TTS, the fused depth decoder (`coreai/DepthDecoder-fp16-gpu.aimodel`: 3 MaskGIT passes + sampling in one
`depth_frame` call per frame) instead of the Core ML LM and depth files. The encoder and codec stay on Core ML (the
Neural Engine is not available to Core AI on macOS 27.0). Plain int8 Core AI bundles pass ASR but flip too many TTS
speech codes.

The Core AI LM loads one `AIModel` per entrypoint (`prefill`, `main`, `hidden`), all bound to the same states: with
one shared `AIModel`, macOS 27.0 aborts in `GPUMemrefOps` ("Failed to resolve dynamic dimensions for memref.alloc")
when one function runs twice at consecutive positions with the same row count and another function then takes the
next shape — the first `main` after speech frames in TTS.

## Architecture

```
16 kHz audio -> log-mel (host, == HF feature extractor) -> causal audio encoder (12.5 Hz rows)
  -> prompt "Perform ASR in <language>." with the rows spliced in -> LM (2048 x 24) greedy text

text -> prompt "Perform TTS in <language> using a <female|male> voice." + <|audio_start|>
  -> per 80 ms frame: LM hidden -> depth decoder (3 MaskGIT passes, 12 codebooks, host sampling)
     -> frame embedding back into the LM;  frames -> LiquidCodec decoder (ANE) -> 24 kHz audio
```

- **Offline ASR** transcribes up to 40 s in one pass (as upstream); longer audio is cut into <= 30 s segments at the
  quietest 20 ms window in each segment's last 5 s.
- **LiveASR** follows upstream `transcribe_live`: 96 mel frames (0.96 s) per chunk through the encoder's `n12`
  function with carried state, up to 48 text tokens per chunk until `<|text_end|>`, and an LM restart every 31 chunks.
- **TTS** samples speech codes with the model's settings (temperature 1.0, top-k 128, presence penalty 1.1 over 4096
  frames). Frames are decoded while the LM keeps generating; `onAudio` gets each chunk for streaming playback.
- The LM context is 4096 positions (about 5 minutes of generated speech).

## Usage

```swift
import FluidAudio

let manager = try await LFM3AudioManager.downloadAndLoad()  // backend: .coreML, includeSpeech: true

// Offline ASR: 16 kHz mono samples or a file; language nil = let the model infer it
let result = try await manager.transcribe(samples, language: "English")
print(result.text, result.rtfx)

// LiveASR (English)
try await manager.startLiveTranscription()
for chunk in microphoneChunks {
    let partial = try await manager.appendLiveAudio(chunk)
}
let transcript = try await manager.finishLiveTranscription()

// TTS: 24 kHz mono
let speech = try await manager.synthesize(
    "Hello from LFM3 Audio.", voice: .male, language: "British English", seed: 0,
    onAudio: { samples in /* play as it arrives */ })
```

One manager owns one LM context: concurrent calls run one at a time in arrival order, and `transcribe` /
`synthesize` throw `LFM3AudioError.liveSessionActive` while a live session is open. `LFM3AudioComputeUnits` overrides
the per-component compute units.

### CLI

```bash
swift run -c release fluidaudiocli lfm3-audio transcribe audio.wav [--language French|auto]
swift run -c release fluidaudiocli lfm3-audio live audio.wav [--chunk-ms 160]
swift run -c release fluidaudiocli lfm3-audio tts "Hello there." --voice female --output hello.wav
swift run -c release fluidaudiocli lfm3-audio benchmark [--live] --max-files 100
# --backend coreai on macOS 27; --model-dir <dir> to load a local copy of the HF repo layout
```

## Benchmarks

Apple M5 Pro (24 GB), macOS 27.0, release build; wall time includes warm-up after load.

| Task | Data | Core ML | Core AI LM |
|---|---|---|---|
| Offline ASR | LibriSpeech test-clean, first 100 utterances (901 s) | WER 1.32%, RTFx 18.4; 100/100 transcripts identical to the original model (HF fp32, greedy) | WER 1.32%, RTFx 24.6; 98/100 identical |
| LiveASR (160 ms pieces) | LibriSpeech test-clean, first 20 utterances | WER 1.46%, RTFx 10.4 | WER 1.46%, RTFx 13.3 |
| TTS | 20 phrases, ~72 s of speech | RTFx 4.3-4.5 (across runs), first audio 37 ms (median), Parakeet round-trip WER 2.5% | RTFx 5.07 (Core AI depth decoder), first audio 37 ms, round-trip WER 2.5% |

Core ML per speech frame: LM 9.4 ms + depth decoder 4.9 ms (GPU), codec in parallel on the ANE. ASR peak memory
footprint: 1.9 GB on both backends. The first Core AI load specializes the graph and caches it (~3.3 GB under
`~/Library/Caches/coreai-cache`); the system purges it under storage pressure or when the bundle changes, and
`LFM3AudioModels.clearCoreAICache()` deletes it on demand.

## First load and disk space

The first Core ML load in an app compiles the GPU functions into the app's Core ML cache
(`~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache`): 5.3 GB and ~11 s for recognition on the M5 Pro. The
`prefill_q13` / `prefill_q16` buckets (LiveASR chunks, short prefill remainders) load on first use instead
(`startLiveTranscription()` loads the live one): ~1.7 s each from a warm cache, and on a cold one ~3 GB more cache
and a few seconds once.

On macOS 27 these cache files are purgeable as soon as they are written. With little free disk the system can delete
them while Core ML is still loading, and the model then fails to load or aborts on its first prediction with
`MPSGraphTensorData shape.count = 0 != strides.count = 3`. Keep ~30 GB free for a first load: `LFM3AudioModels` logs
a warning below that, and a load failure on a low disk throws `LFM3AudioError.lowDiskSpace` and keeps the download
instead of deleting and re-fetching it.

## License

LFM Open License v1.0 (Liquid AI): organizations with USD 10 million or more in annual revenue need a commercial
license from Liquid AI. See the model repo's LICENSE.
