![banner.png](banner.png)

# FluidAudio - Transcription, Text-to-speech, VAD, Speaker diarization with CoreML Models

[![Swift](https://img.shields.io/badge/Swift-6.0+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)
[![Documentation](https://img.shields.io/badge/Documentation-docs.fluidinference.com-008574.svg)](https://docs.fluidinference.com/introduction)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face%20Models-800k%2B%20downloads-brightgreen?logo=huggingface)](https://huggingface.co/FluidInference)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FluidInference/FluidAudio)

FluidAudio is a Swift SDK for fully local, low-latency audio AI on Apple devices, with inference offloaded to the Apple Neural Engine (ANE), resulting in less memory and generally faster inference.

The SDK includes state-of-the-art speaker diarization, transcription, and voice activity detection via open-source models (MIT/Apache 2.0) that can be integrated with just a few lines of code. Models are optimized for background processing, ambient computing and always on workloads by running inference on the ANE, minimizing CPU usage and avoiding GPU/MPS entirely.

For custom use cases, feedback, additional model support, or platform requests, join our [Discord](https://discord.gg/WNsvaCtmDe). We're also bringing visual, language, and TTS models to device and will share updates there.

Below are some featured local AI apps using Fluid Audio models on macOS and iOS:

<p align="left">
  <a href="https://github.com/Beingpax/VoiceInk/"><img src="Documentation/assets/voiceink.png" height="40" alt="Voice Ink"></a>
  <a href="https://spokenly.app/"><img src="Documentation/assets/spokenly.png" height="40" alt="Spokenly"></a>
  <a href="https://slipbox.ai/"><img src="Documentation/assets/slipbox.png" height="40" alt="Slipbox"></a>
  <a href="https://hex.kitlangton.com/"><img src="Documentation/assets/hex.png" height="40" alt="Hex"></a>
  <a href="https://boltai.com/"><img src="Documentation/assets/boltai.png" height="40" alt="BoltAI"></a>
  <a href="https://paraspeech.com"><img src="Documentation/assets/paraspeech.png" height="40" alt="Paraspeech"></a>
  <a href="https://altic.dev/fluid"><img src="Documentation/assets/fluidvoice.png" height="40" alt="Fluid Voice"></a>
  <a href="https://snaply.ai"><img src="Documentation/assets/snaply.png" height="40" alt="Snaply"></a>
  <a href="https://github.com/yazinsai/OpenOats"><img src="Documentation/assets/openoats.png" height="40" alt="OpenOats"></a>
  <a href="https://talat.app"><img src="Documentation/assets/talat.png" height="40" alt="Talat"></a>
<!-- Add your app: submit logo via PR. The Fluid Inference team works to curate this and add new apps to the showcase section every couple of weeks. We appreciate your patience. -->
</p>

Want to convert your own model? Check [möbius](https://github.com/FluidInference/mobius)

## Highlights

- **Automatic Speech Recognition (ASR)**: Parakeet TDT v3 (0.6b) for batch transcription supporting 25 European languages; Parakeet EOU (120m) for streaming ASR with end-of-utterance detection (English only)
- **Inverse Text Normalization (ITN)**: Post-process ASR output to convert spoken-form to written-form ("two hundred" → "200"). See [text-processing-rs](https://github.com/FluidInference/text-processing-rs)
- **Text-to-Speech (TTS)**: Kokoro (82m) for parallel synthesis with SSML and pronunciation control across 9 languages (EN, ES, FR, HI, IT, JA, PT, ZH); PocketTTS for streaming TTS with voice cloning support (English only)
- **Speaker Diarization (Online + Offline)**: Speaker separation and identification across audio streams. Streaming pipeline for real-time processing and offline batch pipeline with advanced clustering.
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering, you can use this for speaker identification
- **Voice Activity Detection (VAD)**: Voice activity detection with Silero models
- **Apple Neural Engine**: Models run efficiently on Apple's ANE for maximum performance with minimal power consumption
- **Open-Source Models**: All models are publicly available on HuggingFace — converted and optimized by our team; permissive licenses

## Video Demos

| Link | Description |
| --- | --- |
| **[Spokenly Real-time ASR](https://www.youtube.com/watch?v=9fXKKkyL8JE)** | Video demonstration of FluidAudio's transcription accuracy and speed |
| **[Senko Integration](https://x.com/hamza_q_/status/1970228971657928995)** | Python Speaker diarization on Mac using FluidAudio's segmentation model |
| **[Kokoro TTS](https://x.com/sach1n/status/1977817056507793521)** | Text-to-speech demo using FluidAudio's Kokoro and Silero models on iOS |
| **[Parakeet Realtime EOU](https://x.com/sach1n/status/2003210626659680762)** | Parakeet streaming ASR with end-of-utterance detection on iOS |
| **[Sortformer Diarization](https://x.com/Alex_tra_memory/status/2010530705667661843)** | Sortformer for speaker diarization with overlapping speech on iOS |
| **[PocketTTS](https://x.com/sach1n/status/2017627657006158296)** | Streaming text-to-speech using PocketTTS on iOS |
| **[Parakeet EOU Ultra-Low Latency](https://x.com/y_earu/status/2038654262608064967)** | Real-time Parakeet EOU transcription on iOS demonstrating ultra-low latency speech-to-text |
| **[Action Phrase Live Production Control](https://www.youtube.com/watch?v=ykcvdTHHmrk)** | Voice-controlled live production workflow using FluidAudio's ASR and speaker diarization to trigger cameras, graphics, and layouts with natural voice commands |

## Showcase

Make a PR if you want to add your app, please keep it in chronological order.

| App | Description |
| --- | --- |
| **[Voice Ink](https://tryvoiceink.com/)** | Local AI for instant, private transcription with near-perfect accuracy. Uses Parakeet ASR. |
| **[Spokenly](https://spokenly.app/)** | Mac dictation app for fast, accurate voice-to-text; supports real-time dictation and file transcription. Uses Parakeet ASR and speaker diarization. |
| **[Senko](https://github.com/narcotic-sh/senko)** | A very fast and accurate speaker diarization pipeline. A [good example](https://github.com/narcotic-sh/senko/commit/51dbd8bde764c3c6648dbbae57d6aff66c5ca15c) for how to integrate FluidAudio into a Python app |
| **[Slipbox](https://slipbox.ai/)** | Privacy-first meeting assistant for real-time conversation intelligence. Uses Parakeet ASR (iOS) and speaker diarization across platforms. |
| **[Whisper Mate](https://whisper.marksdo.com)** | Transcribes movies and audio locally; records and transcribes in real time from speakers or system apps. Uses speaker diarization. |
| **[Altic/Fluid Voice](https://github.com/altic-dev/Fluid-oss)** | Lightweight Fully free and Open Source Voice to Text dictation for macOS built using FluidAudio. Never pay for dictation apps |
| **[Paraspeech](https://paraspeech.com)** | AI powered voice to text. Fully offline. No subscriptions. |
| **[mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest)** | Comparison of different local ASR, including one of the first versions of FluidAudio's ASR models |
| **[Starling](https://github.com/Ryandonofrio3/Starling)** | Open Source, fully local voice-to-text transcription with auto-paste at your cursor. |
| **[BoltAI](https://boltai.com/)** | Write content 10x faster using parakeet models |
| **[Voxeoflow](https://www.voxeoflow.app)** | Mac dictation app with real-time translation. Lightning-fast transcription in over 100 languages, instantly translated to your target language. |
| **[Speakmac](https://speakmac.app)** | Mac app that lets you type anywhere on your Mac using your voice. Fully local, private dictation built on FluidAudio. |
| **[SamScribe](https://github.com/Steven-Weng/SamScribe)** | An open-source macOS app that captures and transcribes audio from your microphone and meeting applications (Zoom, Teams, Chrome) in real-time, with cross-session speaker recognition. |
| **[WhisKey](https://whiskey.asktobuild.app/)** | Privacy-first voice dictation keyboard for iOS and macOS. On-device transcription with 12+ languages, AI meeting summaries, and mindmap generation. Great for daily use and vibe-coding. Uses speaker diarization. |
| **[Dictate Anywhere](https://github.com/hoomanaskari/mac-dictate-anywhere)** | Native macOS dictation app with global Fn key activation. Dictate into any app with 25 language support. Uses Parakeet ASR. |
| **[hongbomiao.com](https://github.com/hongbo-miao/hongbomiao.com)** | A personal R&D lab that facilitates knowledge sharing. Uses Parakeet ASR. |
| **[Hex](https://github.com/kitlangton/Hex)** | macOS app that lets you press-and-hold a hotkey to record your voice, transcribe it, and paste into any application. Uses Parakeet ASR. |
| **[Super Voice Assistant](https://github.com/ykdojo/super-voice-assistant)** | Open-source macOS voice assistant with local transcription. Uses Parakeet ASR. |
| **[VoiceTypr](https://github.com/moinulmoin/voicetypr)** | Open-source voice-to-text dictation for macOS and Windows. Uses Parakeet ASR. |
| **[Summit AI Notes](https://summitnotes.app/)** | Local meeting transcription and summarization with speaker identification. Supports 100+ languages. |
| **[Ora](https://futurelab.studio/ora)** | Local voice assistant for macOS with speech recognition and text-to-speech. |
| **[Flowstay](https://flowstay.app)** | Easy text-to-speech, local post-processing and Claude Code integration for macOS. Free forever. |
| **[macos-speech-server](https://github.com/dokterbob/macos-speech-server)** | OpenAI compatible STT/transcription and TTS/speech API server. |
| **[Snaply](https://snaply.ai)** |Free, Fast, 100% local AI dictation for Mac. |
| **[OpenOats](https://github.com/yazinsai/OpenOats)** | Open-source meeting note-taker that transcribes conversations in real time and surfaces relevant notes from your knowledge base. Uses FluidAudio for local transcription. |
| **[Enconvo](https://enconvo.com)** | AI Agent Launcher for macOS with voice input, live captions, and text-to-speech. Uses Parakeet ASR for local speech recognition. |
| **[Meeting Transcriber](https://github.com/pasrom/meeting-transcriber)** | macOS menu bar app that auto-detects, records, and transcribes meetings (Teams, Zoom, Webex) with dual-track speaker diarization. Uses Parakeet ASR, Qwen3-ASR, and speaker diarization. |
| **[Hitoku Draft](https://hitoku.me/draft)** | A local, private, voice writing assistant on your macOS menu bar. Uses Parakeet ASR. |
| **[Audite](https://github.com/zachatrocity/audite)** | macOS menu-bar app that records meetings and transcribes them locally into Markdown notes for Obsidian. Uses Parakeet ASR via FluidAudio on the Apple Neural Engine. |
| **[Muesli](https://github.com/pHequals7/muesli)** | Native macOS dictation and meeting transcription with ~0.13s latency. Captures microphone and system audio with automatic speaker diarization. Uses Parakeet TDT and Qwen3 ASR. |
| **[NanoVoice](https://apps.apple.com/kz/app/nanovoice/id6760539688)** | Free iOS voice keyboard for fast, private dictation in any app. Uses Parakeet ASR. |
| **[MiniWhisper](https://github.com/andyhtran/MiniWhisper)** | Open-source macOS menu bar for quick local voice-to-text with minimal setup. Pick a shortcut, start talking. Uses Parakeet ASR. |
| **[Talat](https://talat.app)** | Privacy-focused AI meeting notes app. Records and transcribes meetings locally on your Mac with speaker identification and LLM-powered summaries. Featured in [TechCrunch](https://techcrunch.com/2026/03/24/talats-ai-meeting-notes-stay-on-your-machine-not-in-the-cloud/). Uses Parakeet ASR. |
| **[Volocal](https://github.com/fikrikarim/volocal)** | Fully local voice AI on iOS. Uses streaming Parakeet EOU ASR and streaming PocketTTS. |
| **[VivaDicta](https://github.com/n0an/VivaDicta)** | Open-source iOS voice-to-text app with system-wide AI voice keyboard — dictate and AI-process text in any app. 15+ AI providers, 40+ AI presets. Uses Parakeet ASR. |
| **[MimicScribe](https://mimicscribe.app/)** | macOS menu bar app combining Parakeet TDT streaming ASR, PyanNote Community 1 speaker diarization, and cloud LLMs to provide AI-generated talking points during meetings, derived from the live transcript and user-provided instructions. Features meeting summarization, natural language search, an MCP server for agent integration, and a keyboard- and voice-forward UI. |
| **[Action Phrase](https://actionphrase.com/)** | Voice-controlled live production app for iOS, iPadOS, and macOS. Control cameras, graphics, layouts, and production workflows with natural voice commands. Integrates with popular tools including OBS, vMix, ProPresenter, Bitfocus Companion, and more. Uses Parakeet TDT ASR and Sortformer diarization. |

## Installation

Add FluidAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.12.4"),
],
```

**In Xcode:**
1. Add the FluidAudio package to your project
2. In the "Add Package" dialog, select `FluidAudio`
3. Add it to your app target

**In Package.swift:**
```swift
.product(name: "FluidAudio", package: "FluidAudio")
```

**CocoaPods:** We recommend using [cocoapods-spm](https://github.com/trinhngocthuyen/cocoapods-spm) for better SPM integration, but if needed, you can also use our podspec: `pod 'FluidAudio', '~> 0.12.4'`

### Other Frameworks

Building with a different framework? Use one of our official wrappers:

| Platform | Package | Install |
|----------|---------|---------|
| **React Native / Expo** | [@fluidinference/react-native-fluidaudio](https://github.com/FluidInference/react-native-fluidaudio) | `npm install @fluidinference/react-native-fluidaudio` |
| **Rust / Tauri** | [fluidaudio-rs](https://github.com/FluidInference/fluidaudio-rs) | `cargo add fluidaudio-rs` |

### Post-Processing Tools

Enhance ASR output with post-processing:

| Tool | Description | Language |
|------|-------------|----------|
| **[text-processing-rs](https://github.com/FluidInference/text-processing-rs)** | Inverse Text Normalization (ITN) and Text Normalization (TN) across 7 languages (EN, DE, ES, FR, HI, JA, ZH). 100% NeMo test compatibility (3,011 tests). Converts spoken-form ASR output to written form ("two hundred" → "200", "five dollars" → "$5"). Rust port of [NVIDIA NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing) with Swift wrapper. | Rust, Swift |

## Configuration

### Quick Reference

Both solve the same problem: **"I can't reach HuggingFace directly."** They're alternative approaches - pick whichever matches your setup:

| Scenario | Solution | Configuration |
|----------|----------|---|
| You have a **local mirror or internal model server** | Use Registry URL override | `REGISTRY_URL=https://your-mirror.com` |
| You're **behind a corporate firewall** with a proxy that can reach HuggingFace | Use Proxy configuration | `https_proxy=http://proxy.company.com:8080` |

**How they work:**
- **Registry URL** - App requests from `your-mirror.com` instead of `huggingface.co`
- **Proxy** - App still requests `huggingface.co`, but traffic routes through proxy to reach it

In most cases, you only need one. (You'd use both only if your mirror is behind the proxy and unreachable without it.)

<details>
<summary><b>Model Registry URL</b> - Change download destination</summary>

By default, FluidAudio downloads models from HuggingFace. You can override this to use a mirror, local server, or air-gapped environment.

**Programmatic override (recommended for apps):**
```swift
import FluidAudio

// Set custom registry before using any managers
ModelRegistry.baseURL = "https://your-mirror.example.com"

// Models will now download from the custom registry
let diarizer = DiarizerManager()
```

**Environment Variables (recommended for CLI/testing):**
```bash
# Use custom registry
export REGISTRY_URL=https://your-mirror.example.com
swift run fluidaudiocli transcribe audio.wav

# Or use the MODEL_REGISTRY_URL alias
export MODEL_REGISTRY_URL=https://models.internal.corp
swift run fluidaudiocli diarization-benchmark --auto-download
```

**Xcode Scheme Configuration:**
1. Edit Scheme → Run → Arguments
2. Go to **Environment Variables** tab
3. Click `+` and add: `REGISTRY_URL` = `https://your-mirror.example.com`
4. The custom registry will apply to all debug runs

</details>

<details>
<summary><b>Proxy Configuration</b> - Route downloads through a proxy server</summary>

If you're behind a corporate firewall and cannot reach HuggingFace (or your registry) directly, configure a proxy to forward requests:

Set the `https_proxy` environment variable:

```bash
export https_proxy=http://proxy.company.com:8080
# or for authenticated proxies:
export https_proxy=http://user:password@proxy.company.com:8080

swift run fluidaudiocli transcribe audio.wav
```

**Xcode Scheme Configuration for Proxy:**
1. Edit Scheme → Run → Arguments
2. Go to **Environment Variables** tab
3. Click `+` and add: `https_proxy` = `http://proxy.company.com:8080`
4. FluidAudio will automatically route downloads through the proxy

</details>

## Documentation

**[DeepWiki](https://deepwiki.com/FluidInference/FluidAudio)** for auto-generated docs for this repo.

### Documentation Index

- Guides
  - [Audio Conversion for Inference](Documentation/Guides/AudioConversion.md)
  - Manual model download & loading options: [ASR](Documentation/ASR/ManualModelLoading.md), [Diarizer](Documentation/Diarization/GettingStarted.md#manual-model-loading), [VAD](Documentation/VAD/GettingStarted.md#manual-model-loading)
  - Routing Hugging Face (or compatible) requests through a proxy? Set `https_proxy` before running the download helpers (see [Documentation/API.md](Documentation/API.md:9)).
- Models
  - Automatic Speech Recognition/Transcription
    - [Getting Started](Documentation/ASR/GettingStarted.md)
    - [Last Chunk Handling](Documentation/ASR/LastChunkHandling.md)
  - Speaker Diarization
    - [Speaker Diarization Guide](Documentation/Diarization/GettingStarted.md)
  - VAD: [Getting Started](Documentation/VAD/GettingStarted.md)
    - [Segmentation](Documentation/VAD/Segmentation.md)
    - [Model Conversion Code](https://github.com/FluidInference/mobius)
- [Benchmarks](Documentation/Benchmarks.md)
- [API Reference](Documentation/API.md)
- [Command Line Guide](Documentation/CLI.md)

### MCP Server

The repo is indexed by DeepWiki MCP server, so your coding tool can access the docs:

```json
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/mcp"
    }
  }
}
```

For claude code:

```bash
claude mcp add -s user -t http deepwiki https://mcp.deepwiki.com/mcp
```

## Automatic Speech Recognition (ASR) / Transcription

- **Models**:
  - `FluidInference/parakeet-tdt-0.6b-v3-coreml` (multilingual, 25 European languages)
  - `FluidInference/parakeet-tdt-0.6b-v2-coreml` (English-only, highest recall)
- **Processing Mode**: Batch transcription for complete audio files
- **Real-time Factor**: ~190x on M4 Pro (processes 1 hour of audio in ~19 seconds)
- **Streaming Support**: Real-time streaming via `SlidingWindowAsrManager` with sliding window processing and cancellation support
- **Backend**: Same Parakeet TDT v3 model powers our backend ASR

### ASR Quick Start

```swift
import FluidAudio

// Batch transcription from an audio file
Task {
    // 1) Initialize ASR manager and load models
    let models = try await AsrModels.downloadAndLoad(version: .v3)  // Switch to .v2 for English-only work
    let asrManager = AsrManager(config: .default)
    try await asrManager.loadModels(models)

    // 3) Transcribe the audio 16hz, already converted
    let result = try await asrManager.transcribe(samples)

    // 3) Transcribe a file
    // let url = URL(fileURLWithPath: sample.audioPath)

    // 3) Transcribe AVAudioPCMBuffer
    // let result = try await asrManager.transcribe(audioBuffer)
    print("Transcription: \(result.text)")
}
```

```bash
# Transcribe an audio file (batch)
swift run fluidaudiocli transcribe audio.wav

# English-only run with higher recall
swift run fluidaudiocli transcribe audio.wav --model-version v2
```

## Speaker Diarization

### Offline Speaker Diarization Pipeline

Pyannote Community-1 pipeline (powerset segmentation + WeSpeaker + VBx) for offline speaker diarization. Use this for most use cases, see Benchmarks.md for benchmarks.

```swift
import FluidAudio

let config = OfflineDiarizerConfig()
let manager = OfflineDiarizerManager(config: config)
try await manager.prepareModels()  // Downloads + compiles Core ML bundles if they are missing

let samples = try AudioConverter().resampleAudioFile(path: "meeting.wav")
let result = try await manager.process(audio: samples)

for segment in result.segments {
    print("\(segment.speakerId) \(segment.startTimeSeconds)s → \(segment.endTimeSeconds)s")
}
```

For processing audio files, use the file-based API which automatically uses memory-mapped streaming for efficiency:

```swift
let url = URL(fileURLWithPath: "meeting.wav")
let result = try await manager.process(url)

for segment in result.segments {
    print("\(segment.speakerId) \(segment.startTimeSeconds)s → \(segment.endTimeSeconds)s")
}
```

```bash
# Process a meeting with full VBx clustering
swift run fluidaudiocli process ~/FluidAudioDatasets/ami_official/sdm/ES2004a.Mix-Headset.wav \
  --mode offline --threshold 0.6 --output es2004a_offline.json

# Run the AMI single-file benchmark with automatic downloads
swift run fluidaudiocli diarization-benchmark --mode offline --auto-download \
  --single-file ES2004a --threshold 0.6 --output offline_results.json
```

`offline_results.json` contains DER/JER/RTFx along with timing breakdowns for segmentation, embedding extraction, and VBx clustering. CI now runs this workflow on every PR to ensure the offline models stay healthy and the Hugging Face assets remain accessible.

### LS-EEND (LongForm Streaming End-to-End Neural Diarization)

End-to-end streaming diarization with CoreML inference. Default choice for online diarization — single model, no clustering pipeline, up to 10 speakers, 100ms frame updates with 900ms tentative preview. Supports both streaming and complete-buffer processing. See [Documentation/Diarization/GettingStarted.md](Documentation/Diarization/GettingStarted.md) for details.

```swift
import FluidAudio

Task {
    let diarizer = LSEENDDiarizer()
    try await diarizer.initialize(variant: .dihard3)

    let samples = try await loadSamples16kMono(path: "path/to/meeting.wav")
    let timeline = try diarizer.processComplete(samples, sourceSampleRate: 16_000)

    for segment in timeline.segments {
        print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

### Sortformer (End-to-End Neural Diarization)

End-to-end neural diarization using [NVIDIA's Sortformer](https://arxiv.org/abs/2409.06656). Secondary streaming diarizer — trades LS-EEND's higher speaker capacity and benchmark results for better speaker identity stability. Limited to 4 speakers. No separate VAD, segmentation, or clustering needed. Licensed under NVIDIA Open Model License.

Both LS-EEND and Sortformer emit results into a `DiarizerTimeline` with ultra-low-latency updates. See [Documentation/Diarization/Sortformer.md](Documentation/Diarization/Sortformer.md) for usage and comparison.

### Streaming/Online Speaker Diarization (Pyannote)

This pipeline uses segmentation plus speaker embeddings and is the third choice behind LS-EEND and Sortformer. It can be useful if you specifically want the classic multi-stage pipeline, but it is much slower than LS-EEND or Sortformer for live diarization.

Why use the WeSpeaker/Pyannote pipeline:
- More modular pipeline if you want separate segmentation and embedding stages
- Better fit when you need to integrate external speaker identification or clustering logic
- Speaker pre-enrollment is reliable
- Speaker database management is much easier
- Purging or updating individual speakers is straightforward
- Not recommended when low-latency live diarization is the priority

In most applications:
- Use LS-EEND as the default online diarizer
- Use Sortformer as the second choice when its stronger identity stability and participant focus matter more than the 4-speaker limit
- Use the WeSpeaker/Pyannote pipeline only when you specifically need its modular design despite the speed cost

Tradeoffs:
- Slower in both inference time and practical latency than LS-EEND or Sortformer
- Needs larger chunks, with at least 5 seconds usually required for decent results
- Unlike LS-EEND and Sortformer, speaker state is much easier to manipulate explicitly

```swift
import FluidAudio

// Diarize an audio file
Task {
    let models = try await DiarizerModels.downloadIfNeeded()
    let diarizer = DiarizerManager() 
    diarizer.initialize(models: models)

    // Prepare 16 kHz mono samples (see: Audio Conversion)
    let samples = try await loadSamples16kMono(path: "path/to/meeting.wav")

    // Run diarization
    let result = try diarizer.performCompleteDiarization(samples)
    for segment in result.segments {
        print("Speaker \(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

For diarization streaming see [Documentation/Diarization/GettingStarted.md](Documentation/Diarization/GettingStarted.md)

```bash
swift run fluidaudiocli diarization-benchmark --single-file ES2004a \
  --chunk-seconds 3 --overlap-seconds 2
```

### CLI

```bash
# Process an individual file and save JSON
swift run fluidaudiocli process meeting.wav --output results.json --threshold 0.6
```

## Voice Activity Detection (VAD)

Silero VAD powers our on-device detector. The latest release surfaces the same
timestamp extraction and streaming heuristics as the upstream PyTorch
implementation. Ping us on Discord if you need help tuning it for your
environment.

### VAD Quick Start (Offline Segmentation)

Simple call to return chunk-level probabilities every 256 ms hop:

```swift
let results = try await manager.process(samples)
for (index, chunk) in results.enumerated() {
    print(
        String(
            format: "Chunk %02d: prob=%.3f, inference=%.4fs",
            index,
            chunk.probability,
            chunk.processingTime
        )
    )
}
```

The following are higher level APIs better suited to integrate with other systems

```swift
import FluidAudio

Task {
    let manager = try await VadManager(
        config: VadConfig(defaultThreshold: 0.75)
    )

    let audioURL = URL(fileURLWithPath: "path/to/audio.wav")
    let samples = try AudioConverter().resampleAudioFile(audioURL)

    var segmentation = VadSegmentationConfig.default
    segmentation.minSpeechDuration = 0.25
    segmentation.minSilenceDuration = 0.4

    let segments = try await manager.segmentSpeech(samples, config: segmentation)
    for segment in segments {
        print(
            String(format: "Speech %.2f–%.2fs", segment.startTime, segment.endTime)
        )
    }
}
```

### Streaming

```swift
import FluidAudio

Task {
    let manager = try await VadManager()
    var state = await manager.makeStreamState()

    for chunk in microphoneChunks {
        let result = try await manager.processStreamingChunk(
            chunk,
            state: state,
            config: .default,
            returnSeconds: true,
            timeResolution: 2
        )

        state = result.state

        // Access raw probability (0.0-1.0) for custom logic
        print(String(format: "Probability: %.3f", result.probability))

        if let event = result.event {
            let label = event.kind == .speechStart ? "Start" : "End"
            print("\(label) @ \(event.time ?? 0)s")
        }
    }
}
```

### CLI

Start with the general-purpose `process` command, which runs the diarization
pipeline (and therefore VAD) end-to-end on a single file:

```bash
swift run fluidaudiocli process path/to/audio.wav
```

Once you need to experiment with VAD-specific knobs directly, reach for:

```bash
# Inspect offline segments (default mode)
swift run fluidaudiocli vad-analyze path/to/audio.wav

# Streaming simulation only (timestamps printed in seconds by default)
swift run fluidaudiocli vad-analyze path/to/audio.wav --streaming

# Benchmark accuracy/precision trade-offs
swift run fluidaudiocli vad-benchmark --num-files 50 --threshold 0.3
```

`swift run fluidaudiocli vad-analyze --help` lists every tuning option, including
negative-threshold overrides, max-speech splitting, padding, and chunk size.
Offline mode also reports RTFx using the model's per-chunk processing time.

## Text‑To‑Speech (TTS)

> **⚠️ Beta:** TTS currently supports American English only. Additional language support is planned.

FluidAudio ships two TTS backends:

| | PocketTTS | Kokoro |
|---|---|---|
| **GPL dependencies** | None | None |
| **Tokenizer** | SentencePiece | CoreML G2P → IPA phonemes |
| **Generation** | Frame-by-frame autoregressive (80ms) | Parallel (all frames at once) |
| **Streaming** | Yes | No |
| **Voice cloning** | Yes (1–30s audio sample) | No |
| **Pronunciation control** | No | Yes (SSML, custom lexicon) |
| **Output** | 24 kHz mono WAV | 24 kHz mono WAV |

### PocketTTS

Streaming-friendly TTS with voice cloning support from short audio samples.

```swift
import FluidAudio

Task {
    let manager = try await PocketTtsManager()
    let audioData = try await manager.synthesize("Hello from FluidAudio.")
    try audioData.write(to: URL(fileURLWithPath: "out.wav"))
}
```

```bash
# Synthesize with default voice
swift run fluidaudiocli tts "Hello from FluidAudio." --output out.wav --backend pocket

# Clone a voice from an audio sample
swift run fluidaudiocli tts "Hello world." --output out.wav --backend pocket --clone-voice speaker.wav
```

### Kokoro

High-quality parallel TTS with SSML and phoneme-level pronunciation control. Uses a CoreML G2P (grapheme-to-phoneme) model for out-of-vocabulary words — no external dependencies required.

```swift
import FluidAudio

Task {
    let manager = KokoroTtsManager()
    try await manager.initialize()
    let data = try await manager.synthesize(text: "Hello from FluidAudio.")
    try data.write(to: URL(fileURLWithPath: "out.wav"))
}
```

```bash
swift run fluidaudiocli tts "Hello from FluidAudio." --auto-download --output out.wav
```

Dictionary and model assets are cached under `~/.cache/fluidaudio/Models/kokoro`.

## Continuous Integration

- `tests.yml`: Default build matrix covering SwiftPM tests and an iOS archive smoke test.
- `diarizer-benchmark.yml`: Runs the streaming diarization benchmark on ES2004a for regression tracking.
- `offline-pipeline.yml`: Executes the VBx offline pipeline end-to-end (`fluidaudio diarization-benchmark --mode offline`) and fails if DER/JER drift beyond guardrails or if models fail to download. Use this workflow as a reference for provisioning model caches in your own CI.

## Everything Else

### FAQs

- CLI is available on macOS only. For iOS, use the library programmatically.
- Models auto-download on first use. If your network restricts Hugging Face access, set an HTTPS proxy: `export https_proxy=http://127.0.0.1:7890`.
- Windows alternative in development: [fluid-server](https://github.com/FluidInference/fluid-server)
- If you're looking to get the system audio on a Mac, take a look at this repo for reference [AudioCap](https://github.com/insidegui/AudioCap/tree/main)

### License

Apache 2.0 — see `LICENSE` for details.

### Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques.

Pyannote: <https://github.com/pyannote/pyannote-audio>

WeSpeaker: <https://github.com/wenet-e2e/wespeaker>

Parakeet-mlx: <https://github.com/senstella/parakeet-mlx>

silero-vad: <https://github.com/snakers4/silero-vad>

Kokoro-82M: <https://huggingface.co/hexgrad/Kokoro-82M>

### Citation

If you use FluidAudio in your work, please cite:

FluidInference Team. (2025). FluidAudio: Local Speaker Diarization, ASR, and VAD for Apple Platforms (Version 0.12.4) [Computer software]. GitHub. <https://github.com/FluidInference/FluidAudio>

```bibtex
@software{FluidInferenceTeam_FluidAudio_2025,
  author = {{FluidInference Team}},
  title = {{FluidAudio: Local Speaker Diarization, ASR, and VAD for Apple Platforms}},
  year = {2025},
  month = {3},
  version = {0.12.4},
  url = {https://github.com/FluidInference/FluidAudio},
  note = {Computer software}
}
```

---

## Show Your Support

Help the Fluid Inference community grow by adding a "Powered by Fluid Inference" badge to your project!

<p align="center">
  <a href="https://fluidinference.com">
    <img src="https://assets.inference.plus/fi-badge.png" alt="Powered by Fluid Inference" height="80">
  </a>
</p>

Copy and paste this prompt to your coding agent where you host your homepage:

```text
Add a centered 'Powered by Fluid Inference' badge to the footer linking to fluidinference.com. Image: https://assets.inference.plus/fi-badge.png. Whitelist the image hostname in your framework's config.
```

Or use one of these code snippets:

<details>
<summary>React/Next.js</summary>

```jsx
<div className="flex justify-center py-8">
  <a href="https://fluidinference.com">
    <img
      src="https://assets.inference.plus/fi-badge.png"
      alt="Powered by Fluid Inference"
      height={80}
    />
  </a>
</div>
```

</details>

<details>
<summary>HTML</summary>

```html
<div style="text-align: center; padding: 20px;">
  <a href="https://fluidinference.com">
    <img src="https://assets.inference.plus/fi-badge.png" alt="Powered by Fluid Inference" height="80">
  </a>
</div>
```

</details>

<details>
<summary>Markdown</summary>

```markdown
<p align="center">
  <a href="https://fluidinference.com">
    <img src="https://assets.inference.plus/fi-badge.png" alt="Powered by Fluid Inference" height="80">
  </a>
</p>
```

</details>
