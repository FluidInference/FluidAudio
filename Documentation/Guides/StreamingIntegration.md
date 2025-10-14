# Streaming ASR Integration Guide

This checklist distills the stabilized streaming pipeline so automation agents (or humans) can wire it up quickly without digging through every source file.

## Prerequisites
- macOS 13+ or iOS 16+ (Swift Concurrency + Core ML requirements)
- `FluidAudio` imported in your target
- Network access the first time you download ASR + VAD models (or staged bundles in your app)

## Implementation Steps
1. **Download models**
   ```swift
   let models = try await AsrModels.downloadAndLoad(version: .v3)  // or .v2 for English-only
   ```
   - Ships the compiled Core ML artifacts into `~/Library/Application Support/com.fluidaudio`.
   - For offline usage, pre-stage the bundles and call `AsrModels.load(from:)`.

2. **Configure streaming**
   ```swift
   var config = StreamingAsrConfig.streaming  // Balanced stabilization preset
   config = config.withStabilizer(
       StreamingStabilizerConfig
           .preset(.lowLatency)
           .withMaxWaitMilliseconds(650)
   )
   ```
   - `.streaming` preset = 11.2 s chunk, 1.6 s overlap, VAD enabled.
   - Use `.default` when you need backwards compatibility with the old API.

3. **Reuse or inject VAD**
   - Let `StreamingAsrManager` manage VAD for you (`config.vad = .default`), or
   - Create a shared `VadManager` and pass it to the initializer:
     ```swift
     let vadManager = try await VadManager(config: .init(threshold: 0.30))
     let streamingAsr = StreamingAsrManager(config: config, vadManager: vadManager)
     ```
   - Sharing the instance avoids redundant downloads and warm-up across multiple sessions.

4. **Start the session**
   ```swift
   try await streamingAsr.start(models: models, source: .microphone)
   ```
   - Pass `.system` if your audio originates from the OS mix.
   - On resume, `start()` ensures the stabilizer + pipeline queues are primed.

5. **Feed audio buffers**
   ```swift
   inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
       Task { await streamingAsr.streamAudio(buffer) }
   }
   ```
   - Any `AVAudioPCMBuffer` format is accepted; audio is resampled internally.
   - For file playback, stream converted `Float` arrays with `streamAudio(samples:)`.

6. **Consume transcription updates**
   ```swift
   Task {
       for await update in await streamingAsr.transcriptionUpdates {
           if update.isConfirmed {
               confirmedText.append(update.text)
           } else {
               volatileOverlay = update.text
           }
       }
   }
   ```
   - `StreamingTranscriptionUpdate` provides token IDs + timings so you can highlight words or align subtitles.
   - Keep rendering volatile text separately; append confirmed strings to your transcript accumulator.

7. **Finalize gracefully**
   ```swift
   inputNode.removeTap(onBus: 0)
   engine.stop()
   let finalTranscript = try await streamingAsr.finish()
   ```
   - `finish()` flushes pending chunks, closes the async stream, and returns the merged transcript.

## Stabilization Profiles
- `.balanced` (default) – 3-window consensus, waits up to 800 ms to confirm.
- `.lowLatency` – commits quickly (2-window consensus, 450 ms max wait) at the cost of small rewinds.
- `.highStability` – slows confirmation (4-window consensus, 1.2 s wait) for the cleanest text.
- Toggle via CLI (`--stabilize-profile`) or in code (`StreamingStabilizerConfig.preset(_)`).

## Observability & Debugging
- `streamingAsr.metricsSnapshot()` → quick look at chunk counts, average processing time, and latency.
- `.withDebugDumpEnabled(true)` on the stabilizer writes JSONL traces per chunk to `~/Library/Logs/FluidAudio`.
- CLI users can run `swift run fluidaudio transcribe audio.wav --streaming --stabilize-debug` to collect per-window traces.

## Common Pitfalls
- **Clipping leading words:** Ensure VAD `speechPadding` ≥ 0.2 s when trimming silence aggressively.
- **Out-of-order updates:** Always process updates sequentially from the async stream; do not run multiple collectors concurrently.
- **Long pauses:** When streams idle for several seconds, call `streamingAsr.cancel()` to avoid holding buffers indefinitely.
- **Background threads:** No `@unchecked Sendable` usage—call into the actor from tasks and let it hop contexts safely.

For a full architectural deep dive, continue with [ASR/StabilizedStreaming](../ASR/StabilizedStreaming.md).
