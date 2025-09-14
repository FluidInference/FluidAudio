# Voice Activity Detection (VAD)

The current VAD APIs require careful tuning for your specific use case. If you need help integrating VAD, reach out in our Discord channel.

Our goal is to provide a streamlined API similar to Apple's upcoming SpeechDetector in [OS26](https://developer.apple.com/documentation/speech/speechdetector).

## Quick Start (Code)

```swift
import FluidAudio

// Programmatic VAD over an audio file
Task {
    // 1) Initialize VAD (async loads Silero model)
    let vad = try await VadManager(
        config: VadConfig(threshold: 0.85, debugMode: false)
    )

    // 2) Process any supported file; conversion to 16 kHz mono is automatic
    let url = URL(fileURLWithPath: "path/to/audio.wav")
    let results = try await vad.process(url)

    // 3) Convert per-frame decisions into segments (512-sample frames @ 16 kHz)
    let sampleRate = 16000.0
    let frame = 512.0

    var startIndex: Int? = nil
    for (i, r) in results.enumerated() {
        if r.isVoiceActive {
            startIndex = startIndex ?? i
        } else if let s = startIndex {
            let startSec = (Double(s) * frame) / sampleRate
            let endSec = (Double(i + 1) * frame) / sampleRate
            print(String(format: "Speech: %.2f–%.2fs", startSec, endSec))
            startIndex = nil
        }
    }
}
```

Notes:
- You can also call `process(_ buffer: AVAudioPCMBuffer)` or `process(_ samples: [Float])`.
- Frame size is `512` samples (32 ms at 16 kHz). Threshold defaults to `0.85`.

## CLI

```bash
# Run VAD benchmark (mini50 dataset by default)
swift run fluidaudio vad-benchmark --num-files 50 --threshold 0.3

# Save results and enable debug output
swift run fluidaudio vad-benchmark --all-files --output vad_results.json --debug

# VOiCES subset mixed-condition benchmark (high-precision setting)
swift run fluidaudio vad-benchmark --dataset voices-subset --all-files --threshold 0.85

# Download VAD dataset if needed
swift run fluidaudio download --dataset vad
```
