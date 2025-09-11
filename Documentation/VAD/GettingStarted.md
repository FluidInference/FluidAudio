# Voice Activity Detection (VAD)

The current VAD APIs require careful tuning for your specific use case. If you need help integrating VAD, reach out in our Discord channel.

Our goal is to provide a streamlined API similar to Apple's upcoming SpeechDetector in [OS26](https://developer.apple.com/documentation/speech/speechdetector).

## Quick Start (Code)

```swift
import FluidAudio

// Programmatic VAD over an audio file
Task {
    // 1) Initialize VAD (async load of Silero model)
    let vad = try await VadManager(config: VadConfig(threshold: 0.3))

    // 2) Prepare 16 kHz mono samples (see: Audio Conversion)
    let samples = try await loadSamples16kMono(path: "path/to/audio.wav")

    // 3) Run VAD and print speech segments (512-sample frames)
    let results = try await vad.processAudioFile(samples)
    let sampleRate = 16000.0
    let frame = 512.0

    var startIndex: Int? = nil
    for (i, r) in results.enumerated() {
        if r.isVoiceActive {
            if startIndex == nil { startIndex = i }
        } else if let s = startIndex {
            let startSec = (Double(s) * frame) / sampleRate
            let endSec = (Double(i + 1) * frame) / sampleRate
            print(String(format: "Speech: %.2f–%.2fs", startSec, endSec))
            startIndex = nil
        }
    }
}
```

## CLI

```bash
# Run VAD benchmark (mini50 dataset by default)
swift run fluidaudio vad-benchmark --num-files 50 --threshold 0.3

# Save results and enable debug output
swift run fluidaudio vad-benchmark --all-files --output vad_results.json --debug

# Download VAD dataset if needed
swift run fluidaudio download --dataset vad
```

