# Audio Conversion (16 kHz mono)

Most FluidAudio features expect 16 kHz mono Float32 samples. Most of the commands should already be doing this implicitly when receiving input from the client but if needed, you can also convert manually.

Implementation details:

- System APIs: Under the hood, conversion uses `AVAudioConverter` for sample‑rate conversion, sample‑format conversion
  (e.g., Int16 → Float32), and channel mixing (stereo → mono).
- No manual resampling: We do not implement custom resamplers (e.g., linear interpolation/decimation). Always go through
  `AudioConverter` so the system handles quality and buffering correctly.

## Swift Example

```swift
import AVFoundation
import FluidAudio

public func loadSamples16kMono(path: String) async throws -> [Float] {
    let converter = AudioConverter()
    return try converter.resampleAudioFile(path: path)
}
```

Notes:
- Input can be any format readable by `AVAudioFile` (e.g., WAV, M4A, MP3, FLAC).
- Output is 16 kHz mono Float32 samples suitable for ASR/VAD/Diarization.
- Each conversion is stateless, so you can reuse the same converter instance for different audio formats.

## Streaming Example

```swift
import AVFoundation
import FluidAudio

let converter = AudioConverter()

// In your audio capture loop, per incoming chunk:
func processChunk(_ pcmBuffer: AVAudioPCMBuffer) async throws {
    let samples = try converter.resampleBuffer(pcmBuffer)
    // feed `samples` to downstream ASR/VAD/etc.
}

// No need to flush - each conversion is complete and independent
```
