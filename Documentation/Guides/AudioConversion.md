# Audio Conversion (16 kHz mono)

Most FluidAudio features expect 16 kHz mono Float32 samples. Use `AudioConverter` for both CLI and library paths to ensure identical results.

## Swift Example

```swift
import AVFoundation
import FluidAudio

public func loadSamples16kMono(path: String) async throws -> [Float] {
    let converter = AudioConverter()
    return try await converter.convertFileToAsrSamples(path: path)
}
```

Notes:
- Input can be any format readable by `AVAudioFile` (e.g., WAV, M4A, MP3, FLAC).
- Output is 16 kHz mono Float32 samples suitable for ASR/VAD/Diarization.
- For live/streaming audio, call `convertToAsrFormat(buffer, streaming: true)` per chunk and finish with `finishStreamingConversion()` to flush remaining samples.
