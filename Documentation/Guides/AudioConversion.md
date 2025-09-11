# Audio Conversion (16 kHz mono)

Most FluidAudio features expect 16 kHz mono Float32 samples. Use `AudioConverter` to load and convert from any `AVAudioFile` format.

## Swift Example

```swift
import AVFoundation
import FluidAudio

public func loadSamples16kMono(path: String) async throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let file = try AVAudioFile(forReading: url)
    let capacity = AVAudioFrameCount(file.length)
    guard let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: capacity) else {
        return []
    }
    try file.read(into: buf)
    let converter = AudioConverter()
    return try await converter.convertToAsrFormat(buf)
}
```

Notes:
- Input can be any PCM format supported by `AVAudioFile`.
- Output is 16 kHz mono Float32 samples suitable for ASR/VAD/Diarization.
