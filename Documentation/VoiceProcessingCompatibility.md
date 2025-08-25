# Voice Processing Compatibility with FluidAudio

## Problem Description

When Apple's voice processing is enabled on `AVAudioInputNode`, it changes the audio format and can cause timestamp validation errors in FluidAudio's internal voice processor. Specifically:

- **Without voice processing**: Audio is typically 48kHz, 1 channel - works fine
- **With voice processing**: Audio becomes 96kHz, 3 channels - causes timestamp errors

The error message you might see:
```
[vp::vx::Voice_Processor:0x12605d800] failed to process downlink voice proc due to 'Unknown' error at line 79 column 19 in "vp/vx/io/wires/Audio_Pass_Through_Wire.cpp" - audio time stamp does not have valid sample time
```

## Root Cause

The issue occurs because when voice processing is enabled, the `AVAudioTime` object passed to the tap callback doesn't have a valid `sampleTime`, causing internal validation errors.

## Solution

FluidAudio's `StreamingAsrManager.streamAudio(_:)` method **does not require timestamps** - it only needs the audio buffer. The `AudioConverter` automatically handles format conversion from any input format (including 96kHz 3-channel) to the required 16kHz mono format.

## Recommended Implementation

### ❌ Problematic Code (timestamp dependency)

```swift
// Don't rely on AVAudioTime when voice processing is enabled
inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, time in
    // This fails when voice processing creates invalid timestamps
    guard time.sampleTime != AVAudioFramePosition.max else { return }
    self?.processAudioBuffer(buffer, at: time, source: .microphone)
}
```

### ✅ Correct Implementation

```swift
// Enable voice processing
do {
    try inputNode.setVoiceProcessingEnabled(true)
    inputNode.volume = 1.0
    inputNode.isVoiceProcessingBypassed = false
    
    // Configure ducking for macOS 14+
    if #available(macOS 14.0, *) {
        let duckingConfig = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
            enableAdvancedDucking: false,
            duckingLevel: .min
        )
        inputNode.voiceProcessingOtherAudioDuckingConfiguration = duckingConfig
        inputNode.isVoiceProcessingAGCEnabled = false
    }
} catch {
    print("Failed to enable voice processing: \(error)")
}

// Install tap - ignore the timestamp, only use the buffer
inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, time in
    // Key: Only use the buffer, ignore the time parameter
    self?.processAudioBuffer(buffer, source: .microphone)
}

private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, source: AudioSource) {
    guard let audioBuffer = buffer.copy() as? AVAudioPCMBuffer else { return }
    
    // FluidAudio handles format conversion automatically
    // 96kHz 3-channel → 16kHz mono conversion happens internally
    Task {
        await micStream.streamAudio(audioBuffer)
    }
}
```

## FluidAudio's Automatic Format Handling

FluidAudio's `AudioConverter` automatically handles:

1. **Sample rate conversion**: 96kHz → 16kHz (or any rate → 16kHz)
2. **Channel conversion**: 3-channel → mono (averaging across channels)  
3. **Format conversion**: Any PCM format → Float32

This happens transparently in the `StreamingAsrManager` when you call `streamAudio(_:)`.

## Testing Voice Processing Compatibility

Run the voice processing compatibility tests to verify everything works:

```bash
swift test --filter VoiceProcessingCompatibilityTests
```

These tests verify:
- ✅ 96kHz 3-channel format conversion works correctly
- ✅ Streaming works without timestamp dependency
- ✅ Performance remains good with voice processing formats
- ✅ Multiple voice processing buffers can be streamed

## Complete Working Example

```swift
import AVFoundation
import FluidAudio

@available(macOS 13.0, *)
class VoiceProcessingManager {
    private let audioEngine = AVAudioEngine()
    private let inputNode: AVAudioInputNode
    private var streamingManager: StreamingAsrManager?
    
    init() {
        inputNode = audioEngine.inputNode
        setupFluidAudio()
    }
    
    private func setupFluidAudio() {
        Task {
            streamingManager = StreamingAsrManager(config: .default)
            try await streamingManager?.start()
        }
    }
    
    func startRecordingWithVoiceProcessing() throws {
        // Enable voice processing
        try inputNode.setVoiceProcessingEnabled(true)
        inputNode.volume = 1.0
        inputNode.isVoiceProcessingBypassed = false
        
        // Configure for macOS 14+
        if #available(macOS 14.0, *) {
            let duckingConfig = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                enableAdvancedDucking: false,
                duckingLevel: .min
            )
            inputNode.voiceProcessingOtherAudioDuckingConfiguration = duckingConfig
            inputNode.isVoiceProcessingAGCEnabled = false
        }
        
        // Get the format AFTER enabling voice processing
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        print("Voice processing format: \(recordingFormat)")
        // Output: something like "3 ch, 96000 Hz, Float32, deinterleaved"
        
        // Install tap - only use buffer, ignore time parameter
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            // Key: Don't use the time parameter
            self?.processAudioBuffer(buffer)
        }
        
        // Start audio engine
        try audioEngine.start()
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let buffer = buffer.copy() as? AVAudioPCMBuffer else { return }
        
        // FluidAudio automatically converts 96kHz 3-channel to 16kHz mono
        streamingManager?.streamAudio(buffer)
    }
    
    func stopRecording() {
        audioEngine.stop()
        inputNode.removeTap(onBus: 0)
        
        Task {
            await streamingManager?.cancel()
        }
    }
}
```

## Key Takeaways

1. **Don't rely on `AVAudioTime`** when voice processing is enabled
2. **Only pass the `AVAudioPCMBuffer`** to FluidAudio's `streamAudio(_:)` method  
3. **FluidAudio handles format conversion automatically** - no manual conversion needed
4. **Voice processing format changes are transparent** to your application
5. **Test with the provided compatibility tests** to ensure everything works

This approach eliminates the timestamp validation error while maintaining full compatibility with Apple's voice processing features like echo cancellation and noise reduction.