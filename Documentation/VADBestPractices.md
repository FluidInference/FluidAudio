# VAD (Voice Activity Detection) Best Practices

This guide provides best practices for using FluidAudio's Voice Activity Detection system effectively.

## Quick Start

For most use cases, use the optimized configuration:

```swift
let vadManager = VadManager(config: .optimized)
try await vadManager.initialize()
```

This provides:
- **Threshold**: 0.445 (98% accuracy on benchmarks)
- **Adaptive thresholding**: Automatically adjusts to ambient noise
- **SNR filtering**: Robust noise rejection
- **Debug mode**: Off (reduced logging)

## Understanding VAD Output

### Probability Values

The VAD system outputs a probability value between 0.0 and 1.0:
- **0.0 - 0.2**: Very likely non-speech (silence, low noise)
- **0.2 - 0.4**: Possibly speech with low confidence
- **0.4 - 0.6**: Likely speech (around threshold)
- **0.6 - 0.8**: High confidence speech
- **0.8 - 1.0**: Very high confidence speech

### Threshold Selection

The default threshold of 0.445 was determined through extensive benchmarking:
- **98% accuracy** on MUSAN dataset
- Balances false positives and false negatives
- Works well across different noise conditions

For specific use cases:
- **Noisy environments**: Increase threshold to 0.5-0.6
- **Quiet environments**: Decrease threshold to 0.3-0.4
- **Real-time applications**: Use adaptive thresholding

## Configuration Options

### Basic Configuration

```swift
let config = VadConfig(
    threshold: 0.445,            // Detection threshold
    chunkSize: 512,              // 32ms at 16kHz
    sampleRate: 16000,           // Input sample rate
    debugMode: false             // Logging verbosity
)
```

### Advanced Configuration

```swift
let config = VadConfig(
    threshold: 0.445,
    chunkSize: 512,
    sampleRate: 16000,
    debugMode: false,
    
    // Adaptive thresholding
    adaptiveThreshold: true,     // Enable dynamic adjustment
    minThreshold: 0.1,           // Minimum allowed threshold
    maxThreshold: 0.7,           // Maximum allowed threshold
    
    // SNR filtering
    enableSNRFiltering: true,    // Enable noise analysis
    minSNRThreshold: 6.0,        // Minimum SNR in dB
    noiseFloorWindow: 100,       // Samples for noise estimation
    
    // Spectral analysis
    spectralRolloffThreshold: 0.85,
    spectralCentroidRange: (200.0, 8000.0),
    
    // Compute units
    computeUnits: .cpuAndNeuralEngine
)
```

## Performance Optimization

### Chunk Size Selection

- **512 samples (32ms)**: Default, good balance
- **256 samples (16ms)**: Lower latency, slightly less accurate
- **1024 samples (64ms)**: More accurate, higher latency

### Compute Units

- **.cpuAndNeuralEngine**: Best for Apple Silicon
- **.cpuOnly**: More compatible, slower
- **.all**: Let CoreML decide (recommended)

### Real-time Processing

For real-time applications:

```swift
// Configure for low latency
let config = VadConfig(
    threshold: 0.445,
    chunkSize: 256,              // 16ms chunks
    adaptiveThreshold: true,     // Handle varying conditions
    debugMode: false             // Minimize overhead
)

// Process audio stream
func processAudioStream(_ samples: [Float]) async {
    let result = try await vadManager.processChunk(samples)
    
    if result.isVoiceActive {
        // Handle voice activity
        startRecording()
    } else {
        // Handle silence
        stopRecording()
    }
}
```

## Common Issues and Solutions

### Issue: Low Probability Values

**Symptom**: Probabilities always < 0.1

**Solutions**:
1. Ensure audio is 16kHz sample rate
2. Check audio levels (not too quiet)
3. Verify audio is mono, not stereo
4. Update to latest FluidAudio version

### Issue: Too Many False Positives

**Symptom**: Non-speech detected as speech

**Solutions**:
1. Increase threshold (try 0.5-0.6)
2. Enable SNR filtering
3. Adjust `minSNRThreshold` higher

### Issue: Missing Speech

**Symptom**: Speech not detected

**Solutions**:
1. Decrease threshold (try 0.3-0.4)
2. Check audio input levels
3. Ensure proper audio preprocessing

## Audio Preprocessing

For best results, preprocess audio before VAD:

```swift
// 1. Ensure 16kHz sample rate
let resampled = try await AudioProcessor.resampleAudio(
    audioData, 
    from: originalSampleRate, 
    to: 16000
)

// 2. Normalize audio levels
let normalized = AudioProcessor.normalizeAudio(resampled)

// 3. Apply VAD
let result = try await vadManager.processChunk(normalized)
```

## Debugging

Enable debug mode for troubleshooting:

```swift
let config = VadConfig(
    threshold: 0.445,
    debugMode: true  // Enable detailed logging
)
```

This will log:
- Model output shapes
- Probability calculations
- SNR values
- Processing times

## Integration Examples

### With Speaker Diarization

```swift
// Use VAD to filter before diarization
let vadResults = try await vadManager.processAudioFile(audioData)
let speechSegments = extractSpeechSegments(vadResults)
let diarizationResult = try await diarizer.process(speechSegments)
```

### With ASR

```swift
// Only transcribe when voice is detected
if vadResult.isVoiceActive && vadResult.probability > 0.6 {
    let transcription = try await asrManager.transcribe(audioChunk)
}
```

## Benchmarking

Run benchmarks to validate configuration:

```bash
swift run fluidaudio vad-benchmark --threshold 0.445 --all-files
```

Expected results:
- Accuracy: >95%
- Precision: >90%
- Recall: >90%
- F1-Score: >90%
- RTFx: >20x faster than real-time

## Model Information

FluidAudio uses Silero VAD models converted to CoreML:
- Optimized for Apple Neural Engine
- 512-sample chunk processing
- RNN-based architecture
- ~5MB model size

The models are automatically downloaded and cached on first use.