# FluidAudio Bundled Models Examples

This directory contains examples showing how to bundle FluidAudio models with your app for offline use.

## Overview

Instead of downloading models at runtime, you can bundle them with your app to provide:
- ✅ Offline functionality
- ✅ Faster startup times
- ✅ No network dependency
- ✅ Consistent model versions
- ✅ Better user experience

⚠️ **Trade-offs:**
- Increases app size significantly (~600MB total for all models)
- Models become part of app updates
- Less flexibility for model updates

## Examples

### 1. [BundledDiarizationExample.swift](BundledDiarizationExample.swift)
Shows how to bundle and use diarization models (speaker identification).

**Models needed:**
- `pyannote_segmentation.mlmodelc` (~50MB)
- `wespeaker_v2.mlmodelc` (~50MB)

**Features:**
- Load models from app bundle
- Custom configurations
- Performance optimization
- Alternative storage approaches

### 2. [BundledASRExample.swift](BundledASRExample.swift)
Shows how to bundle and use ASR (Automatic Speech Recognition) models.

**Models needed:**
- `Melspectogram.mlmodelc` (~5MB)
- `ParakeetEncoder_v2.mlmodelc` (~200MB)
- `ParakeetDecoder.mlmodelc` (~100MB)
- `RNNTJoint.mlmodelc` (~50MB)
- `parakeet_vocab.json` (~1MB)

**Features:**
- Individual model loading
- Performance configurations
- Background processing support
- Streaming transcription

### 3. [BundledCombinedExample.swift](BundledCombinedExample.swift)
Shows how to use both diarization and ASR together for complete audio analysis.

**Features:**
- Speaker identification + transcription
- Conversation transcript generation
- Per-speaker analysis
- Performance metrics

## Quick Start

### Step 1: Download Models

First, download the models using FluidAudio CLI:

```bash
# Download diarization models
swift run fluidaudio download-models --type diarizer

# Download ASR models  
swift run fluidaudio download-models --type asr
```

Or programmatically:

```swift
// Download diarization models
let diarizerModels = try await DiarizerModels.downloadIfNeeded()

// Download ASR models
let asrModels = try await AsrModels.downloadAndLoad()
```

### Step 2: Locate Downloaded Models

Models are downloaded to:
```
~/Library/Application Support/FluidAudio/Models/
├── pyannote-segmentation-3.0/
│   ├── pyannote_segmentation.mlmodelc/
│   └── wespeaker_v2.mlmodelc/
└── parakeet-tdt-0.6b-v2-coreml/
    ├── Melspectogram.mlmodelc/
    ├── ParakeetEncoder_v2.mlmodelc/
    ├── ParakeetDecoder.mlmodelc/
    ├── RNNTJoint.mlmodelc/
    └── parakeet_vocab.json
```

### Step 3: Add to Xcode Project

1. Drag the model files into your Xcode project
2. Make sure "Add to target" is checked for your app target
3. Choose "Create groups" (not folder references)

### Step 4: Use in Your App

```swift
import FluidAudio

// Diarization example
do {
    try await BundledDiarizationExample.performDiarization()
} catch {
    print("Diarization failed: \(error)")
}

// ASR example
do {
    try await BundledASRExample.performTranscription()
} catch {
    print("Transcription failed: \(error)")
}

// Combined analysis
do {
    try await BundledCombinedExample.demonstrateCompleteWorkflow()
} catch {
    print("Analysis failed: \(error)")
}
```

## Model Sizes and Performance

| Model Type | Size | Purpose | Performance Impact |
|------------|------|---------|-------------------|
| Segmentation | ~50MB | Speaker boundary detection | Low CPU/ANE |
| Embedding | ~50MB | Speaker feature extraction | Medium ANE |
| Melspectrogram | ~5MB | Audio feature extraction | Low CPU/GPU |
| Encoder | ~200MB | Speech encoding | High ANE |
| Decoder | ~100MB | Language modeling | High ANE |
| Joint | ~50MB | Output generation | Medium ANE |
| Vocabulary | ~1MB | Token mapping | N/A |

**Total:** ~456MB for all models

## Performance Tips

### 1. Compute Units
```swift
// Optimal for most devices
config.computeUnits = .cpuAndNeuralEngine

// For iOS background processing
config.computeUnits = .cpuAndNeuralEngine  // Avoid GPU in background

// For maximum accuracy (slower)
config.computeUnits = .all
config.allowLowPrecisionAccumulationOnGPU = false
```

### 2. Memory Management
```swift
// Clean up when done
diarizerManager.cleanup()
asrManager.cleanup()

// Reset decoder state between files
try await asrManager.resetDecoderState(for: .microphone)
```

### 3. Model Loading
```swift
// Load models asynchronously
async let diarizer = BundledDiarizationExample.initializeWithBundledModels()
async let asr = BundledASRExample.initializeWithBundledModels()

let (diarizerManager, asrManager) = try await (diarizer, asr)
```

## Alternative Approaches

### Option 1: Copy to Documents Directory
Copy models from bundle to Documents directory on first launch:

```swift
let modelsURL = try BundledDiarizationExample.copyModelsToDocuments()
let models = try await DiarizerModels.load(from: modelsURL)
```

### Option 2: Hybrid Approach
Bundle critical models, download optional ones:

```swift
// Bundle core models for offline use
let coreModels = try await loadBundledModels()

// Download enhanced models when network available
if hasNetworkConnection {
    let enhancedModels = try await downloadEnhancedModels()
}
```

### Option 3: Model Variants
Bundle different model sizes for different use cases:

```swift
// Bundle quantized/smaller models for basic functionality
// Download full models for premium features
```

## Best Practices

1. **Model Validation**: Always validate bundled models work before shipping
2. **Error Handling**: Provide fallbacks if bundled models fail to load
3. **Memory Management**: Clean up models when not needed
4. **Background Processing**: Use appropriate compute units for iOS background
5. **Testing**: Test on different devices and iOS versions
6. **Update Strategy**: Plan for model updates in future app versions

## Troubleshooting

### Common Issues

1. **Models not found in bundle**
   - Verify models are added to Xcode target
   - Check file names match exactly
   - Ensure models are in app bundle, not frameworks

2. **Model loading fails**
   - Check device compatibility (iOS 16+, macOS 13+)
   - Verify model files aren't corrupted
   - Try different compute unit configurations

3. **Memory issues**
   - Don't load all models simultaneously if not needed
   - Clean up unused models
   - Monitor memory usage in development

4. **Performance issues**
   - Use appropriate compute units for device
   - Enable FP16 optimization where supported
   - Profile performance on target devices

### Debug Tips

```swift
// Enable debug mode
let config = DiarizerConfig(debugMode: true)
let asrConfig = ASRConfig(enableDebug: true)

// Check model availability
print("Diarizer available: \(diarizer.isAvailable)")
print("ASR available: \(asr.isAvailable)")

// Monitor performance
asr.profilePerformance()
```

## See Also

- [FluidAudio Documentation](../Documentation/)
- [Performance Optimization Guide](../CLAUDE.md)
- [API Reference](../Sources/FluidAudio/)