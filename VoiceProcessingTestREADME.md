# Voice Processing Test App

## Overview

This test app reproduces the exact voice processing timestamp issue described in the GitHub issue and validates FluidAudio's handling of the problem.

## The Issue

When Apple's voice processing is enabled on `AVAudioInputNode`:

- **Without voice processing**: Audio is typically 48kHz, 1 channel - works fine
- **With voice processing**: Audio becomes 96kHz, 3 channels - causes timestamp errors

The error message:
```
[vp::vx::Voice_Processor:0x12605d800] failed to process downlink voice proc due to 'Unknown' error at line 79 column 19 in "vp/vx/io/wires/Audio_Pass_Through_Wire.cpp" - audio time stamp does not have valid sample time
```

## The Solution

FluidAudio's `StreamingAsrManager.streamAudio(_:)` method **only uses the audio buffer**, not the timestamp. This bypasses the timestamp validation error while maintaining full functionality.

## Running the Test App

### Option 1: Use the provided script
```bash
./run_voice_processing_test.sh
```

### Option 2: Run directly with Swift Package Manager
```bash
swift run voice-processing-test
```

### Option 3: Build first, then run
```bash
swift build
swift run voice-processing-test
```

## Test Scenarios

The app provides two automated test scenarios:

### Test Scenario A: Voice Processing Enabled
1. Enables voice processing with exact user configuration
2. Records for 10 seconds
3. Monitors timestamp validity and FluidAudio processing
4. Reports any errors or issues

### Test Scenario B: Voice Processing Disabled  
1. Disables voice processing (baseline)
2. Records for 10 seconds
3. Compares behavior with Scenario A

## Interactive Commands

1. **Run Test Scenario A** - Voice Processing Enabled
2. **Run Test Scenario B** - Voice Processing Disabled  
3. **Toggle Voice Processing** - Manual on/off
4. **Start/Stop Recording Manually** - Manual control
5. **Show Current Status** - Display system state
6. **Show Recent Logs** - View diagnostic information
7. **Clear Logs** - Reset log history
**q** - **Quit**

## What the App Tests

### 🔍 Timestamp Validation
- Monitors `AVAudioTime.sampleTime` and `AVAudioTime.hostTime` validity
- Counts valid vs invalid timestamps
- Reports timestamp error rates

### 📊 Format Monitoring
- Displays audio format before/after voice processing
- Shows sample rate, channel count, and format changes
- Logs format transitions in real-time

### 🎙️ FluidAudio Integration
- Tests `StreamingAsrManager` with voice processing formats
- Monitors for any streaming errors
- Displays transcription results (if models are available)
- Validates format conversion (96kHz 3-channel → 16kHz mono)

### 📈 Performance Metrics
- Buffer processing rates
- Timestamp error statistics
- FluidAudio processing success rates

## Expected Results

### ✅ With Voice Processing Enabled
- **Audio Format**: Changes to higher sample rate + multiple channels
- **Timestamps**: May show invalid `sampleTime` values
- **FluidAudio**: Should continue working normally
- **Transcription**: Should work if models are available
- **Errors**: No FluidAudio streaming errors

### ✅ Without Voice Processing  
- **Audio Format**: Standard format (e.g., 48kHz mono)
- **Timestamps**: Should be valid
- **FluidAudio**: Works normally
- **Baseline**: For comparison with voice processing

## Key Insights

1. **FluidAudio doesn't use timestamps** - Only the audio buffer matters
2. **Automatic format conversion** - Handles any input format → 16kHz mono
3. **Voice processing is transparent** - No code changes needed
4. **Timestamp errors are harmless** - Don't affect FluidAudio processing

## Troubleshooting

### Microphone Permission
The app will request microphone permission. Grant it to test properly.

### FluidAudio Models
If models aren't available, transcription won't work, but format/timestamp testing will still function.

### Build Issues
Ensure you're running from the FluidAudio root directory with:
```bash
swift --version  # Check Swift is available
swift build      # Build dependencies first
```

## Example Output

```
🎤 FluidAudio Voice Processing Test App
=====================================

📋 Checking system requirements...
✅ macOS version: 14.2.1
✅ Swift version: swift-driver version: 1.87.3

🎬 Starting recording...
📊 Recording format: <AVAudioFormat 0x12345678:  3 ch,  96000 Hz, Float32, deinterleaved>
  - Sample rate: 96000.0 Hz
  - Channels: 3
  - Format: 1
  - Interleaved: false

⚠️ Timestamp issue #1: sampleTime=INVALID, hostTime=valid
📦 Processed 100 buffers...
✅ FluidAudio initialized successfully
[VOLATILE] hello world (conf: 0.85)

📊 Recording session summary:
  - Total buffers processed: 500
  - Valid timestamps: 0
  - Invalid timestamps: 500
  - Timestamp error rate: 100.0%
```

## Validation

This app validates that:
- ✅ FluidAudio handles voice processing formats correctly
- ✅ Timestamp validation errors don't affect functionality  
- ✅ Format conversion works (multi-channel → mono)
- ✅ Real-time transcription continues working
- ✅ No code changes are needed for voice processing compatibility

The test demonstrates that the original user's code should work without modification when using FluidAudio.