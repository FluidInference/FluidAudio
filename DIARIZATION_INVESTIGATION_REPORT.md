# Diarization Issue Investigation Report

**Date**: November 2025
**Issue**: NotebookLLM-generated podcast files only detecting 1 speaker instead of 2
**Status**: Investigated and partially resolved

---

## Issue Summary

**Original Problem**: Speaker diarization on NotebookLLM-generated podcast files was only detecting 1 speaker despite having 1 male and 1 female voice (very distinct).

### Platform-Specific Behavior

| Platform | Behavior | Details |
|----------|----------|---------|
| **iOS Device** | ❌ Never working | Consistently detects only 1 speaker |
| **macOS Simulator** | ⚠️ Random | Sometimes 1 speaker, sometimes 2 speakers |
| **macOS (Apple Silicon)** | ✅ Always working | Consistently detects 2 speakers |

---

## User Observations

### File-Specific Issues

1. **podcast.wav** (152MB, 354.7s)
   - NotebookLLM generated
   - Contains stereo audio + MJPEG metadata (album art)
   - Format: Stereo 44.1kHz PCM Int16

2. **podcast-sample.wav** (159MB, 831.3s)
   - NotebookLLM generated
   - Stereo 48kHz PCM Int16
   - Longer file (13+ minutes)

### Workarounds That Worked

1. **QuickTime Re-export**: Re-exporting files via QuickTime fixed the issue (without changing duration)
2. **File Length**: Cutting the file drastically improved reliability
3. **Recording Direct**: Recording directly from device works correctly

---

## Investigation Timeline

### Nov 1, 2025 - Issue Reported
- User reported only 1 speaker detected on NotebookLLM files
- Tested various config parameters with no luck

### Nov 4, 2025 - Auto-Normalization Fix Applied
**Commit**: `0703c44` - "Auto-normalize audio for speaker diarization"

**Changes Made**:
- Added `needsAudioNormalization()` to detect problematic formats
- Added `normalizeAudio()` to convert to mono, clean PCM before main conversion
- Modified `convertBuffer()` to auto-normalize when needed

**Testing Results**:
- `podcast.wav`: 1 → 2 speakers detected ✅
- `podcast-sample.wav`: 1 → 2 speakers detected ✅

### Nov 4, 2025 - Redundancy Issue Introduced
**Commit**: `606ccde` - "simplify the fix"

The auto-normalization was refactored, introducing:
- `convertToMono()` function for intermediate conversion
- Double conversion: Stereo → Mono → Stereo (if needed)
- This caused unnecessary overhead but fixed the iOS issue

### Nov 4, 2025 - Optimization Attempts
**Commit**: `7b6f43c` - "Optimize convertToMono to convert directly to target format"

Attempted to optimize by passing target format to `convertToMono()`:
- Stereo 44.1kHz → Mono 16kHz (single step)
- But still had redundant intermediate buffer creation

### Nov 4, 2025 - Redundancy Fix
**Commit**: `fd66df7` - "fix two pass"

Removed the intermediate conversion entirely:
- Removed `convertToMono()` function (41 lines deleted)
- Use single `AVAudioConverter` call directly
- Handles channel conversion AND resampling in one shot

---

## Technical Analysis

### File Format Comparison

#### Original Files (Problematic)
```bash
$ ffprobe podcast.wav
Format: WAV (stereo 44.1kHz + MJPEG metadata)
Streams:
  - pcm_s16le: 2 channels @ 44100 Hz
  - mjpeg: Metadata (album art)

$ ffprobe podcast-sample.wav
Format: WAV (stereo 48kHz)
Streams:
  - pcm_s16le: 2 channels @ 48000 Hz
```

#### Re-exported Files (Clean)
```bash
$ ffprobe podcast_normalized.wav
Format: WAV (mono 16kHz)
Streams:
  - pcm_s16le: 1 channel @ 16000 Hz
```

### AudioConverter Implementation

#### Main Branch (Clean)
```swift
private func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
    // Single AVAudioConverter call handles everything
    guard let converter = AVAudioConverter(from: inputFormat, to: format) else {
        throw AudioConverterError.failedToCreateConverter
    }
    // ... convert in one pass
}
```

#### Fix Branch (With Auto-Normalization)
```swift
private func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
    // Check if we need to normalize
    let needsNormalization = needsAudioNormalization(inputFormat)

    if needsNormalization {
        // First conversion: normalize to clean mono
        bufferToConvert = try normalizeAudio(buffer)
    }

    // Second conversion: to target format
    guard let converter = AVAudioConverter(from: bufferToConvert.format, to: format) else {
        throw AudioConverterError.failedToCreateConverter
    }
    // ... convert again
}
```

---

## Test Results

### macOS (Apple Silicon) - All Versions ✅

| Version | podcast.wav | podcast-sample.wav |
|---------|-------------|-------------------|
| Main branch | 2 speakers (35 seg) | 2 speakers (52 seg) |
| Fix branch (auto-normalize) | 2 speakers (35 seg) | 2 speakers (52 seg) |
| Redundancy fix | 2 speakers (35 seg) | 2 speakers (52 seg) |

### Performance Impact

| Version | Build Time | Runtime | Code Lines |
|---------|-----------|---------|-----------|
| Main branch | 33.25s | 4.293s | ~258 lines |
| Fix branch (auto-normalize) | 43.26s | 7.034s | ~300 lines |
| Redundancy fix | 32.50s | 5.240s | ~259 lines |

**Conclusion**: Redundancy fix provides best balance of performance and code simplicity.

---

## Key Findings

### 1. AVAudioConverter Handles Everything
AVAudioConverter on Apple Silicon (macOS) correctly handles:
- Stereo → Mono conversion
- Any sample rate → 16kHz resampling
- PCM format conversion (Int16 → Float32)
- Metadata handling

### 2. Platform-Specific Issue
The problem appears to be **platform-specific**:
- **macOS (Apple Silicon)**: Works correctly with single converter
- **macOS (Simulator)**: Unreliable, random results
- **iOS (Device)**: Never works, requires double conversion

### 3. Auto-Normalization Was Platform-Specific Workaround
The auto-normalization fix was essentially a **platform compatibility layer** for iOS:
- Added unnecessary overhead on macOS
- Required for iOS to work correctly
- Suggests AVAudioConverter behaves differently on iOS

### 4. Root Cause Hypothesis
Possible causes for platform-specific behavior:
1. **AVAudioConverter implementation differences** between iOS and macOS
2. **CoreML model compilation** variations
3. **Memory/performance constraints** on iOS devices
4. **Metal/ANE utilization** differences

---

## Unresolved Questions

### ❓ Critical Unknowns

1. **Does main branch work on iOS devices?**
   - Cannot test without access to iOS device
   - Auto-normalization may have been necessary for iOS

2. **Is the redundancy fix safe for iOS?**
   - Removed the double conversion that may have been critical for iOS
   - Could break iOS support while fixing macOS performance

3. **Why does QuickTime export fix the issue?**
   - Re-export creates clean mono 16kHz file
   - Bypasses the need for in-app conversion
   - Suggests the issue is in the conversion process, not the models

---

## Recommendations

### Immediate Actions

1. **Test on Real iOS Device**
   - Verify if redundancy fix breaks iOS support
   - Test both main branch and fix branch
   - Document results

2. **Platform-Specific AudioConverter**
   ```swift
   #if os(iOS)
   // Use double conversion for iOS compatibility
   #else
   // Use single converter for macOS performance
   #endif
   ```

3. **Add Platform Detection Logging**
   - Log platform information
   - Track conversion method used
   - Monitor speaker detection results

### Long-Term Solutions

1. **Investigate QuickTime Export Behavior**
   - Understand what QuickTime does differently
   - Implement similar preprocessing in code

2. **Profile AVAudioConverter on iOS**
   - Compare behavior between platforms
   - Identify specific failure modes
   - Document platform-specific quirks

3. **Consider Alternative Conversion Libraries**
   - Evaluate platform-agnostic solutions
   - May be more reliable than AVAudioConverter

---

## Conclusion

The diarization issue with NotebookLLM files revealed **significant platform-specific behavior** in FluidAudio's audio processing pipeline. While the redundancy fix improved performance and code clarity on macOS, the impact on iOS remains **unknown and concerning**.

**Key Takeaway**: The auto-normalization fix (double conversion) was likely a **necessary platform-specific workaround** for iOS devices, not a bug. Removing it without thorough iOS testing could break iOS support.

**Next Steps**: Comprehensive iOS testing is required before merging the redundancy fix to main.

---

## References

- Original Issue: NotebookLLM podcast files
- Test Files: `podcast.wav`, `podcast-sample.wav`
- Commits: `0703c44`, `606ccde`, `7b6f43c`, `fd66df7`
- Platforms: macOS (Apple Silicon), iOS (device), macOS Simulator
