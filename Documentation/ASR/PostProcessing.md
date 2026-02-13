# Post-Processing ASR Output

## Inverse Text Normalization (ITN)

Inverse Text Normalization converts spoken-form ASR output to written form:

| Input (spoken) | Output (written) |
|----------------|------------------|
| "two hundred" | "200" |
| "five dollars and fifty cents" | "$5.50" |
| "january fifth twenty twenty five" | "January 5, 2025" |
| "two thirty pm" | "2:30 p.m." |
| "test at gmail dot com" | "test@gmail.com" |

## Post-Processing Tools

| Tool | Description | Language |
|------|-------------|----------|
| **[text-processing-rs](https://github.com/FluidInference/text-processing-rs)** | Inverse Text Normalization (ITN) - converts spoken-form ASR output to written form. Rust port of [NVIDIA NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing) with Swift wrapper. | Rust, Swift |

## Using ITN with FluidAudio

FluidAudio includes optional support for text-processing-rs through the `TextNormalizer` class. The library uses dynamic loading, so it's completely optional - if not linked, `normalize()` returns the input unchanged.

### Basic Usage

```swift
import FluidAudio

let normalizer = TextNormalizer.shared

// Check if native library is available
if normalizer.isNativeAvailable {
    print("ITN version: \(normalizer.version ?? "unknown")")
}

// Normalize spoken-form text
let result = normalizer.normalize("two hundred dollars")
// Returns "$200" (with native library) or "two hundred dollars" (without)
```

### With ASR Results

```swift
// Transcribe audio
let asrResult = try await asrManager.transcribe(samples, source: .system)

// Normalize the result
let normalizedResult = normalizer.normalize(result: asrResult)
print(normalizedResult.text)  // Written form
```

### Linking the Native Library

To enable ITN support, link your app against `libnemo_text_processing`:

1. Build text-processing-rs for your target platform
2. Add the library to your Xcode project's linker settings
3. `TextNormalizer.isNativeAvailable` will return `true`

See the [text-processing-rs README](https://github.com/FluidInference/text-processing-rs) for build instructions.
