# Text Processing

## Overview

**[text-processing-rs](https://github.com/FluidInference/text-processing-rs)** provides both Inverse Text Normalization (ITN) and Text Normalization (TN) across 7 languages (EN, DE, ES, FR, HI, JA, ZH). 100% NeMo test compatibility (3,011 tests). Rust port of [NVIDIA NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing) with Swift wrapper.

## Inverse Text Normalization (ITN)

ITN converts spoken-form ASR output to written form — useful for post-processing ASR transcriptions:

| Input (spoken) | Output (written) |
|----------------|------------------|
| "two hundred" | "200" |
| "five dollars and fifty cents" | "$5.50" |
| "january fifth twenty twenty five" | "January 5, 2025" |
| "two thirty pm" | "2:30 p.m." |
| "test at gmail dot com" | "test@gmail.com" |

## Text Normalization (TN)

TN converts written-form text to spoken form — useful for TTS preprocessing:

| Input (written) | Output (spoken) |
|-----------------|-----------------|
| "123" | "one hundred twenty three" |
| "$5.50" | "five dollars and fifty cents" |
| "January 5, 2025" | "january fifth twenty twenty five" |
| "2:30 PM" | "two thirty p m" |
| "1st" | "first" |

## Using with FluidAudio

FluidAudio supports text-processing-rs through the `TextNormalizer` class. The native engine ships with the package as the `NemoTextProcessing` binary target and is linked directly — no setup required, it works out of the box for every SwiftPM consumer.

### ITN (Spoken to Written)

```swift
import FluidAudio

let normalizer = TextNormalizer.shared

print("ITN version: \(normalizer.version ?? "unknown")")

// Normalize spoken-form text
let result = normalizer.normalize("two hundred dollars")
// Returns "$200"
```

### TN (Written to Spoken)

```swift
// Convert written text to spoken form for TTS
let spoken = normalizer.tnNormalize("$5.50")
// Returns "five dollars and fifty cents"

let spoken = normalizer.tnNormalize("January 5, 2025")
// Returns "january fifth twenty twenty five"
```

### With ASR Results

```swift
// Transcribe audio
let asrResult = try await asrManager.transcribe(samples, source: .system)

// Normalize the result (ITN: spoken → written)
let normalizedResult = normalizer.normalize(result: asrResult)
print(normalizedResult.text)  // Written form
```

### Native Library

The engine is bundled: `Package.swift` declares a `NemoTextProcessing` binary target (a prebuilt xcframework from [text-processing-rs](https://github.com/FluidInference/text-processing-rs) releases) that SwiftPM downloads and links automatically. `TextNormalizer.isNativeAvailable` always returns `true`; it is kept only for source compatibility with releases ≤ 0.15.6, which resolved the library at runtime and silently returned input unchanged when it was absent.
