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

FluidAudio supports text-processing-rs through the `TextNormalizer` class. The native engine ships with the package as the `NemoTextProcessing` binary target and is linked directly — no setup required, it works out of the box for every SwiftPM consumer. Apps that don't use TTS or ITN can opt out of the ~18 MB engine with a package trait; see [Opting out](#opting-out-of-the-engine).

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

The engine is bundled: `Package.swift` declares a `NemoTextProcessing` binary target (a prebuilt xcframework from [text-processing-rs](https://github.com/FluidInference/text-processing-rs) releases) that SwiftPM downloads and links automatically. It is linked at build time, so `TextNormalizer.isNativeAvailable` is a compile-time constant: `true` whenever the engine is part of the build, `false` only when a consumer opts out (below). Releases ≤ 0.15.6 resolved the library at runtime and silently returned input unchanged when it was absent.

### Opting out of the engine

The engine is a prebuilt Rust static library (~18 MB per slice). ASR/VAD/diarization-only apps, and apps that ship their own Rust runtime (a second copy of the Rust std symbols fails to link), can leave it out with the `NemoTextProcessing` package trait. Requires Swift tools 6.1 / Xcode 16.3 or later; older toolchains read `Package.swift` and always link the engine.

```swift
// Package.swift of the consuming package / app
.package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.15.7", traits: [])
```

With the trait disabled:

- `TextNormalizer` and `NemoTextNormalizer` remain in the API. `isNativeAvailable`, `isTnAvailable`, and `NemoTextNormalizer.isAvailable` report `false`.
- Every normalization call returns its input unchanged; `version` is `nil`; custom rules are ignored (a warning is logged).
- TTS frontends run without NeMo normalization: Kokoro English falls back to the built-in `EnglishTextNormalizer` rules, and Kokoro Mandarin verbalizes numerals with `MandarinNumberNormalizer`. Keep the trait enabled for byte-exact NeMo readings.

To build the package itself without the engine: `swift build --disable-default-traits`.
