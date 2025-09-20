# CEspeakNG Integration Guide (eSpeak NG in Swift)

This document explains how we integrate the eSpeak NG C library into Swift, how to build it with SwiftPM, how the TTS pipeline uses it for G2P (grapheme-to-phoneme) fallback, and known limitations/troubleshooting.

## Summary

- There is no officially published Swift package for eSpeak NG.
- It is fully supported in Swift via a SwiftPM System Library target (aka a Clang module wrapper) that exposes the C headers as a Swift-importable module `CEspeakNG`.
- We ship a minimal wrapper in this repo that makes `import CEspeakNG` possible and provide a small Swift helper (`EspeakG2P`) to call the C API for IPA phonemization.
- The TTS pipeline uses the dictionary first; for out-of-vocabulary (OOV) words, it calls eSpeak NG to get IPA and maps IPA → Kokoro tokens.

## What’s included in this repo

- SwiftPM System Library target:
  - `Package.swift` contains:
    ```swift
    .systemLibrary(
      name: "CEspeakNG",
      pkgConfig: "espeak-ng",
      providers: [ .brew(["espeak-ng"]) ]
    )
    ```
  - Module map and shim header:
    - `Sources/CEspeakNG/module.modulemap`
    - `Sources/CEspeakNG/shim.h` (includes `<espeak-ng/speak_lib.h>` and falls back to `<espeak/speak_lib.h>`)
- Swift usage:
  - `Sources/FluidAudio/TextToSpeech/Kokoro/EspeakG2P.swift`: a tiny actor that initializes eSpeak NG and uses `espeak_TextToPhonemes()` to produce IPA.
  - `Sources/FluidAudio/TextToSpeech/Kokoro/PhonemeMapper.swift`: maps IPA tokens → Kokoro token inventory.
  - `Sources/FluidAudio/TextToSpeech/Kokoro/KokoroChunker.swift`: dictionary-first; OOV → eSpeak NG → IPA map; preserves punctuation tokens.

## Install prerequisites (macOS/Homebrew)

```bash
brew update
brew install espeak-ng pkg-config
``

Apple Silicon notes:
- Homebrew prefix is typically `/opt/homebrew`.
- If SwiftPM doesn’t pick up pkg-config, build with explicit include/lib paths:
  ```bash
  swift build -Xcc -I/opt/homebrew/include -Xlinker -L/opt/homebrew/lib
  ```

Verify headers/libs:
```bash
pkg-config --cflags --libs espeak-ng
ls /opt/homebrew/include/espeak-ng/speak_lib.h
```

## Build and run

```bash
swift build
swift run fluidaudio tts "Your text here" --auto-download --output out.wav
```

During TTS, OOV words print a G2P line by default, e.g.

```
[G2P] word=microchunking | ipa=m aɪ k ɹ oʊ t͡ʃ ʌ ŋ k ɪ ŋ | map=m aɪ k r o ʧ ʌ ŋ k ɪ ŋ
```

We also provide a simple test command:

```bash
swift run fluidaudio g2p "chunking microchunking résumé"
```

## Swift API shape

Using the C module directly:

```swift
import CEspeakNG

// Initialize
espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nil, 0)
"en-us".withCString { espeak_SetVoiceByName($0) }

// Phonemize → IPA string
var ptr: UnsafeRawPointer? = UnsafeRawPointer(("chunking" as NSString).utf8String)
let ipa = espeak_TextToPhonemes(&ptr, Int32(espeakCHARS_AUTO), Int32(espeakPHONEMES_IPA))
if let ipa = ipa { print(String(cString: ipa)) }
```

Convenience actor (in repo):

```swift
let ipaTokens = EspeakG2P.shared.phonemize(word: "chunking")
```

## Integration in TTS

1) Normalize/tokenize text and preserve punctuation tokens present in Kokoro’s vocab.
2) Dictionary lookup: `word_phonemes.json` → Kokoro tokens.
3) OOV fallback: eSpeak NG → IPA tokens → `PhonemeMapper.mapIPA(...)` → Kokoro tokens.
4) Chunk under token budget; synthesize with Kokoro CoreML.

Notes:
- There is no letter-by-letter fallback; if eSpeak NG fails for a word or the mapping yields nothing, the word is skipped (with a log line).
- Punctuation tokens are preserved when present in the model’s vocabulary.

## Is “Swift support” possible?

Yes. While there’s no official Swift package published by the eSpeak NG project, Swift supports C libraries via System Library targets.

- Pros:
  - Zero-copy import of the C API; no bridging overhead beyond normal FFI.
  - Works across macOS/iOS (subject to platform availability of the lib; iOS would require bundling a static lib and app store review considerations).
  - No need to shell out to `espeak-ng` CLI.
- Cons / Caveats:
  - eSpeak NG is GPL-licensed. Ensure your distribution and usage comply with GPL requirements.
  - iOS/macOS app store distribution with GPL-linked code is generally incompatible.
  - For CLI/tools or internal usage on macOS, this is fine.
  - You must ensure headers/libs are available at build time (via Homebrew/pkg-config in development).

## Limitations & Considerations

- Licensing: eSpeak NG is GPL; using it in a distributed product may impose GPL obligations.
- Multilingual: You can switch `espeak_SetVoiceByName()` to other languages (`en-gb`, `fr`, etc.), but our IPA→Kokoro mapping is focused on English. Non-English support will need mapping rules and model coverage.
- Quality: eSpeak NG phonemization is rule-based; accuracy may vary for proper nouns and domain-specific words. Consider caching successful results to a sidecar JSON.
- Performance: `espeak_TextToPhonemes()` is fast per word; the current actor uses a serial queue for thread safety. If you need high-throughput batch G2P, consider batching at the text level or managing your own synchronization.
- Mapping: `PhonemeMapper` approximates IPA → Kokoro’s inventory. Review and extend it for your model’s exact tokens and stress/diphthong handling.

## Troubleshooting

Header not found (`'espeak-ng/speak_lib.h' file not found`):
- Ensure Homebrew is set up and eSpeak NG is installed:
  - `brew install espeak-ng pkg-config`
- Verify headers/libs:
  - `pkg-config --cflags --libs espeak-ng`
  - `ls /opt/homebrew/include/espeak-ng/speak_lib.h`
- Build with explicit include/lib flags if needed:
  - `swift build -Xcc -I/opt/homebrew/include -Xlinker -L/opt/homebrew/lib`

SwiftPM cannot write caches (CI/sandbox):
- Set a writable storage path:
  ```bash
  export SWIFTPM_STORAGE_PATH="$PWD/.swiftpm-storage"
  mkdir -p "$SWIFTPM_STORAGE_PATH"
  swift build
  ```

No IPA output or empty mapping:
- Print the raw IPA string and review tokens.
- Extend `PhonemeMapper` to cover missing symbols; validate against your model’s vocabulary.

## Alternatives

- Shell out to the CLI: `espeak-ng -q --ipa -v en-us "word"` (what we used initially as a fallback). Simple but adds process overhead.
- Other G2P engines:
  - `phonemizer` (Python) with `espeak-ng` backend.
  - `g2p-en` (CMU-based) for ARPAbet.
  - Cloud services (latency/cost/privacy tradeoffs).

## Roadmap / Options

- Optional flag to force G2P-only mode (ignore dictionary) for testing.
- Configurable language (`--voice-lang en-us|en-gb|fr|...`) for G2P.
- Cache OOV G2P results into a user dictionary JSON to avoid recomputation.
- Expand IPA→Kokoro mapping for better coverage (stress, length, diphthongs).

