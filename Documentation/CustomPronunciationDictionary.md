# Custom Pronunciation Dictionary

FluidAudio TTS supports custom pronunciation dictionaries (lexicons) that allow you to override how specific words are pronounced. This is essential for domain-specific terminology, brand names, acronyms, and proper nouns that the default text-to-speech system may not handle correctly.

## Overview

Custom lexicons take **highest priority** in the pronunciation resolution pipeline, ensuring your specified pronunciations are always used when a word matches.

### Priority Order (highest to lowest)

1. **Per-word phonetic overrides** — Inline markup like `[word](/phonemes/)`
2. **Custom lexicon** — Your `word=phonemes` file entries
3. **Case-sensitive built-in lexicon** — Handles abbreviations like `F.B.I`
4. **Standard built-in lexicon** — General English pronunciations
5. **Grapheme-to-phoneme (G2P)** — eSpeak-NG fallback for unknown words

## File Format

Custom lexicon files use a simple line-based format:

```
# This is a comment
word=phonemes
```

### Rules

| Element | Description |
|---------|-------------|
| `#` | Lines starting with `#` are comments |
| `=` | Separator between word and phonemes |
| Phonemes | Compact IPA string (no spaces between phoneme characters) |
| Whitespace in phonemes | Creates word boundaries for multi-word expansions |
| Empty lines | Ignored |

### Phoneme Notation

Phonemes are written as a compact IPA string where each Unicode character (grapheme cluster) becomes one token:

```
kokoro=kəkˈɔɹO
```

This produces tokens: `["k", "ə", "k", "ˈ", "ɔ", "ɹ", "O"]`

For multi-word expansions, use whitespace to separate words:

```
# United Nations
UN=junˈaɪtᵻd nˈeɪʃənz
```

This produces: `["j", "u", "n", "ˈ", "a", "ɪ", "t", "ᵻ", "d", " ", "n", "ˈ", "e", "ɪ", "ʃ", "ə", "n", "z"]`

## Word Matching

The lexicon uses a three-tier matching strategy:

1. **Exact match** — `NASDAQ` matches only `NASDAQ`
2. **Case-insensitive** — `nasdaq` matches `NASDAQ`, `Nasdaq`, `nasdaq`
3. **Normalized** — Strips to letters/digits/apostrophes, lowercased

This allows you to:
- Define case-specific pronunciations when needed
- Use lowercase keys for general entries that match any case variant

```
# Case-specific: only matches uppercase
NASDAQ=nˈæzdæk

# General: matches any case variant of "ketorolac"
ketorolac=kˈɛtɔːɹˌɒlak
```

## Pipeline Integration

### Where Custom Lexicon is Applied

The custom lexicon is consulted during the **chunking phase** in `KokoroChunker.buildChunks()`:

```
Input Text
    │
    ▼
┌─────────────────────┐
│  Text Preprocessing │  ← Inline overrides extracted
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Sentence Splitting │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Word Tokenization  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Phoneme Resolution │  ← Custom lexicon checked HERE
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Chunk Assembly     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Model Inference    │
└─────────────────────┘
    │
    ▼
Audio Output
```

### Resolution Logic

For each word, the chunker:

1. Checks for inline phonetic override (from preprocessing)
2. Looks up the **original word** in custom lexicon (preserves case)
3. Falls back to built-in lexicons and G2P if not found

The custom lexicon's `phonemes(for:)` method handles matching:

```swift
// Exact match first
if let exact = entries[word] { return exact }

// Case-insensitive fallback
if let folded = lowercaseEntries[word.lowercased()] { return folded }

// Normalized fallback (letters/digits/apostrophes only)
let normalized = normalizeForLookup(word)
return normalizedEntries[normalized]
```

## Usage

### CLI

```bash
swift run fluidaudio tts "The NASDAQ index rose today" --lexicon custom.txt --output output.wav
```

### Swift API

```swift
// Load from file
let lexicon = try TtsCustomLexicon.load(from: fileURL)

// Or parse from string
let lexicon = try TtsCustomLexicon.parse("""
    kokoro=kəkˈɔɹO
    xiaomi=ʃaʊˈmiː
""")

// Or create programmatically
let lexicon = TtsCustomLexicon(entries: [
    "kokoro": ["k", "ə", "k", "ˈ", "ɔ", "ɹ", "O"]
])

// Use with TtSManager
let manager = TtSManager(customLexicon: lexicon)
try await manager.initialize()
let audio = try await manager.synthesize(text: "Welcome to Kokoro TTS")

// Or update at runtime
manager.setCustomLexicon(newLexicon)
```

### Merging Lexicons

```swift
let baseLexicon = try TtsCustomLexicon.load(from: baseURL)
let domainLexicon = try TtsCustomLexicon.load(from: domainURL)

// Domain entries override base entries on conflict
let combined = baseLexicon.merged(with: domainLexicon)
```

## Example Lexicon File

Below is a comprehensive example covering multiple domains:

```
# ============================================
# Custom Pronunciation Dictionary
# FluidAudio TTS
# ============================================

# --------------------------------------------
# FINANCE & TRADING
# --------------------------------------------

# Stock exchanges and indices
NASDAQ=nˈæzdæk
Nikkei=nˈɪkA

# Financial terms
EBITDA=iːbˈɪtdɑː
SOFR=sˈoʊfɚ

# Cryptocurrencies
Bitcoin=bˈɪtkɔɪn
DeFi=diːfˈaɪ

# --------------------------------------------
# HEALTHCARE & PHARMACEUTICALS
# --------------------------------------------

# Common medications
acetaminophen=əˌsiːtəmˈɪnəfɛn
omeprazole=ˈOmpɹəzˌOl

# Medical terms
HIPAA=hˈɪpɑː
COPD=kˈɑpt

# Conditions
fibromyalgia=fˌIbɹOmIˈælʤiə
arrhythmia=əɹˈɪðmiə

# --------------------------------------------
# TECHNOLOGY COMPANIES & BRANDS
# --------------------------------------------

# Tech giants
Xiaomi=zˌIəˈOmi
NVIDIA=ɛnvˈɪdiə

# Software & services
Kubernetes=kuːbɚnˈɛtiːz
kubectl=kjˈubɛktᵊl

# --------------------------------------------
# PRODUCT NAMES
# --------------------------------------------

Kokoro=kəkˈɔɹO
FluidAudio=flˈuːɪd ˈɔːdioʊ
```

## Troubleshooting

### Invalid Phonemes Warning

If you see warnings like:

```
Custom lexicon entry for 'word' has no tokens in Kokoro vocabulary
```

Your phonemes contain characters not in the Kokoro vocabulary. Common issues:

- Using X-SAMPA instead of IPA
- Extra spaces between phoneme characters
- Unicode normalization differences

### Word Not Being Matched

Check the matching rules:

1. Is there a typo in the word key?
2. Is case sensitivity affecting the match?
3. Does the word contain punctuation that's being stripped?

Use logging to debug:

```swift
if let phonemes = lexicon.phonemes(for: "problematic_word") {
    print("Found: \(phonemes)")
} else {
    print("Not found in lexicon")
}
```

### Finding Valid Phonemes

The Kokoro vocabulary uses a specific phoneme set. To find valid phonemes:

1. Look at existing entries in the built-in lexicon
2. Use eSpeak-NG's IPA output as a reference
3. Test with short phrases to verify pronunciation

### Phoneme Generator
Here's an example of Python code for a phoneme generator based on [this](
https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md#%EF%B8%8F-from-espeak-to-misaki)
```python
#!/usr/bin/env python3
"""
Phoneme Generator - Converts words to phonemes using espeak and misaki,
then generates audio using Kokoro TTS.

Misaki uses 49 phonemes for English:
- 41 shared between American and British English
- 4 American-only: æ (ash), O (oh), ᵻ (schwa-ish), ɾ (t/d flap)
- 4 British-only: a (ash), Q (oh), ɒ (on), ː (extender)

Key espeak -> misaki conversions:
- e -> A (diphthong)
- r -> ɹ (r-sound)
- aɪ -> I (eye sound)
- aʊ -> W (ow sound)
- dʒ -> ʤ (j/dg cluster)
- tʃ -> ʧ (ch cluster)
- ɔɪ -> Y (oy sound)
- oʊ -> O (American oh)
- əʊ -> Q (British oh)
"""

import sys
import argparse
import scipy.io.wavfile as wavfile
import numpy as np
from misaki import en, espeak
from kokoro import KPipeline


def create_g2p(british: bool = False):
    """Create a grapheme-to-phoneme converter."""
    fallback = espeak.EspeakFallback(british=british)
    return en.G2P(trf=False, british=british, fallback=fallback)


def get_espeak_phonemes(word: str, british: bool = False) -> str:
    """Get raw espeak phonemes for a word."""
    fb = espeak.EspeakFallback(british=british)
    return fb(word)


def generate_audio(pipeline: KPipeline, text: str, voice: str = 'af_heart') -> np.ndarray:
    """Generate audio for given text using Kokoro TTS."""
    audio_chunks = []
    for gs, ps, audio in pipeline(text, voice=voice):
        audio_chunks.append(audio)

    if audio_chunks:
        return np.concatenate(audio_chunks)
    return np.array([])


def process_word(word: str, g2p, pipeline: KPipeline, voice: str,
                 save_audio: bool = True, output_dir: str = ".") -> dict:
    """Process a single word: get phonemes and optionally generate audio."""
    # Get misaki phonemes (converted from espeak)
    misaki_phonemes, _ = g2p(word)

    result = {
        'word': word,
        'misaki_phonemes': misaki_phonemes,
    }

    if save_audio:
        audio = generate_audio(pipeline, word, voice)
        if len(audio) > 0:
            filename = f"{output_dir}/{word}.wav"
            wavfile.write(filename, 24000, audio)
            result['audio_file'] = filename

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Convert words to phonemes and generate audio using Kokoro TTS'
    )
    parser.add_argument(
        'words',
        nargs='*',
        help='Words to process (if none provided, uses default list)'
    )
    parser.add_argument(
        '--british', '-b',
        action='store_true',
        help='Use British English phonemes'
    )
    parser.add_argument(
        '--voice', '-v',
        default='af_heart',
        help='Voice to use for TTS (default: af_heart)'
    )
    parser.add_argument(
        '--no-audio', '-n',
        action='store_true',
        help='Skip audio generation, only show phonemes'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Output directory for audio files (default: current directory)'
    )

    args = parser.parse_args()

    # Default words if none provided
    if not args.words:
        args.words = [
            "Kokoro", "Xiaomi", "Porsche"
        ]

    # Initialize
    dialect = "British" if args.british else "American"
    lang_code = 'b' if args.british else 'a'

    print(f"Phoneme Generator ({dialect} English)")
    print("=" * 50)

    g2p = create_g2p(british=args.british)

    pipeline = None
    if not args.no_audio:
        pipeline = KPipeline(lang_code=lang_code)

    # Process each word
    for word in args.words:
        result = process_word(
            word,
            g2p,
            pipeline,
            args.voice,
            save_audio=not args.no_audio,
            output_dir=args.output_dir
        )

        print(f"\n{result['word']}")
        print(f"  Phonemes: {result['misaki_phonemes']}")

        if 'audio_file' in result:
            print(f"  Audio: {result['audio_file']}")

    print("\n" + "=" * 50)
    print(f"Processed {len(args.words)} word(s)")


if __name__ == '__main__':
    main()
```

## Best Practices

1. **Use lowercase keys** for general entries that should match any case
2. **Add case-specific entries** only when pronunciations differ by case
3. **Comment your entries** to document pronunciation sources
4. **Group by domain** for maintainability
5. **Test incrementally** — add a few entries at a time and verify
6a **Keep backups** of working lexicon files before major changes
