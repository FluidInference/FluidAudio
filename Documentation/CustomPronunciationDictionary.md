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

| Element                | Description                                               |
|------------------------|-----------------------------------------------------------|
| `#`                    | Lines starting with `#` are comments                      |
| `=`                    | Separator between word and phonemes                       |
| Phonemes               | Compact IPA string (no spaces between phoneme characters) |
| Whitespace in phonemes | Creates word boundaries for multi-word expansions         |
| Empty lines            | Ignored                                                   |

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

## Best Practices

1. **Use lowercase keys** for general entries that should match any case
2. **Add case-specific entries** only when pronunciations differ by case
3. **Comment your entries** to document pronunciation sources
4. **Group by domain** for maintainability
5. **Test incrementally** — add a few entries at a time and verify
6a **Keep backups** of working lexicon files before major changes
