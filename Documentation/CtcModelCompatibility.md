# CTC Decoder Model Compatibility Guide

This guide explains which FluidAudio models work with the new CTC greedy/beam search decoders and ARPA language models.

## Quick Reference

| Model | Architecture | Works with CTC Decoder? | Works with ARPA LM? | Best Use Case |
|-------|--------------|-------------------------|---------------------|---------------|
| **Parakeet CTC 110M** | Pure CTC | ✅ YES | ✅ YES | Fast keyword spotting |
| **Parakeet CTC 0.6B** | Pure CTC | ✅ YES | ✅ YES | **ARPA LM beam search** |
| Parakeet TDT v2 | TDT | ❌ NO | ❌ NO | Best offline WER (2.1%) |
| Parakeet TDT v3 | TDT | ❌ NO | ❌ NO | Multilingual (2.5% WER) |
| Parakeet EOU 120M | RNN-T | ❌ NO | ❌ NO | Real-time streaming |
| Nemotron 0.6B | RNN-T | ❌ NO | ❌ NO | Real-time streaming |

---

## Compatible Models (CTC Architecture)

### ✅ Parakeet CTC 110M (Hybrid)

**HuggingFace:** [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml)

**Architecture:** Hybrid TDT+CTC (CTC head is auxiliary loss)

**Vocabulary:** 1024 tokens + blank at index 1024

**Usage:**
```swift
import FluidAudio

// Download and load
let ctcModels = try await CtcModels.downloadAndLoad(variant: .ctc110m)
let blankId = ctcModels.vocabulary.count  // 1024

// Load ARPA model
let lm = try ARPALanguageModel.load(from: arpaURL)

// Beam search with LM
let text = ctcBeamSearch(
    logProbs: logProbs,
    vocabulary: ctcModels.vocabulary,
    lm: lm,
    beamWidth: 100,
    lmWeight: 0.3,
    blankId: blankId
)
```

**Pros:**
- ✅ Smaller (110M parameters)
- ✅ Faster inference
- ✅ Good for keyword spotting

**Cons:**
- ⚠️ Hybrid model (CTC is auxiliary)
- ⚠️ Greedy decoding produces ~113% WER (unusable without LM)
- ⚠️ Needs beam search + LM for good results

**Best for:** Fast keyword spotting with custom vocabulary (earnings benchmark)

**CLI:**
```bash
swift run fluidaudiocli ctc-decode-benchmark \
    --audio speech.wav \
    --arpa medical.arpa \
    --ctc-variant 110m
```

---

### ✅ Parakeet CTC 0.6B (Pure CTC) - **RECOMMENDED**

**HuggingFace:** [FluidInference/parakeet-ctc-0.6b-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-coreml)

**Architecture:** Pure CTC (designed for CTC decoding)

**Vocabulary:** 1024 tokens + blank at index 1024

**Usage:**
```swift
import FluidAudio

// Download and load
let ctcModels = try await CtcModels.downloadAndLoad(variant: .ctc06b)
let blankId = ctcModels.vocabulary.count  // 1024

// Load ARPA model
let lm = try ARPALanguageModel.load(from: arpaURL)

// Beam search with LM
let text = ctcBeamSearch(
    logProbs: logProbs,
    vocabulary: ctcModels.vocabulary,
    lm: lm,
    beamWidth: 100,
    lmWeight: 0.3,
    blankId: blankId
)
```

**Pros:**
- ✅ Pure CTC architecture (designed for this)
- ✅ Better with beam search + LM
- ✅ Larger vocabulary coverage
- ✅ Best choice for ARPA LM usage

**Cons:**
- ⚠️ Larger (0.6B parameters)
- ⚠️ Slower inference than 110M
- ⚠️ CoreML conversion issue: greedy still produces ~158% WER

**Best for:** ARPA language model beam search decoding

**CLI:**
```bash
swift run fluidaudiocli ctc-decode-benchmark \
    --audio speech.wav \
    --arpa medical.arpa \
    --ctc-variant 06b  # Recommended
```

---

## Incompatible Models (Non-CTC Architecture)

### ❌ Parakeet TDT v2 (English-only)

**HuggingFace:** [FluidInference/parakeet-tdt-0.6b-v2-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml)

**Architecture:** TDT (Token-and-Duration Transducer) - RNN-T variant

**Why it doesn't work:**
```swift
// TDT has 4 separate models
struct AsrModels {
    let preprocessor: MLModel
    let encoder: MLModel
    let decoder: MLModel        // ← Has decoder (not CTC)
    let jointDecision: MLModel  // ← Joint network (not CTC)
}

// TDT decoder outputs token probabilities via joint network
// NOT direct CTC log-probabilities from encoder
```

**What to use instead:**
```swift
// Use AsrManager (TDT decoder)
let tdtModels = try await AsrModels.downloadAndLoad(version: .v2)
let asrManager = AsrManager(config: .default)
let result = try await asrManager.transcribe(audioURL)
// 2.1% WER on LibriSpeech - no LM needed!
```

**Performance:** 2.1% WER on LibriSpeech test-clean (better than CTC+LM)

---

### ❌ Parakeet TDT v3 (Multilingual)

**HuggingFace:** [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)

**Architecture:** TDT (Token-and-Duration Transducer)

**Why it doesn't work:** Same as v2 - uses TDT decoder, not CTC

**What to use instead:**
```swift
let tdtModels = try await AsrModels.downloadAndLoad(version: .v3)
let asrManager = AsrManager(config: .default)
let result = try await asrManager.transcribe(audioURL)
// 2.5% WER on LibriSpeech, 14.7% on FLEURS multilingual
```

**Performance:** 2.5% WER (English), supports 23 languages

---

### ❌ Parakeet EOU 120M (Streaming)

**HuggingFace:** [FluidInference/parakeet-realtime-eou-120m-coreml](https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml)

**Architecture:** RNN-T (Recurrent Neural Network Transducer) with EOU detection

**Why it doesn't work:**
```swift
// RNN-T has decoder LSTM with state
class RnntDecoder {
    private let decoderModel: MLModel   // ← Decoder LSTM
    private let jointModel: MLModel     // ← Joint network

    private var hState: MLMultiArray    // ← LSTM hidden state
    private var cState: MLMultiArray    // ← LSTM cell state

    private let blankId: Int32 = 1026   // ← Different blank ID
    private let eouId: Int32 = 1024     // ← Special EOU token
}

// Encoder output is [1, time, 640] (features, NOT vocab probabilities)
// Must go through decoder LSTM + joint network
// This is RNN-T decoding, not CTC
```

**What to use instead:**
```swift
// Use StreamingEouAsrManager (RNN-T decoder)
let eouManager = StreamingEouAsrManager()
try await eouManager.initialize(chunkSize: .ms320)

// Process streaming audio
let result = try await eouManager.processAudioChunk(samples)
// Has EOU detection for real-time use
```

**Performance:** 4.87% WER (320ms chunks), real-time streaming

---

### ❌ Nemotron Speech Streaming 0.6B

**HuggingFace:** [FluidInference/nemotron-speech-streaming-en-0.6b-coreml](https://huggingface.co/FluidInference/nemotron-speech-streaming-en-0.6b-coreml)

**Architecture:** RNN-T streaming (NVIDIA)

**Why it doesn't work:**
```swift
// Nemotron uses RNN-T with cache
extension NemotronStreamingAsrManager {
    func processChunk(_ samples: [Float]) async throws {
        // 1. Preprocessor: audio → mel
        let preprocOutput = try await preprocessor.prediction(...)

        // 2. Encoder with cache (NOT CTC output)
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": ...,
            "cache_channel": ...,  // ← Streaming cache
            "cache_time": ...,
            "cache_len": ...
        ])

        // 3. Decoder LSTM + Joint (RNN-T)
        // NOT CTC decoding
    }
}
```

**What to use instead:**
```swift
// Use NemotronStreamingAsrManager (RNN-T decoder)
let nemotron = NemotronStreamingAsrManager()
try await nemotron.initialize(chunkSize: .ms1120)

// Process streaming audio
let text = try await nemotron.processChunk(samples)
```

**Performance:** Fast streaming with vDSP optimization

---

## Architecture Comparison

### CTC (Connectionist Temporal Classification)
```
Audio → Encoder → CTC log-probs [Time, Vocab]
                       ↓
              ctcGreedyDecode / ctcBeamSearch + ARPA LM
                       ↓
                  Final text
```

**Characteristics:**
- ✅ Simple: encoder → decoder (no RNN state)
- ✅ Parallelizable: can process all timesteps at once
- ✅ Works with external LM: ARPA models plug right in
- ❌ Independence assumption: each frame decoded independently
- ❌ Lower accuracy than RNN-T/TDT without LM

**When to use:**
- Need to apply domain-specific ARPA language model
- Want fast keyword spotting
- Have text corpus for LM training

---

### RNN-T (Recurrent Neural Network Transducer)
```
Audio → Encoder → Encoder frames [Time, Features]
                       ↓
              Decoder LSTM (with state) → Decoder output
                       ↓
              Joint Network (encoder + decoder) → Token probs
                       ↓
              Greedy decoding (built-in)
                       ↓
                  Final text
```

**Characteristics:**
- ✅ Better accuracy: models token dependencies
- ✅ Streaming-friendly: processes incrementally with state
- ✅ No external LM needed: decoder acts as implicit LM
- ❌ Slower: sequential decoding (can't parallelize)
- ❌ Can't use ARPA LM: incompatible architecture

**When to use:**
- Need real-time streaming ASR
- Want good accuracy without external LM
- Don't have domain-specific text corpus

---

### TDT (Token-and-Duration Transducer)
```
Audio → Preprocessor → Mel
           ↓
       Encoder → Encoder frames
           ↓
       Decoder → Decoder output
           ↓
       Joint Decision → Token + Duration probs
           ↓
       Final text (best WER)
```

**Characteristics:**
- ✅ Best accuracy: 2.1-2.5% WER on LibriSpeech
- ✅ Duration modeling: knows how long tokens last
- ✅ No LM needed: achieves great WER out of box
- ❌ Offline only: needs full audio context
- ❌ Can't use ARPA LM: incompatible architecture

**When to use:**
- Need absolute best WER for offline transcription
- Don't need streaming
- Don't have domain-specific LM

---

## Hybrid Pipeline: TDT + CTC (Earnings Benchmark)

You can combine TDT and CTC for best of both worlds:

```swift
// 1. Use TDT for base transcription (low WER)
let tdtModels = try await AsrModels.downloadAndLoad(version: .v2)
let asrManager = AsrManager(config: .default)
let tdtResult = try await asrManager.transcribe(audioURL)
// → "patient has die beetus" (15% WER but misses entities)

// 2. Use CTC for keyword spotting (high recall)
let ctcModels = try await CtcModels.downloadAndLoad(variant: .ctc110m)
let spotter = CtcKeywordSpotter(models: ctcModels, blankId: 1024)

let customVocab = CustomVocabularyContext(terms: [
    CustomVocabularyTerm(text: "Nvidia", weight: 1.0),
    CustomVocabularyTerm(text: "Tesla", weight: 1.0),
    CustomVocabularyTerm(text: "Amazon", weight: 1.0)
])

let spotResult = try await spotter.spotKeywordsWithLogProbs(
    audioSamples: samples,
    customVocabulary: customVocab
)
// → Finds "Nvidia" at 12.3s with high confidence

// 3. Combine with VocabularyRescorer
let rescorer = try await VocabularyRescorer.create(
    spotter: spotter,
    vocabulary: customVocab
)

let finalText = rescorer.ctcTokenRescore(
    transcript: tdtResult.text,
    tokenTimings: tdtResult.tokenTimings,
    logProbs: spotResult.logProbs
)
// → "patient has Nvidia Tesla Amazon" ✅
```

**Results:**
- **WER:** 15.0% (from TDT base transcription)
- **Entity Recall:** 99.3% (from CTC keyword spotting)
- **F-score:** 91.7%

**Use case:** Earnings calls, medical transcription with entity names

---

## How to Choose

### Use **Parakeet CTC 0.6B** if:
- ✅ You have domain-specific text corpus (medical, legal, financial)
- ✅ You can train an ARPA language model
- ✅ You want to improve WER with domain knowledge
- ✅ You need keyword detection

**Example:**
```bash
# Create ARPA model from medical transcripts
lmplz -o 2 < medical_corpus.txt > medical.arpa

# Use with CTC decoder
swift run fluidaudiocli ctc-decode-benchmark \
    --audio patient_recording.wav \
    --arpa medical.arpa \
    --ctc-variant 06b
```

---

### Use **Parakeet TDT v2/v3** if:
- ✅ You need best WER (2.1-2.5%)
- ✅ Offline transcription is OK
- ✅ You don't have domain-specific LM
- ✅ You need multilingual support (v3)

**Example:**
```swift
let tdtModels = try await AsrModels.downloadAndLoad(version: .v3)
let asrManager = AsrManager(config: .default)
let result = try await asrManager.transcribe(audioURL)
// 2.5% WER out of the box
```

---

### Use **Parakeet EOU or Nemotron** if:
- ✅ You need real-time streaming
- ✅ You need EOU (end-of-utterance) detection
- ✅ Latency matters (160-1280ms)
- ✅ You don't have domain-specific LM

**Example:**
```swift
let eouManager = StreamingEouAsrManager()
try await eouManager.initialize(chunkSize: .ms320)

// Real-time streaming
for chunk in audioChunks {
    let result = try await eouManager.processAudioChunk(chunk)
    if result.eouDetected {
        print("Utterance complete: \(result.text)")
    }
}
```

---

### Use **TDT + CTC Hybrid** if:
- ✅ You need both low WER and high entity recall
- ✅ You have specific entity list (company names, drugs, etc.)
- ✅ Offline processing is OK
- ✅ You can run both models

**Example:** Earnings benchmark (15% WER, 99.3% entity recall)

---

## FAQ

### Q: Can I use ARPA LM with TDT models?
**A:** No. TDT uses a different decoder architecture (joint network) that's incompatible with external language models.

### Q: Can I use ARPA LM with streaming models (EOU, Nemotron)?
**A:** No. They use RNN-T architecture with decoder LSTM, which is incompatible with CTC-style external LMs.

### Q: Why is greedy CTC decoding broken for both CTC models?
**A:**
- **110M:** It's a hybrid TDT+CTC model where CTC is just auxiliary loss, not primary
- **0.6B:** CoreML conversion issue - PyTorch greedy works (~14% WER), CoreML doesn't (~158% WER)
- **Solution:** Use beam search + ARPA LM (what we added!)

### Q: Which CTC model is better for ARPA LM?
**A:** Parakeet CTC 0.6B - it's pure CTC and designed for this use case.

### Q: Can I train my own ARPA model?
**A:** Yes! Use KenLM:
```bash
# Install KenLM
brew install kenlm

# Train bigram model
lmplz -o 2 < your_corpus.txt > your_model.arpa

# Use with FluidAudio
swift run fluidaudiocli ctc-decode-benchmark \
    --audio audio.wav \
    --arpa your_model.arpa
```

### Q: What about other CTC models (Wav2Vec2, DeepSpeech)?
**A:** Our `ctcGreedyDecode` and `ctcBeamSearch` are **completely generic**! They work with any CTC model that outputs log-probabilities. Just need:
- CTC log-probs: `[[Float]]` shape `[Time, Vocab]`
- Vocabulary mapping: `[Int: String]`
- Blank ID: usually vocab size or 0

---

## Summary Table

| Need | Use This | Notes |
|------|----------|-------|
| ARPA language model support | **Parakeet CTC 0.6B** | Only CTC models work |
| Best WER (offline) | Parakeet TDT v3 | 2.5% WER, no LM needed |
| Real-time streaming | Parakeet EOU / Nemotron | RNN-T with EOU |
| Keyword spotting | Parakeet CTC 110M | Fast, hybrid model |
| Entity recognition + low WER | TDT + CTC hybrid | Earnings benchmark |
| Multilingual | Parakeet TDT v3 | 23 languages |

---

## See Also

- [CtcDecoderExample.md](CtcDecoderExample.md) - Full usage examples with ARPA LM
- [Benchmarks.md](Benchmarks.md) - Performance metrics for all models
- [Models.md](Models.md) - Complete model catalog
