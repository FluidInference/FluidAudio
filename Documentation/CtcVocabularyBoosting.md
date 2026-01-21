# CTC Vocabulary Boosting Pipeline

This document describes FluidAudio's CTC-based custom vocabulary boosting system, which enables accurate recognition of domain-specific terms (company names, technical jargon, proper nouns) without retraining the ASR model.

## Research Foundation

This implementation is based on the NVIDIA NeMo paper:

> **"CTC-based Word Spotter"**
> arXiv:2406.07096
> https://arxiv.org/abs/2406.07096

The paper introduces a dynamic programming algorithm for CTC-based keyword spotting that:
- Scores vocabulary terms against CTC log-probabilities
- Enables "shallow fusion" rescoring without beam search
- Provides acoustic evidence for vocabulary term matching

## Architecture Overview

```
                  ┌─────────────────────────────────────────┐
                  │            Audio Input                  │
                  │           (16kHz, mono)                 │
                  └─────────────────┬───────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────────┐
              │                                               │
              ▼                                               ▼
    ┌─────────────────┐                             ┌─────────────────┐
    │   TDT Encoder   │                             │   CTC Encoder   │
    │  (Parakeet 0.6B)│                             │ (Parakeet 110M) │
    └────────┬────────┘                             └────────┬────────┘
             │                                               │
             ▼                                               ▼
    ┌─────────────────┐                             ┌─────────────────┐
    │   TDT Decoder   │                             │  CTC Log-Probs  │
    │    (Greedy)     │                             │   [T, V=1024]   │
    └────────┬────────┘                             └────────┬────────┘
             │                                               │
             ▼                                               ▼
    ┌─────────────────┐             Custom          ┌─────────────────┐
    │   Raw Transcript│           Vocabulary ──────►│ Keyword Spotter │
    │  "in video corp"│                             │   (DP Algorithm)│
    └────────┬────────┘                             └────────┬────────┘
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   Vocabulary    │
                            │    Rescorer     │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ Final Transcript│
                            │   "NVIDIA Corp" │
                            └─────────────────┘
```

## Dual Encoder Alignment

The system uses two separate neural network encoders that process the same audio:

### 1. TDT Encoder (Primary Transcription)
- **Model**: Parakeet TDT 0.6B (600M parameters)
- **Architecture**: Token Duration Transducer with FastConformer
- **Output**: High-quality transcription with word timestamps
- **Frame Rate**: ~40ms per frame

### 2. CTC Encoder (Keyword Spotting)
- **Model**: Parakeet CTC 110M (110M parameters)
- **Architecture**: FastConformer with CTC head
- **Output**: Per-frame log-probabilities over 1024 tokens
- **Frame Rate**: ~40ms per frame (aligned with TDT)

### Frame Alignment

Both encoders use the same audio preprocessing (mel spectrogram with identical parameters), producing frames at the same rate. This enables direct timestamp comparison between:
- TDT decoder word timestamps
- CTC keyword detection timestamps

```
Audio:     |-------- 15 seconds --------|
TDT Frames: [0] [1] [2] ... [374] (375 frames @ 40ms)
CTC Frames: [0] [1] [2] ... [374] (375 frames @ 40ms)
                    ↑
            Aligned timestamps
```

## Pipeline Components

### 1. CtcTokenizer (`CtcTokenizer.swift`)

Converts vocabulary terms to CTC token ID sequences using the HuggingFace tokenizer (loaded from `tokenizer.json`).

```swift
// Example: tokenizing a vocabulary term
let tokenizer = try await CtcTokenizer.load()
let tokenIds = tokenizer.encode("NVIDIA")
// Result: [42, 156, 89, 23] (subword token IDs)
```

**Why this matters**: The CTC model outputs probabilities over its learned vocabulary. To match custom terms, we must convert them to the same token space.

### 2. CtcKeywordSpotter (`CtcKeywordSpotter.swift`)

Implements the NeMo CTC word spotting algorithm using dynamic programming.

**Algorithm Overview** (per arXiv:2406.07096):

```
For each vocabulary term with token sequence [t₁, t₂, ..., tₙ]:

1. Initialize DP table: dp[frame][token_position]
2. For each CTC frame f:
   - dp[f][i] = max(
       dp[f-1][i] + log_prob[f][blank],      // Stay (emit blank)
       dp[f-1][i-1] + log_prob[f][tᵢ]        // Advance (emit token)
     )
3. Score = dp[T][n] (final frame, all tokens consumed)
```

**Implementation Details**:

The DP table construction is consolidated into `fillDPTable(logProbs:keywordTokens:)`, which is shared by three entry points:
- `ctcWordSpot` - Basic word spotting with normalized scores
- `ctcWordSpotConstrained` - Constrained spotting within frame ranges
- `ctcWordSpotMultiple` - Batch detection of multiple occurrences

Score normalization uses `nonWildcardCount(_:)` to handle wildcard tokens correctly.

**Key Features**:
- Handles CTC blank tokens correctly
- Supports repeated characters (e.g., "committee" → c-o-m-m-i-t-t-e-e)
- Returns detection timestamps and confidence scores

### 3. VocabularyRescorer (`VocabularyRescorer.swift`)

Performs principled comparison between original transcript words and vocabulary terms.

**Rescoring Logic**:

```swift
// For each word in transcript that might match a vocabulary term:

1. Tokenize original word: "in video" → [token IDs]
2. Compute CTC score for original word using cached log-probs
3. Get CTC score for vocabulary term from keyword spotter
4. Apply context-biasing weight (CBW = 3.0 per NeMo paper)

if (vocabScore + CBW) > (originalScore + minAdvantage):
    replace("in video" → "NVIDIA")
```

**Why CTC-vs-CTC Comparison**:
- Both scores are on the same scale (CTC log-probabilities)
- Prevents false replacements when original word has strong acoustic evidence
- The context-biasing weight (CBW) gives vocabulary terms a controlled boost

### 4. CustomVocabularyContext (`CustomVocabularyContext.swift`)

Defines vocabulary terms to boost:

```swift
let vocabulary = CustomVocabularyContext(terms: [
    CustomVocabularyTerm(text: "NVIDIA"),
    CustomVocabularyTerm(text: "PyTorch"),
    CustomVocabularyTerm(text: "TensorRT"),
])
```

Each term is tokenized and scored against CTC log-probabilities. High-scoring terms are used to correct the TDT transcript.

## Frame-Level Scoring Details

### CTC Log-Probability Extraction

```swift
// CTC model output shape: [1, T, V] where:
// - T = number of frames (~375 for 15s audio)
// - V = vocabulary size (1024 for Parakeet CTC)

let logProbs: [[Float]] = extractLogProbs(ctcOutput)
// logProbs[frame][tokenId] = log probability of token at frame
```

### Keyword Score Computation

For a keyword with tokens `[t₁, t₂, t₃]`:

```
Frame 0:  log_prob[0][t₁] = -2.3
Frame 1:  log_prob[1][blank] = -0.5 (stay on t₁)
Frame 2:  log_prob[2][t₂] = -1.8
Frame 3:  log_prob[3][t₃] = -2.1
          ─────────────────────────
          Total Score: -6.7
```

### Detection Thresholds

Per the NeMo paper, the system uses several thresholds:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minScoreAdvantage` | 2.0 | Vocab term must score 2.0 better than original |
| `minVocabScore` | -12.0 | Minimum absolute CTC score for detection |
| `maxOriginalScoreForReplacement` | -4.0 | Don't replace high-confidence words |
| `contextBiasingWeight` (CBW) | 3.0 | Boost applied to vocabulary terms |

## Usage Example

```swift
// 1. Load models
let asrManager = try await AsrManager.shared
let ctcModels = try await CtcModels.downloadAndLoad()
let ctcSpotter = CtcKeywordSpotter(models: ctcModels)

// 2. Define vocabulary
let vocabulary = CustomVocabularyContext(terms: [
    CustomVocabularyTerm(text: "NVIDIA"),
    CustomVocabularyTerm(text: "TensorRT"),
])

// 3. Transcribe with vocabulary boosting
let result = try await asrManager.transcribe(
    audioSamples,
    customVocabulary: vocabulary
)

// result.text: "NVIDIA announced TensorRT optimizations"
// result.ctcDetectedTerms: ["NVIDIA", "TensorRT"]
// result.ctcAppliedTerms: ["NVIDIA", "TensorRT"]
```

## File Reference

| File | Purpose |
|------|---------|
| `CtcModels.swift` | CTC model loading (MelSpectrogram + AudioEncoder) |
| `CtcKeywordSpotter.swift` | NeMo CTC word spotting algorithm with DP helpers |
| `CtcTokenizer.swift` | Vocabulary term tokenization (HuggingFace tokenizer) |
| `VocabularyRescorer.swift` | CTC-vs-CTC score comparison (async factory API) |
| `CustomVocabularyContext.swift` | Vocabulary definition structures |

## References

1. **NeMo CTC Word Spotter**: arXiv:2406.07096 - "Fast Context-Biasing for CTC and Transducer ASR with CTC-based Word Spotter"
2. **Parakeet TDT**: NVIDIA NeMo Parakeet TDT 0.6B - Token Duration Transducer
3. **Parakeet CTC**: NVIDIA NeMo Parakeet CTC 110M - CTC-based encoder
4. **HuggingFace Tokenizers**: swift-transformers for BPE tokenization
