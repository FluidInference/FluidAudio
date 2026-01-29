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

### Memory Usage

Running two encoders in parallel increases peak memory consumption:

| Configuration | Peak RAM | Notes |
|---------------|----------|-------|
| TDT encoder only | ~66 MB | Standard transcription |
| TDT + CTC encoders | ~130 MB | With vocabulary boosting |

*Measured on iPhone 17 Pro. Memory settles after initial model loading.*

The additional ~64 MB overhead comes from the CTC encoder (Parakeet 110M) being loaded alongside the primary TDT encoder. For memory-constrained scenarios, consider:
- Loading the CTC encoder on-demand rather than at startup
- Unloading the CTC encoder after transcription completes
- Using vocabulary boosting only for files where domain terms are expected

## Pipeline Components

### 1. CtcTokenizer (`TokenSpotting/CtcTokenizer.swift`)

Converts vocabulary terms to CTC token ID sequences using the HuggingFace tokenizer (loaded from `tokenizer.json`).

```swift
// Example: tokenizing a vocabulary term
let tokenizer = try await CtcTokenizer.load()
let tokenIds = tokenizer.encode("NVIDIA")
// Result: [42, 156, 89, 23] (subword token IDs)
```

**Why this matters**: The CTC model outputs probabilities over its learned vocabulary. To match custom terms, we must convert them to the same token space.

### 2. CtcKeywordSpotter (`TokenSpotting/CtcKeywordSpotter.swift`, `+Inference.swift`)

Runs CTC model inference and implements the NeMo CTC word spotting algorithm.

**Inference pipeline** (`+Inference.swift`):
1. Audio → MelSpectrogram model → mel features
2. Mel features → AudioEncoder model → CTC logits
3. Logits → log-softmax → log-probabilities `[T, V]`
4. For long audio (>15s), processes in overlapping chunks and averages log-probs at boundaries

**Keyword spotting** (`CtcKeywordSpotter.swift`):
- `spotKeywordsWithLogProbs()` — public API that returns detections + cached log-probs
- Delegates DP work to `CtcDPAlgorithm`
- Returns `SpotKeywordsResult` with detections (scores, frame ranges, timestamps) and reusable log-probs

### 3. CtcDPAlgorithm (`TokenSpotting/CtcDPAlgorithm.swift`)

Pure dynamic programming algorithms for CTC keyword spotting. No CoreML dependency — operates on raw `[[Float]]` log-prob matrices.

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

**Entry points**:
- `fillDPTable()` — core DP table construction shared by all variants
- `ctcWordSpotConstrained()` — find best alignment within a time window (used by rescorer to score original words)
- `ctcWordSpotMultiple()` — find ALL occurrences above a threshold with local-max detection and overlap merging

Score normalization uses `nonWildcardCount(_:)` to handle wildcard tokens correctly.

### 4. VocabularyRescorer (`Rescorer/VocabularyRescorer.swift` + extensions)

Performs principled comparison between original transcript words and vocabulary terms using a three-pass algorithm.

**Pass 1 — Keyword Spotting**: Calls `spotKeywordsWithLogProbs()` to run CTC inference and find all vocabulary term detections with scores and frame ranges.

**Pass 2 — Alignment**: Maps each transcript word to overlapping keyword detections by timestamp. Groups consecutive words into multi-word spans to match multi-word vocabulary terms (e.g., "in video" → "NVIDIA").

**Pass 3 — Evaluation**: For each candidate replacement:

1. Compute string similarity (Levenshtein-based) between original word and vocabulary term
2. Check similarity meets minimum threshold
3. Apply guards:
   - **Length ratio guard** — if original is much shorter than vocab term (e.g., "and" vs "Andre"), require higher similarity
   - **Short word guard** — words ≤4 chars with low length ratio need ≥80% similarity
   - **Stopword guard** — spans containing "the", "and", "or" etc. need ≥85% similarity
4. Score original word against CTC log-probs using constrained DP alignment
5. Compare: replacement score (detection score + CBW boost) vs original score
6. Replace only when vocabulary term has stronger acoustic evidence

**Rescorer files**:
- `VocabularyRescorer.swift` — struct definition, Config, result types, word timing builder
- `VocabularyRescorer+TokenRescoring.swift` — three-pass orchestration (`ctcTokenRescore()`)
- `VocabularyRescorer+TokenEvaluation.swift` — per-candidate scoring and guard logic
- `VocabularyRescorer+Utilities.swift` — string similarity, normalization, token boundary helpers

### 5. CustomVocabularyContext (`CustomVocabularyContext.swift`)

Defines vocabulary terms to boost:

```swift
let vocabulary = CustomVocabularyContext(terms: [
    CustomVocabularyTerm(text: "NVIDIA"),
    CustomVocabularyTerm(text: "PyTorch"),
    CustomVocabularyTerm(text: "TensorRT"),
])
```

Each term is tokenized and scored against CTC log-probabilities. High-scoring terms are used to correct the TDT transcript.

#### Alias Support

Vocabulary terms can include aliases to handle common misspellings or phonetic variations:

```swift
let vocabulary = CustomVocabularyContext(terms: [
    CustomVocabularyTerm(
        text: "Häagen-Dazs",           // Canonical form (used in output)
        aliases: ["Haagen-Dazs", "Hagen-Das", "Hagen Daz"]  // Recognized variants
    ),
    CustomVocabularyTerm(
        text: "macOS",
        aliases: ["Mac OS", "Mac O S", "Macos"]
    ),
])
```

**How aliases work**:
- The rescorer checks similarity against both the canonical term and all aliases
- When a match is found (via canonical or alias), the **canonical form** is used in the output
- Aliases are useful for terms with accented characters, hyphens, or common ASR mishearings

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `defaultMinSpotterScore` | -15.0 | Minimum CTC score for keyword spotting detections |
| `defaultMinVocabCtcScore` | -12.0 | Minimum CTC score for vocabulary context matching |
| `defaultCbw` (CBW) | 3.0 | Context-biasing weight boost applied to vocabulary terms |
| `minSimilarityFloor` | 0.50 | Absolute minimum string similarity for any match |
| `defaultMinSimilarity` | 0.52 | Default minimum similarity for vocabulary matching |
| `shortWordSimilarity` | 0.80 | Similarity required for short words (≤4 chars) with low length ratio |
| `stopwordSpanSimilarity` | 0.85 | Similarity required when stopwords are present in span |

All constants are defined in `ContextBiasingConstants.swift`.

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

## Streaming Mode Limitations

> **Note**: Vocabulary boosting with streaming mode (`--streaming`) has limitations.

When using `--custom-vocab` with `--streaming`, be aware of the following constraints:

| Feature | File Mode | Streaming Mode |
|---------|-----------|----------------|
| Multi-word compounds | Fully supported | Limited |
| Cross-chunk detection | N/A | Not supported |
| Rescoring accuracy | Optimal | Reduced |

**Why streaming is limited**:
- Vocabulary rescoring requires the complete CTC log-probability matrix for accurate scoring
- In streaming mode, audio is processed in small chunks (~1-2 seconds)
- Keywords that span chunk boundaries may not be detected
- The rescorer cannot look ahead to future frames for optimal alignment

**Recommendations**:
- For maximum vocabulary boosting accuracy, use file-based transcription
- If streaming is required, prefer single-word vocabulary terms over multi-word phrases
- Consider post-processing the streaming transcript with vocabulary boosting on the complete audio

## BK-Tree Approximate String Matching (Experimental)

The rescorer supports an optional **BK-tree** (Burkhard-Keller tree) for efficient approximate string matching. When enabled, the rescorer switches from the default term-centric algorithm to a **word-centric** algorithm.

### How It Works

| Algorithm | Approach | Complexity | Default |
|-----------|----------|------------|---------|
| **Term-centric** | For each vocab term, scan all TDT words | O(V × W) | Yes |
| **Word-centric** | For each TDT word, query BK-tree for candidates | O(W × log V) | No |

The BK-tree organizes vocabulary terms by edit distance, enabling O(log V) fuzzy lookups per word instead of O(V) linear scans. This is beneficial for large vocabularies (100+ terms).

### Enabling BK-Tree

The BK-tree is controlled by `ContextBiasingConstants.useBkTree` (default: `false`). The maximum edit distance for queries is `ContextBiasingConstants.bkTreeMaxDistance` (default: `3`).

### Candidate Matching

When BK-tree is enabled, the word-centric rescorer finds candidates via `findCandidateTermsForWord()`:

1. **Single word match** — query BK-tree with the normalized TDT word
2. **Two-word compound** — concatenate adjacent words (e.g., "Liv" + "Mali" → "livmali" matches "Livmarli")
3. **Three-word compound** — for longer terms (≥6 chars)
4. **Multi-word phrase** — space-separated phrases for multi-word vocabulary terms

All candidates are sorted by similarity (descending) then span length (descending).

### Status

The BK-tree path is experimental. In benchmarks, the default term-centric algorithm produces slightly better results. The BK-tree is primarily useful for very large vocabularies where O(W × log V) lookup provides meaningful speedup over O(V × W) linear scan.

## Vocabulary Size Guidelines

| Vocabulary Size | Performance | Notes |
|-----------------|-------------|-------|
| 1-50 terms | Excellent | Typical use case (company names, products) |
| 50-100 terms | Good | No noticeable latency impact |
| 100-230 terms | Tested | Validated with domain-specific term lists |

**Recommendations**:
- Keep vocabularies focused on domain-specific terms that ASR commonly misrecognizes
- Avoid adding common words that the ASR already handles well
- Terms should be at least 4 characters (configurable via `minTermLength`)
- The system automatically skips stopwords (a, the, and, etc.) to prevent false matches

## File Reference

```
CustomVocabulary/
├── ContextBiasingConstants.swift              — All numeric constants and thresholds
├── CustomVocabularyContext.swift               — Vocabulary term data model and tokenization
├── BKTree/
│   ├── BKTree.swift                           — Burkhard-Keller tree for approximate string matching (experimental)
│   └── VocabularyRescorer+CandidateMatching.swift — Word-centric candidate finding via BK-tree or linear scan
├── Rescorer/
│   ├── VocabularyRescorer.swift               — Core struct, Config, result types, word timing builder
│   ├── VocabularyRescorer+TokenRescoring.swift — Rescoring orchestration (term-centric + word-centric)
│   ├── VocabularyRescorer+TokenEvaluation.swift— Per-candidate scoring and guard logic
│   └── VocabularyRescorer+Utilities.swift     — String similarity, normalization, token boundary helpers
└── TokenSpotting/
    ├── CtcDPAlgorithm.swift                   — Pure DP algorithms (no CoreML dependency)
    ├── CtcKeywordSpotter.swift                — Public spotting API and result types
    ├── CtcKeywordSpotter+Inference.swift       — CoreML inference pipeline (audio → log-probs)
    ├── CtcModels.swift                        — CTC model downloading and loading
    └── CtcTokenizer.swift                     — Text → token ID encoding
```

## References

1. **NeMo CTC Word Spotter**: arXiv:2406.07096 - "Fast Context-Biasing for CTC and Transducer ASR with CTC-based Word Spotter"
2. **Parakeet TDT**: NVIDIA NeMo Parakeet TDT 0.6B - Token Duration Transducer
3. **Parakeet CTC**: NVIDIA NeMo Parakeet CTC 110M - CTC-based encoder
4. **HuggingFace Tokenizers**: swift-transformers for BPE tokenization
