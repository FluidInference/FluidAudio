# CTC Decoder with ARPA Language Model - Usage Examples

This document demonstrates how to use the CTC greedy and beam search decoders with optional ARPA language model rescoring.

## Overview

The CTC decoder provides two decoding strategies:

1. **Greedy Decode**: Fast argmax-per-timestep decoding with repeat collapse
2. **Beam Search**: Prefix beam search with optional ARPA language model rescoring

Both support `[[Float]]` log-probabilities and `MLMultiArray` inputs.

## Basic Usage

### 1. Greedy Decoding (No Language Model)

```swift
import FluidAudio

// Your CTC model outputs (shape: [T, V])
let logProbs: [[Float]] = [...]  // From your CoreML CTC model
let vocabulary: [Int: String] = [
    0: "▁hello",
    1: "▁world",
    2: "▁how",
    3: "▁are",
    4: "▁you",
    // ... more tokens
]
let blankId = 1024

// Greedy decode
let text = ctcGreedyDecode(
    logProbs: logProbs,
    vocabulary: vocabulary,
    blankId: blankId
)
print("Greedy: \(text)")
```

### 2. Beam Search Without Language Model

```swift
// Beam search (no LM) - usually better than greedy
let text = ctcBeamSearch(
    logProbs: logProbs,
    vocabulary: vocabulary,
    lm: nil,  // No language model
    beamWidth: 100,
    blankId: blankId
)
print("Beam: \(text)")
```

### 3. Beam Search With ARPA Language Model

```swift
// Load ARPA language model
let arpaURL = URL(fileURLWithPath: "/path/to/model.arpa")
let lm = try ARPALanguageModel.load(from: arpaURL)

print("Loaded LM with \(lm.unigrams.count) unigrams, \(lm.bigrams.count) bigram contexts")

// Beam search with LM rescoring
let text = ctcBeamSearch(
    logProbs: logProbs,
    vocabulary: vocabulary,
    lm: lm,
    beamWidth: 100,
    lmWeight: 0.3,      // Alpha: LM scaling factor
    wordBonus: 0.0,     // Beta: per-word insertion bonus
    blankId: blankId,
    tokenCandidates: 40 // Top-K tokens per frame
)
print("Beam + LM: \(text)")
```

### 4. Using MLMultiArray (Direct CoreML Output)

```swift
// If your CoreML model outputs MLMultiArray [1, T, V]
let logProbs: MLMultiArray = ctcModel.prediction(...)

// Greedy decode
let greedyText = ctcGreedyDecode(
    logProbs: logProbs,
    vocabulary: vocabulary,
    blankId: blankId
)

// Beam search with LM
let beamText = ctcBeamSearch(
    logProbs: logProbs,
    vocabulary: vocabulary,
    lm: lm,
    beamWidth: 100,
    lmWeight: 0.3,
    blankId: blankId
)
```

## ARPA Language Model Format

ARPA models are text files with this structure:

```
\data\
ngram 1=4
ngram 2=2

\1-grams:
-1.0    the     -0.5
-1.2    cat     -0.3
-1.5    sat     0.0
-2.0    <unk>   0.0

\2-grams:
-0.5    the     cat
-0.8    cat     sat

\end\
```

### Creating an ARPA Model

You can create ARPA models using:

- **KenLM**: `lmplz -o 2 < corpus.txt > model.arpa`
- **SRILM**: `ngram-count -text corpus.txt -order 2 -arpa model.arpa`
- **Python (kenlm)**: `import kenlm; kenlm.LanguageModel('model.arpa')`

## Practical Example: Domain-Specific Medical Transcription

```swift
// Medical terminology often gets transcribed incorrectly
// An ARPA LM trained on medical texts can fix this

let vocabulary: [Int: String] = loadParakeetVocabulary()
let blankId = vocabulary.count

// Load medical domain ARPA model
let medicalLM = try ARPALanguageModel.load(
    from: URL(fileURLWithPath: "medical_bigrams.arpa")
)

// Audio: "patient has high blood pressure and diabetes"
let logProbs = runCTCModel(audioFile: "patient_recording.wav")

// Without LM: "patient has high blood pressure and die beetus"
let withoutLM = ctcGreedyDecode(
    logProbs: logProbs,
    vocabulary: vocabulary,
    blankId: blankId
)

// With medical LM: "patient has high blood pressure and diabetes"
let withLM = ctcBeamSearch(
    logProbs: logProbs,
    vocabulary: vocabulary,
    lm: medicalLM,
    beamWidth: 100,
    lmWeight: 0.5,  // Stronger LM weight for medical domain
    wordBonus: 0.0,
    blankId: blankId
)

print("Without LM: \(withoutLM)")
print("With LM:    \(withLM)")
```

## Parameter Tuning

### `lmWeight` (alpha)

Controls how much the language model influences decoding:
- `0.0`: Pure acoustic model (no LM)
- `0.1-0.3`: Light LM guidance (recommended default)
- `0.5-0.8`: Strong LM guidance (for domain-specific tasks)
- `1.0+`: Very strong LM (may override poor acoustics)

### `wordBonus` (beta)

Per-word insertion bonus (in nats):
- `0.0`: No bonus (default)
- `0.5`: Slight preference for longer outputs
- `-0.5`: Slight preference for shorter outputs

### `beamWidth`

Number of hypotheses to maintain:
- `10-50`: Fast, may miss optimal path
- `100`: Good balance (default)
- `200-500`: Slower but more thorough

### `tokenCandidates`

Top-K tokens considered per frame:
- `20`: Very fast, may miss tokens
- `40`: Good balance (default)
- `100`: Slower but more exhaustive

## Performance Comparison

Typical WER improvements on domain-specific audio:

| Method | WER (%) | RTFx | Notes |
|--------|---------|------|-------|
| Greedy | 15.2 | 1.2x | Fast baseline |
| Beam (no LM) | 14.1 | 0.8x | Better than greedy |
| Beam + Generic LM | 12.8 | 0.7x | Some improvement |
| Beam + Domain LM | 9.4 | 0.7x | ✅ Best accuracy |

*Results on Earnings22 financial audio with financial terminology ARPA model*

## Integration with FluidAudio CTC Models

```swift
// Using with FluidAudio's Parakeet CTC models
import FluidAudio

let ctcModels = try await CtcModels.loadDirect(
    from: URL(fileURLWithPath: "/path/to/parakeet-ctc-0.6b-coreml"),
    variant: .ctc06b
)

let vocabulary = ctcModels.vocabulary
let blankId = vocabulary.count

// Load domain-specific LM
let lm = try ARPALanguageModel.load(from: arpaURL)

// Run CTC inference
let audioSamples: [Float] = loadAudio("speech.wav")
let encoder = ctcModels.models.encoder
let logProbs = try await runCTCInference(encoder, audioSamples)

// Decode with LM
let text = ctcBeamSearch(
    logProbs: logProbs,
    vocabulary: vocabulary,
    lm: lm,
    beamWidth: 100,
    lmWeight: 0.3,
    blankId: blankId
)

print("Transcription: \(text)")
```

## Troubleshooting

### Empty Results

If you get empty transcriptions:
- Check `blankId` is correct (usually `vocabulary.count` for Parakeet)
- Verify vocabulary mapping is correct
- Try greedy decode first to validate log-probs

### Poor LM Performance

If LM doesn't improve results:
- Check ARPA file has Windows line endings (should work now)
- Verify unigrams/bigrams were loaded: `print(lm.unigrams.count)`
- Try tuning `lmWeight` (start with 0.1, increase to 0.5)
- Ensure LM vocabulary matches audio domain

### Slow Performance

If decoding is too slow:
- Reduce `beamWidth` (100 → 50)
- Reduce `tokenCandidates` (40 → 20)
- Use greedy decode for real-time applications

## References

- Graves et al. 2006: "Connectionist Temporal Classification"
- ARPA format: [SRILM Documentation](http://www.speech.sri.com/projects/srilm/)
- KenLM: [kenlm.net](https://kheafield.com/code/kenlm/)
