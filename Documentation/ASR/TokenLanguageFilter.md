# TokenLanguageFilter

## Overview

Parakeet TDT v3 is a multilingual model with a single shared SentencePiece
vocabulary spanning Latin and Cyrillic alphabets. On short utterances with
ambiguous audio, the joint network can emit wrong-language tokens — e.g.
Russian-alphabet tokens while transcribing Polish (issue #512).
`TokenLanguageFilter` picks the highest-logit top-K candidate whose alphabet
matches the caller-provided `language`, suppressing wrong-language leakage
without retraining the model.

Relevant code:
- `Sources/FluidAudio/Shared/TokenLanguageFilter.swift`
- `Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/Decoder/TdtDecoderV3.swift`
- `Tests/FluidAudioTests/Shared/TokenLanguageFilterTests.swift`

## When Filtering Runs

The filter hooks into the v3 TDT decoder at two sites:

1. **Main decode loop** — every step while the decoder is advancing through an
   encoder chunk.
2. **Inner silence-skip loop** — the loop that consumes consecutive blanks
   before committing to a frame.

Empirical testing on the 7 Polish audio samples from issue #512:

| Call site | Swaps across 7 clips |
|---|---|
| Main loop | 7 |
| Inner silence-skip loop | 21 |
| Last-chunk flush loop | 0 |

The flush loop is blank/punct-dominated on short utterances, so it does not
run the filter. See `TdtDecoderV3.swift` for the inline comment.

## API

```swift
public enum Language: String, Sendable, CaseIterable {
    case english = "en"
    case polish = "pl"
    case russian = "ru"
    // ... 18 languages total
}

public enum Script: Sendable {
    case latin
    case cyrillic
}
```

`Language.script` returns the writing script. Pass `language:` on the
transcribe APIs to enable filtering; omit it to disable.

## Algorithm

For every decoder step the joint network produces top-K token IDs and logits.
The filter:

1. Walks the top-K list (does **not** assume CoreML returns it sorted).
2. Looks up each token's text in the vocabulary.
3. Runs `matches(tokenText, script:)` — a per-character Unicode range check.
4. Keeps the highest-logit candidate that passes.
5. Returns that token plus a top-K softmax probability.

If no right-language candidate exists in the top-K, returns `nil` and the
decoder falls back to its unfiltered argmax.

### Script membership rules

**Latin-compatible characters:**
- ASCII (`0x0020`–`0x007F`)
- Latin-1 (`0x00A0`–`0x00FF`)
- Latin Extended-A (`0x0100`–`0x017F`)
- Latin Extended-B (`0x0180`–`0x024F`) — covers Romanian `ș`, `ț`
- Combining Diacritical Marks (`0x0300`–`0x036F`) — handles NFD decomposed
  forms like `e` + U+0301 in case a vocab export doesn't precompose
- Latin Extended Additional (`0x1E00`–`0x1EFF`) — Vietnamese, etc.

**Cyrillic-compatible characters:**
- Cyrillic block (`0x0400`–`0x04FF`)
- Script-neutral ASCII (digits, punctuation, whitespace) — but **ASCII
  letters `A–Z`/`a–z` are explicitly rejected**

### Asymmetric guards

The Latin path does **not** need a "reject Cyrillic" check: the Cyrillic
block (`0x0400`–`0x04FF`) lies outside every Latin range, so `allSatisfy`
naturally fails for any Cyrillic scalar.

The Cyrillic path **does** need an explicit "reject ASCII letters" check,
because ASCII punctuation and digits are script-neutral and shared between
both scripts. Without the guard, a Latin-letter token like `"cat"` would
pass the Cyrillic filter.

### SentencePiece boundary marker

Tokens are prefixed with `▁` (U+2581) as a whitespace indicator. The filter
strips it before the script check — it carries no script information.

Pure-boundary tokens (just `▁`, no letters) return `true` from `matches`,
not `false`. Reason: `filterTopK` uses `matches` as a gate; returning
`false` for neutral tokens would unconditionally exclude them, but they
should compete on logit alone.

## Probability semantics

The probability returned by `filterTopK` is **softmax over the top-K
logits only**, not over the full vocabulary. Because the denominator
excludes ~|vocab|−K terms, this value is systematically larger than a
full-vocab softmax.

For K=64 on an 8k-token vocab it's a reasonable proxy when the model is
confident, but it should not be treated as a drop-in full-vocabulary
probability.

## Edge cases handled

- **`-∞` logit survives argmax.** The sentinel `bestLogit = -.infinity`
  plus a candidate logit of `-.infinity` would never satisfy
  `logit > bestLogit`. The `bestIdx < 0` clause forces the first matching
  candidate to win unconditionally, even if its logit is `-∞`.
- **Non-finite max logit.** If every top-K logit is `-∞`, the softmax
  returns probability 0 rather than producing `NaN`.
- **Array length mismatch.** `topKIds` and `topKLogits` are read
  independently from CoreML outputs; the filter takes `min(count)` and
  returns `nil` if either is empty.
- **NFD decomposed characters.** Combining Diacritical Marks are in the
  Latin allow-list so decomposed forms pass the check.

## Extending

The filter currently partitions by Unicode script only. Per-language token
allowlists (e.g. distinguishing Polish from Czech within the Latin script)
could plug in at the `matches` call site without changing the decoder API.

Adding a new language: extend `Language` and its `script` mapping. If the
new language introduces a third script, extend `Script` and add a
corresponding `matches` case.

## Testing

40 unit tests in `TokenLanguageFilterTests` cover:
- Every supported language's characteristic diacritics
- Cross-script rejection (Latin `"cat"` must fail Cyrillic, Cyrillic `"кот"`
  must fail Latin)
- Boundary marker stripping and neutral-token behavior
- Combining diacritic (NFD) forms
- `-∞` logit edge case in `filterTopK`

Run with:

```bash
swift test --filter TokenLanguageFilterTests
```
