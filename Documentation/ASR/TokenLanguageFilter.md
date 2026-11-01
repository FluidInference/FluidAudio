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
    // ... 28 languages total
}

public enum Script: Sendable {
    case latin
    case cyrillic
    case greek
}
```

`Language.script` returns the writing script. Pass `language:` on the
transcribe APIs to enable filtering; omit it to disable.

## Scope: script, not language

A language hint constrains the script, not the language (#1063). Every
Latin-script `Language` resolves to `.latin`, so `.spanish`, `.english`,
`.german` and the rest filter the identical token set and decode the same
audio identically. On Latin-script audio they usually match no hint as well,
since the filter only acts when the top token is in another script (808 of
808 FLEURS es/it/pt/de clips were byte-identical to no hint). Parakeet
v3 has no language-conditioning token, so a hint cannot stop it from
emitting English words inside Spanish (or Italian, German, …) speech.

`.french` is the one exception: it also runs the English function-word
blocklist (`TdtDecoderV3.applyEnglishBlocklist`, #630). That blocklist is
scoped to French because its pieces (` so`, ` was`, ` her`, ` and`, …) are
words or word starts in other languages (#840, #847). On Spanish it turns
"sociedad" into "suciedad", "hermana" into "armana" and "Washington" into
"Witthington".

Measured on FLEURS es_419 (dev 408 + test 100 clips, v3, macOS 27):

| Decode | WER dev / test | Clips with English function words |
|---|---|---|
| v3, no hint (= `.spanish` = `.english`, byte-identical on all 508) | 4.86% / 4.03% | 6 |
| Canary-1b-v2, source and target `.spanish` | 4.50% / 4.94% | 0 |

In two of those six clips v3 rendered most of the sentence in English ("The
structure that present the plumage suggests that the function is not
relation with the vuelo…"); Canary transcribes them with 0 and 1 word errors. Canary's higher
test WER comes mainly from number formatting ("80.000" vs "800 000").

For a decode conditioned on one language, use Canary
(`CanaryManager(models:source:target:)`, 25 European languages) or Cohere
Transcribe (`CoherePipeline.transcribe(audio:models:language:)`, 14
languages). Both prompt the decoder with the language.

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
Validate any per-language token list against real speech in that language,
including names, titles and code-switching: SentencePiece pieces are shared
across languages, and blocking one also blocks every word it starts (#840).

Adding a new language: extend `Language` and its `script` mapping. If the
new language introduces another script, extend `Script` and add a
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
