# Kokoro Chunker Overview

This document summarizes how `Sources/FluidAudio/TextToSpeech/Kokoro/KokoroChunker.swift`
prepares text for Kokoro synthesis.

## High-level flow
- The public `chunk(text:wordToPhonemes:caseSensitiveLexicon:targetTokens:hasLanguageToken:)` entry
  point trims the
  input, collapses explicit newlines into single spaces, and runs `splitIntoSentences`.
- Sentences are trimmed in `applyRefinements` before any further processing.
- A per-chunk phoneme budget (`capacity`) is derived from the model’s declared input size.
  We subtract two control tokens (BOS/EOS), optionally one language token, and a 12-token
  safety margin.
- Each sentence (aka `segmentsByPeriods`) is evaluated for phoneme cost by
  `tokenCountForSegment`. If the cost is above `capacity`, we attempt punctuation-aware
  splitting with `splitByPunctuation` and `reassembleFragments` before deferring to
  the chunk builder.
- Finally, `buildChunks` constructs `TextChunk` structs, respecting the phoneme budget and
  preserving punctuation/spacing semantics for downstream synthesis.

## Sentence processing
- When the Apple NaturalLanguage framework is available, `splitIntoSentences` uses `NLTokenizer`
  (part of `NLTagger`) to segment text with the detected dominant language. On platforms without
  that framework we fall back to manual splitting on `.?!`.
- `applyRefinements` only removes leading/trailing whitespace and drops empty strings.
- For newline-heavy inputs, `collapseNewlines` reduces any newline runs to single spaces so the
  tokenizer does not emit empty segments.

## Capacity computation
- `computeCapacity(targetTokens:hasLanguageToken:)` subtracts control overhead and a fixed
  12-token safety margin from the model’s `targetTokens`. The caller currently passes the
  15-second model’s limit (~246 tokens), yielding a usable budget of 232 when no language token
  is present.
- The chunker logs the phoneme cost of each sentence relative to this budget to aid debugging.

## Boundary hierarchy
1. **Sentence split:** `splitIntoSentences` uses `NLTokenizer`/`NSLinguisticTagger` to find full
   sentences; we keep entire sentences whenever they fit inside the token budget.
2. **Clause-level splits:** For overflow sentences, `splitByPunctuation` leverages
   `NLTagger`/`NSLinguisticTagger` to cut on commas, semicolons, or colons and
   `reassembleFragments` packs those clauses back under the limit.
3. **Lexical fallback:** When punctuation is missing, the chunk builder still walks the text in
   lexical order. We track part-of-speech/name tags for recent words and prefer to end on
   nouns, verbs, adjectives, names, or conjunctions when a candidate sits near the budget edge.
4. **Hard budget flush:** Only if no suitable lexical breakpoint exists do we cut exactly at the
   model token limit, guaranteeing the chunk never overruns Kokoro’s capacity.

Throughout this process punctuation atoms are preserved if they exist in the Kokoro vocabulary, and
consecutive punctuation characters remain tightly spaced via `noPrespaceCharacters`.

## Tokenization & phoneme lookup
- `tokenizeAtoms` treats runs of letters/digits/apostrophes as words and emits other characters
  as punctuation tokens.
- `resolvePhonemes` first checks the exact-case lexicon, then the lowercase map. When a match is
  still missing it falls back to eSpeak G2P (if the bridge is available), spelled-out number
  handling, or built-in letter pronunciations. Missing entries are logged once per chunk.
- `tokenCountForSegment` mirrors the same logic to compute phoneme cost without allocating
  intermediary chunks.

## Chunk assembly
- `buildChunks` walks atoms and accumulates phoneme tokens until adding the next word/punctuation
  would exceed `capacity`. At that point it flushes the current chunk and continues.
- Inter-word separators are represented explicitly as a single space phoneme. Trailing space
  tokens are trimmed before a chunk is finalized.
- Each emitted `TextChunk` captures the word list, original atoms (words plus punctuation), the
  resolved phoneme sequence, and a synthesized text string created via `appendSegment` so we
  preserve em dashes, quotes, and other punctuation without unwanted whitespace.

## Logging & diagnostics
- The chunker logs the sentence-level segments (`segmentsByPeriods`) and post-punctuation splits
  (`segmentsByPunctuations`), along with phoneme counts. These logs make it easy to see why a
  sentence fell into the 15-second bucket or required additional splitting.
- Missing phoneme lexicon entries are gathered per chunk and emitted as a single warning to avoid
  log spam.

## Interaction with model selection
- Downstream (`KokoroModel.selectVariant`) decides whether to use the 5s or 15s CoreML model
  based on chunk token count. The chunker’s safety margin helps keep the 5s variant within its
  83-token limit while allowing longer sentences to be promoted.
