# Context Boosting (Parakeet v3 · FluidAudio) — Implementation Steps

This guide describes the minimal, safe steps to add decode‑time context boosting to FluidAudio’s Parakeet v3
(RNNT/TDT) pipeline. It assumes you have a phrase list with optional weights, for example the JSON we generated at
`custom_vocab_parakeet_v3/custom_vocab.json`.

## 1) Input Format (Simple JSON)

- File: `custom_vocab_parakeet_v3/custom_vocab.json`
- Shape:
  - `terms: [{ text: String, weight: Float }]`
  - Global knobs: `alpha` (mix), `contextScore` (per‑token bonus), `depthScaling`, `scorePerPhrase` (optional)
- Notes: keep words lowercased, avoid function words and pure numbers to reduce false inserts.

## 2) Public API

- Add a new type `CustomVocabularyContext` (thread‑safe) with fields:
  - `terms: [(text: String, weight: Float)]`
  - `alpha: Float`, `contextScore: Float`, `depthScaling: Float`, `scorePerPhrase: Float`
  - (Advanced later) `tokenIds: [[Int]]` for pre‑tokenized phrases
- Extend the ASR entry point:
  - `AsrManager.transcribe(..., customVocabulary: CustomVocabularyContext?)`
  - File: `Sources/FluidAudio/ASR/AsrManager.swift`

## 3) Trie Engine (Actor)

- Create `CustomVocabularyEngine` (actor) to build and serve a prefix trie over model token IDs.
- Responsibilities:
  - Build trie nodes for each phrase; assign per‑arc scores using `contextScore` and proportional share of
    `scorePerPhrase` (e.g., `scorePerPhrase / phraseLength`). Apply `depthScaling^depth`.
  - Maintain backoff/failure links (Aho–Corasick style) so overlaps continue cleanly.
  - Provide per‑session cursor/state and a method returning a sparse bias for a small candidate set of token IDs.
- Files to add (suggested):
  - `Sources/FluidAudio/ASR/ContextBiasing/CustomVocabularyContext.swift`
  - `Sources/FluidAudio/ASR/ContextBiasing/CustomVocabularyEngine.swift`

## 4) Tokenization Strategy

- Phase A (ship fast): accept only pre‑tokenized IDs (`tokenIds`) per phrase; this avoids wiring a tokenizer now.
- Phase B: integrate SentencePiece/BPE for Parakeet vocab to convert `terms[].text` to model tokens and optionally
  generate multiple tokenizations per phrase (BPE dropout) for robustness.

## 5) Decoder Hook (RNNT/TDT)

- File: `Sources/FluidAudio/ASR/TDT/TdtDecoderV3.swift`
- Where: inside `decodeWithTimings(...)`, after computing token logits and before selecting the next token.
- Steps per decode iteration:
  - Select top‑K token indices from the logits (K ~ 32–64) to bound CPU work.
  - Ask the engine for a sparse bias map over those candidates.
  - Apply `logits[i] += alpha * bonus` for those indices only.
  - Keep duration/blank heads unchanged.
  - If a non‑blank token is emitted, advance the trie cursor and predictor LSTM; always advance frames by predicted
    duration.

## 6) Decoder State

- File: `Sources/FluidAudio/ASR/TDT/TdtDecoderState.swift`
- Add one field to track the trie cursor, e.g. `var vocabNode: Int?` initialized to root when a custom vocabulary is
  provided.

## 7) Wire‑Up in AsrManager

- File: `Sources/FluidAudio/ASR/AsrManager.swift`
- When `customVocabulary` is non‑nil:
  - Build (or reuse) a `CustomVocabularyEngine` for this call/session.
  - Initialize the `vocabNode` to the root and pass both engine + node to the TDT decoder.
- Ensure state reset between independent transcribe calls is still honored.

## 8) CLI Support (Optional)

- File: `Sources/FluidAudioCLI/Commands/ASR/TranscribeCommand.swift`
- Add flags:
  - `--custom-vocab <path>`: load JSON and pass through to `AsrManager.transcribe(...)`.
  - `--alpha`, `--context-score`, `--depth-scaling` to override JSON defaults at runtime.

## 9) Guardrails

- Keep boosts modest: start `alpha ~ 0.5`, `contextScore ~ 1.2`, `depthScaling = 2.0`.
- Penalize unknown beginnings: set `unkScore` ~ 0 (in engine scoring if you add it later).
- Optionally use a small positive `finalEosScore` to reduce partial‑phrase hallucination.
- Prefer longer, more specific phrases; avoid common function words.

## 10) Validation Plan

- Prepare a few positive and negative clips; run A/B:
  - Baseline vs `--custom-vocab custom_vocab_parakeet_v3/custom_vocab.json`.
  - Sweep `alpha` 0.3/0.5/0.7; monitor keyword precision/recall, false inserts, WER, and latency.
- Trim risky terms; keep lists small and session‑specific.

## 10b) Verification Checklist (What to Test)

- Functional
  - JSON loader: accepts well‑formed files; rejects missing/invalid fields; allows overrides via CLI.
  - Trie build: phrases produce expected node/edge counts; backoff/failure links resolve; duplicate phrases deduped.
  - Tokenization (when added): text → token IDs matches vocab; rejects <unk>/blank; supports multiple tokenizations.
  - Bias math: given logits and a synthetic bias map, `logits' = logits + alpha*bias` yields expected argmax changes; duration/blank heads unchanged.
  - State: trie cursor advances on non‑blank tokens; no advance on blanks; resets between transcriptions.

- Integration (Audio)
  - A/B on a small suite (positive clips contain targets; negative clips do not):
    - Metrics: keyword precision/recall/F1, false‑insert rate, WER delta, latency/RTFx delta.
    - Alpha sweep: 0.3/0.5/0.7 → pick the smallest that meets recall target without false inserts.
    - Top‑K sensitivity: K = 32/48/64; confirm negligible change to WER and small CPU overhead.
  - Compound vs spaced: validate representative pairs (e.g., "today/to day", domain compounds) improve with boosting or post‑merge fallback.

- Performance
  - RTFx delta: ≤ 3–5% vs baseline on test‑clean in release build.
  - Bias step cost: O(K) microseconds per iteration; confirm no frame drops in streaming.
  - Memory: trie size and optional `(state, token) → nextState` cache bounded and reclaimed per session.

- Safety/Guardrails
  - Unknown tokens: phrases mapping to <unk> or blank are ignored (no boost applied).
  - Final‑EOS/UNK knobs (when added) reduce partial‑phrase hallucination on synthetic edge cases.
  - Function words: verify filter prevents boosting common stopwords; review top false‑insert offenders and prune.

- CLI/UX
  - `--custom-vocab` loads JSON; `--alpha/--context-score/--depth-scaling` override defaults.
  - Errors are actionable: bad paths, malformed JSON, or incompatible vocab produce clear messages.

## 11) Acceptance Criteria

- Functional: decoder emits identical output to baseline when `customVocabulary` is nil.
- Accuracy: on the positive set, keyword recall increases; on the negative set, false‑insert rate does not increase materially.
- Stability: no regressions in WER beyond agreed tolerance (≤ 0.2–0.5% absolute) on test‑clean; no crashes under streaming.
- Performance: RTFx within 3–5% of baseline; memory stable across many transcriptions.

## 12) Rollout & Controls

- Feature flag (per request): enable/disable `customVocabulary` cleanly.
- Logging: count boosted token applications and top boosted phrases (redacted if needed) for tuning.
- Safe defaults: modest `alpha/contextScore`; session‑scoped phrase lists only.

## 13) Quick Commands

- Baseline benchmark (release):
  - `swift run -c release fluidaudio asr-benchmark --subset test-clean --output baseline.json --no-min-wer --no-sort-wer`
- With custom vocab (JSON above):
  - `swift run -c release fluidaudio asr-benchmark --subset test-clean --output boosted.json --no-min-wer --no-sort-wer --include-diffs` (after wiring CLI)
  - Compare WER, keyword metrics, and RTFx across baseline vs boosted.

## 11) Performance Considerations

- Build the trie once per request/session and reuse across chunks.
- Compute bias only over top‑K candidates each step (O(K) work on CPU).
- Optionally cache `(state, token) -> nextState` transitions in a tiny LRU.

## 12) Optional Fallback (No Decoder Changes)

- If you need a quick safety net for compound/spaced issues, run a post‑merge keyword check using `TokenTiming` and
  replace/insert only when overlap + confidence pass thresholds. This is weaker than decode‑time bias but is fast to
  ship.

## 13) NeMo Parity (for external validation)

- RNNT/TDT (Parakeet v3): set `boosting_tree.key_phrases_file` and `boosting_tree_alpha` in the decoding config.
- Per‑phrase weighting requires prebuilding a boosting‑tree model (`.nemo`); otherwise use global knobs.
- CTC alternatives: PyCTCDecode hotwords (one global weight) or Flashlight + KenLM (`boost.tsv` per‑word weights).

---

Starting point: you can use the generated `custom_vocab_parakeet_v3/custom_vocab.json` (or `vocab.txt`) and follow the
steps above. Once the hook is in, validate on a small set and tune `alpha/contextScore/depthScaling` conservatively.
