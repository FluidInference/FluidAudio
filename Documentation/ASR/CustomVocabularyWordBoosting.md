# Custom Vocabulary and Word Boosting (Parakeet v3 · NeMo/Riva · Argmax)

## Summary

- Word boosting (aka contextual biasing, custom vocabulary) increases the likelihood of specific words/phrases during transcription without retraining or changing model weights.
- Parakeet v3 uses decoder/serving features in NeMo or Riva to apply boosts. The model itself is unchanged.
- Argmax’s Custom Vocabulary is model‑agnostic: it runs a dedicated keyword search and merges hits into any transcript (including Parakeet‑backed pipelines), enabling large vocabularies (~1000 keywords).
- Proper tuning substantially improves name/jargon accuracy while minimizing false inserts.

## How It Works

- Decoder biasing: Add a positive score to target tokens/phrases during search (beam/greedy). Survives pruning, increases output probability of targets.
- Context graph/FST: Build a trie/Aho–Corasick–style graph over BPE tokens with per‑token bonuses, integrated into CTC/RNNT decoding.
- Parallel keyword spotting (KWS): A lightweight spotter runs over acoustic/posterior frames and produces time‑stamped keyword hits. A merger aligns baseline tokens to time and replaces/inserts the boosted words when confidence and overlap thresholds are met.

## Parakeet v3 Support via NeMo/Riva

- NeMo repository (context biasing components):
  - `nemo/collections/asr/parts/context_biasing/context_graph_ctc.py` (CTC context graph)
  - `nemo/collections/asr/parts/context_biasing/context_graph_universal.py` (universal context graph)
  - `nemo/collections/asr/parts/context_biasing/ctc_based_word_spotter.py` (CTC word spotter)
  - `nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py` (GPUBoostingTreeModel)
  - Decoder integration:
    - `nemo/collections/asr/parts/submodules/ctc_decoding.py` (beam config supports `boosting_tree` + `boosting_tree_alpha`)
    - `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` (RNNT fusion with boosting tree)

- Riva serving:
  - Exposes “speech contexts” (phrase lists with boost values) via SDK/gRPC for batch and streaming recognition.
  - Works with Parakeet RNNT/CTC models deployed in Riva. No model changes required.

## Argmax Custom Vocabulary (Model‑Agnostic)

- Runs a KWS pipeline alongside any ASR engine (Parakeet, Whisper, cloud APIs).
- Compiles an index/graph for the provided keywords (supports many items) and detects them with timestamps.
- Merges detected keywords with the baseline transcript using alignment + thresholds; corrects spelling, casing, or inserts missing terms when acoustically present.

## When To Use

- Proper names, brands, acronyms, domain jargon (meetings, support calls, field ops, media with slides).
- Contextual transcription based on meeting invites, CRM, video title/description, or OCR from slides.

## Tuning Guidelines

- Start moderate: Use mid‑range boost weights; raise gradually if targets are still missed.
- Keep lists contextual: Per session/video; avoid large global lists that are rarely relevant.
- Provide variants: Include aliases, common spellings, and pronunciations (when supported) for OOV or hard names.
- Guardrails: Use overlap thresholds with ASR timing, non‑max suppression for duplicates, and cooldowns for repeated hits.

## Fine‑Tuning vs Boosting

- Boosting: Inference‑time only; no model weight changes; suitable for session‑specific terms and rapid iteration.
- Fine‑tuning: Consider when you need permanent, broad domain adaptation that cannot be covered by vocabulary lists.

## Integration in FluidAudio Pipeline

- Placement: `Audio → VAD → Diarization → ASR → CustomVocabularyMerge → Timestamps/Output`
- Inputs: `List<CustomTerm>` where each term may include `text`, optional `weight`, `aliases`, and optional pronunciations.
- Output: Updated transcript (and optionally per‑term matches with timestamps and confidence).

### Example shapes (pseudocode)

```json
{
  "terms": [
    { "text": "Argmax", "weight": 2.0, "aliases": ["ARGMAX", "arg max"] },
    { "text": "Parakeet", "weight": 1.5 },
    { "text": "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch", "weight": 3.0 }
  ],
  "bias": { "defaultWeight": 1.0, "acceptThreshold": 0.6, "overlapPct": 0.3 }
}
```

### NeMo (CTC) decoding with boosting tree (conceptual)

```yaml
# decoding.yaml
beam:
  beam_size: 8
  ngram_lm_alpha: 0.0
  boosting_tree:
    key_phrases_list: ["Argmax", "Parakeet", "Voice Ink"]
    context_score: 1.5
    depth_scaling: 2.0
    unk_score: 0.0
  boosting_tree_alpha: 0.7
```

### Riva (conceptual request fields)

- Provide a list of phrases and per‑phrase boost to recognition request (batch/streaming).
- Tune phrase boosts conservatively to avoid false inserts.

## Validation Plan

- Dataset: Curate names/brands/jargon for your target workflows (meetings, product demos, media).
- Metrics: Keyword precision/recall/F1, WER change, false‑insert rate.
- Ablations: Bias strength sweep; with/without pronunciations; list scaling (100 → 1000 terms); streaming vs batch.
- Ops checks: Latency overhead, per‑speaker vocab (if diarization used), and context activation only when relevant.

## Key Takeaways

- Word boosting is runtime software logic (decoder/merger), not a Parakeet model change.
- NeMo provides context graphs, a GPU boosting tree, and decoder fusion for CTC/RNNT; Riva exposes phrase boosting via APIs.
- Argmax’s Custom Vocabulary is model‑agnostic, merges KWS hits into any transcript, and scales to large vocabularies.
- Use targeted, session‑scoped vocabularies with moderate boosts and clear thresholds to balance misses vs false positives.

## References and Further Reading

- Foundational algorithms for fast keyword matching (basis for context graphs)
  - A. V. Aho and M. J. Corasick. “Efficient String Matching: An Aid to Bibliographic Search.” Communications of the ACM, 1975. https://doi.org/10.1145/360825.360855

- Open-source implementations and docs
  - Icefall Context Graph (hotword/phrase boosting with Aho–Corasick): search “k2-fsa icefall context graph hotword”.
  - NVIDIA NeMo context biasing components (code paths in this repo snapshot):
    - `nemo/collections/asr/parts/context_biasing/context_graph_ctc.py`
    - `nemo/collections/asr/parts/context_biasing/context_graph_universal.py`
    - `nemo/collections/asr/parts/context_biasing/ctc_based_word_spotter.py`
    - `nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py`

- Contextual end‑to‑end ASR (CLAS family and related)
  - Deep CLAS: Deep Contextual Listen, Attend and Spell (2024). https://arxiv.org/abs/2409.17603
  - Contextualized End-to-End Speech Recognition with Contextual Phrase Prediction Network (2023). https://arxiv.org/abs/2305.12493

- RNNT/CTC biasing and hotword boosting
  - Contextual RNN-T For Open Domain ASR (2020). https://arxiv.org/abs/2006.03411
  - Deep Shallow Fusion for RNN-T Personalization (2020). https://arxiv.org/abs/2011.07754
  - Robust Acoustic and Semantic Contextual Biasing in Neural Transducers for Speech Recognition (2023). https://arxiv.org/abs/2305.05271
  - Text Injection for Neural Contextual Biasing (2024). https://arxiv.org/abs/2406.02921
  - CB-Conformer: Contextual biasing Conformer for biased word recognition (2023). https://arxiv.org/abs/2304.09607
  - Improving ASR Contextual Biasing with Guided Attention (2024). https://arxiv.org/abs/2401.08835

- Keyword spotting (KWS) approaches used for model‑agnostic merging
  - DONUT: CTC-based Query-by-Example Keyword Spotting (2018). https://arxiv.org/abs/1811.10736
