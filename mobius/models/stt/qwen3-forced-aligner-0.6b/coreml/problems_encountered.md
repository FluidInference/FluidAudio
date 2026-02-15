# Problems Encountered — Qwen3-ForcedAligner-0.6B CoreML Conversion

Tracking all issues, bugs, trials, and what worked during conversion.
Status: ✅ RESOLVED | ⚠️ PARTIAL | ❌ ABANDONED | 🔵 IN PROGRESS | ⬜ NOT YET TESTED

---

## 1. [✅] RoPE Layout — Interleaved MRoPE vs Standard

**Context:** The ForcedAligner uses `interleaved: true` with `mrope_section: [24, 20, 20]`,
while Qwen3-ASR uses standard non-interleaved RoPE.

**Investigation:** Traced PyTorch source — `interleaved: true` in config refers to MRoPE
frequency interleaving across T/H/W dimensions, NOT the `rotate_half` layout. Both PyTorch
and our wrapper use the same concatenated halves `rotate_half`: `[-x2, x1]`.

**Verification:** With `attn_implementation="eager"`, feeding identical merged embeddings
and cos/sin to the CoreML decoder produces max diff of only 0.0013 vs PyTorch (effectively
identical). The RoPE layout was never the issue.

**Status:** Resolved. RoPE is correctly implemented.

---

## 2. [✅] Audio Encoder — Cross-Chunk Attention

**Context:** ForcedAligner encoder is 24 layers / 1024 dim vs ASR's 18 layers / 896 dim.

**Initial approach:** Monolithic `AudioEncoderFullWrapper` processing one 100-frame mel
window at a time, with each chunk going through conv + transformer independently.

**Discovery:** The native PyTorch encoder processes conv outputs from ALL chunks through
the transformer together with full bidirectional attention. Processing chunks independently
missed cross-chunk attention, causing max diff of ~2.1 in audio embeddings (despite
per-chunk conv diff being only ~0.08). This was the root cause of the 20.7ms AAS error.

**Fix:** Split encoder into two CoreML models:
- `AudioConvWrapper`: mel → conv downsample + positional embedding (per-chunk, [1,128,100] → [1,13,1024])
- `AudioTransformerWrapper`: concatenated conv features → 24-layer transformer + projection ([1,256,1024] → [1,256,1024])

At inference time, all chunk conv outputs are concatenated and fed through the transformer
together, matching the native cross-chunk attention behavior.

**Result:** AAS improved from 20.7ms → 4.4ms, within-20ms from 90.7% → 95.4%,
within-80ms from 92.6% → 99.1%.

**Status:** Resolved.

---

## 3. [✅] LM Head Output Dim — 5000, NOT vocab_size

**Context:** Expected LM head output to be vocab_size (152,064) based on config.json.
Actual lm_head.weight shape is `[5000, 1024]`.

**Discovery:** The ForcedAligner does NOT predict vocab tokens — it predicts raw
timestamp values via argmax. Each output class represents a timestamp bin:
`value × 80ms = absolute time`. With 5000 bins × 80ms = 400 seconds, this covers
up to ~6.7 minutes of audio.

The embedding table IS 152,064 tokens (shared architecture with ASR), but the LM head
is a separate projection to 5000 timestamp classes. Config's `vocab_size` refers to
the embedding, not the LM head.

**Impact:** CoreML output shape is correctly `[1, seq, 5000]`. No fix needed —
this was a documentation/understanding issue, not a conversion bug. Updated README,
metadata, and wrapper docstrings.

---

## 4. [🔵] Prefill Sequence Length

**Context:** ForcedAligner processes up to 5 minutes of audio. With ~100 mel frames
per window and text tokens, sequences could reach 1000+ tokens. Using PREFILL_SEQ_LEN=1024
as initial value.

**Risk:** Fixed-shape prefill may need to be larger for long audio. If 1024 isn't enough,
we'll need to increase or implement chunked prefill.

**Status:** Need to measure typical sequence lengths on test data.

---

## 5. [✅] FP16 Overflow in LM Head

**Context:** The ASR conversion hit a bug where RMSNorm computed x^2 = 300^2 = 90,000
which overflows FP16 max (65,504), producing all-zero logits. Fix was FLOAT32 precision.

**Result:** Using `compute_precision=FLOAT32` for LM head and decoder prefill from
the start. No overflow issues observed. LM head produces correct logits with output
dim 5000 (timestamp values, not vocab tokens).

---

## 6. [⬜] ANE Compilation Time

**Context:** Kokoro TTS showed ANE compilation taking 60-90s for 15s models.
The ForcedAligner decoder (28 layers, 1024 hidden) is a large model.

**Risk:** First-run ANE compilation could take minutes. Need to measure and document.

**Status:** Not yet measured.

---

## 7. [✅] End-to-End Parity Verification

**Context:** Need to verify that the full CoreML pipeline (audio conv → audio transformer →
embedding → decoder prefill → LM head → timestamp extraction) produces correct timestamps
compared to PyTorch reference.

**Method:** Ran 3 LibriSpeech test-clean samples through both PyTorch `Qwen3ForcedAligner`
and the 5-model CoreML pipeline. Compared per-word start/end timestamps.

**Results (54 word boundaries):**
- AAS (mean boundary error): 4.4ms
- Max error: 160ms (single position)
- Within 20ms: 95.4%
- Within 80ms (1 segment): 99.1%
- Within 160ms (2 segments): 100.0%

**Per-component analysis:**
- Decoder: max diff 0.0013 vs PyTorch (essentially identical)
- Audio conv: max diff ~0.08 per chunk (FP16 precision)
- Audio transformer: enables cross-chunk attention matching native behavior
- LM head cos/sin: max diff 1.5e-5 (essentially identical)

**Conclusion:** Conversion is functionally correct with near-identical timestamps.
The remaining few mismatches (< 5%) are at the model's resolution limit (80ms) and
are caused by accumulated FP16 precision differences in the audio conv.

**Status:** Resolved. Parity is excellent for production use.

---

## 8. [✅] Audio Encoder Last Chunk Padding

**Context:** When mel length is not a multiple of 100, the last chunk needs to be
padded to 100 for the fixed-shape CoreML model. But the conv output for a padded
chunk produces spurious frames.

**Example:** Last chunk of 62 mel frames → 8 real output frames. Padded to 100 frames
→ 13 output frames → 5 extra spurious frames.

**Fix:** Calculate expected output frames per chunk using the conv stride formula
`out = (in - 1) // 2 + 1` applied 3 times, then trim the conv output to the
expected count.

**Status:** Resolved.
