# PocketTTS CoreML Conversion — Trial Log

Chronological record of all attempts, failures, and fixes to port PocketTTS from PyTorch to pure CoreML.

---

## Phase 1: Monolithic Conversion Attempts

### Trial 1 — Full model trace (`convert_pocket_tts.py`)
**Approach:** Trace the entire PocketTTS model as one CoreML model.
**Result:** Failed. The model has dynamic control flow (autoregressive loop, EOS checking, variable-length generation) that `torch.jit.trace` cannot capture. CoreML requires static compute graphs.

### Trial 2 — ONNX intermediate (`convert_via_onnx.py`)
**Approach:** Export to ONNX first, then convert ONNX to CoreML.
**Result:** Failed. Same dynamic control flow issues. ONNX export also choked on the streaming KV cache scatter operations.

### Trial 3 — Split into submodels (`convert_pocket_tts_v2.py`, `v3`, `v4`)
**Approach:** Split the pipeline into separate traceable modules:
- Text encoder
- Flow decoder
- Mimi decoder
- EOS detector

**Result:** Partial success. Individual components converted but the orchestration between them still required PyTorch for KV cache management, text preparation, and conditioning.

---

## Phase 2: Step-Based Architecture

### Trial 4 — Traceable FlowLM backbone (`traceable_flowlm.py`)
**Approach:** Create a traceable wrapper for the full transformer backbone that takes `text_embeddings` as a fixed-size input and manages KV cache internally.
**Result:** Converted successfully to `flowlm_backbone_v2.mlpackage`, but required fixed `text_embeddings` shape `[1, 100, 1024]`. This forced zero-padding for shorter inputs, which corrupted the KV cache (see Trial 7).

### Trial 5 — Flexible text_embeddings shape
**Approach:** Use `ct.RangeDim` to allow variable-length `text_embeddings` input `[1, (1-200), 1024]`.
**Result:** Failed. CoreML's `scatter_along_axis` op threw `AssertionError` with dynamic shapes. The scatter operation in the streaming KV cache requires static dimensions.

### Trial 6 — Fixed T_text=150 (`convert_flowlm.py`)
**Approach:** Fix `text_embeddings` to `[1, 150, 1024]` — large enough for any voice+text combination.
**Result:** Converted to `flowlm_backbone_v3.mlpackage`. But still required zero-padding shorter conditioning sequences.

---

## Phase 3: Flow Decoder Fix

### Trial 7 — Flow decoder time values bug
**Bug:** The `TraceableFlowDecoder` was passing wrong time values to `SimpleMLPAdaLN`:
- `s` (start time) was hardcoded to `0` instead of `i/N`
- `t` (end time) received `lsd_step * dt` (the start value) instead of `(lsd_step + 1) * dt`

**Symptom:** CoreML generation produced gibberish audio despite transformer outputs matching PyTorch exactly. The 8-step LSD flow decoding was computing wrong velocity fields at every step.

**Root cause:** `SimpleMLPAdaLN.forward(c, s, t, x)` takes TWO time conditions and averages their embeddings. With `s=0` always, the flow trajectory was wrong.

**Fix:** Updated `traceable_flow_decoder.py` to accept explicit `s` and `t` inputs:
```python
# Before (wrong):
s = torch.zeros_like(t)  # always 0
velocity = self.flow_net(transformer_out, s, t, latent)

# After (correct):
def forward(self, transformer_out, latent, s, t):
    velocity = self.flow_net(transformer_out, s, t, latent)
```

And the generation loop:
```python
# Before (wrong):
t_np = np.array([[lsd_step * dt]])  # this is s, not t!

# After (correct):
s_np = np.array([[lsd_step * dt]])
t_np = np.array([[(lsd_step + 1) * dt]])
```

**Result:** CoreML generation now produced correct speech. Whisper transcribed output as "Hello, this is Pure CoreML Text to Speech Generation." — matching PyTorch reference.

---

## Phase 4: Eliminating PyTorch from Setup

### Trial 8 — Zero-padding conditioning corruption
**Approach:** Use `flowlm_backbone_v3.mlpackage` (T_text=150) with zero-padded conditioning. Pad the 141-token conditioning sequence (125 voice + 16 text) with 9 zeros to fill the fixed 150-slot input.

**Result:** Failed. Zero-padded tokens are NOT ignored — they pass through LayerNorm (which has bias terms) and FFN layers, producing non-zero KV cache entries. The model wrote KV entries at positions 0-149 instead of 0-140, advancing the position counter to 150 instead of 141. Generation started at the wrong position, producing garbage.

**Key insight:** You cannot zero-pad conditioning tokens. Each padded token creates a real (non-zero) KV cache entry because LayerNorm bias + FFN bias transform zeros into non-zero activations.

### Trial 9 — Conditioning step model (`traceable_cond_step.py`, v1)
**Approach:** Create a separate CoreML model that processes ONE conditioning token at a time. Feed all 141 tokens sequentially, no padding needed.

**Result:** Positions now correct (141), EOS triggered at step 21. But audio quality was wrong — Whisper transcribed as "Third is... Yes." instead of expected text.

**Root cause:** The attention implementation in `traceable_cond_step.py` differed from the verified `traceable_flowlm_step.py`:

| Aspect | Step model (correct) | Cond step v1 (wrong) |
|--------|---------------------|----------------------|
| QKV split | `.reshape(B, T, 3, H, D)` slicing | `.chunk(3, dim=-1)` then `.view()` |
| NaN handling | `torch.where(isnan, zeros, keys)` | None |
| Masking | Boolean mask + `F.scaled_dot_product_attention` | Float mask + manual softmax + `-1e9` |
| RoPE | `torch.exp(ds * (-math.log(10000) * 2/D))` | `1.0 / (10000 ** (2*indices/D))` |

While mathematically equivalent, these code paths trace to different CoreML ops, and the missing NaN→0 replacement caused NaN propagation through attention.

### Trial 10 — Conditioning step model (v2, fixed attention)
**Approach:** Copy the exact `_apply_rope_tensor` and `_streaming_attention` methods from the verified `traceable_flowlm_step.py` into `traceable_cond_step.py`.

**Result:** Reconverted `cond_step.mlpackage`. Still produced "Third is... Yes."

**Root cause discovered via verification script:** The **conditioning order** was wrong.

### Trial 11 — Conditioning order fix (voice-first)
**Bug:** `generate_coreml_v4.py` concatenated `[text_emb, voice_emb]` (text first), but the original PocketTTS model processes **voice first** then text:
1. `get_state_for_audio_prompt("alba")` → fills KV cache with 125 voice tokens (positions 0-124)
2. `_run_flow_lm_and_increment_step(text_tokens)` → adds 16 text tokens (positions 125-140)

**Verification:** Wrote a comparison script that ran both orders through the PyTorch cond_step model and compared KV caches against the original model:

| Order | Max diff | Cosine sim |
|-------|----------|-----------|
| Voice-first | 0.000007 | 1.000 |
| Text-first | 4.33–9.11 | 0.60–0.90 |

**Fix:**
```python
# Before (wrong):
combined = np.concatenate([text_emb, voice_emb], axis=1)

# After (correct):
combined = np.concatenate([voice_emb, text_emb], axis=1)
```

**Result:** Correct speech output. Whisper: "Hello, this is Pure CoreML Text to Speech Generation." Duration: 3.52s, 44 frames, EOS at step 41. Zero PyTorch dependency confirmed.

---

## Summary of Bugs Found

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Flow decoder `s` hardcoded to 0 | Gibberish audio | Pass explicit `s = i/N` |
| 2 | Flow decoder `t` off by one | Gibberish audio | Pass `t = (i+1)/N` |
| 3 | Zero-padding conditioning | Wrong position (150 vs 141) | Use per-token cond_step model |
| 4 | Attention implementation mismatch | Wrong KV cache content | Copy exact code from verified step model |
| 5 | Conditioning order (text-first vs voice-first) | "Third is... Yes." | Swap to voice-first, then text |
