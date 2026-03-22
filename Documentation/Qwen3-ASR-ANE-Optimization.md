# Qwen3-ASR ANE Optimization

## Overview

Qwen3-ASR-0.6B is a 2-model CoreML pipeline for automatic speech recognition. Both models are currently loaded with `computeUnits: .all`, which allows CoreML to schedule ops on ANE where it sees fit.

## Models

| Model | Role | Architecture | Size |
|-------|------|-------------|------|
| `qwen3_asr_audio_encoder.mlmodelc` | Mel spectrogram → 1024-d audio features | Conv2D frontend (3 stride-2 layers) + 18 transformer layers + projection | ~200MB |
| `qwen3_asr_decoder_stateful.mlmodelc` | Autoregressive token generation | 28 GQA transformer layers (16 Q heads, 8 KV heads, head_dim=128) + fused LM head (1024 → 151936) | ~1.5GB |
| `qwen3_asr_embeddings.bin` | Token embedding lookup | Swift-side float16 matrix [151936 × 1024] | ~300MB |

## Baseline Profiling Results (March 2026)

Profiled with `fluidaudiocli profile --verbose`, compute units = `.all`.

### Audio Encoder

| Device | Ops | Cost % |
|--------|-----|--------|
| **ANE** | **10** | **34.8%** |
| GPU | 477 | 65.2% |
| Unknown (const) | 709 | 0.0% |

**ANE ops (34.8% of total cost):**

| Op | Cost % | Role |
|----|--------|------|
| `ios17.conv` (×3) | 24.9% | Conv2D frontend (3 stride-2 layers) |
| `ios16.gelu` (×3) | 8.6% | Activations after each conv layer |
| `ios17.transpose` | 0.7% | Reshape between conv and linear |
| `ios17.reshape` | 0.4% | Reshape between conv and linear |
| `ios17.cast` | 0.1% | Type cast |
| `ios17.expand_dims` | 0.1% | Dimension expansion |

**GPU ops (65.2% of total cost) — per transformer layer (×18):**

| Op | Cost each | Count per layer | Role |
|----|-----------|----------------|------|
| `ios17.linear` | 1.07% | 2 (FFN up/down) | FFN projections (896→3584→896) |
| `ios17.linear` | 0.45% | 1 (FFN mid) | FFN intermediate |
| `ios17.linear` | 0.07% | 4 (Q/K/V/out) | Attention projections |
| `ios16.gelu` | 0.20% | 1 | FFN activation |
| `ios17.matmul` | 0.05% | 2 (QK, attn×V) | Attention computation |
| `ios16.softmax` | 0.01% | 1 | Attention weights |
| `ios17.layer_norm` | 0.05% | 2 | Pre-attention + pre-FFN norm |
| `ios17.add` | 0.13% | 2 | Residual connections |
| `ios17.reshape` | 0.05% | 3 | Head reshaping |
| `ios17.transpose` | 0.03% | 4 | Head permutation |
| `ios17.expand_dims` | 0.05% | 3 | Dimension expansion |
| `ios17.mul` | 0.06% | 1 | Attention scaling |

**Final projection (GPU):**

| Op | Cost % | Role |
|----|--------|------|
| `ios17.linear` | 5.65% | proj2: 896→1024 output projection |
| `ios17.linear` | 0.07% | proj1: 896→896 intermediate |
| `ios16.gelu` | 0.05% | proj1 activation |
| `ios17.layer_norm` | 0.05% | ln_post |

### Decoder Stateful

| Device | Ops | Cost % |
|--------|-----|--------|
| GPU | 2717 | **100.0%** |
| Unknown (const) | 2990 | 0.0% |

All 28 GQA transformer layers + fused LM head run entirely on GPU. Zero ANE scheduling. This is expected for a T=1 autoregressive decoder with stateful KV cache — same finding as PocketTTS's cond_step and flowlm_step.

## Phase 1: Conv2d + Einsum Rewrite (March 2026)

Applied Apple's `ml-ane-transformers` patterns to the audio encoder's 18 transformer layers.

### Architecture Changes

- `nn.Linear` → `nn.Conv2d(kernel_size=1)` for all projections (Q/K/V/out, FFN fc1/fc2, conv_out, proj1, proj2)
- `(B, C, 1, S)` tensor layout throughout (S=13 frames, not T=1 streaming)
- Per-head einsum attention: `torch.einsum('bchq,bkhc->bkhq', qi, ki)` with 14 heads × 64 channels
- Manual LayerNorm on channel dimension (mean + variance over dim=1)
- No causal mask — full bidirectional encoder attention
- Separate Q/K/V Conv2d projections (with bias) instead of combined QKV
- Sinusoidal positional embedding added in `(B, C, 1, S)` format

### Key Differences from PocketTTS Attempt

| Factor | PocketTTS cond_step | Qwen3 Audio Encoder |
|--------|-------------------|-------------------|
| Sequence length | T=1 (streaming) | T=13 (fixed window) |
| KV cache | Yes (NaN-masked, 512 slots) | No |
| Causal mask | Yes (masked_fill) | No |
| Architecture | Autoregressive decoder | Batch encoder |
| ANE transfer amortization | Poor (tiny per-op data) | Good (13 frames per op) |

### Profiling Results

| Model | Before (original) | After (Conv2d + einsum) | ANE % |
|-------|-------------------|------------------------|-------|
| audio_encoder | ANE: 34.8%, GPU: 65.2% | **ANE: 100%, GPU: 0%** | **100%** |

**3941 total ops**: 1573 on ANE, 2368 const/unknown. Zero GPU ops.

### Inference Benchmark (M4 Max)

Benchmarked with Swift `MLModel` inference, 100 iterations after 10 warmup iterations. Input: random float16 mel spectrogram (1, 128, 100).

| Metric | Original (GPU+ANE) | ANE 100% |
|--------|-------------------|----------|
| Mean | 12.10 ms | 8.53 ms |
| Median | 11.61 ms | 7.60 ms |
| P95 | 16.79 ms | 9.51 ms |
| Min | 9.74 ms | 6.84 ms |

**Speedup (median): 1.53x — ANE version is 52.8% faster**

### End-to-End Validation (LibriSpeech test-clean, 10 files)

Transcribed 10 LibriSpeech test-clean files (~70s total audio) with the full Qwen3 ASR pipeline using both the original and ANE-optimized audio encoder.

| Metric | Original | ANE |
|--------|----------|-----|
| Overall RTFx | 2.47x | 2.47x |
| Median RTFx | 2.90x | 2.90x |
| Average WER | 1.5% | 1.5% |
| Average CER | 0.6% | 0.6% |

- 9/10 files produced **identical transcriptions**
- 1 file had a minor spelling difference ("alms" vs "alm's") — not a quality regression
- End-to-end speed is dominated by the decoder (28 GQA layers), so the 1.53x encoder speedup has modest pipeline impact

### Numerical Verification

Verified against the original audio encoder with random mel input (1, 128, 100):
- Max absolute difference: **2.61e-07** (within float32 tolerance)
- Mean absolute difference: 3.70e-08
- Output shape matches: (1, 13, 1024)

### Why It Worked (vs. PocketTTS failure)

The Qwen3 audio encoder is a fundamentally different use case than PocketTTS's streaming transformers:

1. **Batch encoder, not streaming decoder**: The encoder processes T=13 frames simultaneously with no autoregressive loop. ANE can parallelize across the sequence dimension.

2. **No KV cache**: PocketTTS's cond_step had complex NaN-masked KV cache operations (tile, mul, add, sub) that pulled the attention subgraph to GPU. The encoder has pure attention with no state.

3. **No causal mask**: PocketTTS used `masked_fill(~mask, -inf)` which isn't ANE-friendly. The encoder uses full bidirectional attention — no mask needed.

4. **Sufficient sequence length**: T=13 gives ANE enough work per attention op to amortize the ANE↔host data transfer overhead. T=1 was too small.

5. **coremltools einsum decomposition still happens**: The einsum is still decomposed into transpose + matmul MIL ops. But without KV cache ops forcing the graph to GPU, CoreML's scheduler keeps the entire computation on ANE.

---

## Analysis

### Audio Encoder — Fully on ANE

After the Conv2d + einsum rewrite, all 1573 compute ops run on ANE (100% of cost). This includes:
- Conv2D frontend (3 stride-2 layers) — was already on ANE
- All 18 transformer layers (attention + FFN) — moved from GPU to ANE
- Post-transformer projection (ln_post → proj1 → GELU → proj2) — moved from GPU to ANE

### Decoder Stateful — GPU is correct

The decoder is a T=1 autoregressive transformer with 28 layers, GQA (8 KV heads → 16 Q heads via repeat_interleave), and a stateful KV cache. This maps poorly to ANE for the same reasons as PocketTTS:
1. T=1 streaming is too small for ANE to amortize transfer overhead
2. Stateful operations (KV cache concat) are GPU-managed
3. The decoder runs per output token (hundreds of iterations) — GPU is efficient for this pattern

## Variant Comparison (10 LibriSpeech test-clean files, M4 Max)

| Variant | Encoder RAM | Decoder RAM | Embeds RAM | **Total RAM** | Overall RTFx | Median RTFx | WER |
|---------|------------|-------------|-----------|-----------|-------------|-------------|-----|
| f32 | 410 MB | 988 MB | 391 MB | **1790 MB** | 3.0x | 3.5x | 0.7% |
| f32-ane | 100 MB | 988 MB | 391 MB | ~1480 MB | 3.1x | 3.5x | 0.7% |
| int8 | 395 MB | 384 MB | 296 MB | **1076 MB** | 2.9x | 3.2x | 0.7% |
| int8-ane (fp16 enc) | 100 MB | 330 MB | 296 MB | **728 MB** | 2.8x | 3.4x | 0.7% |
| int4 | 401 MB | 53 MB | 296 MB | **751 MB** | 2.7x | 3.3x | 0.7% |

All variants produce identical transcriptions on all 10 files. The encoder was never quantized in the original int8/int4 variants — only the decoder was. The ANE encoder (Conv2d + einsum rewrite with fp16 precision) reduces encoder RAM from ~400 MB to 100 MB.

### Int8 Encoder Quantization

Tested int8 quantization on the ANE encoder: same WER (0.7%), same RTFx (~2.9x), same RAM (727 MB). The only difference is half the encoder download size (179 MB vs 356 MB on disk). The encoder is already small relative to the decoder + embeddings, so quantizing it doesn't move the needle on total RAM.

## Deployment

The ANE encoder is uploaded to HuggingFace as `qwen3_asr_audio_encoder_v2` in both `f32/` and `int8/` variants. The original encoder remains available for backward compatibility.

## Next Steps

1. ~~**Integrate the ANE audio encoder into FluidAudio**~~ — Done. Uploaded as `qwen3_asr_audio_encoder_v2` to HuggingFace, Swift code updated to load v2.
2. ~~**Measure actual inference speed**~~ — Done. 1.53x encoder speedup confirmed on M4 Max (11.61ms → 7.60ms median).
3. **Test on different Apple Silicon** — Profile on M1/M2/M3/M4 to verify ANE scheduling is consistent
4. **Decoder optimization** — Not recommended for the Conv2d/einsum approach (T=1 limitation). ANEMLL-style full rewrite would be needed to move the decoder to ANE.

## Files Created

### ANE Audio Encoder
- `mobius/models/stt/qwen3-asr-0.6b/coreml/traceable_audio_encoder_ane.py` — Conv2d + einsum audio encoder wrapper
- `mobius/models/stt/qwen3-asr-0.6b/coreml/convert_audio_encoder_ane.py` — Conversion script
- `mobius/models/stt/qwen3-asr-0.6b/coreml/qwen3_asr_audio_encoder_ane.mlpackage` — Converted model
- `mobius/models/stt/qwen3-asr-0.6b/coreml/qwen3_asr_audio_encoder_ane.mlmodelc` — Compiled model (100% ANE)
- `mobius/models/stt/qwen3-asr-0.6b/coreml/benchmark_encoder.swift` — Swift benchmark script (original vs ANE latency)

## Key Source Files

- `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrModels.swift` — model loading, compute unit config
- `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrManager.swift` — manager interface
- `mobius/models/stt/qwen3-asr-0.6b/coreml/convert-qwen3-asr.py` — original conversion script
- `mobius/models/stt/qwen3-asr-0.6b/coreml/individual_components.py` — PyTorch wrappers

## HuggingFace Repos

- `FluidInference/qwen3-asr-0.6b-coreml/f32` — full precision
- `FluidInference/qwen3-asr-0.6b-coreml/int8` — int8 quantized
