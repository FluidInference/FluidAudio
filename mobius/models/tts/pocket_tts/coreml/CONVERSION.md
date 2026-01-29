# PocketTTS CoreML Conversion Guide

How the PocketTTS PyTorch model is converted to a pure CoreML pipeline with zero PyTorch runtime dependency.

---

## Architecture Overview

PocketTTS uses a **flow-matching language model** architecture:

```
Text → Tokenize → Embed → [Voice Embed] → Transformer (KV Cache) → Flow Decode → Mimi Decode → Audio
```

The pipeline is split into **4 CoreML models** plus numpy/sentencepiece for preprocessing:

```
┌─────────────────────────────────────────────────────────────┐
│                    Pure Python (no PyTorch)                  │
│  SentencePiece tokenization → numpy embedding lookup        │
│  safetensors voice loading → numpy array ops                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  cond_step.mlpackage          (KV cache prefill)            │
│  Input:  1 conditioning token [1, 1, 1024]                  │
│  Output: updated KV caches [2, 1, 200, 16, 64] x6          │
│  Called: 141 times (125 voice + 16 text tokens)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  flowlm_step.mlpackage       (autoregressive generation)    │
│  Input:  latent frame [1, 1, 32] + KV caches               │
│  Output: transformer_out [1, 1, 1024] + EOS logit + caches │
│  Called: ~20-50 times per utterance                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  flow_decoder.mlpackage   (flow decoding, 8 LSD steps)  │
│  Input:  transformer_out [1, 1024] + latent [1, 32] + s, t │
│  Output: velocity [1, 32]                                   │
│  Called: 8 times per generation step                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  mimi_decoder.mlpackage   (audio synthesis)              │
│  Input:  quantized latent [1, 512, 1] + streaming state     │
│  Output: audio frame [1, 1, 1920] (80ms at 24kHz)           │
│  Called: once per generation step                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                   audio.wav
```

---

## Model Conversion Process

Each model follows the same pattern:
1. Load original PocketTTS PyTorch weights
2. Create a `Traceable*` wrapper with explicit inputs (no dynamic control flow)
3. `torch.jit.trace` the wrapper
4. `coremltools.convert` to `.mlpackage`

### 1. Conditioning Step (`cond_step.mlpackage`)

**Source:** `traceable_cond_step.py` → `convert_cond_step.py`

Processes one conditioning token (text or voice embedding) through the 6-layer transformer, updating the KV cache. This is the **prefill** stage — called once per conditioning token before generation begins.

**Why a separate model?** The original model processes all conditioning at once with variable-length input. CoreML's scatter op doesn't support dynamic shapes, so we process one token at a time with a fixed-shape cache.

**Architecture:**
- 6 transformer layers (same weights as the generation step model)
- No `input_linear` — conditioning is already 1024d
- No BOS handling, no EOS output
- Streaming attention with circular KV cache (max 200 positions)
- RoPE positional encoding

**Inputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `conditioning` | `[1, 1, 1024]` | One pre-embedded conditioning token |
| `cache0`–`cache5` | `[2, 1, 200, 16, 64]` | KV caches (key + value) per layer |
| `position0`–`position5` | `[1]` | Current write position per layer |

**Outputs:** Updated caches and positions (12 tensors total).

### 2. Generation Step (`flowlm_step.mlpackage`)

**Source:** `traceable_flowlm_step.py` → `convert_flowlm_step.py`

The autoregressive backbone. Takes one latent frame, runs it through the transformer (attending to the prefilled KV cache), and outputs the transformer hidden state + EOS logit.

**Architecture:**
- `input_linear`: projects 32d latent → 1024d
- 6 transformer layers (shared weights with cond_step)
- `out_norm` + `out_eos`: EOS prediction head
- NaN input → BOS embedding substitution (first frame)
- Streaming attention with circular KV cache

**Inputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `sequence` | `[1, 1, 32]` | Latent frame (NaN for BOS) |
| `bos_emb` | `[32]` | BOS embedding constant |
| `cache0`–`cache5` | `[2, 1, 200, 16, 64]` | KV caches |
| `position0`–`position5` | `[1]` | Current positions |

**Outputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `input` | `[1, 1, 1024]` | Transformer hidden state |
| EOS logit | `[1, 1, 1]` | EOS probability (threshold: -4.0) |
| Updated caches/positions | | 12 tensors |

### 3. Flow Decoder (`flow_decoder.mlpackage`)

**Source:** `traceable_flow_decoder.py` → `convert_flow_decoder.py`

Lagrangian Self-Distillation (LSD) flow decoder. Converts transformer output to a 32d audio latent via 8 iterative denoising steps.

**Architecture:**
- `SimpleMLPAdaLN` network conditioned on transformer output and time
- Takes **two** time inputs: start time `s` and end time `t`
- Averages time embeddings: `time_emb = (embed(s) + embed(t)) / 2`

**Critical detail:** Both `s` and `t` must be passed correctly:
```
Step 0: s=0.000, t=0.125
Step 1: s=0.125, t=0.250
...
Step 7: s=0.875, t=1.000
```

**Inputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `transformer_out` | `[1, 1024]` | Hidden state from generation step |
| `latent` | `[1, 32]` | Current noise/latent |
| `s` | `[1, 1]` | Start time |
| `t` | `[1, 1]` | End time |

**Output:** Velocity field `[1, 32]`. Applied as: `latent = latent + velocity * dt`

### 4. Mimi Decoder (`mimi_decoder.mlpackage`)

**Source:** Converted separately (streaming convolutional decoder).

Converts quantized latent `[1, 512, 1]` to audio `[1, 1, 1920]` (80ms at 24kHz). Uses streaming state for causal convolutions and attention.

**State:** 26 tensors tracking convolutional history, attention caches, and partial upsampling buffers. Initialized from `mimi_init_state.npz`.

---

## Constants Export

**Script:** `export_constants.py`

Extracts model constants from PyTorch and saves as numpy files. This is a one-time operation requiring PyTorch.

| File | Shape | Description |
|------|-------|-------------|
| `bos_emb.npy` | `[32]` | Beginning-of-sequence embedding |
| `emb_mean.npy` | `[32]` | Latent normalization mean |
| `emb_std.npy` | `[32]` | Latent normalization std |
| `quantizer_weight.npy` | `[512, 32, 1]` | Quantizer projection (1D conv kernel) |
| `text_embed_table.npy` | `[4001, 1024]` | Text token embedding table |
| `mimi_init_state.npz` | 26 tensors | Mimi decoder initial streaming state |
| `tokenizer.model` | — | SentencePiece tokenizer (copied from HuggingFace) |
| `alba.safetensors` | `[1, 125, 1024]` | Pre-encoded voice conditioning |

Total: ~28 MB

---

## Generation Pipeline (`generate_coreml_v4.py`)

Zero PyTorch dependency. Imports: `numpy`, `sentencepiece`, `safetensors`, `coremltools`, `scipy`.

### Step-by-step:

```
1. Text preparation (string ops)
   - Normalize whitespace, capitalize first letter, add period
   - Pad short texts with spaces for better prosody
   → prepared_text, frames_after_eos

2. Tokenize (SentencePiece)
   sp.encode(prepared_text) → token_ids [N]

3. Embed text (numpy lookup)
   text_embed_table[token_ids] → text_emb [1, 16, 1024]

4. Load voice (safetensors)
   load_file("alba.safetensors")["audio_prompt"] → voice_emb [1, 125, 1024]

5. Combine conditioning (voice FIRST, then text)
   concat(voice_emb, text_emb) → combined [1, 141, 1024]

6. KV cache prefill (cond_step.mlpackage × 141)
   For each token in combined:
     cond_step.predict(token, caches, positions) → updated caches
   → positions now at 141

7. Autoregressive generation loop:
   a. flowlm_step.predict(latent, bos_emb, caches, positions)
      → transformer_out [1, 1, 1024], eos_logit, updated caches

   b. Check EOS (logit > -4.0 → stop after frames_after_eos extra frames)

   c. Flow decode (flow_decoder × 8 LSD steps):
      latent = randn(1, 32) * sqrt(0.7)
      for i in 0..7:
        velocity = flow_decoder.predict(transformer_out, latent, s=i/8, t=(i+1)/8)
        latent += velocity * (1/8)

   d. Denormalize: latent * emb_std + emb_mean

   e. Quantize: dot(latent, quantizer_weight.T) → [1, 512, 1]

   f. Mimi decode: mimi_decoder.predict(quantized, state) → audio [1, 1, 1920]

   g. Update sequence for next step: latent reshaped to [1, 1, 32]

8. Concatenate audio frames, normalize, save as WAV (24kHz, 16-bit)
```

### Critical ordering detail

Conditioning must be **voice first, then text** — matching the original model's processing order:
1. `get_state_for_audio_prompt()` fills positions 0–124 with voice
2. `_run_flow_lm_and_increment_step()` fills positions 125–140 with text

Reversing this order (text first) produces cosine similarity of only 0.6–0.9 in the KV cache vs the reference, causing unintelligible output.

---

## Disk & Memory Footprint

| Model | Disk | RAM (loaded) |
|-------|------|-------------|
| flowlm_step | 289 MB | ~50 MB |
| cond_step | 253 MB | ~45 MB |
| flow_decoder | 37 MB | ~7 MB |
| mimi_decoder | 20 MB | ~5 MB |
| Constants | 28 MB | ~28 MB |
| **Total** | **627 MB** | **~135 MB** |

Note: cond_step and flowlm_step share the same transformer weights (289 MB each on disk) but are separate CoreML models because they have different input/output signatures.

---

## Environment Setup

The conversion scripts require the upstream PocketTTS PyTorch package. It's installed as an editable dependency via `uv`.

```bash
cd models/tts/pocket_tts

# Clone the upstream PocketTTS repo (provides the pocket_tts/ Python package)
git clone https://github.com/kyutai-labs/pocket-tts.git pocket_tts_upstream
cp -r pocket_tts_upstream/pocket_tts ./pocket_tts
rm -rf pocket_tts_upstream

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync --extra coreml

# Verify
.venv/bin/python -c "from pocket_tts import TTSModel; print('OK')"
```

The `.python-version` file pins Python 3.10 (required for `pocket_tts` type syntax).

## Reproducing the Conversion

Requires PyTorch (one-time only):

```bash
# 1. Export constants
.venv/bin/python coreml/convert_assets/export_constants.py

# 2. Convert models (each creates an .mlpackage)
.venv/bin/python coreml/convert_models/convert/convert_cond_step.py
.venv/bin/python coreml/convert_models/convert/convert_flowlm_step.py
.venv/bin/python coreml/convert_models/convert/convert_flow_decoder.py
# mimi_decoder requires a custom functional wrapper (see convert_mimi_decoder.py)

# 3. Run generation (no PyTorch needed after conversion)
.venv/bin/python coreml/generate_coreml_v4.py
```

---

## Porting to Swift

All components map to native Swift/Apple frameworks:

| Python | Swift |
|--------|-------|
| SentencePiece | swift-sentencepiece or bundled C lib |
| numpy array ops | `[Float]` + Accelerate/vDSP |
| safetensors | Simple binary parser (~50 lines) |
| coremltools inference | `MLModel.prediction(from:)` (native, faster) |
| scipy WAV write | AVFoundation |

The 4 `.mlpackage` files and constants bundle directly into an Xcode project.
