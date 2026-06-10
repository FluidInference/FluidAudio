# ANE Optimization Candidates

Companion to [ANE_Profiler.md](ANE_Profiler.md). Ranked backlog of models that
may still be squeezable onto the Neural Engine, written 2026-06-09 after the
PocketTTS rank-4 campaign (mobius Trials 19–23) and the Kokoro Noise routing
fix (#677). Each entry records the evidence, the applicable playbook, and the
go/no-go gate, so future work starts from data instead of rediscovery.

## The playbook (proven this cycle)

1. **ANECCompile rejections are usually construct problems, not model
   problems.** Rank-5 tensors → split/reshape to rank-4. `scatter` →
   one-hot multiply-add (exact circular-buffer semantics). `masked_fill(-inf)`
   → additive mask. Runtime trig on big tensors → precompute or hoist.
   (Trials 19/20: flowlm 0%→100% ANE, cond_prefill 0%→92%, bit-identical.)
2. **Hoist layer-invariant scalar math out of layer loops** — per-layer
   rebuilds force CPU↔ANE partition transitions (17→9.6 ms on cond_prefill).
3. **Fuse host-side loops into the graph** (Trial 16/21, Nemotron B1):
   fat graphs get accepted where tiny per-step kernels are rejected.
4. **Categorical dead ends:** `ios17.lstm` (no ANE kernel at any precision),
   fp32-by-necessity math (ANE is fp16-only; e.g. Kokoro Noise's
   `sin(cumsum)` phase), genuinely compute-bound stages (PocketTTS mimi).
5. **Always check routing before converting** — Kokoro Noise was 2–3× slower
   purely from a compute-units preset, fixed in one line (#677).
6. **Placement ≠ speed.** `.all` often prefers GPU on M-series; ANE residency
   is a host policy choice. Interleaved A/B (mobius Trial 15 methodology)
   before shipping, always. NaN inputs are mangled by the ANE before `isnan`
   evaluates — never use NaN protocols in ANE-bound graphs.

## Ranked candidates

### 1. Nemotron Multilingual `decoder_joint` — strongest signal
- Today: **54% ANE / 46% CPU** — the only mixed-placement graph in the
  profiler doc. 79 ms/utt (168 calls), the biggest decode cost in that
  pipeline. Sibling `joint` model is already 100% ANE.
- Action: per-op fallback dump (`ane_ops`) to identify the rejected
  constructs; expect a narrow Trial-19-style fix.
- Gate: none — diagnostic is read-only and cheap.

### 2. Pyannote segmentation — biggest non-TTS prize, binary unknown
- Today: 100% CPU, 55.4 ms/call — the heaviest warm stage in offline
  diarization (embedding is already 93% ANE).
- Action: one `ane_ops` run answers LSTM-blocked (dead end) vs
  construct-blocked (fixable). Nobody has looked.
- Note: even a "can't ANE" verdict may surface a routing win (Kokoro lesson).

### 3. Parakeet TDT v3 decoder+joint — RESOLVED 2026-06-09 (LSTM-gated; fusion built)
- Today: 100% CPU, ~32 ms/utt; the joint loop rivals the 28 ms encoder call.
- Action: (a) LSTM check on the prediction network — go/no-go gate;
  (b) B1-style decoder+joint fusion regardless (+15% precedent on Nemotron);
  (c) rank-4 rewrite if the gate passes.

### 4. Parakeet EOU decode loop — RESOLVED 2026-06-09 (LSTM-gated; fusion built, ship)
- Today: encoder 6.5 ms/chunk (97% ANE) vs decoder+joint **58 ms/utt** on CPU
  (229 steps × ~0.13 ms). Nemotron EN's loop (67 ms/utt) is the same shape.
- Action: same campaign as #3 — shared RnntDecoder machinery, same LSTM gate.
- Caveat: G2P data shows tiny AR steps sometimes run fastest on CPU; the
  interleaved A/B decides, not the placement.

### 5. Supertonic VectorEstimator loop fusion — garnish
- Today: 94% ANE but the host dispatches the 8-step denoising loop (8×3.8 ms).
- Action: Trial-16 fusion → 1 dispatch/chunk. Ceiling ~5–10% of an 81 ms
  synth; do it opportunistically, not as a project.

## Settled — do not revisit without new hardware/OS facts

| Model/stage | Verdict | Why |
|---|---|---|
| PocketTTS mimi | CPU forever | fp16 state-feedback artifacts + compute-bound + split regressed (Trials 15/22) |
| PocketTTS flowlm/prefill/flowdec | done | 100%/92%/100% ANE (Trials 19–21); MLState design shipped-ready (Trial 23, iOS18 gate) |
| Kokoro Noise | GPU (#677) | fp32 by necessity (`sin(cumsum)` phase overflows fp16); full-ANE hybrid ceiling 3–4.5% — declined |
| Kokoro PostAlbert | CPU | `ios17.lstm` has no ANE kernel; GPU planner rejects its dynamic shapes; 1.2% of synth |
| Kokoro Tail | GPU | fp32 iSTFT; ANE rejects exp/sin/iSTFT; BNNS segfaults (#667) |
| Parakeet TDT v3 Decoder + EOU decoder | CPU forever | `ios17.lstm` in both prediction networks (no ANE kernel); fusion-only campaign run 2026-06-09 — fused decoder+joint 1.11×/utt (v3) and 1.21× (EOU), fp16 only (fp32 fused regresses on GPU units); see mobius feat/parakeet-decode-fusion OPTIMIZATION.md |
| SenseVoice / Paraformer encoders | done | 97–99% ANE already |
| Silero VAD, preprocessors, G2P, Supertonic DurationPredictor | CPU | under the ~50 MB transfer-overhead floor, or measured fastest on CPU |

## Profiling debt (separate from placement work)

- Power: `powermetrics --samplers ane_power` has never been run — the
  ANE-for-power thesis is unmeasured.
- iPhone: every number is M5; placement trade-offs provably flip on A-series.
- Cold-start: systematic load+compile column per model (anecdotes only today).
- Synthetic-latency rows (TDT ja, CTC 110M) are flagged ~2× low — re-measure
  on real audio.
- PocketTTS pipelining mystery: serial == pipelined on M5 release despite
  independent engines; per-stage timers needed.
- Cohere Transcribe, FSMN-VAD (#653), CAM++ (#652): no published speed
  numbers at all.
