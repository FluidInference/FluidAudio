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
7. **Dispatch isn't always the bottleneck.** A bit-exact `make_pipeline` of
   the EOU decoder+joint measured 0% faster — per-stage execution overhead,
   not host dispatch, was the cost. Only true graph fusion (single MIL
   program) paid off, and at fp16 it is NOT bit-exact: E5RT kernel selection
   below MIL control flips near-tie tokens (bit-exact + fast was proven
   structurally unattainable). Fusion speedups therefore need a WER gate,
   not just tensor parity.

## Ranked candidates

### 1. Nemotron Multilingual `decoder_joint` — RESOLVED 2026-06-10 (LSTM-gated; 54% is the ceiling)
- Was: **54% ANE / 46% CPU** — the only mixed-placement graph in the
  profiler doc. 79 ms/utt (168 calls), the biggest decode cost in that
  pipeline. Sibling `joint` model is already 100% ANE.
- Diagnosis (`ane_ops` per-op dump): the CPU 46% is **exactly the RNN-T
  prediction network** — 2× `ios18.lstm` (640-hidden) plus its inseparable
  glue (embedding `gather`, h/c-state `split`/`stack`, squeezes, IO casts).
  The joint half (3× `linear` + `relu` + `add`, incl. the 640→13088 logits
  matmul) is already 100% ANE inside the fused graph — that's the 54%.
  Categorical, not construct: `ios18.lstm` has no ANE kernel at any
  precision (playbook item 4; same gate that settled Parakeet TDT v3/EOU).
  No rank-5 / scatter / masked_fill / dynamic-shape constructs present.
- Why the standalone models differ: standalone `joint` = only the
  ANE-eligible half → 100% ANE; standalone `decoder` = only the prediction
  network → 100% CPU. The fused B1 graph already achieves the best possible
  split, and B1 fusion already banked the +15% from dispatch savings.
- Side finding: the per-language `decoder_joint` variants (latin/es vocab
  2829, de vocab 796 — vs 13088 multilingual) compile **100% CPU**: once
  the logits matmul shrinks, the whole graph falls under the ANE worth-it
  threshold. Expected to be the fast outcome for those sizes (transfer-floor
  rule below); only revisit if a per-language decode shows up hot.
- Verdict: as good as it gets. Moved to Settled.

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

### 4. Parakeet EOU decode loop — RESOLVED 2026-06-09/10 (LSTM-gated; two fusions built, gate pending)
- Verdicts: decoder `ios17.lstm` (ANE dead end); joint_decision has **zero
  ANE segments under any compute-unit config** — the ~50 MB small-graph
  floor measured, not assumed. Bit-exact `make_pipeline`: 0% (see playbook
  #7).
- Two fused artifacts exist (mobius):
  - traced fusion (`feat/parakeet-decode-fusion`): 1.21× decode loop,
    parity ≤1.2e-7 — the safe default;
  - MIL-lean fusion (`feat/eou-decode-ane`): **−42%/step, 1.59× e2e**, but
    flips near-tie tokens on ~25% of utterances (WER-neutral on 50 files:
    34.85→35.04%).
- PENDING: full LibriSpeech head-to-head WER gate (rule: lean ships iff
  ΔWER ≤ +0.10 pp and no per-file blowup >20 pp; else traced wins).
- Nemotron EN's loop (67 ms/utt) is the same shape — apply the winner there.

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
| Parakeet TDT v3 Decoder + EOU decoder | CPU forever | `ios17.lstm` in both prediction networks (no ANE kernel); fusion-only campaigns 2026-06-09/10 — v3 fused 1.11×/utt; EOU traced fused 1.21× (parity ≤1.2e-7) vs MIL-lean fused 1.59× e2e (not bit-exact, WER gate pending); fp16 only (fp32 fused regresses on GPU units); see OPTIMIZATION.md on mobius feat/parakeet-decode-fusion + feat/eou-decode-ane |
| Parakeet EOU joint_decision (standalone) | CPU forever | zero ANE segments under ALL or CPU_AND_NE — 3 MB small-graph floor confirmed empirically; flat across all compute units |
| Nemotron ML `decoder_joint` | 54% ANE is the ceiling | per-op dump 2026-06-10: CPU 46% = the LSTM prediction network (2× `ios18.lstm`, no ANE kernel) + inseparable state glue; the joint half (3 linears incl. 640→13088 logits) is already 100% ANE in the fused graph. No fixable constructs. Per-language variants (vocab ≤2829) are 100% CPU — under the worth-it floor, leave alone |
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
