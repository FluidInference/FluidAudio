# Paradee (beta)

[Paradee-8M](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.3) (Sahil Mahendrakar, Apache-2.0) is Kokoro-82M
distilled into ~8M parameters: Kokoro's own modules at smaller widths, 24 kHz.
Paper: [arXiv 2610.06817](https://arxiv.org/abs/2610.06817) (v1.0).

| `ParadeeVersion` | Voices | Decoder |
|---|---|---|
| `.v13` (default) | `af_heart`, `bf_isabella`, `am_puck`, `bm_fable`; Hindi `hf_alpha`, `hm_omega`; French `ff_siwis`; Spanish `em_alex` | v1.1's frame-rate source-filter head |
| `.v12` | `af_heart`, `bf_isabella`, `am_puck`, `bm_fable` | same |
| `.v10` | `af_heart` | Kokoro's iSTFTNet generator + phase-lock filter |

The voice picks the language (`ParadeeVoice.language`): `a`/`b` American/British English, `h` Hindi, `f` French,
`e` Spanish. v1.3 is v1.2's graph with an 8-row voice table; the English voices keep rows 0–3.

Core ML models: [FluidInference/paradee-8m-coreml](https://huggingface.co/FluidInference/paradee-8m-coreml)
(v1.3 in `v1.3/int8/` 11 MB default and `v1.3/fp32/` 33 MB, plus the G2P in `v1.3/g2p/` 3.9 MB; v1.2 in `v1.2/`;
v1.0 in `int8/` and `fp32/`). Conversion of v1.0 and v1.2:
[mobius `models/tts/paradee/coreml`](https://github.com/FluidInference/mobius/tree/main/models/tts/paradee/coreml);
v1.3 and the G2P use the same scripts (the model card describes the G2P conversion).

> **Beta:** API, model artifacts and accuracy may change.

## Usage

```swift
let manager = ParadeeManager()                 // v1.3, .int8, .cpuOnly
try await manager.initialize()                 // downloads models, the Paradee G2P and English G2P assets
let samples = try await manager.synthesize(text: "Hello from FluidAudio.", voice: .bmFable, speed: 1.0, noiseSeed: 0)
let french = try await manager.synthesize(text: "Il fait très beau aujourd'hui.", voice: .ffSiwis)
let phonemes = try await manager.phonemes(for: "La biblioteca abre a las nueve.", voice: .emAlex)
let fromPhonemes = try await manager.synthesize(phonemes: phonemes, voice: .emAlex)

let v12 = ParadeeManager(version: .v12)       // the previous release, English only
```

```bash
swift run fluidaudiocli tts "Hello from FluidAudio." --backend paradee --voice bf_isabella --output out.wav
swift run fluidaudiocli tts "आज मौसम बहुत अच्छा है।" --backend paradee --voice hf_alpha --output out.wav
swift run fluidaudiocli tts "Hello." --backend paradee --variant fp32 --speed 1.2 --seed 7 --output out.wav
swift run fluidaudiocli tts "Hello." --backend paradee --paradee-version 1.2 --output out.wav
swift run fluidaudiocli tts-benchmark --backend paradee --corpus minimax-french --voice ff_siwis
```

## Pipeline

```
English:         text -> sentences -> NeMo TN -> Misaki lexicon / Paradee G2P (v1.3; BART before) -> ɾ→T, ʔ→t -> ids
Hindi/Fr/Es:     text -> sentences (also after । ॥) -> Paradee G2P (word lists, numbers, CTC model) -> ids
ParadeeText      ids [1,T], voice [1] (v1.2+)    -> duration [1,T], d [1,224,T], asr_tok [1,512,T]
host             n = max(1, round(duration / speed)); repeat columns; N(0,1) source noise [1,1,600F]
ParadeeAcoustic  en [1,224,F], asr [1,512,F], noise, voice [1] (v1.2+) -> audio [1,600F]
```

- `voice` picks a row of each half's voice table; everything else is shared between the voices.
- Text is synthesized one sentence at a time, as in the upstream package; sentences over 510 phonemes are split.
- The English frontend is KokoroAne's. Its lexicon stores misaki's raw flap `ɾ` and glottal stop `ʔ`; misaki
  rewrites them to `T` / `t` before Kokoro v1.0 sees them, and Paradee was trained only on that output, so the
  Paradee text path applies the same rewrite. Without it "kittens", "satellite" and "patterns" come out garbled.
- `noiseSeed` seeds the source noise; the same seed gives the same audio.
- Compute units: `.cpuOnly` (default) or `.cpuAndNeuralEngine`. `.all` / `.cpuAndGPU` are rejected because the
  LSTMs abort in MPSGraph (`GPURNNOps … JIT not supported`).

## Hindi, French and Spanish (v1.3)

Kokoro, and so Paradee's training data, took the phonemes for these languages from eSpeak NG, which is GPL-3.0.
Upstream 1.3.1 ([paradee#13](https://github.com/sahilmahendrakar/paradee/pull/13)) replaced it with its own
Apache-2.0 G2P, and FluidAudio runs a port of it (`ParadeeG2P`, files in `v1.3/g2p/`):

- `ParadeeG2P.mlmodelc`: a 0.76M-parameter CTC Transformer that reads one word plus the last letter of the word
  before and the first letter of the word after (French liaison, Spanish b/d/g softening, phrase-final stress).
  Upstream ships it as a dynamically quantized int8 ONNX graph; the Core ML model keeps the same int8 weights and
  computes in fp32, which is as close to eSpeak as the ONNX graph (below). Its input lengths vary per word, so it
  always runs on the CPU.
- `paradee_g2p.json`: the model's symbols, the words it gets wrong plus the 3,000 most common words per language
  (some keyed by the neighbouring letters), and number pieces from which numbers are read (lakh/crore for Hindi,
  French `vingt et un`, Spanish `veintiún mil`). `%` is spelled out; French digit groups (`12 000`) are one number.
- In v1.3 it also gives the English words missing from the Misaki lexicon (its own American and British word
  lists, then the model), as upstream does; words with digits still go to the BART G2P.
- Input is NFC-normalized. FLEURS Hindi writes some nukta letters precomposed (`ज़` U+095B, `ड़` U+095C), which the
  model has no symbol for: upstream drops the sound (`बड़ा` → `baː`), the NFC form reads it (`bˈʌr.aː`).

Phoneme character error rate against eSpeak NG on the FLEURS test transcripts (945 sentences):

| | Hindi | French | Spanish |
|---|---:|---:|---:|
| upstream 1.3.1 (int8 ONNX) | 0.71 % | 0.66 % | 0.30 % |
| FluidAudio `ParadeeG2P` | 0.63 % | 0.66 % | 0.30 % |

The Swift port gives the same phonemes as the Python reference with the same model on all 966 test lines (FLEURS
plus number, percent, guillemet and digit-group lines) and 30 English words; Hindi differs from upstream only by the
NFC step. Known upstream behaviour kept as is: a `.` or `,` inside a number splits it (`3.500` is read "tres
quinientos"), so write thousands without separators in Spanish and Hindi.

## British voices

Upstream reads `bf_isabella` and `bm_fable` with misaki's British G2P, which is what Kokoro used when it made their
training data ("bath" `bˈɑːθ`, "go" `ɡˈQ`, "schedule" `ʃˈɛdjuːl`). FluidAudio does the same:

- Lexicon: `gb_lexicon_cache.json` (misaki `gb_gold` + `gb_silver`, in the same format as `us_lexicon_cache.json`),
  from the Kokoro repo. It is downloaded the first time a British voice is used.
- The `-'s` clitic after a sibilant is `ɪz` (`boss's` → `bˈɒsɪz`), as in misaki's British `_s`.
- Words missing from the lexicon go through the BART G2P, which only knows American phonemes. Its output is
  rewritten into misaki's British inventory (`MisakiBritishFallback`): `O`→`Q`, `ᵻ`→`ɪ`, flaps → `t`, non-rhotic
  `ɹ` before a non-vowel (`ɑɹ`→`ɑː`, `ɪɹ`→`ɪə`, `əɹ`→`ə` …), `æ`→`a`, `ɑ`→`ɒ`. On the 153,873 words in both misaki
  lexicons this halves the edit distance to the British entry (1.59 → 0.78 phonemes). Misaki itself falls back to
  eSpeak `en-gb` here.

## Benchmark

`tts-benchmark --backend paradee --corpus minimax-<language>`, 100 phrases per language, Parakeet TDT round trip
(Hindi: Whisper large-v3-turbo over the same run's WAVs, nukta and punctuation folded), M5 Pro, macOS 27,
`.cpuOnly`, release build:

| Release | Voice | Corpus | Variant | WER | CER | RTFx (audio / synth) | synth p50 / p95 |
|---|---|---|---|---:|---:|---:|---:|
| v1.3 | `af_heart` | english | int8 | 1.18 % | 0.16 % | 143× | 54 / 64 ms |
| v1.3 | `bf_isabella` | english | int8 | 0.73 % | 0.17 % | 137× | 55 / 74 ms |
| v1.3 | `am_puck` | english | int8 | 1.15 % | 0.21 % | 139× | 55 / 67 ms |
| v1.3 | `bm_fable` | english | int8 | 0.82 % | 0.17 % | 141× | 52 / 63 ms |
| v1.3 | `ff_siwis` | french | int8 | 2.92 % | 0.97 % | 161× | 43 / 66 ms |
| v1.3 | `em_alex` | spanish | int8 | 1.08 % | 0.59 % | 159× | 35 / 46 ms |
| v1.3 | `hf_alpha` | hindi | int8 | 15.0 % | 6.2 % | 175× | 51 / 64 ms |
| v1.3 | `hm_omega` | hindi | int8 | 15.1 % | 6.0 % | 177× | 52 / 66 ms |
| v1.2 | `af_heart` | english | int8 | 0.93 % | 0.12 % | 108× | 70 / 85 ms |
| v1.2 | `af_heart` | english | fp32 | 0.93 % | 0.12 % | 102× | 75 / 99 ms |
| v1.2 | `bf_isabella` | english | int8 | 0.71 % | 0.18 % | 106× | 69 / 86 ms |
| v1.2 | `am_puck` | english | int8 | 1.00 % | 0.16 % | 110× | 68 / 83 ms |
| v1.2 | `bm_fable` | english | int8 | 0.84 % | 0.23 % | 97× | 74 / 104 ms |
| v1.0 | `af_heart` | english | int8 | 1.20 % | 0.14 % | 97× | 78 / 96 ms |

Synth time covers the text frontend and both models. The models alone run ~200× real time (v1.2, v1.3) and ~150×
(v1.0); the rest is the text frontend. Most of the remaining English WER is the corpus's own text ("will - power",
"date.It's", "five thousand" → "5,000") and homophones ("knights" → "nights"); v1.3 and v1.2 `af_heart` differ in 6
phrases, all ASR compounds ("lighthousekeeper", "hand crafted") either way. Hindi WER is mostly spelling variants,
high for Kokoro too (upstream: 20.7 % for Kokoro's `hf_alpha`, 21.7 % for Paradee).

Against upstream (Python, 10 inputs incl. a 28 s paragraph, each v1.2 voice): 0 duration mismatches and identical
lengths vs PyTorch with matched noise; log-mel distance to the released ONNX ≤ 0.012; Whisper WER equal to the
upstream ONNX for fp32 (3.39 %) and int8 (3.28 %). v1.3 (60 cases over the 8 voices, phonemes from upstream's own
G2P): 0 duration mismatches vs PyTorch, log-mel L1 to the released fp32 ONNX ≤ 0.011, and Whisper large-v3-turbo
WER, Core ML fp32 = ONNX fp32 in every language (English 3.09 %, Hindi 10.74 %, French 7.89 %, Spanish 5.26 %);
int8 3.09 / 10.37 / 7.02 / 5.26 % vs upstream int8 ONNX 3.09 / 12.22 / 7.89 / 5.26 %.
