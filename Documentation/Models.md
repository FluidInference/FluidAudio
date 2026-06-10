# Models

A guide to each CoreML model pipeline in FluidAudio.

## ASR Models

### Sliding-Window Transcription (Near Real-Time)

Long-form audio processed via `SlidingWindowAsrManager` — chunked, overlapped, and stitched. Distinct from the **Streaming Transcription** section below, which uses cache-aware encoders that emit partials as audio arrives.

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet TDT v2** | Batch speech-to-text, English only (0.6B params). TDT architecture. | First ASR model added. |
| **Parakeet TDT v3** | Batch speech-to-text, 25 European languages (0.6B params). Default ASR model. | Released after v2 to add multilingual support. |
| **Parakeet TDT-CTC-110M** | Hybrid TDT-CTC batch model (110M params). 3.01% WER on LibriSpeech test-clean. 96.5x RTFx on M2 Mac. Fused preprocessor+encoder for reduced memory footprint. iOS compatible. | Smaller, faster alternative to v3 with competitive accuracy. |
| **Parakeet TDT Japanese** | Batch speech-to-text, Japanese only (0.6B params). Hybrid model: INT8 CTC-trained preprocessor + encoder paired with a TDT decoder + joint. 6.85% CER on JSUT, 10.8x RTFx on M2. | CTC-only Japanese inference was removed in 846924a1d; only the preprocessor + encoder from the original CTC repo are reused. |
| **Cohere Transcribe** ([FluidAudio#487](https://github.com/FluidInference/FluidAudio/pull/487), [#537](https://github.com/FluidInference/FluidAudio/pull/537)) | Batch encoder-decoder speech-to-text, 14 languages (en/fr/de/es/it/pt/nl/pl/el/ar/ja/zh/ko/vi). 48-layer Conformer encoder + 8-layer transformer decoder with external KV cache. Mixed precision: INT8 encoder (1.8 GB, iOS 18+) + FP32 ANE-resident static-shape decoder (v2, ~1.6× faster on Apple Silicon than the dynamic FP16 v1 decoder). Hard 35 s per-call audio cap (`max_audio_clip_s` from upstream config), 16 384-token SentencePiece vocab. Language must be passed explicitly via the conditioned prompt. | First Cohere Transcribe port; ANE-optimized v2 decoder (#537) lands fixed `[1, 1, 1, 108]` `attention_mask` so the decoder stays on the Neural Engine. |
| **SenseVoiceSmall** (FunASR) | Non-autoregressive multilingual batch speech-to-text (50+ languages). SANM encoder + single CTC head. 3-stage pipeline: fp32 CPU preprocessor (waveform → 560-d LFR features) → fp16 ANE encoder+CTC (with fp32 fallback) → host greedy-CTC decode → SentencePiece detokenize, stripping the leading `<\|lang\|><\|emo\|><\|event\|><\|itn\|>` tags. INT8 encoder variant available. Language is auto-detected by default (`language = 0`). Managed by `SenseVoiceManager`. | Fast non-autoregressive alternative for broad language coverage. |
| **Paraformer-large (zh)** | Non-autoregressive Mandarin Chinese batch speech-to-text. SANM encoder + CIF predictor (host-side integrate-and-fire) + parallel decoder. INT8 encoder/decoder variants available. Managed by `ParaformerManager`. | Non-autoregressive Mandarin model; emits all tokens in parallel rather than one-at-a-time. |

TDT/CTC and the non-autoregressive models above are wrapped by `SlidingWindowAsrManager`, which chunks audio (~15s with overlap) and stitches the per-chunk transcripts.

### Streaming Transcription (True Real-Time)

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet EOU** | Streaming speech-to-text with end-of-utterance detection (120M params). Three chunk-size variants — 160ms / 320ms / 1280ms — for ultra-low-latency to higher-accuracy streaming. | Added after TDT was released & for streaming. Smaller model (120M vs 0.6B). |
| **Nemotron Speech Streaming 0.6B** | RNNT streaming ASR with 3 chunk-size variants — 560ms / 1120ms / 2240ms (2240ms is the default). English-only (0.6B params). Int8 encoder quantization. Trades latency for accuracy across tiers. Managed by `StreamingNemotronAsrManager`. | Larger streaming model for better accuracy and quality |
| **Nemotron Speech Streaming Multilingual 0.6B** | RNNT streaming ASR, multilingual (en/es/fr/it/pt/de/zh/ja) with an `auto` language-detection mode. 0.6B params, 4 chunk-size tiers — 560ms / 1120ms / 2240ms / 4480ms (2240ms recommended). The HF repo is organized as `<lang>/<tier>ms/` subfolders; the variant subdirectory is selected dynamically at download time. Managed by `StreamingNemotronMultilingualAsrManager`. | Multilingual counterpart to the English-only streaming model. |

### Custom Vocabulary / Keyword Spotting

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet CTC 110M** | CTC-based encoder for custom keyword spotting. Runs rescoring alongside TDT to boost domain-specific terms (names, jargon). | |
| **Parakeet CTC 0.6B** | Larger CTC variant (same role as 110M) with better quality |  |

## VAD Models

| Model | Description | Context |
|-------|-------------|---------|
| **Silero VAD** | Voice activity detection; speech vs silence on 256ms windows. Segments audio before ASR or diarization. | Support model that other pipelines build on. Converted at the time being the best model out there |

## Diarization Models

| Model | Description | Context |
|-------|-------------|---------|
| **LS-EEND** | Research prototype end-to-end streaming diarization model from Westlake University. Supports both streaming and complete-buffer inference for up to 10 speakers. Uses frame-in, frame-out processing, requiring 900ms of warmup audio and 100ms per update. | Added after Sortformer to support largers speaker counts. |
| **Sortformer** | NVIDIA's enterprise-grade end-to-end streaming diarization model. Supports both streaming and complete-buffer inference for up to 4 speakers. More stable than LS-EEND, but sometimes misses speech. Processes audio in chunks, requiring 1040ms of warmup audio and 480ms per update for the low latency versions. | Added after Pyannote to support low-latency streaming diarization. |
| **Pyannote CoreML Pipeline** | Speaker diarization. Segmentation model + WeSpeaker embeddings for clustering. Online/streaming pipeline (DiarizerManager) based on pyannote/speaker-diarization-3.1. Offline batch pipeline (OfflineDiarizerManager) based on pyannote/speaker-diarization-community-1. | First diarizer model added. Converted from Pyannote with custom made batching mode |

## TTS Models

| Model | Description | Context |
|-------|-------------|---------|
| **Kokoro ANE (7-stage)** | Kokoro 82M weights split into 7 CoreML stages so the ANE-friendly layers (Albert / Prosody / Vocoder) stay resident on the Neural Engine while PostAlbert / Alignment / Noise / Tail run on CPU. 3-11× RTFx. English (`af_heart`) and Mandarin (`ANE-zh`) variants. ≤510 IPA phonemes per call, no chunker / SSML / custom lexicon. Managed by `KokoroAneManager`. | ANE-optimized variant derived (with permission) from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml). The original single-graph (mono) Kokoro backend was removed in favor of this ANE pipeline; the kokoro repo root is now retained only for shared G2P assets. |
| **PocketTTS** | TTS backend (~155M params). Autoregressive frame-by-frame generation with dynamic audio chunking. No phoneme stage, works directly on text tokens. Managed by `PocketTtsManager`. | Supports streaming, minimal RAM usage, excellent quality |
| **Supertonic-3** | Multilingual TTS, 31 languages (`en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`, plus `na` for numeric/language-agnostic input). 4-stage CoreML pipeline (text_encoder → duration_predictor → vector_estimator → vocoder, ~398 MB). Caller-supplied voice styles loaded from Supertonic preset JSON. 44.1 kHz mono fp32 output. Managed by `Supertonic3Manager`. | CoreML conversion of upstream `Supertone/supertonic-3`; see `Scripts/convert_supertonic3_to_coreml.py`. |
| **StyleTTS2 (LibriTTS, iteration_3)** | Reference-audio–driven zero-shot English TTS. 8-stage CoreML pipeline (`text_encoder → bert → ref_encoder → fused_diffusion_sampler → duration_predictor → fused_f0n_har_source → decoder_pre → decoder_upsample`) with 3 lazily-loaded T = 64 / 128 / 256 bucket variants of `bert` / `fused_diffusion_sampler`. 5-step ADPM2 Karras-σ diffusion sampler with α/β style blending against a speaker reference clip. 24 kHz mono fp32 output. Phonemizer reuses Kokoro's Misaki lexicon cache + BART G2P CoreML model with Misaki uppercase diphthong shorthand (`A O I Y W` → `eɪ oʊ aɪ ɔɪ aʊ`) expanded before encoding so the output matches the espeak IPA the model was trained on. Callers with a higher-quality phonemizer can bypass the stack via `StyleTTS2Manager.synthesize(ipa:...)`. See [StyleTTS2.md](TTS/StyleTTS2.md). | Zero-shot voice cloning from a single reference WAV; English only |

## Evaluated Models (Not Supported)

Models we converted and tested but are not supported: too large for on-device deployment, limitations or superseded by better approaches.

| Model | Status |
|-------|--------|
| **KittenTTS** ([FluidAudio#409](https://github.com/FluidInference/FluidAudio/pull/409), [HF](https://huggingface.co/alexwengg/kittentts-coreml)) | Not supported due to inefficient espeak alternatives. Nano (15M) and Mini (82M) variants. |
| **Qwen3-TTS** ([FluidAudio#290](https://github.com/FluidInference/FluidAudio/pull/290), [mobius#20](https://github.com/FluidInference/mobius/pull/20), [HF](https://huggingface.co/alexwengg/qwen3-tts-coreml)) | Now 1.1GB but too slow. Needs further testing. |
| **Qwen3-ForcedAligner-0.6B** ([FluidAudio#315](https://github.com/FluidInference/FluidAudio/pull/315), [mobius#21](https://github.com/FluidInference/mobius/pull/21), [HF](https://huggingface.co/alexwengg/Qwen3-ForcedAligner-0.6B-Coreml)) | 5-model CoreML pipeline, large footprint. Low upstream adoption (Qwen ASR CoreML model downloads). |

## Model Sources

| Model | HuggingFace Repo |
|-------|-----------------|
| Parakeet TDT v3 | [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) |
| Parakeet TDT v2 | [FluidInference/parakeet-tdt-0.6b-v2-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml) |
| Parakeet TDT-CTC-110M | [FluidInference/parakeet-tdt-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-tdt-ctc-110m-coreml) |
| Parakeet TDT Japanese | [FluidInference/parakeet-0.6b-ja-coreml](https://huggingface.co/FluidInference/parakeet-0.6b-ja-coreml) (hybrid: CTC preprocessor/encoder + TDT decoder/joint) |
| SenseVoiceSmall | [FluidInference/sensevoice-small-coreml](https://huggingface.co/FluidInference/sensevoice-small-coreml) |
| Paraformer-large (zh) | [FluidInference/paraformer-large-zh-coreml](https://huggingface.co/FluidInference/paraformer-large-zh-coreml) |
| Parakeet CTC 110M | [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) |
| Parakeet CTC 0.6B | [FluidInference/parakeet-ctc-0.6b-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-coreml) |
| Parakeet EOU | [FluidInference/parakeet-realtime-eou-120m-coreml](https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml) (subdirs: `/160ms`, `/320ms`, `/1280ms`) |
| Cohere Transcribe (INT8 hybrid, default) | [FluidInference/cohere-transcribe-03-2026-coreml](https://huggingface.co/FluidInference/cohere-transcribe-03-2026-coreml) (variant: `/q8`) |
| Silero VAD | [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml) |
| Diarization (Pyannote) | [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml) |
| LS-EEND | [FluidInference/ls-eend-coreml](https://huggingface.co/FluidInference/ls-eend-coreml) (per-dataset optimized variants: `/optimized/ami`, `/optimized/ch`, `/optimized/dih2`, `/optimized/dih3`) |
| Sortformer | [FluidInference/diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml) |
| Kokoro ANE (7-stage) | [FluidInference/kokoro-82m-coreml/tree/main/ANE](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE) (English: `/ANE`; Mandarin: `/ANE-zh`; shared G2P assets `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`, `g2p_vocab.json` at the repo root) |
| PocketTTS | [FluidInference/pocket-tts-coreml](https://huggingface.co/FluidInference/pocket-tts-coreml) |
| StyleTTS2 (LibriTTS, iteration_3) | [FluidInference/StyleTTS-2-coreml/iteration_3/compiled](https://huggingface.co/FluidInference/StyleTTS-2-coreml/tree/main/iteration_3/compiled) (shared phonemizer assets pulled from [`FluidInference/kokoro-82m-coreml`](https://huggingface.co/FluidInference/kokoro-82m-coreml): `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`, `g2p_vocab.json`, `us_lexicon_cache.json`) |
| Supertonic-3 | [FluidInference/supertonic-3-coreml](https://huggingface.co/FluidInference/supertonic-3-coreml) |
| Nemotron Streaming (English) | [FluidInference/nemotron-speech-streaming-en-0.6b-coreml](https://huggingface.co/FluidInference/nemotron-speech-streaming-en-0.6b-coreml) (subdirs: `/560ms`, `/1120ms`, `/2240ms`) |
| Nemotron Streaming (Multilingual) | [FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML](https://huggingface.co/FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML) (`<lang>/<tier>ms/` subfolders) |
| Multilingual G2P (Charsiu ByT5) | [FluidInference/charsiu-g2p-byt5-coreml](https://huggingface.co/FluidInference/charsiu-g2p-byt5-coreml) |
