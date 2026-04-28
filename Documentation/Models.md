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
| **Parakeet CTC Chinese** | Batch speech-to-text, Mandarin Chinese (0.6B params). CTC architecture. 8.37% mean CER on THCHS-30 dataset. Int8 encoder (0.55GB) or FP32 (1.1GB). | First Mandarin Chinese ASR model. Uses CTC greedy decoder. |

TDT/CTC models above are wrapped by `SlidingWindowAsrManager`, which chunks audio (~15s with overlap) and stitches the per-chunk transcripts.

### Streaming Transcription (True Real-Time)

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet EOU** | Streaming speech-to-text with end-of-utterance detection (120M params). Three chunk-size variants — 160ms / 320ms / 1280ms — for ultra-low-latency to higher-accuracy streaming. | Added after TDT was released & for streaming. Smaller model (120M vs 0.6B). |
| **Nemotron Speech Streaming 0.6B** | RNNT streaming ASR with 4 chunk size variants (80ms, 160ms, 560ms, 1120ms). English-only (0.6B params). Int8 encoder quantization. Supports ultra-low latency (80ms chunks) to high accuracy (1120ms chunks). | Larger streaming model for better accuracy and quality |

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
| **Kokoro TTS** | Text-to-speech synthesis (82M params), 48 voices, minimal RAM usage on iOS. Generates all frames at once via flow matching over mel spectrograms + Vocos vocoder. Uses CoreML G2P model for phonemization. | First TTS backend added + support custom pronounces |
| **Kokoro ANE (7-stage)** | Same Kokoro 82M weights split into 7 CoreML stages so the ANE-friendly layers (Albert / PostAlbert / Alignment / Vocoder) stay resident on the Neural Engine while Prosody / Noise / Tail run on CPU+GPU. 3-11× RTFx vs. the single-graph Kokoro. Single voice (`af_heart`), ≤510 IPA phonemes per call, no chunker / SSML / custom lexicon. | ANE-optimized variant derived (with permission) from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml) |
| **PocketTTS** | Second TTS backend (~155M params). Autoregressive frame-by-frame generation with dynamic audio chunking. No phoneme stage, works directly on text tokens. | Supports streaming, minimal RAM usage, excellent quality |

## Not Production Ready

Models that are functionally complete and shipped, but **not yet recommended for production use** — RTFx or WER limitations that need community assistance to push past. Open to PRs / issue reports / perf investigations.

| Model | Status |
|-------|--------|
| **Magpie TTS Multilingual** ([FluidAudio#541](https://github.com/FluidInference/FluidAudio/pull/541), [mobius#44](https://github.com/FluidInference/mobius/pull/44), [HF](https://huggingface.co/FluidInference/magpie-tts-multilingual-357m-coreml)) | NVIDIA NeMo Magpie TTS Multilingual 357M, 8 languages (en/es/de/fr/it/vi/zh/hi), 5 built-in speakers. 4-model CoreML pipeline (text_encoder + decoder_prefill + decoder_step + nanocodec_decoder) + pure-Swift Local Transformer (Accelerate + BNNS). Custom IPA override via `\|...\|` segments. **Quite slow on Apple Silicon — RTFx ≈ 0.04 (~25× slower than realtime), ~30 s cold first synth, ~96 s warm for an 8-word English sentence on M-series.** Audio is ASR-clean on 4/5 speakers; spk0 has a single trailing-word artifact attributable to fp16 sampler-trajectory drift. Throughput investigation, MLX-backed LocalTransformer, CFG perf, and Japanese support (OpenJTalk + MeCab) are pending. For real-time TTS use Kokoro or PocketTTS instead. |
| **CosyVoice3 (Mandarin)** ([FluidAudio#536](https://github.com/FluidInference/FluidAudio/pull/536), [mobius#42](https://github.com/FluidInference/mobius/pull/42), [HF](https://huggingface.co/FluidInference/CosyVoice3-0.5B-coreml)) | FunAudioLLM CosyVoice3 0.5B Mandarin zero-shot voice cloning. 4-model CoreML pipeline (Qwen2 LLM prefill + stateful decode via `MLState` + CFM Flow + HiFT vocoder) with Swift-native Qwen2 byte-level BPE tokenizer, mmap'd 151 936×896 / 6761×896 fp16 embedding tables, and Mandarin text frontend. Voice prompts (precomputed speech IDs / mel / 192-d spk-emb) are produced offline via `mobius/.../bootstrap_aishell3_voices.py`. **Slow on Apple Silicon — end-to-end RTFx < 1.0 typical, several seconds of latency for short Mandarin utterances.** Bottlenecks: Flow stays fp32 / `cpuAndGPU`-only because fp16 + ANE NaNs through fused `layer_norm` (CoreMLTools limitation, tracked upstream); HiFT sinegen / windowing falls back to CPU. On-device prompt-asset extraction (SpeechTokenizerV3 + CAMPPlus DSP), production-grade Mandarin TN, and a streaming `AsyncStream` API are pending. Backend is flagged `[BETA — slow, RTFx < 1.0]` in the CLI; API / model layout / prompt-asset format may change without deprecation aliases. For real-time TTS use Kokoro or PocketTTS instead. |

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
| Parakeet CTC Chinese | [FluidInference/parakeet-ctc-0.6b-zh-cn-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-zh-cn-coreml) |
| Parakeet CTC 110M | [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) |
| Parakeet CTC 0.6B | [FluidInference/parakeet-ctc-0.6b-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-coreml) |
| Parakeet EOU | [FluidInference/parakeet-realtime-eou-120m-coreml](https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml) |
| Silero VAD | [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml) |
| Diarization (Pyannote) | [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml) |
| LS-EEND | [FluidInference/lseend-coreml](https://huggingface.co/FluidInference/lseend-coreml) |
| Sortformer | [FluidInference/diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml) |
| Kokoro TTS | [FluidInference/kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml) |
| Kokoro ANE (7-stage) | [FluidInference/kokoro-82m-coreml/tree/main/ANE](https://huggingface.co/FluidInference/kokoro-82m-coreml/tree/main/ANE) |
| PocketTTS | [FluidInference/pocket-tts-coreml](https://huggingface.co/FluidInference/pocket-tts-coreml) |
| Magpie TTS Multilingual | [FluidInference/magpie-tts-multilingual-357m-coreml](https://huggingface.co/FluidInference/magpie-tts-multilingual-357m-coreml) |
| CosyVoice3 (Mandarin) | [FluidInference/CosyVoice3-0.5B-coreml](https://huggingface.co/FluidInference/CosyVoice3-0.5B-coreml) |
| Nemotron Streaming | [FluidInference/nemotron-speech-streaming-en-0.6b-coreml](https://huggingface.co/FluidInference/nemotron-speech-streaming-en-0.6b-coreml) |
