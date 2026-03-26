# Models

A guide to each CoreML model pipeline in FluidAudio.

## ASR Models

### Batch Transcription (Near Real-Time)

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet TDT v2** | Batch speech-to-text, English only (0.6B params). TDT architecture. | First ASR model added. |
| **Parakeet TDT v3** | Batch speech-to-text, 25 European languages (0.6B params). Default ASR model. | Released after v2 to add multilingual support. |
| **Parakeet TDT-CTC-110M** | Hybrid TDT-CTC batch model (110M params). 3.01% WER on LibriSpeech test-clean. 96.5x RTFx on M2 Mac. Fused preprocessor+encoder for reduced memory footprint. iOS compatible. | Smaller, faster alternative to v3 with competitive accuracy. |

TDT models process audio in chunks (~15s with overlap) as batch operations. Fast enough for dictation-style workflows. Not suitable for word-by-word live captions.

### Streaming Transcription (True Real-Time)

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet EOU** | Streaming speech-to-text with end-of-utterance detection (120M params). Processes 160ms/320ms frames for true real-time results as the user speaks. | Added after TDT was released. Smaller model (120M vs 0.6B). |
| **Nemotron Speech Streaming 0.6B** | RNNT streaming ASR with 4 chunk size variants (80ms, 160ms, 560ms, 1120ms). English-only (0.6B params). Int8 encoder quantization. Supports ultra-low latency (80ms chunks) to high accuracy (1120ms chunks). | Contributed by Muesli app developer. Larger model than EOU (0.6B vs 120M) with flexible latency tradeoffs. |

### Custom Vocabulary / Keyword Spotting

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet CTC 110M** | CTC-based encoder for custom keyword spotting. Runs rescoring alongside TDT to boost domain-specific terms (names, jargon). | |
| **Parakeet CTC 0.6B** | Larger CTC variant (same role as 110M). |  |

## VAD Models

| Model | Description | Context |
|-------|-------------|---------|
| **Silero VAD** | Voice activity detection — speech vs silence on 256ms windows. Segments audio before ASR or diarization. | Support model that other pipelines build on. |

## Diarization Models

| Model | Description | Context |
|-------|-------------|---------|
| **LS-EEND** | Research prototype end-to-end streaming diarization model from Westlake University. Supports both streaming and complete-buffer inference for up to 10 speakers. Uses frame-in, frame-out processing, requiring 900ms of warmup audio and 100ms per update. | Added after Sortformer to support largers speaker counts. |
| **Sortformer** | NVIDIA's enterprise-grade end-to-end streaming diarization model. Supports both streaming and complete-buffer inference for up to 4 speakers. More stable than LS-EEND, but sometimes misses speech. Processes audio in chunks, requiring 1040ms of warmup audio and 480ms per update for the low latency versions. | Added after Pyannote to support low-latency streaming diarization. |
| **Pyannote CoreML Pipeline** | Speaker diarization. Segmentation model + WeSpeaker embeddings for clustering. Best offline diarization pipeline, but poor for online use, since speaker confusion increases with more frequent updates, and at least 5 seconds of warmup is recommended for acceptable results. | First diarizer model added. Converted from Pyannote. |

## TTS Models

| Model | Description | Context |
|-------|-------------|---------|
| **Kokoro TTS** | Text-to-speech synthesis (82M params), 48 voices, minimal RAM usage on iOS. Generates all frames at once via flow matching over mel spectrograms + Vocos vocoder. Uses CoreML G2P model for phonemization. | First TTS backend added. |
| **PocketTTS** | Second TTS backend (~155M params). Autoregressive frame-by-frame generation with dynamic audio chunking. No phoneme stage — works directly on text tokens. | Different tradeoffs: streaming-capable, simpler chunking, dynamic inputs & longer token counts |

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
| Parakeet CTC 110M | [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) |
| Parakeet CTC 0.6B | [FluidInference/parakeet-ctc-0.6b-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-coreml) |
| Parakeet EOU | [FluidInference/parakeet-realtime-eou-120m-coreml](https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml) |
| Silero VAD | [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml) |
| Diarization (Pyannote) | [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml) |
| LS-EEND | [FluidInference/lseend-coreml](https://huggingface.co/FluidInference/lseend-coreml) |
| Sortformer | [FluidInference/diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml) |
| Kokoro TTS | [FluidInference/kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml) |
| PocketTTS | [FluidInference/pocket-tts-coreml](https://huggingface.co/FluidInference/pocket-tts-coreml) |
| Nemotron Streaming | [FluidInference/nemotron-speech-streaming-en-0.6b-coreml](https://huggingface.co/FluidInference/nemotron-speech-streaming-en-0.6b-coreml) |
