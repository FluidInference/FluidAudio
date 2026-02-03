# Models

A guide to each CoreML model pipeline in FluidAudio.

## ASR Models

### Batch Transcription (Near Real-Time)

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet TDT v2** | Batch speech-to-text, English only (0.6B params). TDT architecture. | First ASR model added. |
| **Parakeet TDT v3** | Batch speech-to-text, 25 European languages (0.6B params). Default ASR model. | Released after v2 to add multilingual support. |

TDT models process audio in chunks (~15s with overlap) as batch operations. Fast enough for dictation-style workflows. Not suitable for word-by-word live captions.

### Streaming Transcription (True Real-Time)

| Model | Description | Context |
|-------|-------------|---------|
| **Parakeet EOU** | Streaming speech-to-text with end-of-utterance detection (120M params). Processes 160ms/320ms frames for true real-time results as the user speaks. | Added after TDT was released. Smaller model (120M vs 0.6B). |

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
| **Pyannote CoreML Pipeline** | Speaker diarization. Segmentation model + WeSpeaker embeddings for clustering. Supports both online (streaming) and offline (VBx) modes. | First diarization system added. Two-stage pipeline. |
| **Sortformer** | End-to-end streaming speaker diarization. Single neural network — no separate segmentation + clustering. Streaming only, no offline mode. | Created & converted after Pyannote was released. |

## TTS Models

| Model | Description | Context |
|-------|-------------|---------|
| **Kokoro TTS** | Text-to-speech synthesis (82M params), 48 voices, minimal RAM usage on iOS. Generates all frames at once via flow matching over mel spectrograms + Vocos vocoder. Requires espeak for phonemization. | First TTS backend added. |
| **PocketTTS** | Second TTS backend (~155M params). Upgrade over Kokoro with much better dynamic audio chunking. No espeak dependency. | |

## Evaluated Models (Not Shipped)

Models we converted and tested but haven't shipped yet — either still in development or superseded by better approaches.

| Model | Status |
|-------|--------|
| **Nemotron Speech Streaming 0.6B** ([#254](https://github.com/FluidInference/FluidAudio/pull/254)) | Streaming model with 1.12s chunks. Not significantly faster or more accurate than existing Parakeet models: streaming (EOU) and batch (TDT) modes. |

## Model Sources

| Model | HuggingFace Repo |
|-------|-----------------|
| Parakeet TDT v3 | [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) |
| Parakeet TDT v2 | [FluidInference/parakeet-tdt-0.6b-v2-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml) |
| Parakeet CTC 110M | [FluidInference/parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) |
| Parakeet CTC 0.6B | [FluidInference/parakeet-ctc-0.6b-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-coreml) |
| Parakeet EOU | [FluidInference/parakeet-realtime-eou-120m-coreml](https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml) |
| Silero VAD | [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml) |
| Diarization (Pyannote) | [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml) |
| Sortformer | [FluidInference/diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml) |
| Kokoro TTS | [FluidInference/kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml) |
| PocketTTS | [FluidInference/pocket-tts-coreml](https://huggingface.co/FluidInference/pocket-tts-coreml) |
