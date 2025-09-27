# Text-To-Speech (TTS) Overview

This document explains how the FluidAudio Text-To-Speech stack is assembled, how it differs from the other inference pipelines, and which moving pieces you need to be aware of when adding features or debugging issues.

## High-Level Pipeline

1. **Input sanitisation** – `TtSManager` trims whitespace and resolves the requested voice before invoking the lower-level `KokoroModel` APIs.
2. **Model preparation** – `KokoroModel` ensures that all required assets are present:
   - `DownloadUtils` pulls the Kokoro Core ML bundles the first time they are requested.
   - `KokoroVocabulary` downloads and caches the phoneme vocabulary.
   - `VoiceEmbeddingDownloader` ensures that a voice embedding JSON is present for the target voice before synthesis begins.
3. **Dictionary loading** – `KokoroModel.loadSimplePhonemeDictionary()` populates the lexicon actor from the `us_gold.json` and `us_silver.json` dictionaries so that the chunker can map words to Kokoro tokens.
4. **Chunking** – `KokoroChunker` splits text into pronounceable segments that respect the Kokoro token budget, spell out numbers when required, and annotate pauses between phrases.
5. **Tokenisation** – The phoneme sequences from each chunk are translated into model input IDs using the vocabulary map.
6. **Model selection** – Each chunk is routed to the 5-second or 15-second Kokoro variant depending on the token count.
7. **Inference** – `KokoroModel` pads/truncates the inputs, injects the selected voice embedding, and calls the Core ML model via `compatPrediction` so older OS versions continue to work.
8. **Post-processing** – The generated sample buffers are cross-faded, normalised, optionally written to per-chunk WAVs, and surfaced as `KokoroModel.SynthesisResult`.

## Key Components

### `TtSManager`
- Provides a simple `initialize` API for the CLI and apps. It loads models, dictionaries, and caches voice embeddings up front.
- Surfaces `synthesizeDetailed` which returns both the audio bytes and the per-chunk metadata used by the CLI metrics output.

### `KokoroModel`
- Owns the synthesis pipeline and reusable resources.
- Uses an internal `LexiconCache` actor to guard concurrent access to the phoneme dictionaries.
- Emits detailed log lines (subsystem `com.fluidaudio.tts`, category `KokoroModel`) so you can trace chunk timings and variant selection in Console.
- Normalises the final waveform and reports prediction timings through `AppLogger`.

### `KokoroChunker`
- Mirrors the MLX Swift chunking behaviour.
- Spell-out logic uses a fresh `NumberFormatter` per invocation to keep thread-safety without introducing extra locks.
- Returns `TextChunk` structs that carry words, atoms, phoneme sequences, and pause metadata for the synthesiser.

### `VoiceEmbeddingDownloader`
- Downloads JSON embeddings directly from HuggingFace when available.
- Falls back to fetching the `.pt` file, saves it under `Models/kokoro/voices`, and throws an actionable error telling the user to convert it with `extract_voice_embeddings.py`.
- Logs all network attempts via `AppLogger` instead of `print`.

### Vocabulary Loading

`KokoroVocabulary` is also actor-backed. It verifies that `vocab_index.json` exists, downloads it if necessary, and throws `TTSError.processingFailed` rather than crashing when the file is missing or malformed.

### Cross-fade Behaviour

Chunks that share a hard boundary are blended with an 8 ms (roughly 192-sample) cross-fade. The fade is linear and now reaches the expected 0→1 range so seams are minimised.

## CLI Nuances

- `fluidaudio tts` records synthesis, model load, and real-time factor metrics. When the `--metrics` flag is provided it also runs ASR on the generated audio and captures WER using the shared `WERCalculator` utility.
- Chunk-level metrics are emitted through `AppLogger(category: "TTSCommand")` and written into the metrics JSON alongside optional per-chunk WAV references when `--chunk-dir` is provided.

## Troubleshooting Checklist

- **Models missing** – Verify `~/Library/Application Support/FluidAudio/Models/kokoro` contains the expected `.mlmodelc` directories. Re-run `TtSManager.initialize()` to trigger downloads.
- **Vocabulary errors** – Call `KokoroVocabulary.shared.getVocabulary()` in isolation; failures will now throw descriptive `TTSError` values.
- **Voice embedding failures** – Ensure the JSON exists under `Models/kokoro/voices/VOICE_ID.json`. If only a `.pt` file exists, run the provided conversion script and re-launch the synthesiser.
- **Unknown words** – `KokoroModel.validateTextHasDictionaryCoverage` throws when out-of-vocabulary words require the eSpeak G2P fallback but the ESpeak data bundle is missing. Run `DownloadUtils.ensureEspeakDataBundle` or bundle the assets with your app.

## Related Files

- `Sources/FluidAudio/TextToSpeech/Kokoro/` – Kokoro Core ML integration.
- `Sources/FluidAudio/TextToSpeech/TtsManager.swift` – High-level orchestration used by the CLI.
- `Sources/FluidAudioCLI/Commands/TTSCommand.swift` – CLI entry point that converts the synthesis result into WAV output and metrics.
- `Documentation/KokoroChunkerOverview.md` – Additional details on the chunking heuristics.

## Future Improvements

- Allow dynamic voice embedding selection at runtime without hitting the filesystem (e.g., in-memory cache or preloading via configuration).
- Surface chunk-level timing data through Swift concurrency tasks so UI clients can progressively stream audio.
- Add unit tests that cover dictionary loading, chunk splitting, and variant selection once the project enables TTS test fixtures.

