# Adding New Models

Step-by-step guide for converting a new model to CoreML and shipping it in FluidAudio. Intended for contributors and coding agents.

## Overview

Adding a new model has three stages across three locations:

1. **[mobius](https://github.com/FluidInference/mobius)** — Convert the source model (PyTorch/ONNX) to CoreML
2. **[HuggingFace](https://huggingface.co/FluidInference)** — Upload and host the converted model artifacts (`.mlmodelc`, `.mlpackage`, vocab JSON, embeddings, etc.)
3. **FluidAudio** — Register the model, write inference code, add CLI command, write tests

Each new model should reference all three:

| Item | Example |
|------|---------|
| mobius PR | [`FluidInference/mobius#21`](https://github.com/FluidInference/mobius/pull/21) (conversion scripts, inference scripts, trial notes) |
| HuggingFace repo | [`FluidInference/qwen3-asr-0.6b-coreml`](https://huggingface.co/FluidInference/qwen3-asr-0.6b-coreml) (name mirrors the base model, model card links back to it) |
| FluidAudio PR | [`FluidInference/FluidAudio#315`](https://github.com/FluidInference/FluidAudio/pull/315) |

---

## Stage 1: Conversion ([mobius](https://github.com/FluidInference/mobius))

mobius is where all model research, experimentation, and conversion happens. Expect trial and error — conversions rarely work on the first attempt. Common issues include tracing failures, shape mismatches, unsupported ops in CoreML, numerical drift between PyTorch and CoreML outputs, and ANE compatibility problems. Document what you tried, what failed, and why, so the next person doesn't repeat the same dead ends.

### 1.1 Create the conversion directory

Each conversion target is self-contained under `mobius/models/{class}/{model-name}/{target}/`:

```
mobius/models/
  asr/
    parakeet-tdt-v3/
      coreml/
        convert-coreml.py    # Conversion script
        pyproject.toml       # Python deps (uv-managed)
        README.md            # Conversion notes, source links, known issues
        src/                 # Helper modules (optional)
```

**Classes:** `asr`, `vad`, `diarization`, `tts`

### 1.2 Write the conversion script

Use `uv` for dependency management. The recommended Python version is 3.10.12 but other versions may work. The script should:

1. Load the source model (PyTorch checkpoint, NeMo, HuggingFace, etc.)
2. Wrap into a traceable `nn.Module` if needed (extract stateful components like LSTM states)
3. Trace with `torch.jit.trace` using representative inputs
4. Convert with `coremltools`:
   ```python
   import coremltools as ct

   coreml_model = ct.convert(
       traced_model,
       inputs=[ct.TensorType(name="audio", shape=(1, 80, 4096))],
       outputs=[ct.TensorType(name="logits")],
       minimum_deployment_target=ct.target.iOS15,
       convert_to="mlprogram",
   )
   ```
5. Set metadata (author, version, description)
6. Save as `.mlpackage`, then compile to `.mlmodelc`
7. Validate outputs against the original PyTorch model (numerical accuracy check)

### 1.3 Open a mobius PR

Include:
- The conversion script and `pyproject.toml`
- A README documenting the source model, conversion steps, known issues, and what was tried
- Link to the HuggingFace repo (once uploaded in Stage 2)

---

## Stage 2: HuggingFace Upload

Upload the converted models to the [`FluidInference`](https://huggingface.co/FluidInference) organization on HuggingFace.

### 2.1 Get access to the FluidInference org

You need write access to the [`FluidInference`](https://huggingface.co/FluidInference) organization to create repos and upload models. To get access:

- **Sign up** at [huggingface.co](https://huggingface.co) if you don't have an account
- **Request access** through the FluidInference org page, or reach out to a repo maintainer directly

### 2.2 Create the repository

- Naming: `{model-name}-coreml` (e.g., `FluidInference/parakeet-tdt-0.6b-v3-coreml`)

### 2.3 Upload model artifacts

- Upload `.mlmodelc` bundles (compiled CoreML models)
- Upload `.mlpackage` files if applicable
- Include supporting files: vocab JSON, embeddings bins, constants, etc.
- If the repo has variants (e.g., frame sizes, precisions), use subdirectories: `160ms/`, `320ms/`, `f32/`, `int8/`

### 2.4 Update the model card

- Source attribution (link to original model)
- License
- Input/output shapes and compute unit recommendations

---

## Stage 3: FluidAudio Integration

### 3.1 Register the model in `ModelNames.swift`

**File:** `Sources/FluidAudio/ModelNames.swift`

Two things to add:

**a) Repo enum case** — points to the HuggingFace repo:

```swift
public enum Repo: String, CaseIterable {
    // ...existing cases...
    case myModel = "FluidInference/my-model-coreml"
}
```

If the repo has subdirectories for variants, include the subpath in the raw value:

```swift
case myModelF32 = "FluidInference/my-model-coreml/f32"
case myModelInt8 = "FluidInference/my-model-coreml/int8"
```

Then implement the required computed properties: `name`, `remotePath`, `subPath`, `folderName`. Follow existing patterns for models with/without subdirectories.

**b) ModelNames enum** — declares expected filenames:

```swift
public enum MyModel {
    public static let encoder = "Encoder"
    public static let decoder = "Decoder"

    public static let encoderFile = encoder + ".mlmodelc"
    public static let decoderFile = decoder + ".mlmodelc"

    public static let requiredModels: Set<String> = [
        encoderFile,
        decoderFile,
    ]
}
```

Also update `getRequiredModelNames(for:variant:)` to return the new model's required set.

### 3.2 Write the Manager / inference code

**Location:** `Sources/FluidAudio/{Component}/` (e.g., `ASR/`, `VAD/`, `TTS/`, `Diarizer/`)

The manager:
- Is an `actor` (thread safety, no `@unchecked Sendable`)
- Downloads models via `DownloadUtils.loadModels()`
- Exposes a public inference API

```swift
public actor MyModelManager {
    private var encoder: MLModel?
    private var decoder: MLModel?

    public init(config: MyModelConfig = .default) async throws {
        let models = try await DownloadUtils.loadModels(
            .myModel,
            modelNames: Array(ModelNames.MyModel.requiredModels),
            directory: cacheDir,
            computeUnits: config.computeUnits
        )
        self.encoder = models[ModelNames.MyModel.encoderFile]
        self.decoder = models[ModelNames.MyModel.decoderFile]
    }

    public func process(_ audio: AVAudioPCMBuffer) async throws -> [Result] {
        // Pre-process audio, run inference, post-process
    }
}
```

Key patterns to follow:
- Use `AppLogger(category:)` for logging (not `print()`)
- Use per-module error enums conforming to `Error, LocalizedError`
- Use guard statements and early returns
- Set compute units per sub-model (e.g., preprocessor on CPU-only, encoder on CPU+ANE)

### 3.3 Add a CLI command

**Location:** `Sources/FluidAudioCLI/Commands/`

Wire the manager into a CLI command so the model can be tested from the terminal:

```swift
enum MyModelCommand {
    static func run(arguments: [String]) async {
        let manager = try await MyModelManager(config: .default)
        let results = try await manager.process(audioURL)
        // Print results
    }
}
```

Register the command in the CLI dispatcher.

### 3.4 Update `Documentation/Models.md`

Add the new model to the appropriate table (ASR, VAD, Diarization, TTS) with:
- Model name and description
- Parameter count
- Context (why it was added)
- HuggingFace repo link in the Model Sources table

If the model was evaluated but not shipped, add it to the "Evaluated Models" table with links to both the FluidAudio PR and mobius PR.

### 3.5 Format and build

```bash
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/
swift build
swift test
```

### 3.6 Run benchmarks

Every new model needs benchmark results before merging. The metrics depend on the model type:

| Model Type | Key Metrics | CLI Command |
|------------|-------------|-------------|
| **ASR (batch)** | WER, CER, RTFx | `swift run -c release fluidaudiocli asr-benchmark` |
| **ASR (streaming)** | WER, RTFx, latency per chunk | `swift run -c release fluidaudiocli parakeet-eou --benchmark` |
| **ASR (multilingual)** | WER/CER per language, RTFx | `swift run -c release fluidaudiocli fleurs-benchmark --languages all` |
| **VAD** | Accuracy, Precision, Recall, F1, RTFx | `swift run fluidaudiocli vad-benchmark` |
| **Diarization** | DER, JER, Miss/FA/SE %, RTFx | `swift run fluidaudiocli diarization-benchmark` |
| **TTS** | RTFx, peak memory, output quality (manual) | `swift run fluidaudiocli tts --benchmark` |
| **G2P** | PER, WER, ms/word | `swift run -c release fluidaudiocli g2p-benchmark` |

**Metric definitions:**

- **WER** (Word Error Rate): Edit distance between hypothesis and reference at the word level
- **CER** (Character Error Rate): Edit distance at the character level. Primary metric for character-based languages (Chinese, Japanese, Korean, Thai)
- **DER** (Diarization Error Rate): Combined missed speech + false alarm + speaker confusion
- **JER** (Jaccard Error Rate): Per-speaker overlap error
- **RTFx** (Real-Time Factor): How many times faster than real-time (higher is better, >1.0 means real-time capable)
- **PER** (Phoneme Error Rate): Character-level Levenshtein distance over reference phonemes

**Datasets:**

Datasets for each model type already exist and most auto-download on first run. We recommend using these, but feel free to use or create new ones as needed.

- **ASR:** LibriSpeech (`test-clean`), FLEURS (24+ languages), AISHELL-1 (Chinese), Earnings22
- **VAD:** Buckeye Corpus, VOiCES Subset, MUSAN Full
- **Diarization:** AMI-SDM, VoxConverse
- **G2P:** CharsiuG2P (9 languages)
- **TTS text processing:** Text normalization test data (`text-processing-rs/tests/data/`)

**What to report:**

- Run benchmarks on Apple Silicon hardware and note the exact device (e.g., "M4 Pro, 48GB, macOS 26")
- Compare against the base PyTorch model if possible (to measure CoreML conversion accuracy loss)
- Compare against existing FluidAudio models in the same category if applicable
- Add results to `Documentation/Benchmarks.md` under the appropriate section

### 3.7 Open a FluidAudio PR

The PR description should include:
- Link to the mobius PR
- Link to the HuggingFace repo
- Link to the source model
- Benchmark results (WER, DER, RTFx, etc.)

---

## Checklist

### mobius

- [ ] Conversion script created at `mobius/models/{class}/{name}/coreml/`
- [ ] `pyproject.toml` with pinned dependencies
- [ ] README with source model link, license, conversion notes
- [ ] mobius PR opened

### HuggingFace

- [ ] Repository created at `FluidInference/{model-name}-coreml`
- [ ] `.mlmodelc` bundles and `.mlpackage` files uploaded
- [ ] Supporting files included (vocab JSON, embeddings, constants)
- [ ] Model card updated with source attribution and license

### FluidAudio

- [ ] `Repo` case added to `ModelNames.swift` (with `name`, `remotePath`, `subPath`, `folderName`)
- [ ] `ModelNames` enum added with filenames and `requiredModels`
- [ ] `getRequiredModelNames(for:variant:)` updated
- [ ] Manager / inference code implemented
- [ ] CLI command added or updated
- [ ] `Documentation/Models.md` updated
- [ ] `swift format` passes
- [ ] `swift build` succeeds
- [ ] `swift test` passes
- [ ] Benchmarks run using existing or new datasets (see table in §3.6) on Apple Silicon with hardware noted
- [ ] Compared against base PyTorch model (accuracy loss check)
- [ ] Compared against existing FluidAudio models in the same category
- [ ] Results added to `Documentation/Benchmarks.md`
- [ ] FluidAudio PR opened with links to mobius PR + HuggingFace repo

---

## Reference: Existing Conversions

See `Documentation/Models.md` for the full list of shipped and evaluated models with their HuggingFace repos, PR links, and status.
