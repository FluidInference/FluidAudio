# CUA-S1-FORMS decision scoring

`CuaS1FormsManager` runs the 706,048-parameter CUA-S1-FORMS classifier locally
through Core ML. It scores supplied options for one form element in a single
stateless prediction. This is a decision component: your application supplies
document entities, describes the UI, constructs candidates, validates the result,
and executes authorized actions.

## Load and score

```swift
import FluidAudio

let manager = try await CuaS1FormsManager.load()
let decision = try await manager.score(
    context: """
        TASK fill the form from the document, then submit
        FORM Contact details
        ELEMENT Edit "Email address" value=""
        """,
    options: ["fill E-mail: person@example.com", "check", "click", "skip"])

print(decision.selectedIndex, decision.selectedOption)
print(decision.probabilities)
```

The default loader uses `Repo.cuaS1Forms`, the shared `ModelHub` cache and offline
policy, and `.cpuAndNeuralEngine`. The model repository is
[FluidInference/cua-s1-forms-coreml](https://huggingface.co/FluidInference/cua-s1-forms-coreml).
[Model PR #1](https://huggingface.co/FluidInference/cua-s1-forms-coreml/discussions/1)
contains the proposed artifacts. Its compiled bundle must be available on the model repository's `main` branch
before automatic downloading can succeed. While reviewing the model PR, download
its PR revision and load the local artifact instead.

```swift
import CoreML
import Foundation

let manager = try await CuaS1FormsManager.load(
    from: URL(fileURLWithPath: "/models/cua_s1_forms_fp16_options32.mlpackage"),
    computeUnits: .cpuAndNeuralEngine)
```

Local loading accepts a portable `.mlpackage` (compiled locally) or an existing
`.mlmodelc`, and does not access the network. An already loaded `MLModel` can be
passed to `try CuaS1FormsManager(model: model)`. The manager checks its input and
output tensor contract before accepting it. Calls on one manager are serialized
by its actor.

## Contract and limitations

- Supply one nonempty context and 2–32 nonempty option strings. Too many options
  raises an error; options are never removed to fit the model.
- Encoding matches upstream: UTF-8 bytes plus one, zero padding, and byte
  truncation at 224 context bytes and 96 bytes per option. This can split a
  multibyte character. Inspect `contextWasTruncated` and `truncatedOptionIndices`
  before relying on a decision about long inputs.
- `selectedIndex` is zero-based in the supplied options. `selectedOption` retains
  the original string. `probabilities` and `logits` contain only supplied options.
- Scores are classifier outputs, not a calibrated authorization signal. The
  caller owns fill/check/click ordering, submit authorization, and execution.
- The artifact targets iOS 17/macOS 14 or newer. Runtime validation here uses an
  Apple silicon Mac; iPhone performance has not been measured.

The pinned upstream demo contains 196 decisions across three forms and three PDFs.
PyTorch and both tested Core ML configurations selected all 196 labeled options
correctly. Maximum probability differences were 0.003099 (`ALL`) and 0.002336
(`CPU_AND_NE`), within the 0.005 conversion tolerance. These are demo conversion
checks, not evidence of general computer-use accuracy.

See the [Mobius conversion toolkit](https://github.com/FluidInference/mobius/tree/codex/cua-s1-forms/models/computer-use/cua-s1-forms/coreml)
for pinned assets, source, licenses, conversion instructions, per-row reports,
and the original Cua evaluator. Exploratory Python timing and device placement
are documented there separately from Swift runtime correctness.

## Reproduce the Swift integration checks

In the Mobius conversion directory, prepare the real assets and reference scores:

```bash
uv sync --frozen
uv run python assets.py
uv run python convert-coreml.py
uv run python export-reference.py
```

Then run from FluidAudio with absolute paths to those outputs. The reference
file contains probabilities from the unmodified PyTorch model and a SHA-256 of
the exact demo bytes; the test checks both the dataset hash and model revision.

```bash
FLUIDAUDIO_CUA_MODEL_PATH=/path/to/coreml/build/cua_s1_forms_fp16_options32.mlpackage \
FLUIDAUDIO_CUA_DEMO_PATH=/path/to/coreml/artifacts/demo.jsonl \
FLUIDAUDIO_CUA_REFERENCE_PATH=/path/to/coreml/build/reference-probabilities.json \
swift test --filter CuaS1Forms
```

The shared-cache check additionally uses `FLUIDAUDIO_CUA_COMPILED_PATH` pointing
to the real `.mlmodelc`. The Mobius README includes its compilation command.
Without these paths, integration tests skip; byte-encoding, limits, rejection,
and model-registry unit tests still run. XCTest requires a full Xcode installation
on macOS. No model or dataset is downloaded by the tests automatically.
