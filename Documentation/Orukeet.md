# Orukeet with local Core ML models

Orukeet is an optional adaptation of NVIDIA Parakeet TDT v3. Its Core ML preview
uses the existing v3 decoder and 8,192-token vocabulary. The weights are
CC BY-SA 4.0; retain the attribution and license files in the bundle.

`AsrModels.loadLocal(from:)` loads the exact compiled directory you supply. It
never downloads NVIDIA weights, changes a default, or resolves a sibling cache.
A missing component is an error. The same path is available in the CLI through
`--local-model-dir`.

Download the baseline profile from [Hugging Face](https://huggingface.co/oruk/orukeet).
It retains the top-K outputs required for language hints. The JSON manifest is
used to verify the pinned archive and participates in Hugging Face's normal
NeMo download counts. Neither loading nor transcription sends a download event.

```sh
python -m pip install huggingface-hub
```

```python
import hashlib
import json
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

revision = "43142dd1897f9ddadcd70173fcb5ff45c08aa951"
manifest = json.loads(Path(hf_hub_download(
    "oruk/orukeet", "coreml/manifest.json", revision=revision
)).read_text())
artifact = manifest["archives"]["baseline"]
assert artifact["sha256"] == "b2a6efc4ed3280c860f29b3e2e2ea242ade14c6482c94f1c8d3e8551d5edb626"
archive = Path(hf_hub_download(
    "oruk/orukeet", "coreml/" + artifact["filename"], revision=revision
))
assert archive.stat().st_size == artifact["bytes"]
with archive.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == artifact["sha256"]
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall("models")
```

Compile on the destination Mac (macOS 14+, Apple Silicon). Save this as
`compile-orukeet.swift` and run `swift compile-orukeet.swift`. Keep the compiled
cache on that device; do not distribute it as a portable model.

```swift
import CoreML
import Foundation

let directory = URL(fileURLWithPath: "models/orukeet-r3-coreml-baseline", isDirectory: true)
for name in ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"] {
    let compiled = try MLModel.compileModel(at: directory.appendingPathComponent("\(name).mlpackage"))
    defer { try? FileManager.default.removeItem(at: compiled) }
    try FileManager.default.copyItem(at: compiled, to: directory.appendingPathComponent("\(name).mlmodelc"))
}
```

```sh
swift run fluidaudiocli transcribe recording.wav --local-model-dir models/orukeet-r3-coreml-baseline --model-version v3
```

For applications, keep the manager loaded between recordings:

```swift
let models = try AsrModels.loadLocal(from: compiledOrukeetDirectory, version: .v3)
let manager = AsrManager(config: .default, models: models)
var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
let result = try await manager.transcribe(mono16kSamples, decoderState: &state)
```

The preview is qualified on a limited Apple Silicon sample. It is an offline
TDT model, not the separate Parakeet EOU live-typing model. This integration
does not establish an accuracy or speed advantage on your audio.

The opt-in local regression runs two decodes of a real recording without any
model download. An optional expected transcript checks export parity:

```sh
FLUIDAUDIO_LOCAL_TEST_MODELS="$PWD/models/orukeet-r3-coreml-baseline" \
FLUIDAUDIO_LOCAL_TEST_AUDIO="$PWD/recording.wav" \
swift test --filter AsrModelsLocalTests
```
