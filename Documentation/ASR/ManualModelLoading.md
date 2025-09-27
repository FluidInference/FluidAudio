# Manual Model Loading for ASR

FluidAudio usually downloads ASR CoreML bundles from HuggingFace with `AsrModels.downloadAndLoad`. When you need to operate in an offline or pre-provisioned environment, you can skip the download helper and point the pipeline at models you staged yourself. This guide shows how to prepare the assets and wire them into the `AsrManager` manually.

## Required assets

Each ASR release ships four CoreML bundles plus the shared vocabulary file:

- `Preprocessor.mlmodelc`
- `Encoder.mlmodelc`
- `Decoder.mlmodelc`
- `JointDecision.mlmodelc`
- `parakeet_vocab.json`

Pick the folder that matches the version you want to serve:

- Multilingual (`AsrModelVersion.v3`): `FluidInference/parakeet-tdt-0.6b-v3-coreml`
- English only (`AsrModelVersion.v2`): `FluidInference/parakeet-tdt-0.6b-v2-coreml`

## Stage the directory layout

`AsrModels.load(from:)` expects the directory you pass to be the staged HuggingFace repo folder itself (the one that contains the `.mlmodelc` bundles and `parakeet_vocab.json`). Place the assets in the structure below (replace `/opt/models` with your storage path):

```
/opt/models
└── parakeet-tdt-0.6b-v3-coreml
    ├── Preprocessor.mlmodelc
    │   ├── coremldata.bin
    │   └── ...
    ├── Encoder.mlmodelc
    ├── Decoder.mlmodelc
    ├── JointDecision.mlmodelc
    └── parakeet_vocab.json
```

If you are deploying the English-only variant, swap the folder name for `parakeet-tdt-0.6b-v2-coreml` and ensure the four `.mlmodelc` bundles plus `parakeet_vocab.json` are present.

### Manual download options

1. Clone from HuggingFace with Git LFS (recommended when you can connect once):
   ```bash
   git lfs install
   git clone https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml
   mv parakeet-tdt-0.6b-v3-coreml /opt/models/
   ```
2. Use the HuggingFace web UI to download the `.tar` archives for each `.mlmodelc` bundle and extract them into the layout above.
3. Copy the prepared directory from another machine that already ran `downloadAndLoad` (the cache lives at `~/Library/Application Support/FluidAudio/Models/<repo>` on macOS).

After staging the files, call `AsrModels.modelsExist(at:)` in your app or a small Swift script to double-check that the four bundles and `parakeet_vocab.json` are readable.

## Loading models without auto-download

The key is to call `AsrModels.load(from:configuration:version:)` with the repo folder URL. This loads the CoreML bundles you staged and never attempts a network download when everything is in place.

```swift
import FluidAudio
import CoreML

@main
struct ManualLoader {
    static func main() async {
        do {
            // Point to the staged repo directory that contains the Core ML bundles
            let repoDirectory = URL(fileURLWithPath: "/opt/models/parakeet-tdt-0.6b-v3-coreml", isDirectory: true)

            // Optional: customize compute units; default uses CPU+ANE
            var configuration = AsrModels.defaultConfiguration()

            let models = try await AsrModels.load(
                from: repoDirectory,
                configuration: configuration,
                version: .v3
            )

            let asrManager = AsrManager()
            try await asrManager.initialize(models: models)

            // ... proceed with transcription workflow
        } catch {
            print("Failed to load ASR models: \(error)")
        }
    }
}
```

### Switching versions at runtime

Pass `.v2` for the English-only repo:

```swift
let englishRepo = URL(fileURLWithPath: "/opt/models/parakeet-tdt-0.6b-v2-coreml", isDirectory: true)
let englishModels = try await AsrModels.load(from: englishRepo, version: .v2)
```

### Troubleshooting tips

- Use `AsrModels.modelsExist(at:)` before calling `load` to confirm the vocabulary file and all four `.mlmodelc` bundles are present.
- `AsrModels.load` reads the vocabulary from the same repo folder. Make sure `parakeet_vocab.json` sits beside the model bundles.
- If you see `AsrModelsError.modelNotFound`, double-check for typos in the folder names or missing `coremldata.bin` files inside each `.mlmodelc` directory.
- `load` still reports helpful diagnostics through `OSLog`. Run your build with the `OS_ACTIVITY_MODE` environment variable cleared so you can see the log lines during bring-up.

With this setup the ASR pipeline stays entirely offline while still using the exact CoreML bundles FluidAudio ships on HuggingFace.
