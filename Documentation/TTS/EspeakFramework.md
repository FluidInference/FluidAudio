# eSpeak-NG Framework Packaging

FluidAudio bundles the eSpeak-NG phoneme resources so Kokoro can fall back to G2P lookups when the US lexicons don’t contain a word. The Core ML pipeline expects the resources under `Resources/espeak-ng/espeak-ng-data.bundle` with the canonical `voices/` directory inside.

## All Platforms (Primary Flow)
- `TtsResourceDownloader.ensureEspeakDataBundle` first attempts to stage the packaged `espeak-ng-data.bundle` from SwiftPM resources (`Sources/FluidAudio/Resources/espeak-ng/`).
- The bundle is copied to `~/.cache/fluidaudio/Models/kokoro/Resources/espeak-ng/`.
- The `voices/` directory is validated after staging; if missing, `TTSError.downloadFailed` is raised.

## Fallback Behavior (macOS Only)
- If the packaged bundle is unavailable, **macOS only** falls back to downloading `espeak-ng.zip` from HuggingFace and extracting it with `/usr/bin/unzip`.
- **iOS/tvOS/watchOS** do not support fallback downloads and will throw `TTSError.downloadFailed` if the packaged bundle is missing.
- For mobile platforms, ensure the packaged bundle is present in the Swift package resources before building.

## Best practices
- Keep the `espeak-ng-data.bundle` (packaged copy) and the optional `espeak-ng.zip` fallback in sync with any updates to the Kokoro phoneme mapper.
- If you customize the cache location, be sure the `Resources/espeak-ng/espeak-ng-data.bundle/voices/` directory is present before running TTS.
- When testing on iOS, bundle the extracted resources with the app or seed the simulator cache in advance to avoid runtime failures.

## CocoaPods integration notes
- The `ESpeakNG.xcframework` now includes support for iOS device (arm64), iOS Simulator (arm64 + x86_64), and macOS (arm64 + x86_64).
- iOS Simulator support is provided via a stub framework that allows building and linking but returns failure values for ESpeakNG function calls.
- Pod validation passes successfully with `pod lib lint FluidAudio.podspec --allow-warnings` for all platforms.
- On iOS Simulator, ESpeakNG initialization will fail gracefully and phonemization requests will return `nil` due to the stub implementation.
- Full ESpeakNG functionality is available on iOS device and macOS platforms.

## Licensing notes
- eSpeak-NG is distributed under the GNU GPL v3 (or later). Both the core library and the `espeak-ng-data` voices inherit the same license.
- The full license text now lives at `Licenses/ESpeakNG_LICENSE.txt`; ship this file (or the upstream `COPYING`) anywhere the framework is redistributed and surface it in your third-party notices UI.
- If you republish the prebuilt `ESpeakNG.xcframework`, keep the license alongside the binary and ensure downstream consumers can obtain the corresponding source per GPL requirements.
