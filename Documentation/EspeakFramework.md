# eSpeak-NG Framework Packaging

FluidAudio bundles the eSpeak-NG phoneme resources so Kokoro can fall back to G2P lookups when the US lexicons don’t contain a word. The Core ML pipeline expects the resources under `Resources/espeak-ng/espeak-ng-data.bundle` with the canonical `voices/` directory inside.

## macOS (and desktop) builds
- `TtsResourceDownloader.ensureEspeakDataBundle` first stages the packaged `espeak-ng-data.bundle` directly from the SwiftPM resources.
- If the packaged copy is removed, macOS falls back to downloading `espeak-ng.zip` and extracts it with `/usr/bin/unzip` into `~/.cache/fluidaudio/Models/kokoro/Resources/`.
- The `voices/` directory is validated after extraction; if it’s missing we raise `TTSError.downloadFailed`.

## iOS / tvOS / watchOS
- The Swift package now looks for a pre-packaged `espeak-ng-data.bundle` under `Sources/FluidAudio/Resources/espeak-ng/` and stages it into the cache on first use.
- If the bundle is missing, we surface `TTSError.downloadFailed`; iOS builds no longer attempt to shell out or download the ZIP on-device.
- Seed the packaged bundle (or pre-populate the on-device cache) before running TTS on these platforms.

## Best practices
- Keep the `espeak-ng-data.bundle` (packaged copy) and the optional `espeak-ng.zip` fallback in sync with any updates to the Kokoro phoneme mapper.
- If you customize the cache location, be sure the `Resources/espeak-ng/espeak-ng-data.bundle/voices/` directory is present before running TTS.
- When testing on iOS, bundle the extracted resources with the app or seed the simulator cache in advance to avoid runtime failures.

## Licensing notes
- eSpeak-NG is distributed under the GNU GPL v3 (or later). Both the core library and the `espeak-ng-data` voices inherit the same license.
- The full license text now lives at `Licenses/ESpeakNG_LICENSE.txt`; ship this file (or the upstream `COPYING`) anywhere the framework is redistributed and surface it in your third-party notices UI.
- If you republish the prebuilt `ESpeakNG.xcframework`, keep the license alongside the binary and ensure downstream consumers can obtain the corresponding source per GPL requirements.
