# eSpeak-NG Framework Packaging

FluidAudio bundles the eSpeak-NG phoneme resources so Kokoro can fall back to G2P lookups when the US lexicons don’t contain a word. The Core ML pipeline expects the resources under `Resources/espeak-ng/espeak-ng-data.bundle` with the canonical `voices/` directory inside.

## macOS (and desktop) builds
- The `DownloadUtils.ensureEspeakDataBundle` helper downloads `espeak-ng.zip` from HuggingFace the first time it’s needed.
- On macOS the archive is extracted with `/usr/bin/unzip` into `~/.cache/fluidaudio/Models/kokoro/Resources/`.
- The `voices/` directory is validated after extraction; if it’s missing we raise `TTSError.downloadFailed`.

## iOS / tvOS / watchOS
- The eSpeak bundle must be pre-packaged in the app or the models cache; the GitHub iOS simulator runner cannot invoke `/usr/bin/unzip`.
- `ensureEspeakDataBundle` skips extraction on these platforms and logs a warning if the bundle is missing.

## Best practices
- Keep the `espeak-ng.zip` artifact in sync with any updates to the Kokoro phoneme mapper.
- If you customize the cache location, be sure the `Resources/espeak-ng/espeak-ng-data.bundle/voices/` directory is present before running TTS.
- When testing on iOS, bundle the extracted resources with the app or seed the simulator cache in advance to avoid runtime failures.

## Licensing notes
- eSpeak-NG is distributed under the GNU GPL v3 (or later). Both the core library and the `espeak-ng-data` voices inherit the same license.
- The full license text now lives at `Licenses/ESpeakNG_LICENSE.txt`; ship this file (or the upstream `COPYING`) anywhere the framework is redistributed and surface it in your third-party notices UI.
- If you republish the prebuilt `ESpeakNG.xcframework`, keep the license alongside the binary and ensure downstream consumers can obtain the corresponding source per GPL requirements.
