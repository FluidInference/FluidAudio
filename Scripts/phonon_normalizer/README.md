# Phonon text normalizer golden fixture

`Sources/FluidAudio/TTS/PocketTTS/Phonon/` is a Swift port of the English text normalization
Gradium's Phonon TTS runs before tokenizing (`ptts` crate, `normalize_text(_, Lang::En,
Rules::DEFAULT)`, plus `prepare_text_prompt` and `split_into_best_sentences`). The test
`Tests/FluidAudioTests/TTS/PocketTTS/PhononTextNormalizerTests.swift` checks every Swift output
against `Tests/FluidAudioTests/TTS/PocketTTS/Fixtures/phonon_normalizer_golden.json`, which the
Rust code itself generated.

Ported from https://github.com/gradium-ai/xn-ptts at `6f42148066dfe3640ab2099d5c2f088c826937d6`.

- `make_corpus.py` writes `corpus.json`, the inputs (every English case from the Rust tests plus
  cases for each rule and character path).
- `phonon_normalize_golden.rs` is a `ptts` example that reads `corpus.json` and writes the golden
  file with the real Rust functions. The chunker is run under a deterministic toy tokenizer
  (mirrored in the Swift test) at budgets of 50 and 12 tokens.

To regenerate (from an `xn-ptts` checkout):

```sh
cp Scripts/phonon_normalizer/phonon_normalize_golden.rs <xn-ptts>/ptts/examples/
printf '\n[[example]]\nname = "phonon_normalize_golden"\n' >> <xn-ptts>/ptts/Cargo.toml
python3 Scripts/phonon_normalizer/make_corpus.py
(cd <xn-ptts> && cargo run --release -p ptts --example phonon_normalize_golden -- \
  "$OLDPWD/Scripts/phonon_normalizer/corpus.json" \
  "$OLDPWD/Tests/FluidAudioTests/TTS/PocketTTS/Fixtures/phonon_normalizer_golden.json")
DEVELOPER_DIR=/Applications/Xcode-27.0.0.app/Contents/Developer swift test --filter PhononTextNormalizerTests
```
