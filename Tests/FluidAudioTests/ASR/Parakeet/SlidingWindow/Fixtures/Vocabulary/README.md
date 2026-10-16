# Vocabulary span regressions

These are human speech recordings for vocabulary rescoring, stored as 16 kHz
mono PCM16 WAV. No speech synthesis, gain adjustment or time stretching is used.
`tdt-v3.json` fixes the uncorrected Parakeet TDT v3 text and token timings for
each recording. Tests run the installed CTC 110M model against the audio without
depending on a particular TDT decoder version. They skip when CTC is absent and
never download models.

- `quilter.wav`: LibriSpeech `1272-128104-0000`, “Mister Quilter is the apostle
  of the middle classes, and we are glad to welcome his gospel.” Protects the
  title from becoming a second copy of the already recognized name.
- `quilter-possessive.wav`: LibriSpeech `1272-128104-0001`, “Nor is Mister
  Quilter's manner less interesting than his matter.” Protects the title and
  possessive ending. Spelling, alias, split-word and curly-apostrophe test
  inputs perturb the recorded text/tokens; the audio and timestamps stay fixed.
  The repetition control joins the complete name spans from `quilter.wav`
  (samples 8,960..<20,480) and this clip (17,920..<30,720). It copies original
  human PCM samples without synthesis, gain, silence or time stretching. This
  edited two-name control tests normal/coarse timing and is not a natural
  continuous utterance or an independent accuracy sample.
- `solanki.wav`: FLEURS `en_us:1669:8173060820032255320.wav`, “Police
  superintendent Chandra Shekhar Solanki said the accused appeared in court
  with covered faces.” Keeps the actual TDT `Solansky` → `Solanki` correction
  working with the short-term bias taper enabled.

## Attribution and license

The audio is licensed under [CC BY 4.0](LICENSE_CC_BY_4_0.txt). The repository's
code license does not replace the recording licenses.

- **LibriSpeech**, Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev
  Khudanpur: [OpenSLR corpus and license](https://www.openslr.org/12).
  Source: [validation subset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be9148/clean),
  revision `5be9148`, `validation-00000-of-00001.parquet`.
- **FLEURS**, Google and the FLEURS authors:
  [dataset card, authors and license](https://huggingface.co/datasets/google/fleurs).
  Revision `4683b04`, `data/en_us/audio/test.tar.gz`; the transcription comes
  from the corresponding normalized reference in `data/en_us/test.tsv`.

SHA-256 integrity hashes:

| Recording | SHA-256 |
| --- | --- |
| quilter.wav | `799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46` |
| quilter-possessive.wav | `473b1bb573e8f7cdcd639c6e318036236190107b66bfe6892286121a92d464c1` |
| solanki.wav | `509bc6279b6549a1ed01a712950f0593e43380204f77f39a4fdd11bd216c68c0` |
