# MiniMax Multilingual TTS Test Set

This directory holds the per-language text files for the FluidAudio
TTS-benchmark corpus, sourced on demand from the
[MiniMaxAI/TTS-Multilingual-Test-Set](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set)
Hugging Face dataset and converted to the harness format (one phrase
per non-empty, non-`#` line).

The `.txt` files are **not vendored** — they're CC-BY-SA-4.0 derivative
content fetched on demand by `fluidaudio minimax-corpus` (see
[Fetching](#fetching) below). Only this README is checked in.

| Field    | Value |
|----------|-------|
| Source   | https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set |
| Revision | `cb416f0ac3658da0577e97873065e19fe6488917` (initial public release) |
| License  | [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Citation | MiniMax-Speech tech report — [arXiv 2505.07916](https://arxiv.org/pdf/2505.07916) |
| Languages | 24 (arabic, cantonese, chinese, czech, dutch, english, finnish, french, german, greek, hindi, indonesian, italian, japanese, korean, polish, portuguese, romanian, russian, spanish, thai, turkish, ukrainian, vietnamese) |
| Phrases   | 100 per language (2400 total) |

The fetched text files are derivative works of the upstream dataset
and remain under **CC-BY-SA-4.0**. The rest of the FluidAudio
repository is licensed separately (see top-level `LICENSE`); only the
contents of `Benchmarks/tts/corpus/minimax/` are share-alike-bound to
CC-BY-SA-4.0.

## Why this corpus?

MiniMax positions this as *"a public benchmark used in a number of
recent TTS papers, which makes our numbers directly comparable to
existing work"* (Gradium, MiniMax-Speech, seed-tts-eval, etc.).
FluidAudio's `tts-benchmark` ships exclusively against this corpus
so the resulting RTFx / WER numbers land on the same axis as
published TTS work.

## Format conversion

Upstream lines have a `<cloning_audio_filename>|<text>` pipe-delimited
shape because the dataset also ships per-speaker reference audio for
zero-shot voice cloning. The FluidAudio harness only needs the text —
voice selection is a per-backend concern (Kokoro / PocketTTS / Magpie /
StyleTTS2 each have their own voice plumbing). The leading
`<filename>|` is stripped at fetch time; if you need the cloning audio
later, fetch it from the upstream HF repo's `audio/` directory.

## Fetching

The `fluidaudio minimax-corpus` CLI subcommand pins the upstream
revision to the value above so re-runs are deterministic. From the
package root:

```bash
# All 24 languages
swift run fluidaudio minimax-corpus

# Subset
swift run fluidaudio minimax-corpus --languages english,spanish,hindi

# Refresh against a newer release
swift run fluidaudio minimax-corpus --revision <commit-sha>
```

Output lands in `Benchmarks/tts/corpus/minimax/<lang>.txt` (relative
to the package root) by default; override with `--out-dir <path>`.
Auth-gated revisions are honored via the standard `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` env vars (same as every other HF asset pull
in the project).

## Using the corpus

Fetch the language(s) you need first (see [Fetching](#fetching)),
then run the benchmark:

```bash
swift run fluidaudio minimax-corpus --languages english
swift run fluidaudio tts-benchmark \
  --backend kokoro-ane \
  --corpus minimax-english \
  --output-json bench.json
```

Available `--corpus` names: `minimax-arabic`, `minimax-cantonese`,
`minimax-chinese`, …, `minimax-vietnamese` (24 total). The harness
maps `minimax-<lang>` to `Benchmarks/tts/corpus/minimax/<lang>.txt`,
which is written by `minimax-corpus`. Running `tts-benchmark` against
a language whose `.txt` hasn't been fetched yet will fail with a
"corpus not found" error — re-run `minimax-corpus` to populate it.

### Language ↔ backend coverage

Not every backend handles every language. The pragmatic subsets:

| Backend     | Supported MiniMax languages |
|-------------|-----------------------------|
| Kokoro / Kokoro ANE | `english` (af_heart only) |
| PocketTTS   | `english`, `german`, `italian`, `portuguese`, `spanish`, `french` |
| StyleTTS2   | `english` (LibriTTS multi-speaker) |
| Magpie      | `english`, `spanish`, `german`, `french`, `italian`, `vietnamese`, `chinese`, `hindi` |
| CosyVoice3  | `chinese` |

## WER caveats

Per the [open community discussion on the upstream
dataset](https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set/discussions/10),
WER on this corpus is sensitive to the ASR + text-normalization stack:

- Whisper-v3 (and similarly Parakeet) often need text normalization on
  the reference (`"32"` → `"thirty two"`) before comparing against the
  hypothesis to get a clean WER.
- For non-Latin-script languages (Hindi, Japanese, Cantonese, etc.) the
  ASR may emit transliterated forms that don't match the reference
  script, inflating WER even when the synthesis is intelligible.
- For non-word-segmented languages (Chinese, Japanese, Thai), CER is
  the more meaningful metric — `tts-benchmark` already reports both.

This means **MiniMax WER is best read relatively (FluidAudio backend
A vs. backend B on the same corpus + same ASR), not absolutely**, and
side-by-side comparison with published numbers requires matching the
upstream ASR + normalizer choice.
