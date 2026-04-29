#!/usr/bin/env python3
"""
Fetch the MiniMax Multilingual TTS Test Set text files and convert them
to the FluidAudio TTS-benchmark corpus format.

Source dataset:  https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set
License:         CC-BY-SA-4.0

Upstream format (per-language `.txt` under `text/`):

    <cloning_audio_filename>|<text_to_synthesize>

For the FluidAudio harness we only need the text — voice cloning is a
per-backend concern (Kokoro / PocketTTS / Magpie / StyleTTS2 each have
their own voice plumbing) and the cloning audio lives separately. We
strip the leading `<filename>|` and emit one phrase per non-empty line
into `Benchmarks/tts/corpus/minimax/<lang>.txt`, prefixed with a header
that documents the source + license.

Usage:

    python Scripts/fetch_minimax_tts_corpus.py
    python Scripts/fetch_minimax_tts_corpus.py --languages english spanish hindi
    python Scripts/fetch_minimax_tts_corpus.py --revision <commit>

The default revision pin matches what's vendored in-tree so re-running
this script is reproducible.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "MiniMaxAI/TTS-Multilingual-Test-Set"
# Pin to the initial public commit so re-runs reproduce the vendored files.
DEFAULT_REVISION = "cb416f0ac3658da0577e97873065e19fe6488917"

# All 24 languages in the upstream `text/` directory.
ALL_LANGUAGES = [
    "arabic", "cantonese", "chinese", "czech", "dutch", "english",
    "finnish", "french", "german", "greek", "hindi", "indonesian",
    "italian", "japanese", "korean", "polish", "portuguese", "romanian",
    "russian", "spanish", "thai", "turkish", "ukrainian", "vietnamese",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "Benchmarks" / "tts" / "corpus" / "minimax"


def hf_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def fetch(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "fluidaudio-minimax-corpus/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def convert(raw: str) -> list[str]:
    """Strip `<filename>|` prefix and return a list of trimmed phrases."""
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "<cloning_audio_filename>|<text>". Some lines may have
        # extra `|` inside the text — keep only the first split.
        sep = line.find("|")
        if sep == -1:
            text = line
        else:
            text = line[sep + 1:].strip()
        if text:
            out.append(text)
    return out


def write_corpus(lang: str, phrases: list[str], out_dir: Path,
                 revision: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{lang}.txt"
    header = [
        f"# MiniMax Multilingual TTS Test Set — {lang}",
        f"# Source:   https://huggingface.co/datasets/{REPO}",
        f"# Revision: {revision}",
        f"# License:  CC-BY-SA-4.0 (Creative Commons "
        "Attribution-ShareAlike 4.0)",
        "# Phrases:  " + str(len(phrases)),
        "#",
        "# Cloning-audio filenames have been stripped — we only need the",
        "# text for the FluidAudio TTS benchmark harness. Voice selection",
        "# is per-backend (see Benchmarks/tts/corpus/minimax/README.md).",
        "",
    ]
    out_path.write_text("\n".join(header + phrases) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--languages", nargs="+", default=ALL_LANGUAGES,
        help="Subset of languages to fetch (default: all 24).")
    ap.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help="HuggingFace dataset revision (commit SHA or branch).")
    ap.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory for converted corpus files.")
    args = ap.parse_args()

    unknown = sorted(set(args.languages) - set(ALL_LANGUAGES))
    if unknown:
        print(f"unknown language(s): {unknown}", file=sys.stderr)
        print(f"available: {ALL_LANGUAGES}", file=sys.stderr)
        return 2

    print(f"Fetching MiniMax TTS Multilingual Test Set @ {args.revision}")
    print(f"  out_dir: {args.out_dir.relative_to(REPO_ROOT)}")
    print(f"  langs:   {len(args.languages)}")
    print()

    total = 0
    for lang in args.languages:
        url = hf_url(REPO, args.revision, f"text/{lang}.txt")
        try:
            raw = fetch(url)
        except Exception as exc:
            print(f"  [{lang}] FAILED: {exc}", file=sys.stderr)
            return 1
        phrases = convert(raw)
        path = write_corpus(lang, phrases, args.out_dir, args.revision)
        print(f"  [{lang}] {len(phrases):3d} phrases -> "
              f"{path.relative_to(REPO_ROOT)}")
        total += len(phrases)

    print()
    print(f"OK — {total} phrases across {len(args.languages)} language(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
