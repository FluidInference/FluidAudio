#!/usr/bin/env python3
"""
Fetch FLEURS test splits for languages not hosted on FluidInference/fleurs
by downloading from google/fleurs and transforming to the flat
FluidInference layout:

  <cache_root>/fleurs/<lang>/<lang>_NNNN.wav
  <cache_root>/fleurs/<lang>/<lang>.trans.txt

google/fleurs layout:
  data/<lang>/test.tsv                      (id, file, raw, normalized, ...)
  data/<lang>/audio/test.tar.gz             (hash-named wavs, e.g. 1234567890.wav)

This script renames tar-wavs to `<lang>_NNNN.wav` in TSV order so downstream
Swift tooling that expects FluidInference's naming convention works unchanged.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_LANGS = ["ar_eg", "ja_jp", "cmn_hans_cn", "ko_kr", "vi_vn"]
GOOGLE_FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
DEFAULT_CACHE = Path.home() / "Library" / "Application Support" / "FluidAudio" / "Datasets" / "fleurs"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.name} ({dest.stat().st_size} bytes)")
        return
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fluidaudio-fleurs-adapter/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"  saved: {dest} ({dest.stat().st_size} bytes)")


def parse_tsv(tsv_path: Path) -> list[tuple[str, str]]:
    """Return list of (hash_filename, normalized_transcript) in TSV row order."""
    rows: list[tuple[str, str]] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 4:
                continue
            fname = row[1].strip()
            # Column 3 is normalized (no punctuation). Fallback to raw if missing.
            transcript = (row[3] if len(row) > 3 and row[3].strip() else row[2]).strip()
            if fname and transcript:
                rows.append((fname, transcript))
    return rows


def process_language(lang: str, cache_root: Path, work_root: Path) -> None:
    print(f"[{lang}] fetching from google/fleurs")
    work_dir = work_root / lang
    work_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = work_dir / "test.tsv"
    tar_path = work_dir / "test.tar.gz"
    download(f"{GOOGLE_FLEURS_BASE}/{lang}/test.tsv", tsv_path)
    download(f"{GOOGLE_FLEURS_BASE}/{lang}/audio/test.tar.gz", tar_path)

    rows = parse_tsv(tsv_path)
    print(f"[{lang}] {len(rows)} rows in test.tsv")

    extract_dir = work_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    print(f"[{lang}] extracting tar to {extract_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    # Audio may be under extract_dir/test/*.wav or flat under extract_dir/*.wav
    candidate_dirs = [extract_dir, extract_dir / "test"]
    wav_index: dict[str, Path] = {}
    for d in candidate_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.wav"):
            wav_index[p.name] = p
    print(f"[{lang}] {len(wav_index)} wavs found in tar")

    dest_dir = cache_root / lang
    dest_dir.mkdir(parents=True, exist_ok=True)
    trans_path = dest_dir / f"{lang}.trans.txt"

    written = 0
    with open(trans_path, "w", encoding="utf-8") as trans_out:
        for idx, (hash_name, transcript) in enumerate(rows):
            src = wav_index.get(hash_name)
            if src is None:
                # google/fleurs sometimes lists multiple TSV rows per wav; skip dups.
                continue
            sample_id = f"{lang}_{idx:04d}"
            dest_wav = dest_dir / f"{sample_id}.wav"
            shutil.copyfile(src, dest_wav)
            # Collapse any stray whitespace/newlines in transcript.
            clean = " ".join(transcript.split())
            trans_out.write(f"{sample_id} {clean}\n")
            written += 1

    print(f"[{lang}] wrote {written} wavs + {trans_path.name}")
    # Clean up intermediate extraction to save space (keep tsv/tar for re-runs).
    shutil.rmtree(extract_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", nargs="*", default=DEFAULT_LANGS,
                        help=f"Language codes to fetch (default: {DEFAULT_LANGS})")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE,
                        help=f"FluidInference fleurs cache root (default: {DEFAULT_CACHE})")
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/fluidaudio-fleurs-work"),
                        help="Scratch dir for downloads and extraction")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if <lang>.trans.txt already exists")
    args = parser.parse_args()

    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        trans_path = args.cache_root / lang / f"{lang}.trans.txt"
        if trans_path.exists() and not args.force:
            # Check that at least one wav exists as a sanity check
            wavs = list((args.cache_root / lang).glob("*.wav"))
            if wavs:
                print(f"[{lang}] already populated ({len(wavs)} wavs), skipping. Use --force to redo.")
                continue
        process_language(lang, args.cache_root, args.work_root)

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
