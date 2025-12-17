#!/usr/bin/env python3
"""Run FluidAudio Swift ASR on a set of WAVs and compare to reference text.

This CoreML-only script:
- Looks for WAV files in `tts_context_boosting_kokoro/`.
- Uses a hard-coded mapping from filename -> reference text for the six Kokoro sentences.
- Calls `swift run -c release fluidaudio transcribe` for each file.
- Writes a JSON report with reference + hypothesis per file.

Usage (from FluidAudioSwift root):
  python Scripts/run_kokoro_swift_eval.py \
    --wav-dir tts_context_boosting_kokoro \
    --out kokoro_swift_eval.json \
    [--custom-words custom_vocab_parakeet_v3/custom_words.txt]
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import json
from typing import Dict, List, Optional

# Default references for the six test sentences
DEFAULT_REFERENCES: Dict[str, str] = {
    "01_saoirse_ronan_netflix_af_heart.wav": "Saoirse Ronan and Timothée Chalamet starred in that new Netflix series.",
    "02_wojciechowski_xarelto_af_heart.wav": "Dr. Wojciechowski prescribed Xarelto for my atrial fibrillation condition.",
    "03_oculus_quest_psvr_af_heart.wav": "Can you order the Oculus Quest from Amazon or should we get the PlayStation VR?",
    "04_advil_zyrtec_chipotle_siobhan_af_heart.wav": "I took two Advil and one Zyrtec, then ate at Chipotle with Siobhan.",
    "05_macbook_airpods_schaumburg_af_heart.wav": "I bought a MacBook Pro and AirPods Max at the Apple Store in Schaumburg.",
    "06_xavier_haagen_dazs_disney_af_heart.wav": "Xavier ordered Postmates delivery of Häagen-Dazs and watched Disney Plus.",
}


def run_swift_transcribe(
    wav_path: Path,
    custom_words: Optional[Path] = None,
    custom_vocab: Optional[Path] = None,
) -> str:
    """Call `swift run -c release fluidaudio transcribe` and return final transcription text.

    Raises RuntimeError if we cannot parse a transcription.
    """

    swift_cmd = os.environ.get("SWIFT_CMD", "/usr/bin/swift")

    cmd: List[str] = [
        swift_cmd,
        "run",
        "-c",
        "release",
        "fluidaudio",
        "transcribe",
        str(wav_path),
    ]
    if custom_words is not None:
        cmd.extend(["--custom-words", str(custom_words)])
    if custom_vocab is not None:
        cmd.extend(["--custom-vocab", str(custom_vocab)])

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    output = proc.stdout

    # For current CLI output, the transcription is printed as the last non-empty line.
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"No output for {wav_path}")
    hyp = lines[-1]
    return hyp


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Swift ASR on WAVs and compare to references")
    ap.add_argument("--wav-dir", default="tts_context_boosting_kokoro", help="Directory containing WAV files")
    ap.add_argument("--out", default="kokoro_swift_eval.json", help="Output JSON path")
    ap.add_argument(
        "--custom-words",
        default=None,
        help="Optional path to newline-delimited custom words for post-processing",
    )
    ap.add_argument(
        "--custom-vocab",
        default=None,
        help="Optional structured custom vocab JSON for CTC keyword boosting",
    )
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    custom_words = Path(args.custom_words).expanduser().resolve() if args.custom_words else None
    custom_vocab = Path(args.custom_vocab).expanduser().resolve() if args.custom_vocab else None

    if not wav_dir.is_dir():
        raise SystemExit(f"WAV directory not found: {wav_dir}")

    if custom_words is not None and not custom_words.is_file():
        raise SystemExit(f"Custom words file not found: {custom_words}")
    if custom_vocab is not None and not custom_vocab.is_file():
        raise SystemExit(f"Custom vocab JSON not found: {custom_vocab}")

    results: List[dict] = []

    for wav in sorted(wav_dir.glob("*.wav")):
        ref = DEFAULT_REFERENCES.get(wav.name)
        if ref is None:
            # Skip files without a known reference; or you could require a ref.
            continue
        print(f"Transcribing {wav} ...")

        # Baseline Swift (no context boosting)
        swift_baseline = run_swift_transcribe(wav, custom_words=None, custom_vocab=None)
        # Optional CTC boosting and/or <unk> rewrite
        if custom_vocab is not None or custom_words is not None:
            hyp = run_swift_transcribe(
                wav,
                custom_words=custom_words,
                custom_vocab=custom_vocab,
            )
        else:
            hyp = swift_baseline
        results.append(
            {
                "file": str(wav),
                "filename": wav.name,
                "reference": ref,
                "swift_baseline": swift_baseline,
                "hypothesis": hyp,
            }
        )

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved evaluation report to {out_path} (items={len(results)})")


if __name__ == "__main__":
    main()
