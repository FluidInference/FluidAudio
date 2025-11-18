#!/usr/bin/env python3
"""Run FluidAudio Swift ASR (and optionally NeMo ASR) on a set of WAVs and compare to reference text.

By default, this script:
- Looks for WAV files in `tts_context_boosting_kokoro/`.
- Uses a hard-coded mapping from filename -> reference text for the six Kokoro sentences.
- Calls `swift run -c release fluidaudio transcribe` for each file.
- Writes a JSON report with reference + hypothesis per file.

With `--with-nemo`, it will also:
- Run a NeMo ASR model on the same WAVs.
- Append `nemo_model` and `nemo_hypothesis` fields to each JSON entry.

Usage (from FluidAudioSwift root):
  python Scripts/run_kokoro_swift_eval.py \
    --wav-dir tts_context_boosting_kokoro \
    --out kokoro_swift_eval.json \
    [--custom-vocab custom_vocab_parakeet_v3/custom_vocab.json] \
    [--with-nemo --nemo-model-name nvidia/parakeet-rnnt-0.6b]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
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


def run_nemo_transcribe(
    model_name: str,
    paths: List[Path],
    key_phrases: Optional[List[str]] = None,
    context_score: float = 1.0,
    depth_scaling: float = 2.0,
    boosting_tree_alpha: float = 0.5,
) -> List[str]:
    """Load a NeMo ASR model and transcribe the given audio files.

    If key_phrases is provided, enable contextual RNNT decoding using a boosting tree
    configured similarly to the NeMo word_boosting docs.
    """

    import torch  # Imported lazily so Swift-only runs don't require PyTorch/NeMo.
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis as RNNT_Hypothesis

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=device)
    if device == "cuda":
        model = model.to(device)

    # Enable contextual decoding if phrases are provided and the config supports it.
    if key_phrases:
        from omegaconf import DictConfig, OmegaConf

        cfg = model.cfg
        # Most RNNT ASR models expose decoding config under `rnnt_decoding`.
        if "rnnt_decoding" in cfg and "greedy" in cfg.rnnt_decoding:
            dec = cfg.rnnt_decoding
            # Ensure we are using greedy (non-batch) decoding for this simple eval.
            if not hasattr(dec, "strategy") or dec.strategy is None:
                dec.strategy = "greedy"

            # Build a small boosting tree config with key_phrases_list.
            greedy = dec.greedy
            bt = getattr(greedy, "boosting_tree", None)
            if bt is None:
                bt = OmegaConf.create({})
                greedy.boosting_tree = bt

            bt.key_phrases_list = list(key_phrases)
            bt.context_score = context_score
            bt.depth_scaling = depth_scaling
            bt.score_per_phrase = 0.0

            # Global weight for the boosting tree.
            dec.boosting_tree_alpha = boosting_tree_alpha

            # Apply the updated decoding config back to the model.
            model.change_decoding_strategy(dec)

    audio_paths = [str(p) for p in paths]
    raw_outputs = model.transcribe(audio_paths, batch_size=1, return_hypotheses=True)

    texts: List[str] = []
    for item in raw_outputs:
        if isinstance(item, RNNT_Hypothesis):
            texts.append(item.text or "")
            continue

        if isinstance(item, list) and item:
            first = item[0]
            if isinstance(first, RNNT_Hypothesis):
                texts.append(first.text or "")
                continue

        if isinstance(item, str):
            texts.append(item)
            continue

        if isinstance(item, dict) and "text" in item:
            texts.append(str(item["text"]))
            continue

        if hasattr(item, "text"):
            texts.append(str(getattr(item, "text") or ""))
            continue

        texts.append(str(item))

    return texts


def run_swift_transcribe(
    wav_path: Path,
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
        "--custom-vocab",
        default=None,
        help="Optional path to custom_vocab JSON for context boosting",
    )
    ap.add_argument(
        "--with-nemo",
        action="store_true",
        help="Also run NeMo ASR and add nemo_hypothesis/nemo_model to the JSON output",
    )
    ap.add_argument(
        "--nemo-model-name",
        default="nvidia/parakeet-rnnt-0.6b",
        help="NeMo ASR model name to use when --with-nemo is set",
    )
    ap.add_argument(
        "--nemo-use-context",
        action="store_true",
        help="Enable NeMo boosting_tree context biasing using phrases from custom_vocab.json",
    )
    ap.add_argument(
        "--nemo-context-score",
        type=float,
        default=1.0,
        help="NeMo boosting_tree context_score (per-arc weight)",
    )
    ap.add_argument(
        "--nemo-depth-scaling",
        type=float,
        default=2.0,
        help="NeMo boosting_tree depth_scaling",
    )
    ap.add_argument(
        "--nemo-alpha",
        type=float,
        default=0.5,
        help="NeMo rnnt_decoding.boosting_tree_alpha",
    )
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    custom_vocab = Path(args.custom_vocab).expanduser().resolve() if args.custom_vocab else None

    if not wav_dir.is_dir():
        raise SystemExit(f"WAV directory not found: {wav_dir}")

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
        swift_baseline = run_swift_transcribe(wav, custom_vocab=None)
        # Boosted Swift (with custom vocab if provided)
        if custom_vocab is not None:
            hyp = run_swift_transcribe(wav, custom_vocab=custom_vocab)
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

    if args.with_nemo and results:
        wav_paths = [Path(item["file"]).expanduser().resolve() for item in results]
        for p in wav_paths:
            if not p.is_file():
                raise SystemExit(f"WAV file not found for NeMo transcription: {p}")

        print(f"Running NeMo ASR model: {args.nemo_model_name}")

        nemo_key_phrases: Optional[List[str]] = None
        if args.nemo_use_context and custom_vocab is not None and custom_vocab.is_file():
            try:
                cv_data = json.loads(custom_vocab.read_text(encoding="utf-8"))
                terms = cv_data.get("terms") or []
                nemo_key_phrases = [t.get("text", "") for t in terms if t.get("text")]
                print(f"Loaded {len(nemo_key_phrases)} context phrases from {custom_vocab}")
            except Exception as exc:
                print(f"Failed to load context phrases from {custom_vocab}: {exc}")
        try:
            # Baseline NeMo (no context boosting)
            nemo_baseline = run_nemo_transcribe(
                args.nemo_model_name,
                wav_paths,
                key_phrases=None,
            )
            # Context-boosted NeMo (if enabled), otherwise same as baseline
            if args.nemo_use_context and nemo_key_phrases:
                nemo_texts = run_nemo_transcribe(
                    args.nemo_model_name,
                    wav_paths,
                    key_phrases=nemo_key_phrases,
                    context_score=args.nemo_context_score,
                    depth_scaling=args.nemo_depth_scaling,
                    boosting_tree_alpha=args.nemo_alpha,
                )
            else:
                nemo_texts = nemo_baseline
        except ImportError as exc:
            print(f"Could not import NeMo; skipping NeMo transcription: {exc}")
        else:
            if len(nemo_texts) != len(results) or len(nemo_baseline) != len(results):
                raise SystemExit(
                    f"Mismatch between NeMo transcripts "
                    f"(baseline={len(nemo_baseline)}, boosted={len(nemo_texts)}) and Swift results ({len(results)})"
                )
            for idx, item in enumerate(results):
                nb = nemo_baseline[idx]
                nh = nemo_texts[idx]
                item["nemo_model"] = args.nemo_model_name
                item["nemo_baseline"] = nb
                item["nemo_model"] = args.nemo_model_name
                item["nemo_hypothesis"] = nh

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved evaluation report to {out_path} (items={len(results)})")


if __name__ == "__main__":
    main()
