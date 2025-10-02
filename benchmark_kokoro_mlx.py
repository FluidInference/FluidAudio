#!/usr/bin/env python3
"""Run the Kokoro MLX TTS benchmark used in Documentation/Benchmarks.md."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import sys

REPO_ROOT = Path(__file__).resolve().parent
MLX_AUDIO_PATH = REPO_ROOT / "mlx-audio"
if MLX_AUDIO_PATH.exists() and MLX_AUDIO_PATH.is_dir():
    sys.path.insert(0, str(MLX_AUDIO_PATH))

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This benchmark requires the `mlx` package. Install it with `uv pip install --python /usr/bin/python3 --user mlx` "
        "or run the script via `uv run --with mlx python benchmark_kokoro_mlx.py`."
    ) from exc

try:
    from loguru import logger  # noqa: F401  (imported for Kokoro dependency)
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Kokoro depends on the `loguru` package. Install it with `uv pip install --python /usr/bin/python3 --user loguru` "
        "or run the script via `uv run --with mlx --with loguru python benchmark_kokoro_mlx.py`."
    ) from exc

from mlx_audio.tts.models.kokoro.kokoro import Model as KokoroModel
from mlx_audio.tts.utils import load_model


DEFAULT_PROMPTS: List[str] = [
    "Fast bridges hum softly over moonlit rail.",
    (
        "FluidAudio Kokoro stack narrates notebooks during focus sprints, keeping teams aligned "
        "while staging updates for demo review time"
    ),
    (
        "When an all-hands update needs polish, Kokoro rehearses the copy so stakeholders hear exactly what changed, "
        "why the release matters, and how to try the new pathways without leaving their private workflows."
    ),
    (
        "Morning smoke tests stay on schedule because the voice bot narrates every regression ticket before stand-up, "
        "highlighting owners, context, and the path to green builds."
    ),
    (
        "Ship cycles feel calmer when change logs and launch briefs keep a consistent, friendly narrator walking "
        "through risks, mitigations, and next milestones for partner teams."
    ),
    (
        "Benchmark notebooks make it effortless to compare MLX and Core ML pipelines side-by-side, capturing token "
        "timings, audio duration, and memory usage for stakeholders."
    ),
    (
        "Late-night benchmarking sessions stay motivating because Kokoro keeps the updates short, upbeat, and precise, "
        "reminding the crew which experiments are landing before sunrise deployments."
    ),
    "Focus!",
    (
        "Losurdo reminds us that freedom has always been contested, so the team documents trade-offs behind every "
        "product decision and revisits them when feedback rolls in from beta partners."
    ),
    (
        "Across partner meetings, open betas, and support calls, Kokoro explains how on-device speech keeps customer "
        "data private while still delivering responsive experiences for long-form narration."
    ),
    (
        "Domenico Losurdo's Liberalism: A Counter-History reframes progress by foregrounding the communities that were "
        "kept outside the sacred space of rights.  The counter-history lens surfaces contradictions-like how abolition, "
        "labor protections, and decolonization advanced because those excluded movements weaponized liberal language.  "
        "FluidAudio's product notes borrow the same humility: document who benefits, who is marginalized, and what "
        "guardrails keep the new voice tools accountable when they ship beyond the lab."
    ),
]


@dataclass
class BenchmarkRow:
    index: int
    chars: int
    audio_seconds: float
    inference_seconds: float
    rtf: float
    peak_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="prince-canuma/Kokoro-82M",
        help="HuggingFace repo ID or local path to the Kokoro model",
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Voice embedding identifier (default: af_heart)",
    )
    parser.add_argument(
        "--lang",
        default="a",
        help="Kokoro language code (default: American English)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier",
    )
    parser.add_argument(
        "--split-pattern",
        default=r"\n+",
        help="Regex used to split paragraphs for long-form prompts",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Optional path to a file with prompts separated by blank lines",
    )
    return parser.parse_args()


def load_prompts(path: Path | None) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS
    text = path.read_text(encoding="utf-8")
    prompts = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def warm_up(model: KokoroModel, prompt: str, args: argparse.Namespace) -> float:
    start = time.perf_counter()
    for _ in model.generate(
        prompt,
        voice=args.voice,
        speed=args.speed,
        lang_code=args.lang,
        split_pattern=args.split_pattern,
    ):
        pass
    mx.clear_cache()
    return time.perf_counter() - start


def run_prompt(
    model: KokoroModel, prompt: str, args: argparse.Namespace, index: int
) -> BenchmarkRow:
    mx.reset_peak_memory()
    audio_seconds = 0.0
    inference_seconds = 0.0

    for segment in model.generate(
        prompt,
        voice=args.voice,
        speed=args.speed,
        lang_code=args.lang,
        split_pattern=args.split_pattern,
    ):
        mx.eval(segment.audio)
        audio_seconds += segment.audio.shape[0] / segment.sample_rate
        inference_seconds += segment.processing_time_seconds

    peak_gb = mx.get_peak_memory() / (1024**3)

    rtf = audio_seconds / inference_seconds if inference_seconds > 0 else float("inf")
    return BenchmarkRow(
        index=index,
        chars=len(prompt),
        audio_seconds=audio_seconds,
        inference_seconds=inference_seconds,
        rtf=rtf,
        peak_gb=peak_gb,
    )


def format_seconds(value: float) -> str:
    return f"{value:>7.3f}"


def format_rtf(value: float) -> str:
    if value == float("inf"):
        return "   inf"
    return f"{value:>8.3f}x"


def print_table(rows: Iterable[BenchmarkRow]) -> None:
    rows = list(rows)
    print("Test   Chars    Output (s)   Inf(s)       RTFx       Peak GB")

    total_audio = 0.0
    total_inference = 0.0
    peak_max = 0.0

    for row in rows:
        total_audio += row.audio_seconds
        total_inference += row.inference_seconds
        peak_max = max(peak_max, row.peak_gb)
        print(
            f"{row.index:<6}{row.chars:<9}{format_seconds(row.audio_seconds)}"
            f"   {format_seconds(row.inference_seconds)}"
            f"   {format_rtf(row.rtf)}   {row.peak_gb:>8.2f}"
        )

    aggregate_rtf = total_audio / total_inference if total_inference > 0 else float("inf")
    print(
        f"Total  -        {format_seconds(total_audio)}   {format_seconds(total_inference)}"
        f"   {format_rtf(aggregate_rtf)}   {peak_max:>8.2f}"
    )


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)

    model = load_model(args.model)
    if not isinstance(model, KokoroModel):
        raise SystemExit(
            "This benchmark currently supports Kokoro models only. Provide a Kokoro checkpoint via --model."
        )

    warmup_duration = warm_up(model, prompts[0], args) if prompts else 0.0

    results: List[BenchmarkRow] = []
    for idx, text in enumerate(prompts, 1):
        results.append(run_prompt(model, text, args, idx))

    print(
        f"\nTTS benchmark for voice {args.voice} (warm-up took an extra {warmup_duration:.3f}s)"
        f" using model {args.model}"
    )
    print_table(results)


if __name__ == "__main__":
    main()
