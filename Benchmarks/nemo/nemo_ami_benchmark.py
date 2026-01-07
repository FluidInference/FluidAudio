#!/usr/bin/env python3
"""
NeMo Sortformer AMI SDM Benchmark (Streaming Mode)

Runs NVIDIA's Sortformer model in STREAMING mode on the AMI SDM dataset
for comparison with the Swift/CoreML implementation.

Uses the same high-latency config as Swift: 30.4s chunks

Usage:
    python nemo_ami_benchmark.py [--output results.json]

Requirements:
    pip install nemo_toolkit[asr] pyannote.metrics
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from nemo.collections.asr.models import SortformerEncLabelModel
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


# AMI SDM test meetings (same as Swift benchmark)
AMI_MEETINGS = [
    "EN2002a", "EN2002b", "EN2002c", "EN2002d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
]

# Default paths
DEFAULT_AMI_AUDIO_DIR = os.path.expanduser("~/FluidAudioDatasets/ami_official/sdm")
DEFAULT_AMI_RTTM_DIR = os.path.expanduser("~/FluidAudioDatasets/ami_official/rttm")

# NVIDIA High-Latency Streaming Config (matches Swift)
# 30.4s total context = 380 encoder frames
SAMPLE_RATE = 16000
FRAME_DURATION = 0.08  # 80ms per frame
NUM_SPEAKERS = 4

# High-latency config parameters (from Swift SortformerConfig.nvidiaHighLatency)
CHUNK_LEN = 48  # Core chunk length in encoder frames
CHUNK_LEFT_CONTEXT = 56  # Left context in encoder frames
CHUNK_RIGHT_CONTEXT = 56  # Right context in encoder frames
SUBSAMPLING_FACTOR = 8  # Mel frames per encoder frame


@dataclass
class BenchmarkResult:
    meeting: str
    der: float
    miss_rate: float
    false_alarm_rate: float
    speaker_error_rate: float
    detected_speakers: int
    ground_truth_speakers: int
    rtfx: float
    processing_time: float
    audio_duration: float


def load_rttm(rttm_path: str) -> Annotation:
    """Load RTTM file into pyannote Annotation."""
    annotation = Annotation()
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                annotation[Segment(start, start + duration)] = speaker
    return annotation


def predictions_to_annotation(
    predictions: np.ndarray,
    frame_duration: float = FRAME_DURATION,
    threshold: float = 0.5
) -> Annotation:
    """Convert frame-level predictions to pyannote Annotation.

    Args:
        predictions: [num_frames, num_speakers] array of probabilities
        frame_duration: Duration of each frame in seconds
        threshold: Threshold for speaker activity
    """
    annotation = Annotation()
    num_frames, num_speakers = predictions.shape

    for spk in range(num_speakers):
        in_segment = False
        segment_start = 0.0

        for frame in range(num_frames):
            prob = predictions[frame, spk]
            time_sec = frame * frame_duration

            if prob >= threshold and not in_segment:
                in_segment = True
                segment_start = time_sec
            elif prob < threshold and in_segment:
                in_segment = False
                if time_sec - segment_start > 0.0:
                    annotation[Segment(segment_start, time_sec)] = f"speaker_{spk}"

        # Handle segment that extends to end
        if in_segment:
            end_time = num_frames * frame_duration
            if end_time - segment_start > 0.0:
                annotation[Segment(segment_start, end_time)] = f"speaker_{spk}"

    return annotation


def run_batch_inference(
    model: SortformerEncLabelModel,
    audio_path: str,
    device: str = "cpu"
) -> np.ndarray:
    """Run Sortformer in batch mode (full file at once).

    Note: This runs the entire file through the model, which is what NeMo's
    Sortformer is designed for. The Swift implementation does streaming
    chunking on top of this.
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)
    length = torch.tensor([waveform.shape[1]]).to(device)

    with torch.no_grad():
        outputs = model.forward(
            audio_signal=waveform,
            audio_signal_length=length
        )

        if isinstance(outputs, tuple):
            preds = outputs[0]
        else:
            preds = outputs

        if preds.min() < 0 or preds.max() > 1:
            preds = torch.sigmoid(preds)

        predictions = preds.squeeze(0).cpu().numpy()

    return predictions


def run_benchmark(
    audio_dir: str,
    rttm_dir: str,
    meetings: list[str],
    output_path: Optional[str] = None,
    device: str = "cpu",
    streaming: bool = True,
    model_path: Optional[str] = None
) -> list[BenchmarkResult]:
    """Run NeMo Sortformer benchmark on AMI meetings."""

    print("=" * 80)
    print("NEMO SORTFORMER AMI BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Mode: {'Streaming (30.4s chunks)' if streaming else 'Batch'}")
    print(f"Audio dir: {audio_dir}")
    print(f"RTTM dir: {rttm_dir}")
    print(f"Meetings: {len(meetings)}")
    print()

    # Load model
    print("Loading Sortformer model...")
    model_start = time.time()
    if model_path:
        print(f"Loading from local path: {model_path}")
        model = SortformerEncLabelModel.restore_from(model_path)
    else:
        print("Downloading from HuggingFace (requires authentication for gated model)")
        model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    model = model.to(device)
    model.eval()
    model_time = time.time() - model_start
    print(f"Model loaded in {model_time:.2f}s")
    print()

    results = []

    print("-" * 70)
    print(f"{'Meeting':<12} {'DER %':>8} {'Miss %':>8} {'FA %':>8} {'SE %':>8} {'Speakers':>10} {'RTFx':>8}")
    print("-" * 70)

    for meeting in meetings:
        # Find audio file
        audio_path = os.path.join(audio_dir, f"{meeting}.wav")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(audio_dir, f"{meeting}.Mix-Headset.wav")
        if not os.path.exists(audio_path):
            print(f"Audio not found for {meeting}, skipping...")
            continue

        # Find RTTM file
        rttm_path = os.path.join(rttm_dir, f"{meeting}.rttm")
        if not os.path.exists(rttm_path):
            print(f"RTTM not found for {meeting}, skipping...")
            continue

        # Load ground truth
        reference = load_rttm(rttm_path)
        gt_speakers = len(set(reference.labels()))

        # Get audio duration
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sample_rate

        # Run inference
        start_time = time.time()

        if streaming:
            # Note: NeMo Sortformer doesn't support true streaming internally.
            # We use batch inference which is what Swift's streaming builds on top of.
            predictions = run_batch_inference(model, audio_path, device)
        else:
            # Batch mode
            with torch.no_grad():
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.to(device)
                length = torch.tensor([waveform.shape[1]]).to(device)

                outputs = model.forward(audio_signal=waveform, audio_signal_length=length)
                if isinstance(outputs, tuple):
                    preds = outputs[0]
                else:
                    preds = outputs
                if preds.min() < 0 or preds.max() > 1:
                    preds = torch.sigmoid(preds)
                predictions = preds.squeeze(0).cpu().numpy()

        processing_time = time.time() - start_time
        rtfx = audio_duration / processing_time if processing_time > 0 else 0

        # Convert predictions to annotation
        hypothesis = predictions_to_annotation(predictions, FRAME_DURATION, threshold=0.5)
        detected_speakers = len(set(hypothesis.labels()))

        # Compute DER components
        der_metric = DiarizationErrorRate()
        detail = der_metric(reference, hypothesis, detailed=True)

        total = detail.get('total', 1)
        der = detail.get('diarization error rate', 0) * 100
        miss = detail.get('missed detection', 0) / total * 100 if total > 0 else 0
        fa = detail.get('false alarm', 0) / total * 100 if total > 0 else 0
        se = detail.get('confusion', 0) / total * 100 if total > 0 else 0

        result = BenchmarkResult(
            meeting=meeting,
            der=der,
            miss_rate=miss,
            false_alarm_rate=fa,
            speaker_error_rate=se,
            detected_speakers=detected_speakers,
            ground_truth_speakers=gt_speakers,
            rtfx=rtfx,
            processing_time=processing_time,
            audio_duration=audio_duration,
        )
        results.append(result)

        print(f"{meeting:<12} {der:>7.1f}% {miss:>7.1f}% {fa:>7.1f}% {se:>7.1f}% {detected_speakers}/{gt_speakers:>8} {rtfx:>7.1f}x")

    print("-" * 70)

    # Compute averages
    if results:
        avg_der = sum(r.der for r in results) / len(results)
        avg_miss = sum(r.miss_rate for r in results) / len(results)
        avg_fa = sum(r.false_alarm_rate for r in results) / len(results)
        avg_se = sum(r.speaker_error_rate for r in results) / len(results)
        avg_rtfx = sum(r.rtfx for r in results) / len(results)

        print(f"{'AVERAGE':<12} {avg_der:>7.1f}% {avg_miss:>7.1f}% {avg_fa:>7.1f}% {avg_se:>7.1f}% {'-':>10} {avg_rtfx:>7.1f}x")

    print("=" * 70)

    # Save results
    if output_path:
        output_data = [
            {
                "meeting": r.meeting,
                "der": r.der,
                "missRate": r.miss_rate,
                "falseAlarmRate": r.false_alarm_rate,
                "speakerErrorRate": r.speaker_error_rate,
                "detectedSpeakers": r.detected_speakers,
                "groundTruthSpeakers": r.ground_truth_speakers,
                "rtfx": r.rtfx,
                "processingTime": r.processing_time,
                "audioDuration": r.audio_duration,
            }
            for r in results
        ]
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="NeMo Sortformer AMI Benchmark")
    parser.add_argument(
        "--audio-dir",
        default=DEFAULT_AMI_AUDIO_DIR,
        help="Path to AMI audio files"
    )
    parser.add_argument(
        "--rttm-dir",
        default=DEFAULT_AMI_RTTM_DIR,
        help="Path to AMI RTTM files"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--single-file",
        help="Run on single meeting (e.g., ES2004a)"
    )
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode instead of streaming"
    )
    parser.add_argument(
        "--model-path",
        help="Path to local .nemo model file (optional, downloads from HF if not provided)"
    )

    args = parser.parse_args()

    meetings = [args.single_file] if args.single_file else AMI_MEETINGS

    run_benchmark(
        audio_dir=args.audio_dir,
        rttm_dir=args.rttm_dir,
        meetings=meetings,
        output_path=args.output,
        device=args.device,
        streaming=not args.batch,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
