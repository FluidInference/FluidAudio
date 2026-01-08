#!/usr/bin/env python3
"""
NeMo Sortformer Benchmark

Benchmarks the NeMo streaming Sortformer model on the same files as:
- SortformerBenchmark.swift
- single_file.py

Uses streaming parameters:
- chunk_len = 340
- left_context = 1  
- right_context = 40
- fifo_len = 40
- spkcache_len = 188
- spkcache_update_period = 300
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import argparse
import urllib.request
from pathlib import Path
from itertools import permutations
import numpy as np
import torch
from nemo.collections.asr.models import SortformerEncLabelModel


# ============================================================
# AMI RTTM Download
# ============================================================
# pyannote AMI-diarization-setup repository
AMI_RTTM_URL = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test"

def download_ami_rttm(meeting_name: str, output_dir: Path) -> str:
    """Download AMI RTTM file from pyannote AMI-diarization-setup repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{meeting_name}.rttm"
    
    if output_path.exists():
        return str(output_path)
    
    # Files in pyannote repo are named {meeting}.rttm (not {meeting}.Mix-Headset.rttm)
    url = f"{AMI_RTTM_URL}/{meeting_name}.rttm"
    try:
        print(f"   Downloading RTTM from {url}...")
        urllib.request.urlretrieve(url, output_path)
        return str(output_path)
    except Exception as e:
        print(f"   Failed to download RTTM: {e}")
        return None

# ============================================================
# Benchmark Configuration
# ============================================================
STREAMING_CONFIG = {
    'chunk_len': 340,
    'chunk_left_context': 1,
    'chunk_right_context': 40,
    'fifo_len': 40,
    'spkcache_len': 188,
    'spkcache_update_period': 300,
}

FRAME_SHIFT = 0.08  # 80ms per frame (matches Swift)
SAMPLE_RATE = 16000
NUM_SPEAKERS = 4


# ============================================================
# Data Paths (matches SortformerBenchmark.swift)
# ============================================================
def get_home_dir():
    return Path.home()


def get_audio_path(meeting_name: str, dataset: str) -> str:
    """Get audio file path for a meeting."""
    home = get_home_dir()
    
    if dataset == "ami":
        return str(home / f"FluidAudioDatasets/ami_official/sdm/{meeting_name}.Mix-Headset.wav")
    elif dataset == "voxconverse":
        return str(home / f"FluidAudioDatasets/voxconverse/voxconverse_test_wav/{meeting_name}.wav")
    elif dataset == "callhome":
        return str(home / f"FluidAudioDatasets/callhome_eng/{meeting_name}.wav")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_rttm_path(meeting_name: str, dataset: str, auto_download: bool = True) -> str:
    """Get RTTM ground truth path for a meeting."""
    home = get_home_dir()
    script_dir = Path(__file__).parent
    
    if dataset == "ami":
        # First try local RTTMs in cache
        cache_dir = script_dir / "rttm_cache" / "ami"
        cached_rttm = cache_dir / f"{meeting_name}.rttm"
        if cached_rttm.exists():
            return str(cached_rttm)
        
        # Try local project RTTM
        local_rttm = script_dir / f"Streaming-Sortformer-Conversion/{meeting_name}.rttm"
        if local_rttm.exists():
            return str(local_rttm)
        
        # Try dataset RTTM
        dataset_rttm = home / f"FluidAudioDatasets/ami_official/rttm/{meeting_name}.rttm"
        if dataset_rttm.exists():
            return str(dataset_rttm)
        
        # Auto-download if enabled
        if auto_download:
            downloaded = download_ami_rttm(meeting_name, cache_dir)
            if downloaded:
                return downloaded
        
        return str(cached_rttm)  # Return path even if not downloaded (will fail later)
        
    elif dataset == "voxconverse":
        return str(home / f"FluidAudioDatasets/voxconverse/rttm_repo/test/{meeting_name}.rttm")
    elif dataset == "callhome":
        return str(home / f"FluidAudioDatasets/callhome_eng/rttm/{meeting_name}.rttm")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_ami_files(max_files: int = None) -> list:
    """Get list of AMI test set meetings (matches Swift benchmark)."""
    # Official AMI SDM test set (16 meetings) - matches NeMo evaluation
    all_meetings = [
        "EN2002a", "EN2002b", "EN2002c", "EN2002d",
        "ES2004a", "ES2004b", "ES2004c", "ES2004d",
        "IS1009a", "IS1009b", "IS1009c", "IS1009d",
        "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    ]
    
    available = []
    for meeting in all_meetings:
        if Path(get_audio_path(meeting, "ami")).exists():
            available.append(meeting)
    
    if max_files:
        return available[:max_files]
    return available


def get_voxconverse_files(max_files: int = None) -> list:
    """Get list of VoxConverse test files."""
    home = get_home_dir()
    vox_dir = home / "FluidAudioDatasets/voxconverse/voxconverse_test_wav"
    
    if not vox_dir.exists():
        return []
    
    available = []
    for wav_file in sorted(vox_dir.glob("*.wav")):
        name = wav_file.stem
        rttm_path = home / f"FluidAudioDatasets/voxconverse/rttm_repo/test/{name}.rttm"
        if rttm_path.exists():
            available.append(name)
    
    if max_files:
        return available[:max_files]
    return available


def get_callhome_files(max_files: int = None) -> list:
    """Get list of CALLHOME files."""
    home = get_home_dir()
    callhome_dir = home / "FluidAudioDatasets/callhome_eng"
    
    if not callhome_dir.exists():
        return []
    
    available = []
    for wav_file in sorted(callhome_dir.glob("*.wav")):
        name = wav_file.stem
        rttm_path = callhome_dir / f"rttm/{name}.rttm"
        if rttm_path.exists():
            available.append(name)
    
    if max_files:
        return available[:max_files]
    return available


# ============================================================
# RTTM Ground Truth Loading
# ============================================================
def load_rttm(rttm_path: str) -> list:
    """
    Load RTTM file and return list of segments.
    Format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    if not Path(rttm_path).exists():
        return []
    
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            
            try:
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                end_time = start_time + duration
                
                segments.append({
                    'speaker_id': speaker_id,
                    'start': start_time,
                    'end': end_time,
                })
            except (ValueError, IndexError):
                continue
    
    speakers = set(s['speaker_id'] for s in segments)
    print(f"   [RTTM] Loaded {len(segments)} segments, speakers: {sorted(speakers)}")
    return segments


# ============================================================
# DER Calculation (matches Swift implementation)
# ============================================================
def calculate_der(predictions: np.ndarray, ground_truth: list, 
                  threshold: float = 0.5, frame_shift: float = 0.08) -> dict:
    """
    Calculate DER using simple frame-level binary comparison.
    This matches the NeMo/Swift evaluation approach.
    
    Args:
        predictions: [num_frames, num_speakers] probability array
        ground_truth: List of RTTM segments with 'speaker_id', 'start', 'end'
        threshold: Speaker activity threshold
        frame_shift: Time per frame in seconds
    
    Returns:
        dict with 'der', 'miss', 'fa', 'se' percentages
    """
    num_frames = predictions.shape[0]
    num_speakers = predictions.shape[1]
    
    # Create reference binary matrix [num_frames, num_speakers]
    ref_binary = np.zeros((num_frames, num_speakers), dtype=np.float32)
    
    # Map ground truth speakers to indices
    speaker_labels = sorted(set(s['speaker_id'] for s in ground_truth))
    speaker_map = {label: idx for idx, label in enumerate(speaker_labels) if idx < num_speakers}
    
    # Fill reference binary from ground truth segments
    for segment in ground_truth:
        spk_id = segment['speaker_id']
        if spk_id not in speaker_map:
            continue
        spk_idx = speaker_map[spk_id]
        start_frame = max(0, min(int(segment['start'] / frame_shift), num_frames))
        end_frame = max(0, min(int(segment['end'] / frame_shift), num_frames))
        ref_binary[start_frame:end_frame, spk_idx] = 1.0
    
    # Create prediction binary matrix
    pred_binary = (predictions > threshold).astype(np.float32)
    
    # Try all permutations to find best DER
    best_der = float('inf')
    best_miss = 0
    best_fa = 0
    best_se = 0
    
    for perm in permutations(range(num_speakers)):
        miss_frames = 0
        fa_frames = 0
        se_frames = 0
        total_ref_speech = 0
        
        for frame in range(num_frames):
            ref_speech = ref_binary[frame].any()
            pred_speech_permuted = any(pred_binary[frame, perm[spk]] > 0 for spk in range(num_speakers))
            
            if ref_speech:
                total_ref_speech += 1
            
            if ref_speech and not pred_speech_permuted:
                miss_frames += 1
            elif not ref_speech and pred_speech_permuted:
                fa_frames += 1
            elif ref_speech and pred_speech_permuted:
                # Calculate speaker error
                ref_spks = set(spk for spk in range(num_speakers) if ref_binary[frame, spk] > 0)
                pred_spks = set(spk for spk in range(num_speakers) if pred_binary[frame, perm[spk]] > 0)
                sym_diff = ref_spks.symmetric_difference(pred_spks)
                se_frames += len(sym_diff) / 2.0
        
        if total_ref_speech > 0:
            der = (miss_frames + fa_frames + se_frames) / total_ref_speech * 100
            if der < best_der:
                best_der = der
                best_miss = miss_frames / total_ref_speech * 100
                best_fa = fa_frames / total_ref_speech * 100
                best_se = se_frames / total_ref_speech * 100
    
    return {
        'der': best_der,
        'miss': best_miss,
        'fa': best_fa,
        'se': best_se,
    }


# ============================================================
# NeMo Sortformer Inference
# ============================================================
def run_inference(model, audio_path: str) -> tuple:
    """
    Run NeMo Sortformer streaming inference on an audio file.
    
    Returns:
        (predictions, duration, processing_time)
        - predictions: [num_frames, num_speakers] probability array
        - duration: Audio duration in seconds
        - processing_time: Inference time in seconds
    """
    start_time = time.time()
    
    # Run inference
    predicted_segments, predicted_probs = model.diarize(
        audio=audio_path,
        batch_size=1,
        include_tensor_outputs=True
    )
    
    processing_time = time.time() - start_time
    
    # Process output probabilities
    probs = predicted_probs[0].squeeze().cpu().numpy()  # [num_frames, num_speakers]
    
    # Calculate duration from number of frames
    num_frames = probs.shape[0]
    duration = num_frames * FRAME_SHIFT
    
    return probs, duration, processing_time


def process_audio_file(model, audio_path: str, threshold: float, verbose: bool) -> dict:
    """Process a single audio file without ground truth (inference only)."""
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    try:
        print(f"   Running inference on {audio_path}...")
        probs, duration, processing_time = run_inference(model, audio_path)
        
        rtfx = duration / processing_time
        
        # Print probability statistics
        min_val = probs.min()
        max_val = probs.max()
        mean_val = probs.mean()
        above_05 = (probs > 0.5).sum()
        total_vals = probs.size
        
        print(f"   Audio duration: {duration:.2f}s")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   RTFx: {rtfx:.1f}x")
        print(f"   Prob stats: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}")
        print(f"   Activity: {above_05}/{total_vals} values ({above_05/total_vals*100:.1f}%) above 0.5")
        
        # Count detected speakers
        detected_speakers = sum(1 for spk in range(probs.shape[1]) if (probs[:, spk] > threshold).any())
        print(f"   Detected speakers: {detected_speakers}")
        
        return {
            'file': audio_path,
            'duration': duration,
            'processing_time': processing_time,
            'rtfx': rtfx,
            'num_frames': probs.shape[0],
            'detected_speakers': detected_speakers,
            'prob_min': float(min_val),
            'prob_max': float(max_val),
            'prob_mean': float(mean_val),
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error processing {audio_path}: {e}")
        traceback.print_exc()
        return None


def process_meeting(model, meeting_name: str, dataset: str, threshold: float, verbose: bool) -> dict:
    """Process a single meeting and return benchmark results."""
    audio_path = get_audio_path(meeting_name, dataset)
    rttm_path = get_rttm_path(meeting_name, dataset, auto_download=True)
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return None
    
    try:
        # Run inference
        print(f"   Running inference on {audio_path}...")
        probs, duration, processing_time = run_inference(model, audio_path)
        
        rtfx = duration / processing_time
        
        # Print probability statistics
        min_val = probs.min()
        max_val = probs.max()
        mean_val = probs.mean()
        above_05 = (probs > 0.5).sum()
        total_vals = probs.size
        
        print(f"   Prob stats: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}")
        print(f"   Activity: {above_05}/{total_vals} values ({above_05/total_vals*100:.1f}%) above 0.5")
        
        # Load ground truth
        ground_truth = load_rttm(rttm_path)
        if not ground_truth:
            print(f"⚠️ No ground truth found for {meeting_name}")
            return None
        
        # Calculate DER
        metrics = calculate_der(probs, ground_truth, threshold=threshold, frame_shift=FRAME_SHIFT)
        
        # Count speakers
        detected_speakers = sum(1 for spk in range(probs.shape[1]) if (probs[:, spk] > threshold).any())
        gt_speakers = len(set(s['speaker_id'] for s in ground_truth))
        
        return {
            'meeting': meeting_name,
            'der': metrics['der'],
            'miss': metrics['miss'],
            'fa': metrics['fa'],
            'se': metrics['se'],
            'rtfx': rtfx,
            'processing_time': processing_time,
            'duration': duration,
            'num_frames': probs.shape[0],
            'detected_speakers': detected_speakers,
            'gt_speakers': gt_speakers,
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error processing {meeting_name}: {e}")
        traceback.print_exc()
        return None


# ============================================================
# Main Benchmark
# ============================================================
def run_benchmark(args):
    """Run the full benchmark."""
    print("🚀 Starting NeMo Sortformer Benchmark")
    print(f"   Dataset: {args.dataset}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Device: {args.device}")
    print()
    
    # Load model
    print("🔧 Loading NeMo Sortformer model...")
    model_load_start = time.time()
    
    device = torch.device(args.device)
    model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1",
        map_location=device
    )
    model.eval()
    model.to(device)
    
    # Apply streaming configuration
    modules = model.sortformer_modules
    modules.chunk_len = STREAMING_CONFIG['chunk_len']
    modules.chunk_left_context = STREAMING_CONFIG['chunk_left_context']
    modules.chunk_right_context = STREAMING_CONFIG['chunk_right_context']
    modules.fifo_len = STREAMING_CONFIG['fifo_len']
    modules.spkcache_len = STREAMING_CONFIG['spkcache_len']
    modules.spkcache_update_period = STREAMING_CONFIG['spkcache_update_period']
    
    # Validate streaming parameters
    modules._check_streaming_parameters()
    
    model_load_time = time.time() - model_load_start
    print(f"✅ Model loaded in {model_load_time:.2f}s")
    print(f"   chunk_len={modules.chunk_len}, left_ctx={modules.chunk_left_context}, right_ctx={modules.chunk_right_context}")
    print(f"   fifo_len={modules.fifo_len}, spkcache_len={modules.spkcache_len}, update_period={modules.spkcache_update_period}")
    print()
    
    # Get files to process
    if args.single_file:
        files_to_process = [args.single_file]
    else:
        if args.dataset == "ami":
            files_to_process = get_ami_files(args.max_files)
        elif args.dataset == "voxconverse":
            files_to_process = get_voxconverse_files(args.max_files)
        elif args.dataset == "callhome":
            files_to_process = get_callhome_files(args.max_files)
        else:
            print(f"❌ Unknown dataset: {args.dataset}")
            return
    
    if not files_to_process:
        print("❌ No files found to process")
        return
    
    print(f"📂 Processing {len(files_to_process)} file(s)")
    print()
    
    # Process each file
    all_results = []
    
    for i, meeting in enumerate(files_to_process):
        print("=" * 60)
        print(f"[{i+1}/{len(files_to_process)}] Processing: {meeting}")
        print("=" * 60)
        
        result = process_meeting(model, meeting, args.dataset, args.threshold, args.verbose)
        
        if result:
            all_results.append(result)
            print(f"📊 Results for {meeting}:")
            print(f"   DER: {result['der']:.1f}%")
            print(f"   RTFx: {result['rtfx']:.1f}x")
            print(f"   Speakers: {result['detected_speakers']} detected / {result['gt_speakers']} truth")
        print()
    
    # Print final summary
    if all_results:
        print_summary(all_results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Results saved to: {args.output}")


def print_summary(results: list):
    """Print benchmark summary."""
    print()
    print("=" * 80)
    print("NEMO SORTFORMER BENCHMARK SUMMARY")
    print("=" * 80)
    
    print("📋 Results Sorted by DER:")
    print("-" * 70)
    print(f"{'Meeting':<14} {'DER %':>8} {'Miss %':>8} {'FA %':>8} {'SE %':>8} {'Speakers':>10} {'RTFx':>8}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x['der']):
        speaker_info = f"{result['detected_speakers']}/{result['gt_speakers']}"
        print(f"{result['meeting']:<14} {result['der']:>8.1f} {result['miss']:>8.1f} {result['fa']:>8.1f} {result['se']:>8.1f} {speaker_info:>10} {result['rtfx']:>8.1f}")
    
    print("-" * 70)
    
    # Calculate averages
    n = len(results)
    avg_der = sum(r['der'] for r in results) / n
    avg_miss = sum(r['miss'] for r in results) / n
    avg_fa = sum(r['fa'] for r in results) / n
    avg_se = sum(r['se'] for r in results) / n
    avg_rtfx = sum(r['rtfx'] for r in results) / n
    
    print(f"{'AVERAGE':<14} {avg_der:>8.1f} {avg_miss:>8.1f} {avg_fa:>8.1f} {avg_se:>8.1f} {'-':>10} {avg_rtfx:>8.1f}")
    print("=" * 70)
    
    print()
    print("✅ Target Check:")
    if avg_der < 15:
        print(f"   ✅ DER < 15% (achieved: {avg_der:.1f}%)")
    elif avg_der < 20:
        print(f"   🟡 DER < 20% (achieved: {avg_der:.1f}%)")
    else:
        print(f"   ❌ DER > 20% (achieved: {avg_der:.1f}%)")
    
    if avg_rtfx > 1:
        print(f"   ✅ RTFx > 1x (achieved: {avg_rtfx:.1f}x)")
    else:
        print(f"   ❌ RTFx < 1x (achieved: {avg_rtfx:.1f}x)")


def run_single_audio(args):
    """Run inference on a single audio file without ground truth."""
    print("🚀 Starting NeMo Sortformer Inference")
    print(f"   Audio: {args.audio}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Device: {args.device}")
    print()
    
    # Load model
    print("🔧 Loading NeMo Sortformer model...")
    model_load_start = time.time()
    
    device = torch.device(args.device)
    model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1",
        map_location=device
    )
    model.eval()
    model.to(device)
    
    # Apply streaming configuration
    modules = model.sortformer_modules
    modules.chunk_len = STREAMING_CONFIG['chunk_len']
    modules.chunk_left_context = STREAMING_CONFIG['chunk_left_context']
    modules.chunk_right_context = STREAMING_CONFIG['chunk_right_context']
    modules.fifo_len = STREAMING_CONFIG['fifo_len']
    modules.spkcache_len = STREAMING_CONFIG['spkcache_len']
    modules.spkcache_update_period = STREAMING_CONFIG['spkcache_update_period']
    modules._check_streaming_parameters()
    
    model_load_time = time.time() - model_load_start
    print(f"✅ Model loaded in {model_load_time:.2f}s")
    print(f"   chunk_len={modules.chunk_len}, left_ctx={modules.chunk_left_context}, right_ctx={modules.chunk_right_context}")
    print()
    
    print("=" * 60)
    result = process_audio_file(model, args.audio, args.threshold, args.verbose)
    print("=" * 60)
    
    if result and args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="NeMo Sortformer Benchmark")
    parser.add_argument("--dataset", choices=["ami", "voxconverse", "callhome"], 
                        default="ami", help="Dataset to benchmark on")
    parser.add_argument("--single-file", type=str, default=None,
                        help="Process a specific meeting (e.g., ES2004a)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Process a single audio file (no ground truth, inference only)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Speaker activity threshold")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.audio:
        run_single_audio(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
