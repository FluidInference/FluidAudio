#!/usr/bin/env python3
"""Benchmark NeMo Sortformer on CALLHOME English dataset with low-latency config."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import torchaudio
from scipy.ndimage import median_filter
from itertools import permutations
import math
import json
import time
from glob import glob

CALLHOME_DIR = os.path.expanduser("~/FluidAudioDatasets/callhome_eng")
RTTM_DIR = os.path.join(CALLHOME_DIR, "rttm")
PROGRESS_FILE = "callhome_nemo_progress.json"

print("=" * 70)
print("NeMo LOW-LATENCY Config - CALLHOME Benchmark")
print("=" * 70)

from nemo.collections.asr.models import SortformerEncLabelModel

print("\nLoading NeMo model...")
nemo_model = SortformerEncLabelModel.from_pretrained(
    'nvidia/diar_streaming_sortformer_4spk-v2.1', map_location='cpu'
)
nemo_model.eval()
nemo_model.streaming_mode = True

modules = nemo_model.sortformer_modules

# LOW-LATENCY CONFIG (matching Swift)
modules.chunk_len = 6
modules.chunk_right_context = 1  # Low latency!
modules.chunk_left_context = 1
modules.fifo_len = 40           # Smaller than NVIDIA
modules.spkcache_len = 120      # Smaller than NVIDIA
modules.spkcache_update_period = 30

if hasattr(nemo_model.preprocessor, 'featurizer'):
    if hasattr(nemo_model.preprocessor.featurizer, 'dither'):
        nemo_model.preprocessor.featurizer.dither = 0.0


def load_rttm(path):
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append((start, start + duration, speaker))
    return segments


def calculate_der(pred_probs, ref_segments, frame_shift=0.08, threshold=0.5):
    num_frames, num_speakers = pred_probs.shape
    speaker_labels = sorted(set(s[2] for s in ref_segments))
    speaker_map = {label: i for i, label in enumerate(speaker_labels)}

    ref_binary = np.zeros((num_frames, len(speaker_labels)), dtype=np.float32)
    for start, end, speaker in ref_segments:
        if speaker not in speaker_map:
            continue
        spk_idx = speaker_map[speaker]
        start_frame = max(0, min(int(start / frame_shift), num_frames))
        end_frame = max(0, min(int(end / frame_shift), num_frames))
        ref_binary[start_frame:end_frame, spk_idx] = 1.0

    if ref_binary.shape[1] < num_speakers:
        ref_binary = np.pad(ref_binary, ((0, 0), (0, num_speakers - ref_binary.shape[1])))

    pred_binary = (pred_probs > threshold).astype(np.float32)

    best_der = float('inf')
    for perm in permutations(range(num_speakers)):
        pred_permuted = pred_binary[:, perm]
        ref_speech = ref_binary.sum(axis=1) > 0
        pred_speech = pred_permuted.sum(axis=1) > 0

        miss_frames = np.sum(ref_speech & ~pred_speech)
        fa_frames = np.sum(~ref_speech & pred_speech)

        both_speech = ref_speech & pred_speech
        se_frames = 0
        for i in range(num_frames):
            if both_speech[i]:
                ref_spks = set(np.where(ref_binary[i] > 0)[0])
                pred_spks = set(np.where(pred_permuted[i] > 0)[0])
                se_frames += len(ref_spks.symmetric_difference(pred_spks)) / 2

        total_ref = np.sum(ref_speech)
        if total_ref > 0:
            der = (miss_frames + fa_frames + se_frames) / total_ref * 100
            if der < best_der:
                best_der = der

    return best_der


def process_file(wav_path, rttm_path):
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    start_time = time.time()

    with torch.no_grad():
        audio_len = torch.tensor([waveform.shape[1]])
        processed_signal, processed_signal_length = nemo_model.process_signal(
            audio_signal=waveform, audio_signal_length=audio_len
        )
    processed_signal = processed_signal[:, :, :processed_signal_length.max()]

    batch_size = processed_signal.shape[0]
    processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long, device='cpu')

    streaming_loader = modules.streaming_feat_loader(
        feat_seq=processed_signal,
        feat_seq_length=processed_signal_length,
        feat_seq_offset=processed_signal_offset,
    )

    streaming_state = modules.init_streaming_state(batch_size=1, device='cpu')
    all_preds = []

    for chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
        with torch.no_grad():
            chunk_pre_encode_embs, chunk_pre_encode_lengths = nemo_model.encoder.pre_encode(
                x=chunk_feat_seq_t, lengths=feat_lengths
            )

            spkcache_fifo_chunk_pre_encode_embs = modules.concat_embs(
                [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs],
                dim=1, device='cpu'
            )
            spkcache_fifo_chunk_pre_encode_lengths = (
                streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1] + chunk_pre_encode_lengths
            )

            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = nemo_model.frontend_encoder(
                processed_signal=spkcache_fifo_chunk_pre_encode_embs,
                processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
                bypass_pre_encode=True,
            )

            preds = nemo_model.forward_infer(
                emb_seq=spkcache_fifo_chunk_fc_encoder_embs,
                emb_seq_length=spkcache_fifo_chunk_fc_encoder_lengths
            )

            lc = round(left_offset / nemo_model.encoder.subsampling_factor)
            rc = math.ceil(right_offset / nemo_model.encoder.subsampling_factor)

            streaming_state, chunk_preds = modules.streaming_update(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                preds=preds,
                lc=lc,
                rc=rc,
            )

        all_preds.append(chunk_preds.numpy())

    proc_time = time.time() - start_time
    duration = waveform.shape[1] / sr
    rtfx = duration / proc_time

    all_preds_concat = np.concatenate(all_preds, axis=1)[0]
    filtered_preds = median_filter(all_preds_concat, size=(7, 1))

    ref_segments = load_rttm(rttm_path)
    der = calculate_der(filtered_preds, ref_segments, threshold=0.5)

    return der, rtfx, duration, len(set(s[2] for s in ref_segments))


# Load progress if exists
completed = {}
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        completed = json.load(f)
    print(f"\nResuming from {len(completed)} completed files...")

# Get all test files
audio_files = sorted(glob(os.path.join(CALLHOME_DIR, "*.wav")))
print(f"\nTotal CALLHOME files: {len(audio_files)}")
print("Config: chunk_len=6, right_context=1, fifo=40, spkcache=120")
print()

results = []
for i, wav_path in enumerate(audio_files):
    name = os.path.splitext(os.path.basename(wav_path))[0]
    rttm_path = os.path.join(RTTM_DIR, f"{name}.rttm")

    # Skip if already completed
    if name in completed:
        results.append(completed[name])
        continue

    if not os.path.exists(rttm_path):
        print(f"[{i+1:3d}/{len(audio_files)}] {name}: SKIPPED (no RTTM)")
        continue

    try:
        der, rtfx, duration, gt_speakers = process_file(wav_path, rttm_path)
        result = {
            "session": name,
            "der": float(der),
            "rtfx": float(rtfx),
            "duration": float(duration),
            "gt_speakers": gt_speakers
        }
        results.append(result)
        completed[name] = result

        print(f"[{i+1:3d}/{len(audio_files)}] {name}: DER={der:.1f}%, RTFx={rtfx:.1f}x, spks={gt_speakers}")

        # Save progress
        with open(PROGRESS_FILE, "w") as f:
            json.dump(completed, f, indent=2)

    except Exception as e:
        print(f"[{i+1:3d}/{len(audio_files)}] {name}: ERROR - {e}")

# Summary
print("\n" + "=" * 70)
print("CALLHOME NeMo BENCHMARK SUMMARY")
print("=" * 70)

if results:
    avg_der = np.mean([r["der"] for r in results])
    avg_rtfx = np.mean([r["rtfx"] for r in results])

    # Filter by speaker count (model supports max 4)
    results_4spk = [r for r in results if r["gt_speakers"] <= 4]
    if results_4spk:
        avg_der_4spk = np.mean([r["der"] for r in results_4spk])
        print(f"\nResults (≤4 speakers, {len(results_4spk)} files):")
        print(f"  Average DER: {avg_der_4spk:.1f}%")

    print(f"\nAll files ({len(results)} files):")
    print(f"  Average DER: {avg_der:.1f}%")
    print(f"  Average RTFx: {avg_rtfx:.1f}x")

    # Save final results
    with open("callhome_nemo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to callhome_nemo_results.json")
