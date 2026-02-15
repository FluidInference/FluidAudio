#!/usr/bin/env python3
"""End-to-end CoreML inference for Qwen3-ForcedAligner-0.6B.

Replicates the PyTorch ForcedAligner pipeline using the 4 converted CoreML models:
  1. Audio Encoder:      mel → audio embeddings
  2. Token Embedding:    input_ids → text embeddings
  3. Decoder Prefill:    merged embeddings + RoPE → hidden states (NAR, single pass)
  4. LM Head:            hidden states → logits → argmax → timestamps

Usage:
  uv run python run_coreml_inference.py --audio-file audio.wav --text "hello world"
  uv run python run_coreml_inference.py --compare-pytorch  # compare against PyTorch reference
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# Constants from metadata / model config
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
MEL_WINDOW_SIZE = 100  # frames per encoder window
HIDDEN_SIZE = 1024
HEAD_DIM = 128
PREFILL_SEQ_LEN = 1024
TIMESTAMP_TOKEN_ID = 151705
TIMESTAMP_SEGMENT_TIME_MS = 80
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
ROPE_THETA = 1000000.0
MROPE_SECTION = [24, 20, 20]  # T, H, W interleaving
CONV_DOWNSAMPLE_FACTOR = 8
N_WINDOW = 50
ATTENTION_SCALING = 1.0  # default rope type
AUDIO_TRANSFORMER_SEQ_LEN = 256  # max frames for audio transformer


BUILD_DIR = Path(__file__).parent / "build" / "forced-aligner"


@dataclass
class AlignmentResult:
    text: str
    start_ms: float
    end_ms: float


# ---------------------------------------------------------------------------
# Mel Spectrogram (reuse transformers WhisperFeatureExtractor)
# ---------------------------------------------------------------------------

def compute_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute 128-bin log-mel spectrogram using WhisperFeatureExtractor.

    Returns: [128, T] float32 numpy array.
    """
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor(
        feature_size=NUM_MEL_BINS,
        sampling_rate=sr,
        padding_value=0.0,
    )
    features = extractor(audio, sampling_rate=sr, return_tensors="np")
    mel = features["input_features"][0]  # [128, T]
    return mel.astype(np.float32)


# ---------------------------------------------------------------------------
# MRoPE Position Computation
# ---------------------------------------------------------------------------

def compute_rope_inv_freq() -> np.ndarray:
    """Compute inv_freq for RoPE: 1 / (theta^(2i/d)) for i in [0, d/2)."""
    dim = HEAD_DIM
    inv_freq = 1.0 / (
        ROPE_THETA ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    )
    return inv_freq.astype(np.float32)  # [64]


def apply_interleaved_mrope(freqs_3d: np.ndarray, mrope_section: list) -> np.ndarray:
    """Apply interleaved MRoPE: reorganize from [T,H,W] grids to interleaved layout.

    Args:
        freqs_3d: [3, seq_len, head_dim//2]  (T, H, W frequency grids)
        mrope_section: [24, 20, 20]

    Returns:
        freqs: [seq_len, head_dim//2]  (interleaved)
    """
    freqs_t = freqs_3d[0].copy()  # start with T grid
    for dim_idx, offset in enumerate((1, 2), start=1):  # H=1, W=2
        length = mrope_section[dim_idx] * 3
        # Select every 3rd element starting at offset
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs_3d[dim_idx, ..., idx]
    return freqs_t


def compute_mrope_cos_sin(
    position_ids_3d: np.ndarray,  # [3, seq_len]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MRoPE cos/sin embeddings for prefill.

    Args:
        position_ids_3d: [3, seq_len] position IDs for T, H, W grids

    Returns:
        cos: [1, seq_len, head_dim] float32
        sin: [1, seq_len, head_dim] float32
    """
    inv_freq = compute_rope_inv_freq()  # [64]
    seq_len = position_ids_3d.shape[1]

    # freqs[d, s, f] = position_ids_3d[d, s] * inv_freq[f]
    # Shape: [3, seq_len, 64]
    freqs_3d = np.zeros((3, seq_len, len(inv_freq)), dtype=np.float32)
    for d in range(3):
        freqs_3d[d] = np.outer(position_ids_3d[d].astype(np.float32), inv_freq)

    # Apply interleaved mrope
    freqs = apply_interleaved_mrope(freqs_3d, MROPE_SECTION)  # [seq_len, 64]

    # Duplicate: [seq_len, 64] → [seq_len, 128]
    emb = np.concatenate([freqs, freqs], axis=-1)

    cos = np.cos(emb) * ATTENTION_SCALING  # [seq_len, 128]
    sin = np.sin(emb) * ATTENTION_SCALING

    return cos[np.newaxis], sin[np.newaxis]  # [1, seq_len, 128]


def compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    """Compute 3D position IDs from attention mask.

    For forced alignment (no padding), this is simply [0, 1, 2, ..., seq_len-1]
    replicated for all 3 dimensions.

    Args:
        attention_mask: [seq_len] int

    Returns:
        position_ids: [3, seq_len] int
    """
    positions = np.cumsum(attention_mask.astype(np.float32)) - 1
    positions = np.maximum(positions, 0).astype(np.int64)
    # All 3 grids get the same positions for forced alignment
    return np.stack([positions, positions, positions], axis=0)


# ---------------------------------------------------------------------------
# Text Processing (simplified for English)
# ---------------------------------------------------------------------------

def tokenize_for_alignment(text: str, language: str = "English"):
    """Tokenize text and create aligner input with <timestamp> delimiters.

    Returns:
        word_list: list of words
        input_text: formatted text with audio placeholders and timestamps
    """
    # Simple space-based tokenization for English
    words = []
    for seg in text.split():
        # Clean: keep only letters, numbers, apostrophes
        cleaned = "".join(ch for ch in seg if ch.isalnum() or ch == "'")
        if cleaned:
            words.append(cleaned)

    input_text = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
    input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text

    return words, input_text


def fix_timestamp(data: np.ndarray) -> List[int]:
    """Fix non-monotonic timestamps using LIS (Longest Increasing Subsequence).

    Replicates Qwen3ForceAlignProcessor.fix_timestamp().
    """
    data = data.tolist()
    n = len(data)

    # LIS with parent tracking
    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_length = max(dp)
    max_idx = dp.index(max_length)

    lis_indices = []
    idx = max_idx
    while idx != -1:
        lis_indices.append(idx)
        idx = parent[idx]
    lis_indices.reverse()

    is_normal = [False] * n
    for idx in lis_indices:
        is_normal[idx] = True

    result = data.copy()
    i = 0

    while i < n:
        if not is_normal[i]:
            j = i
            while j < n and not is_normal[j]:
                j += 1

            anomaly_count = j - i

            if anomaly_count <= 2:
                left_val = None
                for k in range(i - 1, -1, -1):
                    if is_normal[k]:
                        left_val = result[k]
                        break

                right_val = None
                for k in range(j, n):
                    if is_normal[k]:
                        right_val = result[k]
                        break

                for k in range(i, j):
                    if left_val is None:
                        result[k] = right_val
                    elif right_val is None:
                        result[k] = left_val
                    else:
                        result[k] = left_val if (k - (i - 1)) <= ((j) - k) else right_val

            else:
                left_val = None
                for k in range(i - 1, -1, -1):
                    if is_normal[k]:
                        left_val = result[k]
                        break

                right_val = None
                for k in range(j, n):
                    if is_normal[k]:
                        right_val = result[k]
                        break

                if left_val is not None and right_val is not None:
                    step = (right_val - left_val) / (anomaly_count + 1)
                    for k in range(i, j):
                        result[k] = left_val + step * (k - i + 1)
                elif left_val is not None:
                    for k in range(i, j):
                        result[k] = left_val
                elif right_val is not None:
                    for k in range(i, j):
                        result[k] = right_val

            i = j
        else:
            i += 1

    return [int(res) for res in result]


# ---------------------------------------------------------------------------
# CoreML Model Loading
# ---------------------------------------------------------------------------

class CoreMLAligner:
    """Manages the 4 CoreML models for forced alignment inference."""

    def __init__(self, model_dir: Path):
        typer.echo(f"Loading CoreML models from {model_dir}...")

        # Try split encoder (audio_conv + audio_transformer) first, fall back to monolithic
        conv_path = model_dir / "forced_aligner_audio_conv.mlpackage"
        transformer_path = model_dir / "forced_aligner_audio_transformer.mlpackage"
        monolithic_path = model_dir / "forced_aligner_audio_encoder.mlpackage"

        if conv_path.exists() and transformer_path.exists():
            self.audio_conv = ct.models.MLModel(str(conv_path))
            self.audio_transformer = ct.models.MLModel(str(transformer_path))
            self.audio_encoder = None
            self._use_split_encoder = True
            typer.echo("  ✓ Audio conv (split encoder)")
            typer.echo("  ✓ Audio transformer (split encoder)")
        elif monolithic_path.exists():
            self.audio_encoder = ct.models.MLModel(str(monolithic_path))
            self.audio_conv = None
            self.audio_transformer = None
            self._use_split_encoder = False
            typer.echo("  ✓ Audio encoder (monolithic)")
        else:
            raise FileNotFoundError(
                f"No audio encoder found in {model_dir}. "
                "Need either audio_conv+audio_transformer or audio_encoder."
            )

        self.embedding = ct.models.MLModel(
            str(model_dir / "forced_aligner_embedding.mlpackage")
        )
        typer.echo("  ✓ Token embedding")

        self.decoder = ct.models.MLModel(
            str(model_dir / "forced_aligner_decoder_prefill.mlpackage")
        )
        typer.echo("  ✓ Decoder prefill")

        self.lm_head = ct.models.MLModel(
            str(model_dir / "forced_aligner_lm_head.mlpackage")
        )
        typer.echo("  ✓ LM head")

        # Load the HF processor for tokenization
        self._init_processor()

    def _init_processor(self):
        """Initialize HF processor for tokenization + feature extraction."""
        import sys
        import types
        import importlib.util

        qwen_asr_path = Path(__file__).resolve().parents[6] / "qwen3-asr"
        if not qwen_asr_path.exists():
            raise FileNotFoundError(f"qwen3-asr not found at {qwen_asr_path}")

        tb_dir = qwen_asr_path / "qwen_asr" / "core" / "transformers_backend"

        for pkg_name, pkg_path in [
            ("qwen_asr", qwen_asr_path / "qwen_asr"),
            ("qwen_asr.core", qwen_asr_path / "qwen_asr" / "core"),
            ("qwen_asr.core.transformers_backend", tb_dir),
        ]:
            if pkg_name not in sys.modules:
                mod = types.ModuleType(pkg_name)
                mod.__path__ = [str(pkg_path)]
                mod.__package__ = pkg_name
                sys.modules[pkg_name] = mod

        config_fqn = "qwen_asr.core.transformers_backend.configuration_qwen3_asr"
        if config_fqn not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                config_fqn, tb_dir / "configuration_qwen3_asr.py"
            )
            config_mod = importlib.util.module_from_spec(spec)
            sys.modules[config_fqn] = config_mod
            spec.loader.exec_module(config_mod)
        else:
            config_mod = sys.modules[config_fqn]

        proc_fqn = "qwen_asr.core.transformers_backend.processing_qwen3_asr"
        if proc_fqn not in sys.modules:
            spec2 = importlib.util.spec_from_file_location(
                proc_fqn, tb_dir / "processing_qwen3_asr.py"
            )
            proc_mod = importlib.util.module_from_spec(spec2)
            sys.modules[proc_fqn] = proc_mod
            spec2.loader.exec_module(proc_mod)
        else:
            proc_mod = sys.modules[proc_fqn]

        from transformers import AutoConfig, AutoProcessor
        try:
            AutoConfig.register("qwen3_asr", config_mod.Qwen3ASRConfig)
        except Exception:
            pass
        try:
            AutoProcessor.register(config_mod.Qwen3ASRConfig, proc_mod.Qwen3ASRProcessor)
        except Exception:
            pass

        # Full Qwen3ASRProcessor with feature extractor + tokenizer
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            fix_mistral_regex=True,
        )
        typer.echo(f"  ✓ HF processor ({type(self.processor).__name__})")

    def align(
        self,
        audio_path: str,
        text: str,
        language: str = "English",
    ) -> Tuple[List[AlignmentResult], float]:
        """Run forced alignment using CoreML models.

        Returns: (alignments, latency_ms)
        """
        start_time = time.perf_counter()

        # 1. Load audio
        audio, sr = sf.read(audio_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 2. Tokenize text with timestamp delimiters
        word_list, input_text = tokenize_for_alignment(text, language)

        # 3. Use HF processor to get input_ids + mel features
        inputs = self.processor(
            text=[input_text],
            audio=[audio],
            return_tensors="np",
            padding=True,
        )
        input_ids = inputs["input_ids"][0].astype(np.int64)  # [seq_len]
        input_features = inputs["input_features"]  # [1, 128, T]
        feature_attention_mask = inputs.get("feature_attention_mask", None)
        if feature_attention_mask is not None:
            feature_len = int(feature_attention_mask[0].sum())
        else:
            feature_len = input_features.shape[2]

        typer.echo(f"  Input IDs: {input_ids.shape}, Features: {input_features.shape}")

        # 4. Run audio encoder on mel
        mel_features = input_features[0, :, :feature_len]  # [128, feature_len]
        if self._use_split_encoder:
            audio_embeddings = self._run_split_audio_encoder(mel_features)
        else:
            audio_embeddings = self._run_audio_encoder(mel_features)
        typer.echo(f"  Audio embeddings: {audio_embeddings.shape}")

        # 5. Run token embedding
        text_embeddings = self._run_embedding(input_ids)  # [1, seq_len, 1024]
        typer.echo(f"  Text embeddings: {text_embeddings.shape}")

        # 6. Merge: replace audio_pad positions with audio features
        audio_mask = (input_ids == AUDIO_PAD_TOKEN_ID)
        num_audio_pads = audio_mask.sum()
        typer.echo(f"  Audio pads: {num_audio_pads}, Audio features: {audio_embeddings.shape[0]}")

        merged = text_embeddings[0].copy()  # [seq_len, 1024]

        # Scatter audio embeddings into pad positions
        pad_indices = np.where(audio_mask)[0]
        n_audio = min(len(pad_indices), audio_embeddings.shape[0])
        for i in range(n_audio):
            merged[pad_indices[i]] = audio_embeddings[i]

        seq_len = merged.shape[0]
        typer.echo(f"  Merged sequence: {seq_len}")

        # 7. Pad/truncate to PREFILL_SEQ_LEN
        if seq_len > PREFILL_SEQ_LEN:
            typer.echo(f"  WARNING: seq_len {seq_len} > PREFILL_SEQ_LEN {PREFILL_SEQ_LEN}, truncating")
            merged = merged[:PREFILL_SEQ_LEN]
            input_ids = input_ids[:PREFILL_SEQ_LEN]
            seq_len = PREFILL_SEQ_LEN

        padded = np.zeros((PREFILL_SEQ_LEN, HIDDEN_SIZE), dtype=np.float32)
        padded[:seq_len] = merged

        # 8. Compute MRoPE position embeddings
        attention_mask = np.zeros(PREFILL_SEQ_LEN, dtype=np.int32)
        attention_mask[:seq_len] = 1
        position_ids_3d = compute_position_ids(attention_mask)  # [3, PREFILL_SEQ_LEN]
        cos, sin = compute_mrope_cos_sin(position_ids_3d)  # [1, PREFILL_SEQ_LEN, 128]

        # 9. Run decoder prefill
        decoder_output = self._run_decoder(
            padded[np.newaxis],  # [1, PREFILL_SEQ_LEN, 1024]
            cos.astype(np.float32),
            sin.astype(np.float32),
        )
        typer.echo(f"  Decoder output: {decoder_output.shape}")

        # 10. Run LM head on actual sequence positions only
        hidden_states = decoder_output[0, :seq_len]  # [seq_len, 1024]
        logits = self._run_lm_head(hidden_states[np.newaxis])  # [1, seq_len, 5000]
        typer.echo(f"  Logits: {logits.shape}")

        # 11. Extract timestamps at <timestamp> positions
        output_ids = np.argmax(logits[0], axis=-1)  # [seq_len]
        timestamp_mask = (input_ids[:seq_len] == TIMESTAMP_TOKEN_ID)
        masked_output_ids = output_ids[timestamp_mask]
        timestamp_ms = masked_output_ids * TIMESTAMP_SEGMENT_TIME_MS

        typer.echo(f"  Timestamp tokens: {timestamp_mask.sum()}, Raw values: {masked_output_ids[:10]}...")

        # 12. Fix monotonicity
        timestamp_fixed = fix_timestamp(timestamp_ms)

        # 13. Parse into word-level alignments
        alignments = []
        for i, word in enumerate(word_list):
            start_ms = timestamp_fixed[i * 2]
            end_ms = timestamp_fixed[i * 2 + 1]
            alignments.append(AlignmentResult(
                text=word,
                start_ms=float(start_ms),
                end_ms=float(end_ms),
            ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return alignments, elapsed_ms

    def _run_audio_encoder(self, mel: np.ndarray) -> np.ndarray:
        """Run audio encoder on mel features, chunking into 100-frame windows.

        Replicates PyTorch's Qwen3ASRAudioEncoder.forward() chunking:
        - Split mel into n_window*2=100 frame chunks
        - Last chunk keeps its actual length (not padded to 100)
        - Conv stride-2 x3 layers: output_len = (input_len - 1) // 2 + 1, repeated 3x
        - Trim last chunk's output to the correct frame count

        Args:
            mel: [128, T] mel spectrogram

        Returns:
            audio_features: [N, 1024] concatenated audio embeddings
        """
        T = mel.shape[1]
        all_features = []

        # Process in 100-frame chunks
        for start in range(0, T, MEL_WINDOW_SIZE):
            end = min(start + MEL_WINDOW_SIZE, T)
            actual_chunk_len = end - start
            chunk = mel[:, start:end]  # [128, chunk_len]

            # Compute expected output frames for this chunk
            # Conv stride-2, 3 layers: out = (in - 1) // 2 + 1, repeated 3x
            expected_out = actual_chunk_len
            for _ in range(3):
                expected_out = (expected_out - 1) // 2 + 1

            # Pad to exactly MEL_WINDOW_SIZE for CoreML (fixed input shape)
            if chunk.shape[1] < MEL_WINDOW_SIZE:
                pad_width = MEL_WINDOW_SIZE - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')

            # [1, 128, 100]
            chunk_input = chunk[np.newaxis].astype(np.float32)
            result = self.audio_encoder.predict({"mel_input": chunk_input})
            features = result["audio_features"][0]  # [T', 1024]

            # Trim to expected output length (removes padding artifacts)
            features = features[:expected_out]
            all_features.append(features)

        audio_features = np.concatenate(all_features, axis=0)  # [N, 1024]
        return audio_features

    def _run_split_audio_encoder(self, mel: np.ndarray) -> np.ndarray:
        """Run split audio encoder: conv per chunk, then transformer on all frames.

        This matches the native PyTorch encoder behavior where all conv outputs
        are concatenated and processed through the transformer with full
        bidirectional attention, enabling cross-chunk attention.

        Args:
            mel: [128, T] mel spectrogram

        Returns:
            audio_features: [N, 1024] final audio embeddings
        """
        T = mel.shape[1]
        all_conv_features = []
        chunk_frame_counts = []

        # Step 1: Run conv on each 100-frame chunk
        for start in range(0, T, MEL_WINDOW_SIZE):
            end = min(start + MEL_WINDOW_SIZE, T)
            actual_chunk_len = end - start
            chunk = mel[:, start:end]  # [128, chunk_len]

            # Expected output frames for this chunk
            expected_out = actual_chunk_len
            for _ in range(3):
                expected_out = (expected_out - 1) // 2 + 1

            # Pad to 100 for CoreML fixed input shape
            if chunk.shape[1] < MEL_WINDOW_SIZE:
                pad_width = MEL_WINDOW_SIZE - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')

            chunk_input = chunk[np.newaxis].astype(np.float32)  # [1, 128, 100]
            result = self.audio_conv.predict({"mel_input": chunk_input})
            features = result["conv_features"][0]  # [13, 1024]

            # Trim to expected output length
            features = features[:expected_out]
            all_conv_features.append(features)
            chunk_frame_counts.append(expected_out)

        # Step 2: Concatenate all conv features
        conv_features = np.concatenate(all_conv_features, axis=0)  # [N, 1024]
        total_frames = conv_features.shape[0]

        # Step 3: Pad to AUDIO_TRANSFORMER_SEQ_LEN and run transformer
        if total_frames > AUDIO_TRANSFORMER_SEQ_LEN:
            typer.echo(f"  WARNING: audio frames {total_frames} > max {AUDIO_TRANSFORMER_SEQ_LEN}, truncating")
            conv_features = conv_features[:AUDIO_TRANSFORMER_SEQ_LEN]
            total_frames = AUDIO_TRANSFORMER_SEQ_LEN

        padded = np.zeros((AUDIO_TRANSFORMER_SEQ_LEN, 1024), dtype=np.float32)
        padded[:total_frames] = conv_features

        result = self.audio_transformer.predict({
            "features": padded[np.newaxis].astype(np.float32),
        })
        audio_embeddings = result["audio_embeddings"][0]  # [AUDIO_TRANSFORMER_SEQ_LEN, 1024]

        # Trim to actual frame count
        audio_embeddings = audio_embeddings[:total_frames]
        return audio_embeddings

    def _run_embedding(self, input_ids: np.ndarray) -> np.ndarray:
        """Run token embedding.

        Args:
            input_ids: [seq_len] int

        Returns:
            embeddings: [1, seq_len, 1024]
        """
        ids = input_ids[np.newaxis].astype(np.int32)  # [1, seq_len]
        result = self.embedding.predict({"input_ids": ids})
        return result["embeddings"]

    def _run_decoder(
        self,
        hidden_states: np.ndarray,
        position_cos: np.ndarray,
        position_sin: np.ndarray,
    ) -> np.ndarray:
        """Run decoder prefill (NAR, single pass).

        Args:
            hidden_states: [1, PREFILL_SEQ_LEN, 1024]
            position_cos: [1, PREFILL_SEQ_LEN, 128]
            position_sin: [1, PREFILL_SEQ_LEN, 128]

        Returns:
            output_hidden: [1, PREFILL_SEQ_LEN, 1024]
        """
        result = self.decoder.predict({
            "hidden_states": hidden_states.astype(np.float32),
            "position_cos": position_cos.astype(np.float32),
            "position_sin": position_sin.astype(np.float32),
        })
        return result["output_hidden"]

    def _run_lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run LM head to get logits.

        Args:
            hidden_states: [1, seq_len, 1024]

        Returns:
            logits: [1, seq_len, 5000]
        """
        result = self.lm_head.predict({
            "hidden_states": hidden_states.astype(np.float32),
        })
        return result["logits"]


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def infer(
    audio_file: Path = typer.Option(..., "--audio-file", help="Path to audio file"),
    text: str = typer.Option(..., "--text", help="Transcript text"),
    language: str = typer.Option("English", "--language", help="Language"),
    model_dir: Path = typer.Option(BUILD_DIR, "--model-dir", help="CoreML model directory"),
) -> None:
    """Run CoreML forced alignment on a single file."""
    aligner = CoreMLAligner(model_dir)

    typer.echo(f"\nAligning: {audio_file.name}")
    typer.echo(f"Text: {text[:80]}...")

    alignments, latency = aligner.align(str(audio_file), text, language)

    typer.echo(f"\nResults ({latency:.0f}ms):")
    for a in alignments:
        typer.echo(f"  {a.text:15s} {a.start_ms:8.1f} - {a.end_ms:8.1f} ms")


@app.command()
def compare(
    model_dir: Path = typer.Option(BUILD_DIR, "--model-dir", help="CoreML model directory"),
    reference_file: Path = typer.Option(
        BUILD_DIR / "pytorch_reference.json",
        "--reference",
        help="PyTorch reference JSON",
    ),
    num_files: int = typer.Option(3, "--num-files", help="Number of files to compare"),
) -> None:
    """Compare CoreML output against PyTorch reference timestamps."""
    if not reference_file.exists():
        typer.echo(f"ERROR: Reference file not found: {reference_file}")
        typer.echo("Run PyTorch reference first:")
        typer.echo("  uv run python run_coreml_inference.py pytorch-reference")
        raise typer.Exit(1)

    with open(reference_file) as f:
        ref_data = json.load(f)

    aligner = CoreMLAligner(model_dir)

    # Build a map from filename to full path in test-clean
    test_clean = (
        Path.home() / "Library" / "Application Support" / "FluidAudio"
        / "Datasets" / "LibriSpeech" / "test-clean"
    )
    flac_map = {}
    if test_clean.exists():
        for f in test_clean.rglob("*.flac"):
            flac_map[f.name] = str(f)

    all_errors = []
    for sample in ref_data[:num_files]:
        audio_path = sample["audio"]
        # Resolve relative/short paths
        if not Path(audio_path).exists():
            fname = Path(audio_path).name
            if fname in flac_map:
                audio_path = flac_map[fname]
            else:
                typer.echo(f"  WARNING: Cannot find {audio_path}, skipping")
                continue
        text = sample["text"]

        typer.echo(f"\n=== {Path(audio_path).name} ===")

        coreml_alignments, coreml_latency = aligner.align(audio_path, text)

        ref_alignments = sample["alignments"]
        pytorch_latency = sample["latency_ms"]

        n = min(len(ref_alignments), len(coreml_alignments))
        if len(ref_alignments) != len(coreml_alignments):
            typer.echo(f"  WARNING: word count mismatch: ref={len(ref_alignments)}, coreml={len(coreml_alignments)}")

        sample_errors = []
        for i in range(n):
            ref = ref_alignments[i]
            hyp = coreml_alignments[i]
            start_err = abs(ref["start_ms"] - hyp.start_ms)
            end_err = abs(ref["end_ms"] - hyp.end_ms)
            sample_errors.extend([start_err, end_err])
            all_errors.extend([start_err, end_err])

        errors = np.array(sample_errors)
        typer.echo(f"  Words: {n}")
        typer.echo(f"  AAS: {errors.mean():.1f}ms")
        typer.echo(f"  Max error: {errors.max():.1f}ms")
        typer.echo(f"  Within 20ms: {(errors <= 20).mean() * 100:.1f}%")
        typer.echo(f"  Within 80ms: {(errors <= 80).mean() * 100:.1f}%")
        typer.echo(f"  PyTorch: {pytorch_latency:.0f}ms, CoreML: {coreml_latency:.0f}ms")

        # Show per-word comparison for first 5 words
        typer.echo(f"  Per-word comparison (first 5):")
        for i in range(min(5, n)):
            ref = ref_alignments[i]
            hyp = coreml_alignments[i]
            typer.echo(
                f"    {ref['text']:12s}  "
                f"PT: {ref['start_ms']:7.1f}-{ref['end_ms']:7.1f}  "
                f"CM: {hyp.start_ms:7.1f}-{hyp.end_ms:7.1f}  "
                f"Δ: {abs(ref['start_ms'] - hyp.start_ms):5.1f}/{abs(ref['end_ms'] - hyp.end_ms):5.1f}ms"
            )

    # Overall summary
    if all_errors:
        errors = np.array(all_errors)
        typer.echo(f"\n=== Overall Parity ({len(all_errors)//2} boundaries) ===")
        typer.echo(f"  AAS (mean boundary error): {errors.mean():.1f}ms")
        typer.echo(f"  Max error: {errors.max():.1f}ms")
        typer.echo(f"  Within 20ms: {(errors <= 20).mean() * 100:.1f}%")
        typer.echo(f"  Within 80ms (1 segment): {(errors <= 80).mean() * 100:.1f}%")
        typer.echo(f"  Within 160ms (2 segments): {(errors <= 160).mean() * 100:.1f}%")


if __name__ == "__main__":
    app()
