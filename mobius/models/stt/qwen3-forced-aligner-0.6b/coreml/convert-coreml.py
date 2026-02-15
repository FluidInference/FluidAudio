#!/usr/bin/env python3
"""CLI for exporting Qwen3-ForcedAligner-0.6B components to CoreML.

Architecture:
  Qwen3ASRForConditionalGeneration
    └── thinker
          ├── audio_tower  → AudioEncoderFullWrapper  → forced_aligner_audio_encoder.mlpackage
          ├── model         → PrefillDecoderWrapper    → forced_aligner_decoder_prefill.mlpackage
          │   ├── embed_tokens → TextEmbeddingWrapper  → forced_aligner_embedding.mlpackage
          │   └── norm         (fused into lm_head)
          └── lm_head       → LMHeadWrapper            → forced_aligner_lm_head.mlpackage

Key difference from Qwen3-ASR: this is NAR (non-autoregressive).
The full input (audio + text with <timestamp> tokens) is processed in a single
prefill pass. No KV cache, no decode loop. Logits at timestamp positions are
argmax'd to produce ms-resolution timestamps.

Usage:
  uv run python convert-coreml.py
  uv run python convert-coreml.py --components audio_encoder
  uv run python convert-coreml.py --output-dir ./build/forced-aligner
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import coremltools as ct
import numpy as np
import torch
import typer

from individual_components import (
    AudioConvWrapper,
    AudioEncoderFullWrapper,
    AudioTransformerWrapper,
    ExportSettings,
    LMHeadWrapper,
    PrefillDecoderWrapper,
    TextEmbeddingWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
AUTHOR = "Fluid Inference"
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _load_qwen3_asr_modules():
    """Load Qwen3-ASR configuration and modeling modules via importlib.

    Bypasses qwen_asr/__init__.py which imports heavy inference dependencies
    (nagisa, soynlp, qwen-omni-utils) not needed for CoreML conversion.
    """
    qwen_asr_path = Path(__file__).resolve().parents[6] / "qwen3-asr"
    if not qwen_asr_path.exists():
        raise FileNotFoundError(
            f"qwen3-asr source not found at {qwen_asr_path}\n"
            "Clone it: git clone https://github.com/QwenLM/Qwen3-ASR.git qwen3-asr"
        )

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
    spec = importlib.util.spec_from_file_location(
        config_fqn, tb_dir / "configuration_qwen3_asr.py"
    )
    config_mod = importlib.util.module_from_spec(spec)
    sys.modules[config_fqn] = config_mod
    spec.loader.exec_module(config_mod)

    model_fqn = "qwen_asr.core.transformers_backend.modeling_qwen3_asr"
    spec2 = importlib.util.spec_from_file_location(
        model_fqn, tb_dir / "modeling_qwen3_asr.py"
    )
    model_mod = importlib.util.module_from_spec(spec2)
    sys.modules[model_fqn] = model_mod
    spec2.loader.exec_module(model_mod)

    return config_mod, model_mod


def _load_model(model_id: str):
    """Load Qwen3-ForcedAligner model via transformers."""
    typer.echo(f"Loading model: {model_id}")

    config_mod, model_mod = _load_qwen3_asr_modules()
    typer.echo("  Loaded Qwen3-ASR source modules (bypassed heavy deps)")

    from transformers import AutoConfig, AutoModel

    # Patch ROPE_INIT_FUNCTIONS for transformers 5.x compatibility.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope_init(config, device=None, **kwargs):
            base = config.rope_theta
            dim = config.head_dim
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
        typer.echo("  Patched ROPE_INIT_FUNCTIONS: added 'default' rope type")

    AutoConfig.register("qwen3_asr", config_mod.Qwen3ASRConfig)
    AutoConfig.register("qwen3_asr_audio_encoder", config_mod.Qwen3ASRAudioEncoderConfig)
    AutoModel.register(config_mod.Qwen3ASRConfig, model_mod.Qwen3ASRForConditionalGeneration)
    typer.echo("  Registered Qwen3ASR custom classes with AutoConfig/AutoModel")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    def _ensure_attr(cfg, attr, default):
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    _ensure_attr(config, "pad_token_id", None)
    if hasattr(config, "thinker_config"):
        _ensure_attr(config.thinker_config, "pad_token_id", None)

    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    typer.echo(f"  Model loaded. Type: {type(model).__name__}")

    # Log ForcedAligner-specific config
    typer.echo(f"  timestamp_token_id: {getattr(config, 'timestamp_token_id', 'N/A')}")
    typer.echo(f"  timestamp_segment_time: {getattr(config, 'timestamp_segment_time', 'N/A')}ms")

    return model


def _get_audio_encoder(model):
    if hasattr(model, "thinker"):
        return model.thinker.audio_tower
    return model.audio_tower


def _get_text_model(model):
    if hasattr(model, "thinker"):
        return model.thinker.model
    return model.model


def _get_lm_head(model):
    if hasattr(model, "thinker"):
        return model.thinker.lm_head
    return model.lm_head


def _get_text_norm(model):
    return _get_text_model(model).norm


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Audio Encoder Conversion
# ---------------------------------------------------------------------------

def convert_audio_encoder(model, settings: ExportSettings) -> Path:
    """Convert the audio encoder (24-layer, 1024 dim) to CoreML."""
    typer.echo("\n=== Converting Audio Encoder ===")

    audio_encoder = _get_audio_encoder(model)
    audio_encoder.eval()

    wrapper = AudioEncoderFullWrapper(audio_encoder)
    wrapper.eval()

    # Single window: 100 mel frames (n_window * 2)
    max_mel_frames = 100
    mel_input = torch.randn(1, NUM_MEL_BINS, max_mel_frames, dtype=torch.float32)

    typer.echo(f"  Trace input shape: {mel_input.shape}")

    with torch.inference_mode():
        ref_output = wrapper(mel_input)
        typer.echo(f"  Reference output shape: {ref_output.shape}")

    mel_input = mel_input.clone()

    typer.echo("  Tracing audio encoder...")
    traced = torch.jit.trace(wrapper, (mel_input,), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="mel_input",
            shape=(1, NUM_MEL_BINS, max_mel_frames),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="audio_features", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=None,  # default FP16
    )

    path = settings.output_dir / "forced_aligner_audio_encoder.mlpackage"
    _save_mlpackage(coreml_model, path, "Qwen3-ForcedAligner audio encoder (24 layers, 1024 dim)")

    return path


# ---------------------------------------------------------------------------
# Audio Conv Conversion (split encoder: conv frontend only)
# ---------------------------------------------------------------------------

def convert_audio_conv(model, settings: ExportSettings) -> Path:
    """Convert the audio encoder conv frontend (no transformer) to CoreML."""
    typer.echo("\n=== Converting Audio Conv (split encoder) ===")

    audio_encoder = _get_audio_encoder(model)
    audio_encoder.eval()

    wrapper = AudioConvWrapper(audio_encoder)
    wrapper.eval()

    max_mel_frames = 100
    mel_input = torch.randn(1, NUM_MEL_BINS, max_mel_frames, dtype=torch.float32)

    typer.echo(f"  Trace input shape: {mel_input.shape}")

    with torch.inference_mode():
        ref_output = wrapper(mel_input)
        typer.echo(f"  Reference output shape: {ref_output.shape}")

    mel_input = mel_input.clone()

    typer.echo("  Tracing audio conv...")
    traced = torch.jit.trace(wrapper, (mel_input,), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="mel_input",
            shape=(1, NUM_MEL_BINS, max_mel_frames),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="conv_features", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=None,  # default FP16
    )

    path = settings.output_dir / "forced_aligner_audio_conv.mlpackage"
    _save_mlpackage(coreml_model, path, "Qwen3-ForcedAligner audio conv frontend (3x stride-2 conv)")

    return path


# ---------------------------------------------------------------------------
# Audio Transformer Conversion (split encoder: transformer + projection)
# ---------------------------------------------------------------------------

def convert_audio_transformer(model, settings: ExportSettings) -> Path:
    """Convert the audio encoder transformer + projection to CoreML.

    This takes concatenated conv features from multiple chunks and runs
    the 24-layer transformer with full bidirectional attention, matching
    the native encoder's cross-chunk attention behavior.
    """
    typer.echo("\n=== Converting Audio Transformer (split encoder) ===")

    audio_encoder = _get_audio_encoder(model)
    audio_encoder.eval()

    wrapper = AudioTransformerWrapper(audio_encoder)
    wrapper.eval()

    AUDIO_SEQ = AudioTransformerWrapper.AUDIO_TRANSFORMER_SEQ_LEN
    hidden_size = 1024
    features = torch.randn(1, AUDIO_SEQ, hidden_size, dtype=torch.float32)

    typer.echo(f"  Trace input shape: {features.shape}")
    typer.echo(f"  Max audio frames: {AUDIO_SEQ}")

    with torch.inference_mode():
        ref_output = wrapper(features)
        typer.echo(f"  Reference output shape: {ref_output.shape}")

    features = features.clone()

    typer.echo("  Tracing audio transformer...")
    traced = torch.jit.trace(wrapper, (features,), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="features",
            shape=(1, AUDIO_SEQ, hidden_size),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="audio_embeddings", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=ct.precision.FLOAT32,
    )

    path = settings.output_dir / "forced_aligner_audio_transformer.mlpackage"
    _save_mlpackage(
        coreml_model, path,
        f"Qwen3-ForcedAligner audio transformer (24 layers, bidirectional, max {AUDIO_SEQ} frames)"
    )

    return path


# ---------------------------------------------------------------------------
# Text Embedding Conversion
# ---------------------------------------------------------------------------

def convert_embedding(model, settings: ExportSettings) -> Path:
    """Convert the token embedding layer to CoreML."""
    typer.echo("\n=== Converting Token Embedding ===")

    text_model = _get_text_model(model)
    wrapper = TextEmbeddingWrapper(text_model)
    wrapper.eval()

    seq_len = 32
    input_ids = torch.zeros(1, seq_len, dtype=torch.int32)

    with torch.inference_mode():
        ref_output = wrapper(input_ids)
        typer.echo(f"  Reference output shape: {ref_output.shape}")

    input_ids = input_ids.clone()

    typer.echo("  Tracing embedding layer...")
    traced = torch.jit.trace(wrapper, (input_ids,), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="input_ids",
            shape=(1, ct.RangeDim(1, settings.max_seq_length)),
            dtype=np.int32,
        ),
    ]
    outputs = [
        ct.TensorType(name="embeddings", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
    )

    path = settings.output_dir / "forced_aligner_embedding.mlpackage"
    _save_mlpackage(coreml_model, path, "Qwen3-ForcedAligner token embedding (152064 vocab → 1024 dim)")

    return path


# ---------------------------------------------------------------------------
# LM Head Conversion
# ---------------------------------------------------------------------------

def convert_lm_head(model, settings: ExportSettings) -> Path:
    """Convert the LM head (norm + linear) to CoreML.

    For ForcedAligner, the LM head processes the FULL sequence at once
    (not token by token). The logits at timestamp positions are argmax'd
    to get ms-resolution timestamps.
    """
    typer.echo("\n=== Converting LM Head ===")

    lm_head = _get_lm_head(model)
    norm = _get_text_norm(model)
    wrapper = LMHeadWrapper(lm_head, norm)
    wrapper.eval()

    hidden_size = 1024
    # Realistic magnitude for tracing: hidden states have values in ~[-300, 300]
    hidden_states = torch.randn(1, 1, hidden_size, dtype=torch.float32) * 200.0

    with torch.inference_mode():
        ref_output = wrapper(hidden_states)
        typer.echo(f"  Reference output shape: {ref_output.shape}")
        typer.echo(f"  Trace input range: [{hidden_states.min():.1f}, {hidden_states.max():.1f}]")

    hidden_states = hidden_states.clone()

    typer.echo("  Tracing LM head...")
    traced = torch.jit.trace(wrapper, (hidden_states,), strict=False)
    traced.eval()

    # Use RangeDim for seq_len since this processes full sequences
    inputs = [
        ct.TensorType(
            name="hidden_states",
            shape=(1, ct.RangeDim(1, settings.max_seq_length), hidden_size),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=ct.precision.FLOAT32,
    )

    path = settings.output_dir / "forced_aligner_lm_head.mlpackage"
    _save_mlpackage(coreml_model, path, "Qwen3-ForcedAligner LM head (norm + linear → 5000 timestamp values)")

    return path


# ---------------------------------------------------------------------------
# Prefill Decoder Conversion (NAR — single pass)
# ---------------------------------------------------------------------------

def convert_decoder_prefill(model, settings: ExportSettings) -> Path:
    """Convert the full decoder stack for NAR prefill.

    Unlike Qwen3-ASR which needs autoregressive decode with KV cache,
    the ForcedAligner processes everything in one shot:
      1. Audio embeddings + text tokens → concatenated sequence
      2. Single prefill pass → hidden states for all positions
      3. LM head → logits → argmax at timestamp positions → ms timestamps

    No KV cache management needed.
    """
    typer.echo("\n=== Converting Decoder Prefill (NAR) ===")

    text_model = _get_text_model(model)
    num_layers = len(text_model.layers)
    hidden_size = 1024
    head_dim = 128

    wrapper = PrefillDecoderWrapper(text_model)
    wrapper.eval()

    PREFILL_SEQ = PrefillDecoderWrapper.PREFILL_SEQ_LEN
    typer.echo(f"  {num_layers} layers, prefill seq_len={PREFILL_SEQ}")

    hidden_states = torch.randn(1, PREFILL_SEQ, hidden_size, dtype=torch.float32)
    position_cos = torch.randn(1, PREFILL_SEQ, head_dim, dtype=torch.float32)
    position_sin = torch.randn(1, PREFILL_SEQ, head_dim, dtype=torch.float32)

    with torch.inference_mode():
        ref_output = wrapper(hidden_states, position_cos, position_sin)
        typer.echo(f"  Output hidden: {ref_output.shape}")

    trace_inputs = (
        hidden_states.clone(),
        position_cos.clone(),
        position_sin.clone(),
    )

    typer.echo("  Tracing decoder prefill...")
    traced = torch.jit.trace(wrapper, trace_inputs, strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="hidden_states",
            shape=(1, PREFILL_SEQ, hidden_size),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="position_cos",
            shape=(1, PREFILL_SEQ, head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="position_sin",
            shape=(1, PREFILL_SEQ, head_dim),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="output_hidden", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=ct.precision.FLOAT32,
    )

    path = settings.output_dir / "forced_aligner_decoder_prefill.mlpackage"
    _save_mlpackage(
        coreml_model, path,
        f"Qwen3-ForcedAligner decoder prefill ({num_layers} layers, NAR)"
    )

    return path


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata(
    settings: ExportSettings,
    component_paths: Dict[str, object],
    model_id: str,
) -> Path:
    metadata = {
        "model_id": model_id,
        "architecture": "Qwen3ASRForConditionalGeneration",
        "inference_mode": "NAR (non-autoregressive prefill-only)",
        "sample_rate": SAMPLE_RATE,
        "num_mel_bins": NUM_MEL_BINS,
        "max_audio_seconds": settings.max_audio_seconds,
        "max_seq_length": settings.max_seq_length,
        "audio_encoder": {
            "n_window": 50,
            "n_window_infer": 800,
            "mel_window_size": 100,
            "conv_downsample_factor": 8,
            "d_model": 1024,
            "output_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "ffn_dim": 4096,
        },
        "text_decoder": {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_layers": 28,
            "num_attention_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 152064,  # embedding table vocab
            "lm_head_output_dim": 5000,  # timestamp prediction dim (NOT vocab)
            "rope_theta": 1000000,
            "rope_interleaved": True,
            "mrope_section": [24, 20, 20],
        },
        "special_tokens": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676,
            "timestamp_token_id": 151705,
            "timestamp_segment_time_ms": 80,
        },
        "components": component_paths,
        "export_settings": {
            "compute_units": settings.compute_units.name,
            "compute_precision": (
                settings.compute_precision.name
                if settings.compute_precision is not None
                else "FLOAT32"
            ),
        },
    }

    path = settings.output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str))
    typer.echo(f"\nMetadata written to {path}")
    return path


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

@app.command()
def convert(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model-id", help="HuggingFace model ID"),
    output_dir: Path = typer.Option(
        Path("build/forced-aligner"),
        "--output-dir",
        help="Output directory for CoreML packages",
    ),
    components: Optional[str] = typer.Option(
        None,
        "--components",
        help="Comma-separated: audio_encoder,embedding,lm_head,decoder_prefill. Default: all.",
    ),
    max_seq_length: int = typer.Option(1024, "--max-seq-length", help="Max sequence length for decoder"),
    max_audio_seconds: float = typer.Option(300.0, "--max-audio-seconds", help="Max audio duration (5 min)"),
    no_ane: bool = typer.Option(
        False, "--no-ane",
        help="Target CPU+GPU only (exclude ANE).",
    ),
    no_optimize: bool = typer.Option(
        False, "--no-optimize",
        help="Skip MIL optimization passes.",
    ),
) -> None:
    """Export Qwen3-ForcedAligner-0.6B components to CoreML."""

    target_units = ct.ComputeUnit.CPU_AND_GPU if no_ane else ct.ComputeUnit.ALL
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=target_units,
        deployment_target=ct.target.iOS17,
        compute_precision=None,
        max_audio_seconds=max_audio_seconds,
        max_seq_length=max_seq_length,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Export configuration:")
    typer.echo(f"  Model: {model_id}")
    typer.echo(f"  Output: {output_dir}")
    typer.echo(f"  Max seq length: {max_seq_length}")
    typer.echo(f"  Max audio seconds: {max_audio_seconds}")
    if no_ane:
        typer.echo(f"  Compute units: CPU_AND_GPU (no ANE)")

    if components is not None:
        convert_list = [c.strip() for c in components.split(",")]
    else:
        convert_list = ["audio_conv", "audio_transformer", "embedding", "lm_head", "decoder_prefill"]

    typer.echo(f"  Components: {convert_list}")

    model = _load_model(model_id)

    component_paths: Dict[str, object] = {}

    if "audio_encoder" in convert_list:
        path = convert_audio_encoder(model, settings)
        component_paths["audio_encoder"] = {"path": path.name}

    if "audio_conv" in convert_list:
        path = convert_audio_conv(model, settings)
        component_paths["audio_conv"] = {"path": path.name}

    if "audio_transformer" in convert_list:
        path = convert_audio_transformer(model, settings)
        component_paths["audio_transformer"] = {"path": path.name}

    if "embedding" in convert_list:
        path = convert_embedding(model, settings)
        component_paths["embedding"] = {"path": path.name}

    if "lm_head" in convert_list:
        path = convert_lm_head(model, settings)
        component_paths["lm_head"] = {"path": path.name}

    if "decoder_prefill" in convert_list:
        path = convert_decoder_prefill(model, settings)
        component_paths["decoder_prefill"] = {"path": path.name, "num_layers": 28}

    write_metadata(settings, component_paths, model_id)

    typer.echo("\n=== Conversion complete ===")


if __name__ == "__main__":
    app()
