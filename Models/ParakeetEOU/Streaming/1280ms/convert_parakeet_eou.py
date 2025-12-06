#!/usr/bin/env python3
"""CLI for exporting Parakeet Realtime EOU 120M components to CoreML.

This model is a cache-aware streaming FastConformer-RNNT model optimized for
low-latency speech recognition with end-of-utterance detection.

Key differences from Parakeet TDT v3:
- Smaller model (120M vs 600M params)
- No duration outputs (standard RNNT, not TDT)
- Cache-aware streaming encoder (17 layers, attention context [70,1])
- Special <EOU> token for end-of-utterance detection
- Optimized for 80-160ms latency

Reference: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import (
    DecoderWrapper,
    EncoderWrapper,
    ExportSettings,
    JointWrapper,
    JointDecisionWrapper,
    JointDecisionSingleStep,
    PreprocessorWrapper,
    MelEncoderWrapper,
    _coreml_convert,
)

def apply_stft_patch():
    # Monkey patch coremltools.stft to handle extra arguments from newer torch versions
    try:
        import coremltools.converters.mil.frontend.torch.ops as torch_ops
        _original_stft = torch_ops.stft

        def patched_stft(context, node):
            if len(node.inputs) > 8:
                node.inputs = node.inputs[:8]
            return _original_stft(context, node)

        torch_ops.stft = patched_stft
        if "stft" in torch_ops._TORCH_OPS_REGISTRY:
            torch_ops._TORCH_OPS_REGISTRY["stft"] = patched_stft
        print("Monkey patched coremltools.stft for compatibility.")
    except Exception as e:
        print(f"Warning: Could not monkey patch stft: {e}")

DEFAULT_MODEL_ID = "nvidia/parakeet_realtime_eou_120m-v1"
AUTHOR = "Fluid Inference"


def _compute_length(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def _prepare_audio(
    validation_audio: Optional[Path],
    sample_rate: int,
    max_samples: int,
    seed: Optional[int],
) -> torch.Tensor:
    if validation_audio is None:
        if seed is not None:
            torch.manual_seed(seed)
        audio = torch.randn(1, max_samples, dtype=torch.float32)
        return audio

    data, sr = sf.read(str(validation_audio), dtype="float32")
    if sr != sample_rate:
        raise typer.BadParameter(
            f"Validation audio sample rate {sr} does not match model rate {sample_rate}"
        )

    if data.ndim > 1:
        data = data[:, 0]

    if data.size == 0:
        raise typer.BadParameter("Validation audio is empty")

    if data.size < max_samples:
        pad_width = max_samples - data.size
        data = np.pad(data, (0, pad_width))
    elif data.size > max_samples:
        data = data[:max_samples]

    audio = torch.from_numpy(data).unsqueeze(0).to(dtype=torch.float32)
    return audio


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _parse_compute_units(name: str) -> ct.ComputeUnit:
    """Parse a human-friendly compute units string into ct.ComputeUnit."""
    normalized = str(name).strip().upper()
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_NEURALENGINE": ct.ComputeUnit.CPU_AND_NE,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute units '{name}'. Choose from: " + ", ".join(mapping.keys())
        )
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    """Parse compute precision string into ct.precision or None."""
    if name is None:
        return None
    normalized = str(name).strip().upper()
    if normalized == "":
        return None
    mapping = {
        "FLOAT32": ct.precision.FLOAT32,
        "FLOAT16": ct.precision.FLOAT16,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute precision '{name}'. Choose from: "
            + ", ".join(mapping.keys())
        )
    return mapping[normalized]


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    nemo_path: Optional[Path] = typer.Option(
        None,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to parakeet_realtime_eou_120m-v1.nemo checkpoint (skip to auto-download)",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Model identifier to download when --nemo-path is omitted",
    ),
    output_dir: Path = typer.Option(
        Path("parakeet_eou_coreml"),
        help="Directory where mlpackages and metadata will be written",
    ),
    preprocessor_cu: str = typer.Option(
        "CPU_ONLY",
        "--preprocessor-cu",
        help="Compute units for preprocessor (default CPU_ONLY)",
    ),
    mel_encoder_cu: str = typer.Option(
        "CPU_ONLY",
        "--mel-encoder-cu",
        help="Compute units for fused mel+encoder (default CPU_ONLY)",
    ),
    compute_precision: Optional[str] = typer.Option(
        None,
        "--compute-precision",
        help="Export precision: FLOAT32 (default) or FLOAT16 to shrink non-quantized weights.",
    ),
    max_audio_seconds: float = typer.Option(
        15.0,
        "--max-audio-seconds",
        help="Maximum audio duration in seconds for the fixed window export",
    ),
    validation_audio: Optional[Path] = typer.Option(
        None,
        "--validation-audio",
        exists=True,
        resolve_path=True,
        help="Path to a 16kHz WAV file for tracing (uses random if not provided)",
    ),
) -> None:
    """Export all Parakeet Realtime EOU sub-modules to CoreML.

    This exports the cache-aware streaming FastConformer-RNNT model for
    low-latency speech recognition with end-of-utterance detection.
    """
    export_settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=_parse_compute_precision(compute_precision),
        max_audio_seconds=max_audio_seconds,
        max_symbol_steps=1,
    )

    typer.echo("Export configuration:")
    typer.echo(asdict(export_settings))

    output_dir.mkdir(parents=True, exist_ok=True)
    pre_cu = _parse_compute_units(preprocessor_cu)
    melenc_cu = _parse_compute_units(mel_encoder_cu)

    if nemo_path is not None:
        typer.echo(f"Loading NeMo model from {nemo_path}…")
        # Try loading as generic ASRModel first, then specific class
        try:
            asr_model = nemo_asr.models.ASRModel.restore_from(
                str(nemo_path), map_location="cpu"
            )
        except Exception:
            # Fallback to EncDecRNNTBPEModel
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
                str(nemo_path), map_location="cpu"
            )
        checkpoint_meta = {
            "type": "file",
            "path": str(nemo_path),
        }
    else:
        typer.echo(f"Downloading NeMo model via {model_id}…")
        # Use ASRModel.from_pretrained as recommended for this model
        try:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_id, map_location="cpu"
            )
        except Exception:
            # Fallback to EncDecRNNTBPEModel
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_id, map_location="cpu"
            )
        checkpoint_meta = {
            "type": "pretrained",
            "model_id": model_id,
        }
    asr_model.eval()

    # Print model info
    typer.echo(f"Model class: {type(asr_model).__name__}")
    typer.echo(f"Encoder class: {type(asr_model.encoder).__name__}")

    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = _compute_length(export_settings.max_audio_seconds, sample_rate)

    # Prepare audio for tracing
    if validation_audio is not None:
        typer.echo(f"Using validation audio: {validation_audio}")
        audio_tensor = _prepare_audio(validation_audio, sample_rate, max_samples, seed=None)
    else:
        typer.echo("Using random audio for tracing (seed=42)")
        audio_tensor = _prepare_audio(None, sample_rate, max_samples, seed=42)

    audio_length = torch.tensor([max_samples], dtype=torch.int32)

    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder = EncoderWrapper(asr_model.encoder.eval())
    decoder = DecoderWrapper(asr_model.decoder.eval())
    joint = JointWrapper(asr_model.joint.eval())

    decoder_export_flag = getattr(asr_model.decoder, "_rnnt_export", False)
    asr_model.decoder._rnnt_export = True

    try:
        with torch.no_grad():
            mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)
            mel_length_ref = mel_length_ref.to(dtype=torch.int32)
            encoder_ref, encoder_length_ref, frame_times_ref = encoder(
                mel_ref, mel_length_ref
            )
            encoder_length_ref = encoder_length_ref.to(dtype=torch.int32)

            # Clone tensors to drop inference flags
            mel_ref = mel_ref.clone().detach()
            mel_length_ref = mel_length_ref.clone().detach()
            encoder_ref = encoder_ref.clone().detach()
            encoder_length_ref = encoder_length_ref.clone().detach()
            frame_times_ref = frame_times_ref.clone().detach()

        vocab_size = int(asr_model.tokenizer.vocab_size)
        decoder_hidden = int(asr_model.decoder.pred_hidden)
        decoder_layers = int(asr_model.decoder.pred_rnn_layers)

        # Check if model has extra outputs (TDT-style duration)
        num_extra = getattr(asr_model.joint, "num_extra_outputs", 0)
        typer.echo(f"Vocab size: {vocab_size}, num_extra_outputs: {num_extra}")

        targets = torch.full(
            (1, export_settings.max_symbol_steps),
            fill_value=asr_model.decoder.blank_idx,
            dtype=torch.int32,
        )
        target_lengths = torch.tensor(
            [export_settings.max_symbol_steps], dtype=torch.int32
        )
        zero_state = torch.zeros(
            decoder_layers,
            1,
            decoder_hidden,
            dtype=torch.float32,
        )

        with torch.no_grad():
            decoder_ref, h_ref, c_ref = decoder(
                targets, target_lengths, zero_state, zero_state
            )
            joint_ref = joint(encoder_ref, decoder_ref)

        decoder_ref = decoder_ref.clone()
        h_ref = h_ref.clone()
        c_ref = c_ref.clone()
        joint_ref = joint_ref.clone()

        typer.echo(f"Encoder output shape: {encoder_ref.shape}")
        typer.echo(f"Decoder output shape: {decoder_ref.shape}")
        typer.echo(f"Joint output shape: {joint_ref.shape}")

        # === Export Preprocessor ===
        typer.echo("Tracing and converting preprocessor…")
        preprocessor = preprocessor.cpu()
        audio_tensor = audio_tensor.cpu()
        audio_length = audio_length.cpu()
        traced_preprocessor = torch.jit.trace(
            preprocessor, (audio_tensor, audio_length), strict=False
        )
        traced_preprocessor.eval()
        preprocessor_inputs = [
            ct.TensorType(
                name="audio_signal",
                shape=(1, ct.RangeDim(1, max_samples)),
                dtype=np.float32,
            ),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ]
        preprocessor_outputs = [
            ct.TensorType(name="mel", dtype=np.float32),
            ct.TensorType(name="mel_length", dtype=np.int32),
        ]
        preprocessor_model = _coreml_convert(
            traced_preprocessor,
            preprocessor_inputs,
            preprocessor_outputs,
            export_settings,
            compute_units_override=pre_cu,
        )
        preprocessor_path = output_dir / "parakeet_eou_preprocessor.mlpackage"
        _save_mlpackage(
            preprocessor_model,
            preprocessor_path,
            f"Parakeet EOU preprocessor ({max_audio_seconds}s window)",
        )

        # === Export Encoder ===
        typer.echo("Tracing and converting encoder…")
        traced_encoder = torch.jit.trace(
            encoder, (mel_ref, mel_length_ref), strict=False
        )
        traced_encoder.eval()
        encoder_inputs = [
            ct.TensorType(
                name="mel", shape=_tensor_shape(mel_ref), dtype=np.float32
            ),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ]
        encoder_outputs = [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
            ct.TensorType(name="frame_times", dtype=np.float32),
        ]
        encoder_model = _coreml_convert(
            traced_encoder,
            encoder_inputs,
            encoder_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        encoder_path = output_dir / "parakeet_eou_encoder.mlpackage"
        _save_mlpackage(
            encoder_model,
            encoder_path,
            f"Parakeet EOU encoder ({max_audio_seconds}s window)",
        )

        # === Export Fused Mel+Encoder ===
        typer.echo("Tracing and converting fused mel+encoder…")
        mel_encoder = MelEncoderWrapper(preprocessor, encoder)
        traced_mel_encoder = torch.jit.trace(
            mel_encoder, (audio_tensor, audio_length), strict=False
        )
        traced_mel_encoder.eval()
        mel_encoder_inputs = [
            ct.TensorType(
                name="audio_signal", shape=(1, max_samples), dtype=np.float32
            ),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ]
        mel_encoder_outputs = [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
            ct.TensorType(name="frame_times", dtype=np.float32),
        ]
        mel_encoder_model = _coreml_convert(
            traced_mel_encoder,
            mel_encoder_inputs,
            mel_encoder_outputs,
            export_settings,
            compute_units_override=melenc_cu,
        )
        mel_encoder_path = output_dir / "parakeet_eou_mel_encoder.mlpackage"
        _save_mlpackage(
            mel_encoder_model,
            mel_encoder_path,
            f"Parakeet EOU fused Mel+Encoder ({max_audio_seconds}s window)",
        )

        # === Export Decoder ===
        typer.echo("Tracing and converting decoder…")
        traced_decoder = torch.jit.trace(
            decoder,
            (targets, target_lengths, zero_state, zero_state),
            strict=False,
        )
        traced_decoder.eval()
        decoder_inputs = [
            ct.TensorType(
                name="targets", shape=_tensor_shape(targets), dtype=np.int32
            ),
            ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
            ct.TensorType(
                name="h_in", shape=_tensor_shape(zero_state), dtype=np.float32
            ),
            ct.TensorType(
                name="c_in", shape=_tensor_shape(zero_state), dtype=np.float32
            ),
        ]
        decoder_outputs = [
            ct.TensorType(name="decoder", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ]
        decoder_model = _coreml_convert(
            traced_decoder,
            decoder_inputs,
            decoder_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        decoder_path = output_dir / "parakeet_eou_decoder.mlpackage"
        _save_mlpackage(
            decoder_model,
            decoder_path,
            "Parakeet EOU decoder (RNNT prediction network)",
        )

        # === Export Joint ===
        typer.echo("Tracing and converting joint…")
        traced_joint = torch.jit.trace(
            joint,
            (encoder_ref, decoder_ref),
            strict=False,
        )
        traced_joint.eval()
        joint_inputs = [
            ct.TensorType(
                name="encoder", shape=_tensor_shape(encoder_ref), dtype=np.float32
            ),
            ct.TensorType(
                name="decoder", shape=_tensor_shape(decoder_ref), dtype=np.float32
            ),
        ]
        joint_outputs = [
            ct.TensorType(name="logits", dtype=np.float32),
        ]
        joint_model = _coreml_convert(
            traced_joint,
            joint_inputs,
            joint_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        joint_path = output_dir / "parakeet_eou_joint.mlpackage"
        _save_mlpackage(
            joint_model,
            joint_path,
            "Parakeet EOU joint network (RNNT)",
        )

        # === Export Joint Decision Head ===
        typer.echo("Tracing and converting joint decision head…")
        joint_decision = JointDecisionWrapper(joint, vocab_size=vocab_size)
        traced_joint_decision = torch.jit.trace(
            joint_decision,
            (encoder_ref, decoder_ref),
            strict=False,
        )
        traced_joint_decision.eval()
        joint_decision_inputs = [
            ct.TensorType(
                name="encoder", shape=_tensor_shape(encoder_ref), dtype=np.float32
            ),
            ct.TensorType(
                name="decoder", shape=_tensor_shape(decoder_ref), dtype=np.float32
            ),
        ]
        joint_decision_outputs = [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
        ]
        joint_decision_model = _coreml_convert(
            traced_joint_decision,
            joint_decision_inputs,
            joint_decision_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        joint_decision_path = output_dir / "parakeet_eou_joint_decision.mlpackage"
        _save_mlpackage(
            joint_decision_model,
            joint_decision_path,
            "Parakeet EOU joint + decision head (softmax, argmax)",
        )

        # === Export Single-Step Joint Decision ===
        typer.echo("Tracing and converting single-step joint decision…")
        jd_single = JointDecisionSingleStep(joint, vocab_size=vocab_size)
        # Create single-step slices from refs
        enc_step = encoder_ref[:, :, :1].contiguous()
        dec_step = decoder_ref[:, :, :1].contiguous()
        traced_jd_single = torch.jit.trace(
            jd_single,
            (enc_step, dec_step),
            strict=False,
        )
        traced_jd_single.eval()
        jd_single_inputs = [
            ct.TensorType(
                name="encoder_step",
                shape=(1, enc_step.shape[1], 1),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="decoder_step",
                shape=(1, dec_step.shape[1], 1),
                dtype=np.float32,
            ),
        ]
        jd_single_outputs = [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
            ct.TensorType(name="top_k_ids", dtype=np.int32),
            ct.TensorType(name="top_k_logits", dtype=np.float32),
        ]
        jd_single_model = _coreml_convert(
            traced_jd_single,
            jd_single_inputs,
            jd_single_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        jd_single_path = output_dir / "parakeet_eou_joint_decision_single_step.mlpackage"
        _save_mlpackage(
            jd_single_model,
            jd_single_path,
            "Parakeet EOU single-step joint decision (current frame)",
        )

        # === Save Metadata ===
        metadata: Dict[str, object] = {
            "model_id": model_id,
            "model_name": "parakeet_realtime_eou_120m-v1",
            "model_class": type(asr_model).__name__,
            "encoder_class": type(asr_model.encoder).__name__,
            "sample_rate": sample_rate,
            "max_audio_seconds": export_settings.max_audio_seconds,
            "max_audio_samples": max_samples,
            "max_symbol_steps": export_settings.max_symbol_steps,
            "vocab_size": vocab_size,
            "vocab_with_blank": vocab_size + 1,
            "decoder_hidden": decoder_hidden,
            "decoder_layers": decoder_layers,
            "num_extra_outputs": num_extra,
            "has_eou_token": True,
            "checkpoint": checkpoint_meta,
            "coreml": {
                "compute_units": export_settings.compute_units.name,
                "compute_precision": (
                    export_settings.compute_precision.name
                    if export_settings.compute_precision is not None
                    else "FLOAT32"
                ),
            },
            "components": {
                "preprocessor": {
                    "inputs": {
                        "audio_signal": [1, max_samples],
                        "audio_length": [1],
                    },
                    "outputs": {
                        "mel": list(_tensor_shape(mel_ref)),
                        "mel_length": [1],
                    },
                    "path": preprocessor_path.name,
                },
                "encoder": {
                    "inputs": {
                        "mel": list(_tensor_shape(mel_ref)),
                        "mel_length": [1],
                    },
                    "outputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "encoder_length": [1],
                        "frame_times": [1, _tensor_shape(encoder_ref)[2]],
                    },
                    "path": encoder_path.name,
                },
                "mel_encoder": {
                    "inputs": {
                        "audio_signal": [1, max_samples],
                        "audio_length": [1],
                    },
                    "outputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "encoder_length": [1],
                        "frame_times": [1, _tensor_shape(encoder_ref)[2]],
                    },
                    "path": mel_encoder_path.name,
                },
                "decoder": {
                    "inputs": {
                        "targets": list(_tensor_shape(targets)),
                        "target_length": [1],
                        "h_in": list(_tensor_shape(zero_state)),
                        "c_in": list(_tensor_shape(zero_state)),
                    },
                    "outputs": {
                        "decoder": list(_tensor_shape(decoder_ref)),
                        "h_out": list(_tensor_shape(h_ref)),
                        "c_out": list(_tensor_shape(c_ref)),
                    },
                    "path": decoder_path.name,
                },
                "joint": {
                    "inputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "decoder": list(_tensor_shape(decoder_ref)),
                    },
                    "outputs": {
                        "logits": list(_tensor_shape(joint_ref)),
                    },
                    "path": joint_path.name,
                },
                "joint_decision": {
                    "inputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "decoder": list(_tensor_shape(decoder_ref)),
                    },
                    "outputs": {
                        "token_id": [
                            _tensor_shape(encoder_ref)[0],
                            _tensor_shape(encoder_ref)[2],
                            _tensor_shape(decoder_ref)[2],
                        ],
                        "token_prob": [
                            _tensor_shape(encoder_ref)[0],
                            _tensor_shape(encoder_ref)[2],
                            _tensor_shape(decoder_ref)[2],
                        ],
                    },
                    "path": joint_decision_path.name,
                },
                "joint_decision_single_step": {
                    "inputs": {
                        "encoder_step": [1, _tensor_shape(encoder_ref)[1], 1],
                        "decoder_step": [1, _tensor_shape(decoder_ref)[1], 1],
                    },
                    "outputs": {
                        "token_id": [1, 1, 1],
                        "token_prob": [1, 1, 1],
                        "top_k_ids": [1, 1, 1, 64],
                        "top_k_logits": [1, 1, 1, 64],
                    },
                    "path": jd_single_path.name,
                },
            },
        }

        # Export tokenizer vocab if available
        try:
            tokenizer = asr_model.tokenizer
            vocab = {
                "blank_id": int(asr_model.decoder.blank_idx),
                "vocab_size": vocab_size,
            }
            # Try to get special tokens
            if hasattr(tokenizer, "tokenizer"):
                inner_tokenizer = tokenizer.tokenizer
                if hasattr(inner_tokenizer, "get_vocab"):
                    full_vocab = inner_tokenizer.get_vocab()
                    # Find EOU token
                    eou_token = None
                    for token, idx in full_vocab.items():
                        if "<EOU>" in token.upper() or "eou" in token.lower():
                            eou_token = {"token": token, "id": idx}
                            break
                    if eou_token:
                        vocab["eou_token"] = eou_token
            metadata["tokenizer"] = vocab
        except Exception as e:
            typer.echo(f"Warning: Could not export tokenizer info: {e}")

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        typer.echo(f"\nExport complete. Metadata written to {metadata_path}")
        typer.echo(f"Output directory: {output_dir}")

    finally:
        asr_model.decoder._rnnt_export = decoder_export_flag


if __name__ == "__main__":
    app()
