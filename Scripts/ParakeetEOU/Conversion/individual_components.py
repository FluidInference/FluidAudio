#!/usr/bin/env python3
"""Export Parakeet Realtime EOU 120M RNNT components into CoreML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import torch


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[ct.target]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float
    max_symbol_steps: int


class PreprocessorWrapper(torch.nn.Module):
    """Wrapper for the audio preprocessor (mel spectrogram extraction)."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.module(
            input_signal=audio_signal, length=length.to(dtype=torch.long)
        )
        return mel, mel_length


class EncoderWrapper(torch.nn.Module):
    """Wrapper for the cache-aware FastConformer encoder.

    Note: For the realtime EOU model, the encoder is cache-aware which means
    it can operate in a streaming fashion. For CoreML export, we export
    without cache state for simplicity (full-context mode).
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, features: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, encoded_lengths = self.module(
            audio_signal=features, length=length.to(dtype=torch.long)
        )
        # Synthesize per-frame timestamps (seconds) using the 80 ms encoder stride.
        # Shape: [B, T_enc]
        frame_times = (
            torch.arange(encoded.shape[-1], device=encoded.device, dtype=torch.float32)
            * 0.08
        )
        return encoded, encoded_lengths, frame_times


class DecoderWrapper(torch.nn.Module):
    """Wrapper for the RNNT prediction network (decoder)."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = [h_in, c_in]
        decoder_output, _, new_state = self.module(
            targets=targets.to(dtype=torch.long),
            target_length=target_lengths.to(dtype=torch.long),
            states=state,
        )
        return decoder_output, new_state[0], new_state[1]


class JointWrapper(torch.nn.Module):
    """Wrapper for the RNNT joint network.

    Note: Unlike Parakeet TDT v3, the realtime EOU model does NOT have
    duration outputs (num_extra_outputs). The joint network outputs only
    token logits over the vocabulary + blank.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        # Input: encoder_outputs [B, D, T], decoder_outputs [B, D, U]
        # Transpose to match what projection layers expect
        encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D]
        decoder_outputs = decoder_outputs.transpose(1, 2)  # [B, U, D]

        # Apply projections
        enc_proj = self.module.enc(encoder_outputs)  # [B, T, joint_hidden]
        dec_proj = self.module.pred(decoder_outputs)  # [B, U, joint_hidden]

        # Explicit broadcasting along T and U to avoid converter ambiguity
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, joint_hidden]
        x = self.module.joint_net[0](x)  # ReLU
        x = self.module.joint_net[1](x)  # Dropout (no-op in eval)
        out = self.module.joint_net[2](x)  # Linear -> logits [B, T, U, vocab+blank]
        return out


class MelEncoderWrapper(torch.nn.Module):
    """Fused wrapper: waveform -> mel -> encoder.

    Inputs:
      - audio_signal: [B, S]
      - audio_length: [B]

    Outputs:
      - encoder: [B, D, T_enc]
      - encoder_length: [B]
      - frame_times: [T_enc]
    """

    def __init__(
        self, preprocessor: PreprocessorWrapper, encoder: EncoderWrapper
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(
        self, audio_signal: torch.Tensor, audio_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel, mel_length = self.preprocessor(audio_signal, audio_length)
        encoded, enc_len, frame_times = self.encoder(mel, mel_length.to(dtype=torch.int32))
        return encoded, enc_len, frame_times


class JointDecisionWrapper(torch.nn.Module):
    """Joint + decision head: outputs label id and label prob.

    Unlike Parakeet TDT v3, this model does NOT have duration outputs.

    Inputs:
      - encoder_outputs: [B, D, T]
      - decoder_outputs: [B, D, U]

    Returns:
      - token_id: [B, T, U] int32
      - token_prob: [B, T, U] float32
    """

    def __init__(self, joint: JointWrapper, vocab_size: int) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1

    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor):
        logits = self.joint(encoder_outputs, decoder_outputs)

        # Token selection
        token_ids = torch.argmax(logits, dim=-1).to(dtype=torch.int32)
        token_probs_all = torch.softmax(logits, dim=-1)
        # gather expects int64 (long) indices; cast only for gather
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)

        return token_ids, token_prob


class JointDecisionSingleStep(torch.nn.Module):
    """Single-step variant for streaming: encoder_step -> token decision.

    Inputs:
      - encoder_step: [B=1, D, T=1]
      - decoder_step: [B=1, D, U=1]

    Returns:
      - token_id: [1, 1, 1] int32
      - token_prob: [1, 1, 1] float32
      - top_k_ids: [1, 1, 1, K] int32
      - top_k_logits: [1, 1, 1, K] float32
    """

    def __init__(self, joint: JointWrapper, vocab_size: int, top_k: int = 64) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.top_k = int(top_k)

    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        # Reuse JointWrapper which expects [B, D, T] and [B, D, U]
        logits = self.joint(encoder_step, decoder_step)  # [1, 1, 1, V+blank]

        token_ids = torch.argmax(logits, dim=-1, keepdim=False).to(dtype=torch.int32)
        token_probs_all = torch.softmax(logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)

        # Also expose top-K candidates for host-side processing
        topk_logits, topk_ids_long = torch.topk(
            logits, k=min(self.top_k, logits.shape[-1]), dim=-1
        )
        topk_ids = topk_ids_long.to(dtype=torch.int32)
        return token_ids, token_prob, topk_ids, topk_logits


def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
    compute_precision: Optional[ct.precision] = None,
) -> ct.models.MLModel:
    cu = (
        compute_units_override
        if compute_units_override is not None
        else settings.compute_units
    )
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
    }
    print("Converting:", traced.__class__.__name__)
    print("Conversion kwargs:", kwargs)
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    
    # Priority: explicit argument > settings
    if compute_precision is not None:
        kwargs["compute_precision"] = compute_precision
    elif settings.compute_precision is not None:
        kwargs["compute_precision"] = settings.compute_precision
        
    return ct.convert(traced, **kwargs)
