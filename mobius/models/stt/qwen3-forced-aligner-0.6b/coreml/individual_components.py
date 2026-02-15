#!/usr/bin/env python3
"""Wrapper modules for Qwen3-ForcedAligner-0.6B CoreML export.

Architecture overview:
  Qwen3ASRForConditionalGeneration
    └── thinker: Qwen3ASRThinkerForConditionalGeneration
          ├── audio_tower: Qwen3ASRAudioEncoder (24 layers, 1024 dim)
          ├── model: Qwen3ASRThinkerTextModel (28-layer Qwen3 LLM)
          └── lm_head: Linear(1024, 152064)

Key differences from Qwen3-ASR-0.6B:
  - Audio encoder: 24 layers (vs 18), d_model=1024 (vs 896), 16 heads (vs 14)
  - Vocab size: 152,064 (vs 151,936)
  - RoPE: interleaved mrope with section [24, 20, 20] (vs standard)
  - Inference: NAR prefill-only (vs autoregressive decode)
  - Special token: timestamp_token_id=151705, timestamp_segment_time=80ms
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnemllRMSNorm(nn.Module):
    """ANEMLL-style RMSNorm using native LayerNorm for better CoreML precision.

    Standard RMSNorm is decomposed by coremltools into pow → mean → rsqrt → mul,
    each step losing FP16 precision. Concatenating [x, -x] forces mean to zero,
    making LayerNorm mathematically equivalent to RMSNorm.

    Reference: https://huggingface.co/blog/anemll/anemll-style-rms-ane
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps
        self.dim = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, [doubled.shape[-1]], eps=self.eps)
        normed = normed[..., : self.dim]
        return normed * self.weight


def patch_rms_norms(module: nn.Module) -> None:
    """Replace all Qwen3RMSNorm instances with AnemllRMSNorm."""
    for name, child in list(module.named_children()):
        class_name = type(child).__name__
        if class_name == "AnemllRMSNorm":
            continue
        if "RMSNorm" in class_name and hasattr(child, "weight"):
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-6))
            replacement = AnemllRMSNorm(child.weight.data, eps=eps)
            setattr(module, name, replacement)
        else:
            patch_rms_norms(child)


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[ct.target]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float
    max_seq_length: int


# ---------------------------------------------------------------------------
# Audio Encoder Wrapper
# ---------------------------------------------------------------------------

class AudioConvWrapper(nn.Module):
    """Audio encoder conv frontend: mel → conv downsample → positional embedding.

    Processes a single 100-frame mel window through the conv layers.
    The transformer layers are separated into AudioTransformerWrapper
    to enable cross-chunk attention matching the native encoder.

    Input:
      - mel_input: [1, 128, 100] mel spectrogram (single window)

    Output:
      - features: [1, 13, 1024] conv features with positional embedding
    """

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.conv2d1 = audio_encoder.conv2d1
        self.conv2d2 = audio_encoder.conv2d2
        self.conv2d3 = audio_encoder.conv2d3
        self.conv_out = audio_encoder.conv_out
        self.positional_embedding = audio_encoder.positional_embedding

    def forward(self, mel_input: torch.Tensor) -> torch.Tensor:
        # mel_input: [1, 128, 100]
        x = mel_input.unsqueeze(1)  # [1, 1, 128, 100]

        # Conv downsampling (3 layers, stride 2 each → 8x reduction)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        # x: [1, 480, 17, 13]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.conv_out(x)  # [1, 13, 1024]

        # Positional embedding (each chunk gets positions 0..12, matching native)
        pos_emb = self.positional_embedding(t)  # [13, 1024]
        x = x + pos_emb.unsqueeze(0).to(x.dtype)

        return x  # [1, 13, 1024]


class AudioTransformerWrapper(nn.Module):
    """Audio encoder transformer + projection: conv features → final embeddings.

    Takes concatenated conv features from multiple chunks and runs them through
    the 24-layer transformer with full bidirectional attention + projection.
    This matches the native encoder behavior where frames within a window
    attend to each other.

    Input:
      - features: [1, N, 1024] concatenated conv features (N = total frames across chunks)

    Output:
      - embeddings: [1, N, 1024] final audio embeddings
    """

    AUDIO_TRANSFORMER_SEQ_LEN = 256  # max frames (covers ~19.6 chunks × 13 ≈ 255)

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.layers = audio_encoder.layers
        self.ln_post = audio_encoder.ln_post
        self.proj1 = audio_encoder.proj1
        self.proj2 = audio_encoder.proj2
        self.num_layers = len(self.layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [1, N, 1024]
        hs = features.squeeze(0)  # [N, 1024]
        seq_len = hs.shape[0]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=hs.device)

        for layer in self.layers:
            layer_outputs = layer(hs, cu_seqlens=cu_seqlens)
            hs = layer_outputs[0]

        hs = self.ln_post(hs)
        hs = self.proj1(hs)
        hs = F.gelu(hs)
        hs = self.proj2(hs)  # [N, 1024]
        return hs.unsqueeze(0)  # [1, N, 1024]


class AudioEncoderFullWrapper(nn.Module):
    """Full audio encoder: mel → conv downsample → transformer → projection.

    DEPRECATED: Use AudioConvWrapper + AudioTransformerWrapper instead.
    This wrapper processes each chunk independently through the transformer,
    missing cross-chunk attention. Kept for backward compatibility.

    Input:
      - mel_input: [1, 128, T] mel spectrogram (128 bins, T frames)

    Output:
      - features: [1, T', 1024] where T' = output frames after 8x conv downsampling
    """

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.conv2d1 = audio_encoder.conv2d1
        self.conv2d2 = audio_encoder.conv2d2
        self.conv2d3 = audio_encoder.conv2d3
        self.conv_out = audio_encoder.conv_out
        self.positional_embedding = audio_encoder.positional_embedding
        self.layers = audio_encoder.layers
        self.ln_post = audio_encoder.ln_post
        self.proj1 = audio_encoder.proj1
        self.proj2 = audio_encoder.proj2

    def forward(self, mel_input: torch.Tensor) -> torch.Tensor:
        # mel_input: [1, 128, T]
        x = mel_input.unsqueeze(1)  # [1, 1, 128, T]

        # Conv downsampling (3 layers, stride 2 each → 8x reduction)
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        # x: [1, 480, F', T'] where F'=17, T'=ceil(T/8)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.conv_out(x)  # [1, T', 1024]

        # Positional embedding
        pos_emb = self.positional_embedding(t)  # [T', 1024]
        x = x + pos_emb.unsqueeze(0).to(x.dtype)

        # Transformer layers — flat sequence, full attention (no windowing)
        hs = x.squeeze(0)  # [T', 1024]
        seq_len = hs.shape[0]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=hs.device)

        for layer in self.layers:
            layer_outputs = layer(hs, cu_seqlens=cu_seqlens)
            hs = layer_outputs[0]

        hs = self.ln_post(hs)
        hs = self.proj1(hs)
        hs = F.gelu(hs)
        hs = self.proj2(hs)  # [T', 1024]
        return hs.unsqueeze(0)  # [1, T', 1024]


# ---------------------------------------------------------------------------
# Text Embedding Wrapper
# ---------------------------------------------------------------------------

class TextEmbeddingWrapper(nn.Module):
    """Token embedding layer.

    Input:
      - input_ids: [1, seq_len] int32

    Output:
      - embeddings: [1, seq_len, 1024]
    """

    def __init__(self, text_model: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = text_model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids.long())


# ---------------------------------------------------------------------------
# LM Head Wrapper
# ---------------------------------------------------------------------------

class LMHeadWrapper(nn.Module):
    """LM head: hidden states → logits over vocab.

    Input:
      - hidden_states: [1, seq_len, 1024]

    Output:
      - logits: [1, seq_len, 5000] (raw timestamp values, NOT vocab tokens)
    """

    def __init__(self, lm_head: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hs = self.norm(hidden_states)
        return self.lm_head(hs)


# ---------------------------------------------------------------------------
# Prefill Decoder Wrapper (NAR — single forward pass)
# ---------------------------------------------------------------------------

class PrefillDecoderWrapper(nn.Module):
    """Full decoder stack for NAR prefill (single forward pass, no autoregressive loop).

    The ForcedAligner processes the entire sequence (audio embeddings + text tokens
    with <timestamp> markers) in one prefill call. No KV cache growth, no decode loop.

    Input:
      - hidden_states: [1, PREFILL_SEQ_LEN, 1024]
      - position_cos: [1, PREFILL_SEQ_LEN, 128]
      - position_sin: [1, PREFILL_SEQ_LEN, 128]

    Output:
      - output_hidden: [1, PREFILL_SEQ_LEN, 1024]
    """

    PREFILL_SEQ_LEN = 1024  # max sequence length for forced alignment

    def __init__(self, text_model: nn.Module) -> None:
        super().__init__()
        self.layers = text_model.layers
        self.num_layers = len(self.layers)

        # Baked causal mask: [1, 1, N, N] — each position attends to itself and all previous
        N = self.PREFILL_SEQ_LEN
        mask = torch.full((N, N), -1e9, dtype=torch.float32)
        for i in range(N):
            mask[i, : i + 1] = 0.0
        self.register_buffer("_causal_mask", mask.view(1, 1, N, N))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = position_cos.unsqueeze(1)  # [1, 1, N, 128]
        sin = position_sin.unsqueeze(1)
        attention_mask = self._causal_mask

        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = self._layer_forward(
                layer, hidden_states, cos, sin, attention_mask
            )

        return hidden_states

    @staticmethod
    def _layer_forward(
        layer: nn.Module,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn = layer.self_attn

        residual = hidden_states
        hs = layer.input_layernorm(hidden_states)

        input_shape = hs.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_norm(attn.q_proj(hs).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hs).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(hs).view(hidden_shape).transpose(1, 2)

        # RoPE
        q = (q * cos) + (PrefillDecoderWrapper._rotate_half(q) * sin)
        k = (k * cos) + (PrefillDecoderWrapper._rotate_half(k) * sin)

        # GQA: expand KV heads (8 → 16)
        num_groups = attn.num_key_value_groups
        k_expanded = k.repeat_interleave(num_groups, dim=1)
        v_expanded = v.repeat_interleave(num_groups, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * attn.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = attn.o_proj(attn_output)

        hs = residual + attn_output

        # MLP
        residual = hs
        hs = layer.post_attention_layernorm(hs)
        hs = layer.mlp(hs)
        hs = residual + hs

        return hs

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
    compute_precision_override: Optional[ct.precision] = None,
    no_optimize: bool = False,
) -> ct.models.MLModel:
    cu = compute_units_override if compute_units_override is not None else settings.compute_units
    cp = compute_precision_override if compute_precision_override is not None else settings.compute_precision
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
    }
    print(f"Converting with compute_units={cu}, compute_precision={cp}, no_optimize={no_optimize}")
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    if cp is not None:
        kwargs["compute_precision"] = cp
    if no_optimize:
        minimal_passes = [
            "common::sanitize_input_output_names",
            "common::dedup_op_and_var_names",
            "common::dead_code_elimination",
            "common::const_elimination",
            "common::noop_elimination",
            "common::update_output_dtypes",
            "common::topological_reorder",
            "common::canonicalize_inplace_pattern",
        ]
        kwargs["pass_pipeline"] = ct.PassPipeline(
            pass_names=minimal_passes, pipeline_name="minimal"
        )
    return ct.convert(traced, **kwargs)
