#!/usr/bin/env python3
"""
Split encoder export for true streaming inference.

This script exports the encoder in separate components:
1. PreEncode (ConvSubsampling) - with pre_encode cache for mel frame overlap
2. ConformerStack - 17 conformer layers with attention/time caches

This allows proper streaming inference by:
- Processing fixed-size mel chunks through pre_encode
- Feeding pre_encode output through conformer layers with persistent caches
"""

import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import os
import shutil
import sys
import argparse
import torch
import typer
from torch import nn
import coremltools as ct
import numpy as np

from convert_parakeet_eou import ExportSettings, _coreml_convert, _save_mlpackage, apply_stft_patch
from individual_components import (
    DecoderWrapper,
    JointDecisionSingleStep,
    JointWrapper,
    PreprocessorWrapper,
)


class BypassPreEncode(nn.Module):
    """Helper to bypass pre_encode in ConformerEncoder."""
    def forward(self, *args, **kwargs):
        # If positional args, return first two
        if args:
            return args[0], args[1] if len(args) > 1 else kwargs.get('length') or kwargs.get('lengths')
        
        # If kwargs, look for 'audio_signal' or 'x'
        x = kwargs.get('audio_signal') or kwargs.get('x')
        length = kwargs.get('length') or kwargs.get('lengths')
        return x, length


class MyPreEncodeWrapper(nn.Module):
    """Wrapper for pre_encode (ConvSubsampling) with pre-encode cache.

    The pre_encode module performs 4x subsampling via two conv layers:
    - Conv2d(1, 256, kernel=(3,3), stride=(2,2))
    - Conv2d(256, 256, kernel=(3,3), stride=(2,2))
    - Linear(256 * (mel_dim // 4), hidden_dim)

    For streaming, we need to cache the last few mel frames to handle
    the convolution overlap at chunk boundaries.
    """

    def __init__(self, pre_encode: nn.Module, mel_dim: int = 128, pre_cache_size: int = 9):
        super().__init__()
        self.pre_encode = pre_encode
        # self.pre_encode = nn.Linear(mel_dim, mel_dim) # Dummy for debugging trace
        self.mel_dim = mel_dim
        self.pre_cache_size = pre_cache_size

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor, pre_cache: torch.Tensor) -> Dict[str, torch.Tensor]:
        # if len(args) == 3:
        #     mel, mel_length, pre_cache = args
        # elif len(args) == 2:
        #     mel, mel_length = args
        #     pre_cache = kwargs.get('pre_cache')
        # else:
        #     mel = kwargs.get('mel')
        #     mel_length = kwargs.get('mel_length')
        #     pre_cache = kwargs.get('pre_cache')
            
        if pre_cache is None:
             # print("DEBUG: pre_cache is None! Creating zeros.", file=sys.stderr)
             batch_size = mel.shape[0]
             pre_cache = torch.zeros(batch_size, self.pre_cache_size, self.mel_dim, device=mel.device, dtype=mel.dtype)

        return self._forward_impl(mel, mel_length, pre_cache)
        
        # Dummy return
        # return mel, mel_length, pre_cache

    def _forward_impl(
        self,
        mel: torch.Tensor,
        mel_length: torch.Tensor,
        pre_cache: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: [B, mel_dim, T] - new mel frames (channel-major from preprocessor)
            mel_length: [B] - length of mel
            pre_cache: [B, pre_cache_size, mel_dim] - cached mel frames from previous chunk

        Returns:
            encoded: [B, T', hidden_dim] - subsampled and projected output
            encoded_length: [B] - output length
            new_cache: [B, pre_cache_size, mel_dim] - new cache for next chunk
        """
        batch_size = mel.shape[0]

        # mel is [B, D, T] (channel-first from preprocessor)
        # pre_cache is [B, T_cache, D] (channel-last from CoreML/Swift)
        
        with open("/tmp/debug_log.txt", "a") as f:
            f.write(f"DEBUG: MyPreEncodeWrapper input mel: {mel.shape}\n")
            f.write(f"DEBUG: MyPreEncodeWrapper input pre_cache: {pre_cache.shape}\n")

        # mel is [B, D, T]. pre_cache is [B, T_cache, D].
        # pre_encode expects [B, D, T].
        
        # Transpose pre_cache to [B, D, T_cache]
        pre_cache_T = pre_cache.transpose(1, 2)

        # Concatenate cache with new mel along time (dim 2)
        if self.pre_cache_size > 0:
            mel_with_cache = torch.cat([pre_cache_T, mel], dim=2)  # [B, D, cache+T]
            adjusted_length = mel_length + self.pre_cache_size
        else:
            mel_with_cache = mel
            adjusted_length = mel_length
            
        with open("/tmp/debug_log.txt", "a") as f:
            f.write(f"DEBUG: MyPreEncodeWrapper mel_with_cache: {mel_with_cache.shape}\n")

        # Run pre_encode - expects [B, D, T] or [B, T, D]?
        # Based on crash analysis (17x1024 vs 4352x512), it seems pre_encode expects [B, T, D].
        # mel_with_cache is [B, D, T].
        mel_with_cache_T = mel_with_cache.transpose(1, 2) # [B, T, D]
        
        out = self.pre_encode(mel_with_cache_T, adjusted_length)
        # Output is [B, T_out, D_out] (channel-last)
        full_encoded = out[0]
        full_encoded_length = out[1]
        
        with open("/tmp/debug_log.txt", "a") as f:
            f.write(f"DEBUG: MyPreEncodeWrapper full_encoded: {full_encoded.shape}\n")

        # Slice output to remove cache frames
        # Cache size 9 -> 8x subsampling -> 2 frames to drop
        # L1: (9+2*1-3)/2 + 1 = 5
        # L2: (5+2*1-3)/2 + 1 = 3
        # L3: (3+2*1-3)/2 + 1 = 2
        frames_to_drop = 2
        
        if self.pre_cache_size > 0:
            encoded = full_encoded[:, frames_to_drop:, :]
            encoded_length = full_encoded_length - frames_to_drop
        else:
            encoded = full_encoded
            encoded_length = full_encoded_length

        # Extract new cache from end of original mel (before pre_encode)
        # mel is [B, D, T]. We want last 9 frames.
        if self.pre_cache_size > 0:
            # Take last pre_cache_size frames from input mel
            # mel is [B, 128, 16]. We want last 9 frames along dim 2.
            # start_idx = 16 - 9 = 7. length = 9.
            new_cache_T = mel.narrow(2, mel.shape[2] - self.pre_cache_size, self.pre_cache_size) # [B, D, 9]
            # Transpose back to [B, 9, D] for output
            new_cache = new_cache_T.transpose(1, 2)
        else:
            new_cache = torch.zeros(batch_size, 0, self.mel_dim, dtype=mel.dtype)

        # Optimization: Don't return length, calculate it in Swift (Input / 4)
        # This avoids the inhomogeneous shape error entirely.
        return encoded, new_cache


class ConformerStackWrapper(nn.Module):
    """Wrapper for conformer layers with cache-aware streaming.

    This wraps the 17 conformer layers and handles:
    - cache_last_channel: Attention context cache [layers, B, cache_size, hidden]
    - cache_last_time: Time convolution cache [layers, B, hidden, time_cache]
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_layers: int = 17,
        hidden_dim: int = 512,
        cache_channel_size: int = 70,
        cache_time_size: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        
        # We do NOT need to replace pre_encode with Identity because we use bypass_pre_encode=True
        # and we must NOT modify the encoder because copy.copy is shallow and affects the original.
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cache_channel_size = cache_channel_size
        self.cache_time_size = cache_time_size

    def forward(
        self,
        pre_encoded: torch.Tensor,
        pre_encoded_length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pre_encoded: [B, T', hidden_dim] - output from pre_encode
            pre_encoded_length: [B] - sequence length
            cache_last_channel: [layers, B, cache_size, hidden_dim] - attention cache
            cache_last_time: [layers, B, hidden_dim, time_cache] - time conv cache
            cache_last_channel_len: [B] - current cache usage length

        Returns:
            encoded: [B, hidden_dim, T_out] - encoder output (channel-last for decoder)
            encoded_length: [B] - output length
            new_cache_channel: [layers, B, cache_size, hidden_dim]
            new_cache_time: [layers, B, hidden_dim, time_cache]
            new_cache_len: [B]
        """
        # Use the encoder's cache_aware_stream_step but only for the conformer part
        # We need to call it with the pre-encoded features

        # The FastConformer encoder's cache_aware_stream_step expects:
        # - processed_signal: [B, hidden, T] (channel-first)
        # - processed_signal_length: [B]
        # - cache_last_channel, cache_last_time, cache_last_channel_len

        # Since pre_encoded is [B, T', hidden_dim], transpose to [B, hidden_dim, T']
        x = pre_encoded.transpose(1, 2)  # [B, hidden, T']

        # HACK: Temporarily replace pre_encode with BypassPreEncode to bypass it
        # since bypass_pre_encode=True is not supported in this NeMo version
        original_pre_encode = self.encoder.pre_encode
        self.encoder.pre_encode = BypassPreEncode()
        
        try:
            print("DEBUG: Calling cache_aware_stream_step...", file=sys.stderr)
            # Call cache_aware_stream_step
            outputs = self.encoder.cache_aware_stream_step(
                processed_signal=x,
                processed_signal_length=pre_encoded_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                # bypass_pre_encode=True, # Removed
            )
            print("DEBUG: cache_aware_stream_step returned successfully", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: cache_aware_stream_step failed: {e}", file=sys.stderr)
            raise e
        finally:
            # Restore original pre_encode
            self.encoder.pre_encode = original_pre_encode

        # Outputs: (encoded, encoded_len, new_cache_channel, new_cache_time, new_cache_len)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]



class SimpleConformerWrapper(nn.Module):
    """Simpler approach: Just wrap the full encoder's cache_aware_stream_step.

    This avoids splitting pre_encode since the cache_aware_stream_step
    handles everything internally including the pre_encode cache.

    The mel input must be in [B, mel_dim, T] format (channel-first).
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        mel: torch.Tensor,
        mel_length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: [B, mel_dim, T] - mel spectrogram (channel-first)
            mel_length: [B] - length
            cache_last_channel: [layers, B, cache_size, hidden]
            cache_last_time: [layers, B, hidden, time_cache]
            cache_last_channel_len: [B]

        Returns:
            [encoded_out, encoded_len_out, new_cache_channel, new_cache_time, new_cache_len]
        """
        outputs = self.encoder.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


class FixedChunkPreEncodeWrapper(nn.Module):
    """Pre-encode with FIXED chunk size to avoid dynamic shape issues.

    The ConvSubsampling linear layer expects exactly 4352 input features,
    which comes from: 256 channels * (mel_dim // 4 - 1) = 256 * 17 = 4352

    For mel_dim=128: floor(128/4) - 1 = 31, but the actual calculation is:
    After two conv2d with stride 2: T' = ((T - 3) // 2 + 1 - 3) // 2 + 1
    And freq: F' = ((128 - 3) // 2 + 1 - 3) // 2 + 1 = 30
    Then 256 * 30 / 2 = 3840... let me check the actual model.

    Actually the subsampling_factor is 4, so T_out = T_in // 4.
    And the linear expects: hidden_channels * (feat_in // subsampling_factor)
    = 256 * (128 // 4) = 256 * 32 = 8192... but that doesn't match 4352.

    Let me just trace with the actual chunk size the model expects.
    """

    def __init__(self, pre_encode: nn.Module, mel_dim: int = 128):
        super().__init__()
        self.pre_encode = pre_encode
        self.mel_dim = mel_dim

    def forward(
        self,
        mel: torch.Tensor,
        mel_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: [B, T, mel_dim] - mel spectrogram (time-major)
            mel_length: [B] - length

        Returns:
            encoded: [B, T', hidden_dim] - subsampled output
            encoded_length: [B] - output length
        """
        # Input is [B, D, T] (channel-major)
        # ConvSubsampling expects [B, T, D], so we transpose
        mel = mel.transpose(1, 2)
        return self.pre_encode(mel, mel_length)


class ConformerBatchWrapper(nn.Module):
    """Process pre_encoded features through conformer layers (batch mode)."""

    def __init__(self, encoder):
        super().__init__()
        self.pos_enc = encoder.pos_enc if hasattr(encoder, 'pos_enc') else None
        self.layers = encoder.layers
        self.norm = encoder.norm if hasattr(encoder, 'norm') else None

    def forward(self, x: torch.Tensor, input_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, hidden_dim] - pre_encoded features
            input_length: [B] - sequence lengths
        Returns:
            out: [B, hidden_dim, T] - encoder output (transposed for decoder)
            output_length: [B] - output length
        """
        # Input x is [B, T, D] (from PreEncodeWrapper)
        
        # Input x is [B, T, D] (from PreEncodeWrapper)
        # pos_enc expects [B, T, D]
        # x = x.transpose(1, 2)
        
        # Add positional encoding - returns (x + pos_emb, pos_emb)
        pos_emb = None
        if self.pos_enc is not None:
            result = self.pos_enc(x)
            if isinstance(result, tuple):
                x, pos_emb = result
            else:
                x = result
        
        # x is already [B, T, D]
        # x = x.transpose(1, 2)

        # CRITICAL FIX: Don't create attention mask to avoid transpose rank mismatch in CoreML
        # The mask creation causes 5D tensor issues during conversion (perm rank 4 != input rank 5)
        # For batch processing with fixed-length input, we can pass None
        # This works because we're processing padded input with known length
        
        # Process through layers without mask
        for layer in self.layers:
            x = layer(x, att_mask=None, pos_emb=pos_emb)

        # Final normalization
        if self.norm is not None:
            x = self.norm(x)

        # Transpose back to [B, D, T] for Joint
        x = x.transpose(1, 2)
        
        output_length = input_length * 1  # Force separate computation
        return x, output_length


def inspect_encoder_structure(encoder):
    """Print the encoder's internal structure for debugging."""
    print("\n=== Encoder Structure ===")
    print(f"Type: {type(encoder)}")

    for name, module in encoder.named_children():
        print(f"  {name}: {type(module).__name__}")
        if hasattr(module, 'named_children'):
            for subname, submodule in module.named_children():
                print(f"    {subname}: {type(submodule).__name__}")

    if hasattr(encoder, 'streaming_cfg'):
        cfg = encoder.streaming_cfg
        print(f"\nStreaming Config:")
        print(f"  chunk_size: {cfg.chunk_size}")
        print(f"  shift_size: {cfg.shift_size}")
        print(f"  pre_encode_cache_size: {cfg.pre_encode_cache_size}")
        print(f"  last_channel_cache_size: {cfg.last_channel_cache_size}")
        if hasattr(cfg, 'last_time_cache_size'):
            print(f"  last_time_cache_size: {cfg.last_time_cache_size}")

    print()



class SingleStreamingEncoderWrapper(torch.nn.Module):
    """
    Combines Pre-Encode and Conformer into a SINGLE streaming model.
    
    Inputs:
      - mel: [B, D, T] (Mel Spectrogram)
      - mel_length: [B]
      - cache_last_channel: [B, D, C, T_cache]
      - cache_last_time: [B, D, T_cache, D_time]
      - cache_last_channel_len: [B, D]
      
    Outputs:
      - encoded: [B, D, T_enc]
      - encoded_len: [B]
      - new_cache_last_channel
      - new_cache_last_time
      - new_cache_last_channel_len
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        mel,
        mel_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    ):
        # Ensure lengths are long
        mel_length = mel_length.to(torch.long)
        cache_last_channel_len = cache_last_channel_len.to(torch.long)
        
        # 1. Run Pre-Encode (Mel -> Subsampled Features)
        # Note: We rely on the client to provide the "Convolution Context" (overlapping audio/mel).
        # So we just run the standard pre_encode.
        
        # We call the internal cache_aware_stream_step method of the ConformerEncoder.
        # It expects 'processed_signal' which is the input to pre_encode (Mel) if bypass_pre_encode=False.
        
        outputs = self.encoder.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            # bypass_pre_encode=False # Removed: Not supported in installed NeMo
        )
        
        # outputs: (encoded, encoded_len, new_cache_last_channel, new_cache_last_time, new_cache_last_channel_len)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


def test_pre_encode_shapes(encoder, mel_dim: int = 128):
    """Test what shapes pre_encode expects and produces."""
    print("\n=== Testing Pre-Encode Shapes ===")

    pre_encode = encoder.pre_encode

    for T in [10, 20, 40, 80, 160]:
        mel = torch.randn(1, T, mel_dim)
        mel_len = torch.tensor([T], dtype=torch.long)
        try:
            out, out_len = pre_encode(mel, mel_len)
            print(f"  Input [1, {T}, {mel_dim}] -> Output {list(out.shape)}, len={out_len.item()}")
        except Exception as e:
            print(f"  Input [1, {T}, {mel_dim}] -> ERROR: {e}")


def main(
    output_dir: str = typer.Option("Models/ParakeetEOU/ShortBatch", help="Output directory"),
    model_id: str = typer.Option(
        "nvidia/parakeet_realtime_eou_120m-v1", help="Model ID"
    ),
    inspect_only: bool = typer.Option(False, help="Only inspect encoder structure"),
):
    """Export Parakeet EOU with split encoder for streaming."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import nemo.collections.asr as nemo_asr

    typer.echo(f"Loading model {model_id}...")
    if Path(model_id).exists():
        asr_model = nemo_asr.models.ASRModel.restore_from(model_id, map_location="cpu")
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()
    
    # Set streaming params
    # chunk_size=32 (128 mel frames)
    # shift_size=30 (120 mel frames) -> 2 frame overlap (8 mel frames)
    # chunk_size=4 (16 mel frames)
    # shift_size=2 (8 mel frames)
    asr_model.encoder.setup_streaming_params(chunk_size=4, shift_size=2)
    print(f"DEBUG: Streaming Config: {asr_model.encoder.streaming_cfg}", flush=True)

    encoder = asr_model.encoder
    preprocessor = asr_model.preprocessor
    
    # Disable dither and padding for deterministic export
    if hasattr(preprocessor, 'featurizer'):
        if hasattr(preprocessor.featurizer, 'dither'):
            print(f"DEBUG: Disabling dither (was {preprocessor.featurizer.dither})")
            preprocessor.featurizer.dither = 0.0
        if hasattr(preprocessor.featurizer, 'pad_to'):
            print(f"DEBUG: Disabling pad_to (was {preprocessor.featurizer.pad_to})")
            preprocessor.featurizer.pad_to = 0

    # Inspect structure
    # inspect_encoder_structure(encoder)
    import sys
    print(f"DEBUG: Pre-Encode Structure: {encoder.pre_encode}", flush=True)
    print("DEBUG: Starting main...", flush=True)

    # Get streaming config
    streaming_cfg = encoder.streaming_cfg
    mel_dim = int(asr_model.cfg.preprocessor.features)
    hidden_dim = int(encoder.d_model)
    num_layers = len(encoder.layers)

    # Cache sizes from streaming config
    cache_channel_size = 70
    cache_time_size = 8
    # Force pre_encode_cache_size to 16 to cover full receptive field (15)
    # Original config says 9, but that leads to garbage output.
    pre_cache_size = 16 # Original
    # pre_cache_size = 0 # For 1s window test 
    
    if streaming_cfg:
        if streaming_cfg.last_channel_cache_size:
            lcc = streaming_cfg.last_channel_cache_size
            cache_channel_size = int(lcc[0]) if isinstance(lcc, (list, tuple)) else int(lcc)
        if hasattr(streaming_cfg, 'last_time_cache_size') and streaming_cfg.last_time_cache_size:
            ltc = streaming_cfg.last_time_cache_size
            cache_time_size = int(ltc[0]) if isinstance(ltc, (list, tuple)) else int(ltc)
            
    # Update trace_len for new cache size
    # trace_len = 16 + pre_cache_size

    typer.echo(f"\nEncoder config:")
    typer.echo(f"  mel_dim: {mel_dim}")
    typer.echo(f"  hidden_dim: {hidden_dim}")
    typer.echo(f"  num_layers: {num_layers}")
    typer.echo(f"  cache_channel_size: {cache_channel_size}")
    typer.echo(f"  cache_time_size: {cache_time_size}")

    # Test pre_encode shapes
    test_pre_encode_shapes(encoder, mel_dim)

    if inspect_only:
        return

    # Get chunk size from streaming config
    chunk_size = 2  # Default for 160ms experiment (16 mel frames / 8 subsampling)
    if streaming_cfg and streaming_cfg.chunk_size:
        cs = streaming_cfg.chunk_size
        chunk_size = int(cs[0]) if isinstance(cs, (list, tuple)) else int(cs)

    typer.echo(f"  chunk_size: {chunk_size}")

    # Calculate mel frames needed for one chunk
    # The encoder expects mel in [B, mel_dim, T] format
    # chunk_size is in encoder frames (after 8x subsampling)
    # So we need ~chunk_size * 8 mel frames
    mel_frames_per_chunk = chunk_size * 8 + 9  # Add pre_encode cache size buffer

    typer.echo(f"  mel_frames_per_chunk: {mel_frames_per_chunk}")

    export_settings = ExportSettings(
        output_dir=output_path,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=None,
        max_audio_seconds=30,
        max_symbol_steps=1,
    )

    # ========== Export Preprocessor ==========
    typer.echo("\n=== Exporting Preprocessor ===")

    prep_wrapper = PreprocessorWrapper(preprocessor)

    sample_rate = 16000
    test_audio = torch.randn(1, sample_rate * 2, dtype=torch.float32)
    test_length = torch.tensor([sample_rate * 2], dtype=torch.int32)

    traced_prep = torch.jit.trace(prep_wrapper, (test_audio, test_length), strict=False)
    traced_prep.eval()

    prep_inputs = [
        ct.TensorType(
            name="audio_signal",
            shape=(1, ct.RangeDim(1, sample_rate * 30)),
            dtype=np.float32,
        ),
        ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
    ]
    prep_outputs = [
        ct.TensorType(name="mel", dtype=np.float32),
        ct.TensorType(name="mel_length", dtype=np.int32),
    ]

    # Apply monkey patch before conversion starts
    apply_stft_patch()

    try:
        prep_model = _coreml_convert(
            traced_prep, prep_inputs, prep_outputs, export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )

        prep_path = output_path / "preprocessor.mlpackage"
        _save_mlpackage(prep_model, prep_path, "Preprocessor")
        typer.echo(f"Saved: {prep_path}")
    except Exception as e:
        typer.echo(f"Preprocessor export failed: {e}")
        typer.echo("Continuing with other components...")

    # ========== Export Pre-Encode (ConvSubsampling) ==========
    typer.echo("\n=== Exporting Pre-Encode ===")

    pre_encode = encoder.pre_encode
    
    # Trace pre_encode separately to avoid interference
    print("DEBUG: Tracing pre_encode (ConvSubsampling) separately...")
    # Trace with exact size expected during streaming (Chunk=16 + Cache=pre_cache_size)
    trace_len = 16 + pre_cache_size
    # pre_encode expects [B, T, D]
    pe_mel = torch.randn(1, trace_len, mel_dim)
    pe_len = torch.tensor([trace_len], dtype=torch.long)
    traced_conv_subsampling = torch.jit.trace(pre_encode, (pe_mel, pe_len), strict=False, check_trace=False)
    print("DEBUG: Tracing pre_encode done.")

    # Use PreEncodeWrapper with caching, using the TRACED sub-module
    pre_encode_wrapper = MyPreEncodeWrapper(traced_conv_subsampling, mel_dim, pre_cache_size=pre_cache_size)

    # Chunk size for input (0.16s = 16 frames)
    chunk_size_in = 16
    # pre_cache_size = 0 # Disable cache for 1s window test 
    
    # Test inputs
    # CRITICAL: Must match PreEncodeWrapper expectation [B, D, T]
    test_mel = torch.randn(1, mel_dim, chunk_size_in, dtype=torch.float32)
    test_mel_len = torch.tensor([chunk_size_in], dtype=torch.int32)
    test_pre_cache = torch.zeros(1, pre_cache_size, mel_dim, dtype=torch.float32)

    print(f"DEBUG: Calling pre_encode_wrapper with 3 args")
    print(f"DEBUG: test_mel: {test_mel.shape}")
    print(f"DEBUG: test_mel_len: {test_mel_len.shape}, dtype: {test_mel_len.dtype}")
    print(f"DEBUG: test_pre_cache: {test_pre_cache.shape}")
    print(f"DEBUG: pre_encode_wrapper type: {type(pre_encode_wrapper)}")

    with torch.no_grad():
        # test_out, test_out_len, test_new_cache = pre_encode_wrapper(test_mel, test_mel_len, test_pre_cache)
        test_out, test_new_cache = pre_encode_wrapper.forward(test_mel, test_mel_len, test_pre_cache)
    
    typer.echo(f"Pre-encode test: [{chunk_size_in}x{mel_dim}] -> {list(test_out.shape)}")
    # typer.echo(f"Pre-encode len: {test_out_len.shape}")
    typer.echo(f"Pre-encode cache: {test_new_cache.shape}")

    example_inputs = (test_mel, test_mel_len, test_pre_cache)
    print(f"DEBUG: example_inputs len: {len(example_inputs)}")
    print(f"DEBUG: example_inputs types: {[type(x) for x in example_inputs]}")
    
    print(f"DEBUG: pre_encode type: {type(pre_encode)}")
    print(f"DEBUG: is ScriptModule: {isinstance(pre_encode, torch.jit.ScriptModule)}")

    print("DEBUG: Tracing pre_encode_wrapper with trace...")
    traced_pre = torch.jit.trace(pre_encode_wrapper, example_inputs, strict=False, check_trace=False)
    # traced_pre = torch.jit.trace_module(pre_encode_wrapper, {'forward': example_inputs}, strict=False, check_trace=False)
    print("DEBUG: Tracing done.")
    traced_pre.eval()
    print("DEBUG: Traced graph:")
    print(traced_pre.graph)
    print("DEBUG: Traced graph:")
    print(traced_pre.graph)

    print(f"DEBUG: pre_cache_size for CoreML: {pre_cache_size}")
    pre_inputs = [
        ct.TensorType(
            name="mel",
            shape=(1, 128, chunk_size_in),
            dtype=np.float32,
        ),
        ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="pre_cache", shape=(1, pre_cache_size, 128), dtype=np.float32),
    ]
    pre_outputs = [
        ct.TensorType(name="pre_encoded", shape=(1, chunk_size_in // 4, hidden_dim), dtype=np.float32),
        # ct.TensorType(name="pre_encoded_len", dtype=np.float32), # Removed
        ct.TensorType(name="new_pre_cache", shape=(1, pre_cache_size, mel_dim), dtype=np.float32),
    ]

    try:
        # Let CoreML infer outputs to avoid "inhomogeneous shape" error
        pre_model = _coreml_convert(
            traced_pre, pre_inputs, pre_outputs, export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT32,
        )

        prep_path = output_path / "pre_encode.mlpackage"
        _save_mlpackage(pre_model, prep_path, "Pre-Encode")
        typer.echo(f"Saved: {prep_path}")
    except Exception as e:
        typer.echo(f"Pre-encode export failed: {e}")
        typer.echo("Continuing with other components...")

    # ========== Export Conformer Layers (Streaming) ==========
    typer.echo("\n=== Exporting Conformer Layers (Streaming) ===")

    # Use the streaming wrapper
    conformer_wrapper = ConformerStackWrapper(
        encoder,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        cache_channel_size=cache_channel_size,
        cache_time_size=cache_time_size,
    )

    # Test input shape (output from pre_encode)
    with torch.no_grad():
        pre_out, pre_out_len, _ = pre_encode_wrapper(test_mel, test_mel_len, test_pre_cache)

    # For streaming, we process chunks. Let's use the pre_out as a "chunk"
    # pre_out is [B, T', hidden_dim]
    test_conformer_in = pre_out
    test_conformer_len = pre_out_len.to(torch.long)

    # Create dummy caches
    batch_size = 1
    test_cache_channel = torch.zeros(
        num_layers, batch_size, cache_channel_size, hidden_dim, dtype=torch.float32
    )
    test_cache_time = torch.zeros(
        num_layers, batch_size, hidden_dim, cache_time_size, dtype=torch.float32
    )
    test_cache_len = torch.zeros(batch_size, dtype=torch.long)

    typer.echo(f"Conformer input shape: {list(test_conformer_in.shape)}")
    typer.echo(f"Cache channel shape: {list(test_cache_channel.shape)}")
    print(f"Cache time shape: {list(test_cache_time.shape)}", flush=True)

    print("Calling conformer_wrapper directly...", flush=True)
    try:
        with torch.no_grad():
            (
                conf_out,
                conf_len,
                new_cache_channel,
                new_cache_time,
                new_cache_len,
            ) = conformer_wrapper(
                test_conformer_in,
                test_conformer_len,
                test_cache_channel,
                test_cache_time,
                test_cache_len,
            )
        typer.echo(f"Conformer output shape: {list(conf_out.shape)}")

        typer.echo(f"Conformer output shape: {list(conf_out.shape)}")

        example_inputs = (
            test_conformer_in,
            test_conformer_len,
            test_cache_channel,
            test_cache_time,
            test_cache_len,
        )
        typer.echo(f"Example inputs type: {type(example_inputs)}")
        typer.echo(f"Example inputs len: {len(example_inputs)}")
        for i, inp in enumerate(example_inputs):
            typer.echo(f"Input {i} type: {type(inp)}")
            if isinstance(inp, torch.Tensor):
                typer.echo(f"Input {i} shape: {inp.shape}")

        traced_conf = torch.jit.trace(
            conformer_wrapper,
            example_inputs,
            strict=False,
        )
        traced_conf.eval()

        # Use fixed shapes for inputs, but allow flexible time dimension for the chunk
        T_pre = test_conformer_in.shape[1]
        conf_inputs = [
            ct.TensorType(
                name="pre_encoded",
                shape=(1, T_pre, hidden_dim),
                dtype=np.float32,
            ),
            ct.TensorType(name="pre_encoded_length", shape=(1,), dtype=np.int32),
            ct.TensorType(
                name="cache_last_channel",
                shape=test_cache_channel.shape,
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_last_time",
                shape=test_cache_time.shape,
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_last_channel_len",
                shape=test_cache_len.shape,
                dtype=np.int32,  # CoreML usually prefers int32 for lengths/indices
            ),
        ]
        conf_outputs = [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
            ct.TensorType(name="new_cache_channel", dtype=np.float32),
            ct.TensorType(name="new_cache_time", dtype=np.float32),
            ct.TensorType(name="new_cache_len", dtype=np.int32),
        ]

        conf_model = _coreml_convert(
            traced_conf,
            conf_inputs,
            conf_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )

        conf_path = output_path / "conformer_streaming.mlpackage"
        _save_mlpackage(conf_model, conf_path, "ConformerStreaming")
        typer.echo(f"Saved: {conf_path}")
    except Exception as e:
        typer.echo(f"Conformer export failed: {e}")
        import traceback

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # 4. Single Streaming Encoder (Combined)
    # -------------------------------------------------------------------------
    print("Tracing Single Streaming Encoder...")
    single_encoder_wrapper = SingleStreamingEncoderWrapper(encoder)
    single_encoder_wrapper.eval()
    
    # We need Mel input for tracing
    # Use test_mel from PreEncode section
    
    traced_single_encoder = torch.jit.trace(
        single_encoder_wrapper,
        (
            test_mel,
            test_mel_len,
            test_cache_channel,
            test_cache_time,
            test_cache_len
        ),
        strict=False
    )
    
    print("Verifying traced model...")
    with torch.no_grad():
        out = traced_single_encoder(
            test_mel,
            test_mel_len,
            test_cache_channel,
            test_cache_time,
            test_cache_len
        )
    print(f"Traced model output shape: {out[0].shape}")
    
    single_encoder_inputs = [
        ct.TensorType(name="mel", shape=(1, mel_dim, 128), dtype=np.float32), 
        ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="cache_last_channel", shape=test_cache_channel.shape, dtype=np.float32),
        ct.TensorType(name="cache_last_time", shape=test_cache_time.shape, dtype=np.float32),
        ct.TensorType(name="cache_last_channel_len", shape=test_cache_len.shape, dtype=np.int32),
    ]
    
    single_encoder_outputs = [
        ct.TensorType(name="encoded", dtype=np.float32),
        ct.TensorType(name="encoded_len", dtype=np.int32),
        ct.TensorType(name="new_cache_last_channel", dtype=np.float32),
        ct.TensorType(name="new_cache_last_time", dtype=np.float32),
        ct.TensorType(name="new_cache_last_channel_len", dtype=np.int32),
    ]
    
    single_encoder_mlmodel = ct.convert(
        traced_single_encoder,
        inputs=single_encoder_inputs,
        outputs=single_encoder_outputs,
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.CPU_ONLY, # Encoder often safer on CPU for stability
        skip_model_load=True,
    )
    
    single_encoder_path = output_path / "streaming_encoder.mlpackage"
    if single_encoder_path.exists():
        import shutil
        shutil.rmtree(single_encoder_path)
        
    single_encoder_mlmodel.save(str(single_encoder_path))
    print(f"Saved Single Streaming Encoder to {single_encoder_path}")


    # -------------------------------------------------------------------------
    # 5. Decoder (RNNT Prediction Network)
    # -------------------------------------------------------------------------

    # ========== Export Decoder ==========
    typer.echo("\n=== Exporting Decoder ===")

    decoder = asr_model.decoder
    decoder_wrapper = DecoderWrapper(decoder)

    decoder_hidden = int(decoder.pred_hidden)
    decoder_layers = 1

    test_target = torch.tensor([[0]], dtype=torch.int32)
    test_target_len = torch.tensor([1], dtype=torch.int32)
    test_h = torch.zeros(decoder_layers, 1, decoder_hidden, dtype=torch.float32)
    test_c = torch.zeros(decoder_layers, 1, decoder_hidden, dtype=torch.float32)

    traced_decoder = torch.jit.trace(
        decoder_wrapper, (test_target, test_target_len, test_h, test_c), strict=False
    )
    traced_decoder.eval()

    decoder_inputs = [
        ct.TensorType(name="targets", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="h_in", shape=(decoder_layers, 1, decoder_hidden), dtype=np.float32),
        ct.TensorType(name="c_in", shape=(decoder_layers, 1, decoder_hidden), dtype=np.float32),
    ]
    decoder_outputs = [
        ct.TensorType(name="decoder", dtype=np.float32),
        ct.TensorType(name="h_out", dtype=np.float32),
        ct.TensorType(name="c_out", dtype=np.float32),
    ]

    decoder_model = _coreml_convert(
        traced_decoder, decoder_inputs, decoder_outputs, export_settings,
        compute_units_override=ct.ComputeUnit.CPU_ONLY,
    )

    decoder_path = output_path / "decoder.mlpackage"
    _save_mlpackage(decoder_model, decoder_path, "Decoder")
    typer.echo(f"Saved: {decoder_path}")

    # ========== Export Joint Decision ==========
    typer.echo("\n=== Exporting Joint Decision ===")

    joint = asr_model.joint
    joint_wrapper = JointWrapper(joint)
    vocab_size = int(asr_model.cfg.joint.num_classes)

    jd_single = JointDecisionSingleStep(joint_wrapper, vocab_size=vocab_size)

    # Get test encoder output
    with torch.no_grad():
        # Use pre_encode + conformer for encoder output
        # pre_out, pre_len = pre_encode_wrapper(test_mel, test_mel_len)
        # enc_out, enc_len = conformer_wrapper(pre_out, pre_len.to(torch.long))
        # enc_out = conf_out
        # enc_len = conf_len
        # Create dummy encoder output if conf_out is not available
        enc_out = torch.randn(1, hidden_dim, 1, dtype=torch.float32)
        enc_len = torch.tensor([1], dtype=torch.int32)
        dec_out, _, _ = decoder_wrapper(test_target, test_target_len, test_h, test_c)

    enc_step = enc_out[:, :, :1].contiguous()
    dec_step = dec_out[:, :, :1].contiguous()

    traced_jd = torch.jit.trace(jd_single, (enc_step, dec_step), strict=False)
    traced_jd.eval()

    jd_inputs = [
        ct.TensorType(name="encoder_step", shape=(1, enc_step.shape[1], 1), dtype=np.float32),
        ct.TensorType(name="decoder_step", shape=(1, dec_step.shape[1], 1), dtype=np.float32),
    ]
    jd_outputs = [
        ct.TensorType(name="token_id", dtype=np.int32),
        ct.TensorType(name="token_prob", dtype=np.float32),
        ct.TensorType(name="top_k_ids", dtype=np.int32),
        ct.TensorType(name="top_k_logits", dtype=np.float32),
    ]

    jd_model = _coreml_convert(
        traced_jd, jd_inputs, jd_outputs, export_settings,
        compute_units_override=ct.ComputeUnit.CPU_ONLY,
    )

    jd_path = output_path / "joint_decision.mlpackage"
    _save_mlpackage(jd_model, jd_path, "JointDecision")
    typer.echo(f"Saved: {jd_path}")

    # ========== Save Metadata ==========
    typer.echo("\n=== Saving Metadata ===")

    metadata = {
        "model_id": model_id,
        "model_name": "parakeet_realtime_eou_120m-v1-split",
        "streaming_mode": "split_encoder",
        "sample_rate": sample_rate,
        "mel_dim": mel_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "mel_frames_per_chunk": mel_frames_per_chunk,
        "vocab_size": vocab_size,
        "blank_id": vocab_size,
        "decoder_hidden": decoder_hidden,
        "decoder_layers": decoder_layers,
        "cache_channel_size": cache_channel_size,
        "cache_time_size": cache_time_size,
        "components": {
            "preprocessor": "preprocessor.mlpackage",
            "pre_encode": "pre_encode.mlpackage",
            "conformer": "conformer_streaming.mlpackage",
            "decoder": "decoder.mlpackage",
            "joint_decision": "joint_decision.mlpackage",
        },
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    typer.echo(f"Saved: {output_path / 'metadata.json'}")

    # Copy vocabulary
    tokenizer = asr_model.tokenizer
    vocab = {}
    for i in range(tokenizer.vocab_size):
        vocab[str(i)] = tokenizer.ids_to_tokens([i])[0]

    with open(output_path / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    typer.echo(f"Saved: {output_path / 'vocab.json'}")

    typer.echo("\n=== Export Complete ===")
    typer.echo(f"Output directory: {output_path}")


if __name__ == "__main__":
    typer.run(main)
