
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import typer
from pathlib import Path
from typing import Tuple, List, Optional
import json
import shutil

# Iimport torch
import coremltools as ct
import numpy as np
import argparse
from nemo.collections.asr.models import EncDecRNNTBPEModel

app = typer.Typer()

class LoopbackEncoderWrapper(nn.Module):
    """
    Wraps the entire Parakeet Encoder (PreEncode + Conformer) for CoreML Loopback Streaming.
    
    Inputs:
      - audio_signal: [B, D, T] (Mel spectrogram chunk)
      - audio_length: [B]
      - pre_cache: [B, D, pre_cache_size] (Previous audio context)
      - cache_last_channel: [layers, B, cache_size, hidden]
      - cache_last_time: [layers, B, hidden, time_cache]
      - cache_last_channel_len: [B]
      
    Outputs:
      - encoded_output: [B, D_out, T_out]
      - encoded_length: [B]
      - new_pre_cache: [B, D, pre_cache_size]
      - new_cache_last_channel
      - new_cache_last_time
      - new_cache_last_channel_len
    """
    def __init__(self, encoder, pre_cache_size=16):
        super().__init__()
        self.encoder = encoder
        self.pre_cache_size = pre_cache_size
        
    def forward(
        self, 
        audio_signal: torch.Tensor, 
        audio_length: torch.Tensor,
        pre_cache: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 1. Prepend pre_cache to audio_signal
        # audio_signal: [B, D, T]
        # pre_cache: [B, D, T_cache]
        full_input = torch.cat([pre_cache, audio_signal], dim=2)
        full_length = audio_length + self.pre_cache_size
        
        # 2. Extract NEW pre_cache (last N frames of full_input)
        # Note: We do this BEFORE processing because we want the raw audio context
        new_pre_cache = full_input[:, :, -self.pre_cache_size:]
        
        # 3. Process with Encoder
        # Reconstruct NeMo cache object
        current_cache = [cache_last_channel, cache_last_time, cache_last_channel_len]
        
        encoded, encoded_len, new_cache_channel, new_cache_time, new_cache_len = self.encoder.cache_aware_stream_step(
            processed_signal=full_input,
            processed_signal_length=full_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len
        )
        
        # 4. Drop the first few frames corresponding to pre_cache?
        # NeMo's cache_aware_stream_step usually handles the "valid" output frames.
        # But since we manually prepended, we might get extra output frames.
        # However, for streaming, we usually want the model to see the context but only output the new tokens.
        # Let's trust NeMo's streaming logic for now, or check if we need to slice.
        # Given we are using 'cache_aware_stream_step', it expects the full context window?
        # Actually, standard usage is: input IS the new chunk, but internal convolution looks at past.
        # But since we are stateless, we MUST provide the past.
        # So passing (pre_cache + chunk) is correct.
        
        # Cast lengths to Int32 for CoreML
        encoded_len_32 = encoded_len.to(dtype=torch.int32)
        new_channel_len_32 = new_cache_len.to(dtype=torch.int32)
        
        return encoded, encoded_len_32, new_pre_cache, new_cache_channel, new_cache_time, new_channel_len_32

def _coreml_convert(
    traced_model,
    inputs,
    outputs,
    compute_units=ct.ComputeUnit.CPU_ONLY
):
    return ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS14,
    )

def main():
    model_id: str = "nvidia/parakeet_realtime_eou_120m-v1"
    output_dir: str = "temp_swift_models/StreamingLoopback"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_id}...")
    asr_model = EncDecRNNTBPEModel.from_pretrained(model_name=model_id)
    asr_model.eval()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-frames", type=int, default=17, help="Number of frames in the input chunk (e.g. 17 for 160ms, 129 for 1.28s)")
    args = parser.parse_args()

    encoder = asr_model.encoder
    
    # --- Configuration ---
    # 160ms chunk = 16 frames (but preprocessor produces 17 with padding/centering)
    # 1.28s chunk = 128 frames (preprocessor produces 129)
    chunk_size_in = args.chunk_frames 
    mel_dim = 128
    hidden_dim = encoder.d_model # 512
    num_layers = len(encoder.layers) # 17
    
    # Cache sizes
    cache_channel_size = 70
    cache_time_size = 8
    pre_cache_size = 16
    
    print(f"Config: Chunk={chunk_size_in}, Mel={mel_dim}, Hidden={hidden_dim}, Layers={num_layers}")
    print(f"Cache: Channel={cache_channel_size}, Time={cache_time_size}, Pre={pre_cache_size}")

    # --- Wrapper ---
    wrapper = LoopbackEncoderWrapper(encoder, pre_cache_size=pre_cache_size)
    wrapper.eval()
    
    # --- Test Inputs (for Tracing) ---
    batch_size = 1
    test_mel = torch.randn(batch_size, mel_dim, chunk_size_in)
    test_mel_len = torch.tensor([chunk_size_in], dtype=torch.int32)
    test_pre_cache = torch.zeros(batch_size, mel_dim, pre_cache_size)
    
    # Initial Cache (Zeros)
    test_cache_channel = torch.zeros(num_layers, batch_size, cache_channel_size, hidden_dim)
    test_cache_time = torch.zeros(num_layers, batch_size, hidden_dim, cache_time_size)
    test_cache_len = torch.zeros(batch_size, dtype=torch.int32) 
    
    print("Tracing model...")
    traced_model = torch.jit.trace(
        wrapper,
        (test_mel, test_mel_len, test_pre_cache, test_cache_channel, test_cache_time, test_cache_len),
        strict=False
    )
    
    # --- CoreML Conversion ---
    print("Converting to CoreML...")
    
    inputs = [
        ct.TensorType(name="audio_signal", shape=(1, 128, chunk_size_in), dtype=np.float32),
        ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="pre_cache", shape=(1, 128, pre_cache_size), dtype=np.float32),
        ct.TensorType(name="cache_last_channel", shape=(num_layers, 1, cache_channel_size, hidden_dim), dtype=np.float32),
        ct.TensorType(name="cache_last_time", shape=(num_layers, 1, hidden_dim, cache_time_size), dtype=np.float32),
        ct.TensorType(name="cache_last_channel_len", shape=(1,), dtype=np.int32),
    ]
    
    outputs = [
        ct.TensorType(name="encoded_output", dtype=np.float32),
        ct.TensorType(name="encoded_length", dtype=np.int32),
        ct.TensorType(name="new_pre_cache", dtype=np.float32),
        ct.TensorType(name="new_cache_last_channel", dtype=np.float32),
        ct.TensorType(name="new_cache_last_time", dtype=np.float32),
        ct.TensorType(name="new_cache_last_channel_len", dtype=np.int32),
    ]
    
    mlmodel = _coreml_convert(traced_model, inputs, outputs)
    
    save_path = output_path / "streaming_encoder.mlpackage"
    mlmodel.save(str(save_path))
    print(f"Saved: {save_path}")
    
    # Also export Preprocessor, Decoder, Joint for completeness? 
    # For now, let's assume we reuse the existing ones or export them separately if needed.
    # But the user asked specifically for the Encoder loopback.

if __name__ == "__main__":
    main()
