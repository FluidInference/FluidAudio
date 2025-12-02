
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
import coremltools as ct
import numpy as np
from typing import Tuple

class StreamingEncoderWrapper(nn.Module):
    """Wrapper for cache-aware streaming encoder."""

    def __init__(self, encoder: nn.Module, keep_all_outputs: bool = True):
        super().__init__()
        self.encoder = encoder
        self.keep_all_outputs = keep_all_outputs

        if encoder.streaming_cfg is None:
            encoder.setup_streaming_params()
        self.streaming_cfg = encoder.streaming_cfg

    def forward(
        self,
        mel: torch.Tensor,
        mel_length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        # Call encoder with cache
        outputs = self.encoder.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        # Handle cache updates (ring buffer)
        # NeMo returns only the updated part of the cache
        # We need to concatenate it with the previous cache (shifted)
        
        # 1. cache_last_channel: [layers, 1, T, D] -> dim 2
        new_channel_cache = outputs[2]
        update_len = new_channel_cache.size(2)
        if update_len < cache_last_channel.size(2):
            # Shift and append
            full_channel_cache = torch.cat([
                cache_last_channel[:, :, update_len:, :],
                new_channel_cache
            ], dim=2)
        else:
            full_channel_cache = new_channel_cache

        # 2. cache_last_time: [layers, 1, D, T] -> dim 3
        new_time_cache = outputs[3]
        update_len_time = new_time_cache.size(3)
        if update_len_time < cache_last_time.size(3):
            # Shift and append
            full_time_cache = torch.cat([
                cache_last_time[:, :, :, update_len_time:],
                new_time_cache
            ], dim=3)
        else:
            full_time_cache = new_time_cache
            
        # Construct new outputs tuple
        # (encoder, encoder_len, full_channel_cache, full_time_cache, cache_len)
        return (outputs[0], outputs[1], full_channel_cache, full_time_cache, outputs[4])

def export_streaming_encoder(model_id="nvidia/parakeet_realtime_eou_120m-v1", output_path="streaming_encoder.mlpackage", frames=16, shift=None, streaming_chunk_size=None):
    print(f"Loading model: {model_id}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()
    
    encoder = asr_model.encoder
    
    # Configure streaming params
    # If streaming_chunk_size is provided, use it. Otherwise use frames.
    c_size = streaming_chunk_size if streaming_chunk_size is not None else frames
    s_size = shift if shift is not None else c_size
    
    print(f"Setting up streaming params: chunk_size={c_size}, shift_size={s_size}")
    encoder.setup_streaming_params(chunk_size=c_size, shift_size=s_size)
    
    wrapper = StreamingEncoderWrapper(encoder)
    wrapper.eval()

    # Define input shapes
    # 16 frames = 160ms
    print(f"Exporting for chunk size: {frames} frames ({frames*10}ms)")
    if shift:
        print(f"Shift size: {shift} frames ({shift*10}ms)")
        
    mel_dim = 128  # Parakeet uses 128 mel features, not 80
    
    # Cache shapes: number of layers = 17 (FastConformer architecture)
    num_layers = 17
    
    example_input = (
        torch.randn(1, mel_dim, frames),
        torch.tensor([frames], dtype=torch.int32),
        torch.randn(num_layers, 1, 70, 512), # cache_last_channel
        torch.randn(num_layers, 1, 512, 8),  # cache_last_time
        torch.tensor([0], dtype=torch.int32) # cache_last_channel_len
    )

    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, example_input, strict=False)

    print("Converting to CoreML...")
    inputs = [
        ct.TensorType(name="mel", shape=(1, mel_dim, frames), dtype=np.float32),
        ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="cache_last_channel", shape=(num_layers, 1, 70, 512), dtype=np.float32),
        ct.TensorType(name="cache_last_time", shape=(num_layers, 1, 512, 8), dtype=np.float32),
        ct.TensorType(name="cache_last_channel_len", shape=(1,), dtype=np.int32),
    ]

    outputs = [
        ct.TensorType(name="encoder", dtype=np.float32),
        ct.TensorType(name="encoder_length", dtype=np.int32),
        ct.TensorType(name="cache_last_channel_out", dtype=np.float32),
        ct.TensorType(name="cache_last_time_out", dtype=np.float32),
        ct.TensorType(name="cache_last_channel_len_out", dtype=np.int32),
    ]

    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )

    print(f"Saving to {output_path}")
    mlmodel.save(output_path)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16, help="Number of frames per chunk (10ms per frame)")
    parser.add_argument("--shift", type=int, default=None, help="Shift size in frames (default: same as frames)")
    parser.add_argument("--model-chunk-size", type=int, default=None, help="Chunk size for model setup (output steps). If None, uses frames.")
    parser.add_argument("--output", type=str, default="streaming_encoder.mlpackage", help="Output path")
    args = parser.parse_args()
    
    export_streaming_encoder(frames=args.frames, shift=args.shift, streaming_chunk_size=args.model_chunk_size, output_path=args.output)
