
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

        return outputs

def export_streaming_encoder(model_id="nvidia/parakeet_realtime_eou_120m-v1", output_path="streaming_encoder.mlpackage", frames=16):
    print(f"Loading model: {model_id}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()
    
    encoder = asr_model.encoder
    wrapper = StreamingEncoderWrapper(encoder)
    wrapper.eval()

    # Define input shapes
    # 16 frames = 160ms
    print(f"Exporting for chunk size: {frames} frames ({frames*10}ms)")
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
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
    )

    print(f"Saving to {output_path}")
    mlmodel.save(output_path)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=16, help="Number of frames per chunk (10ms per frame)")
    parser.add_argument("--output", type=str, default="streaming_encoder.mlpackage", help="Output path")
    args = parser.parse_args()
    
    export_streaming_encoder(frames=args.frames, output_path=args.output)
