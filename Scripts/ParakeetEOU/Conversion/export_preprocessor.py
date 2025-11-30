import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
import coremltools as ct
import numpy as np
import argparse

class PreprocessorWrapper(nn.Module):
    """Wrapper for audio preprocessor."""
    
    def __init__(self, preprocessor: nn.Module):
        super().__init__()
        self.preprocessor = preprocessor
    
    def forward(
        self,
        input_signal: torch.Tensor,
        length: torch.Tensor,
    ):
        # Call preprocessor
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal,
            length=length
        )
        return processed_signal, processed_signal_length

def export_preprocessor(
    model_id="nvidia/parakeet_realtime_eou_120m-v1",
    output_path="preprocessor.mlpackage",
    chunk_ms=160
):
    print(f"Loading model: {model_id}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()
    
    preprocessor = asr_model.preprocessor
    # Disable dither and padding for consistent inference
    if hasattr(preprocessor, 'dither'):
        preprocessor.dither = 0.0
    if hasattr(preprocessor, 'pad_to'):
        preprocessor.pad_to = 0
        
    wrapper = PreprocessorWrapper(preprocessor)
    wrapper.eval()
    
    # Calculate audio samples for chunk
    # 160ms at 16kHz = 2560 samples
    chunk_samples = int(chunk_ms / 1000 * 16000)
    
    print(f"Chunk: {chunk_ms}ms = {chunk_samples} samples")
    
    # Create example input
    example_input = (
        torch.randn(1, chunk_samples),
        torch.tensor([chunk_samples], dtype=torch.int64),
    )
    
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, example_input, strict=False)
    
    print("Converting to CoreML...")
    # Use RangeDim for variable-length audio input
    inputs = [
        ct.TensorType(
            name="input_signal",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1600, upper_bound=16000, default=chunk_samples))),
            dtype=np.float32
        ),
        ct.TensorType(name="length", shape=(1,), dtype=np.int32),
    ]
    
    outputs = [
        ct.TensorType(name="mel", dtype=np.float32),
        ct.TensorType(name="mel_length", dtype=np.int32),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk size in milliseconds")
    parser.add_argument("--output-path", type=str, default="preprocessor.mlpackage", help="Output path")
    args = parser.parse_args()
    
    export_preprocessor(chunk_ms=args.chunk_ms, output_path=args.output_path)
