"""Convert traceable flow decoder to CoreML."""
import torch
import numpy as np
import coremltools as ct
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))  # for: from traceable_* import ...

from traceable_flow_decoder import TraceableFlowDecoder


def convert_flow_decoder():
    print("Loading model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Creating traceable flow decoder...")
    flow_decoder = TraceableFlowDecoder.from_flowlm(model.flow_lm)
    flow_decoder.eval()

    print("Creating example inputs...")
    transformer_out = torch.randn(1, 1024)
    latent = torch.randn(1, 32)
    s = torch.tensor([[0.0]])
    t = torch.tensor([[0.125]])

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(flow_decoder, (transformer_out, latent, s, t))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="transformer_out", shape=(1, 1024)),
            ct.TensorType(name="latent", shape=(1, 32)),
            ct.TensorType(name="s", shape=(1, 1)),
            ct.TensorType(name="t", shape=(1, 1)),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "flow_decoder.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    # Test
    print("\nTesting CoreML model...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    test_transformer = np.random.randn(1, 1024).astype(np.float32)
    test_latent = np.random.randn(1, 32).astype(np.float32)
    test_s = np.array([[0.0]], dtype=np.float32)
    test_t = np.array([[0.125]], dtype=np.float32)

    outputs = coreml_model.predict({
        'transformer_out': test_transformer,
        'latent': test_latent,
        's': test_s,
        't': test_t,
    })

    print(f"Output keys: {list(outputs.keys())}")
    velocity = list(outputs.values())[0]
    print(f"Velocity shape: {velocity.shape}")
    print(f"Velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    convert_flow_decoder()
