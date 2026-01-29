"""Convert FlowLM step model to CoreML."""
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

from traceable_flowlm_step import TraceableFlowLMStep


def convert_flowlm_step():
    print("Loading model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Creating traceable step model...")
    max_seq_len = 512
    step_model = TraceableFlowLMStep.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    step_model.eval()

    print("Creating example inputs...")
    B = 1
    T = 1
    H = 16
    D = 64

    sequence = torch.randn(B, T, 32)
    bos_emb = model.flow_lm.bos_emb.data

    # Create example caches and positions
    caches = []
    positions = []
    for i in range(6):
        cache = torch.zeros(2, B, max_seq_len, H, D)
        # Fill some positions with data (simulating voice/text conditioning)
        cache[:, :, :136, :, :] = torch.randn(2, B, 136, H, D)
        caches.append(cache)
        positions.append(torch.tensor([136.0]))

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(step_model, (
            sequence, bos_emb,
            caches[0], positions[0],
            caches[1], positions[1],
            caches[2], positions[2],
            caches[3], positions[3],
            caches[4], positions[4],
            caches[5], positions[5],
        ))

    print("Converting to CoreML...")
    inputs = [
        ct.TensorType(name="sequence", shape=(1, 1, 32)),
        ct.TensorType(name="bos_emb", shape=(32,)),
    ]
    for i in range(6):
        inputs.append(ct.TensorType(name=f"cache{i}", shape=(2, 1, max_seq_len, H, D)))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(1,)))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "flowlm_step.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    # Test
    print("\nTesting CoreML model...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    # Create test inputs
    test_seq = np.random.randn(1, 1, 32).astype(np.float32)
    test_bos = bos_emb.numpy().astype(np.float32)

    test_caches = {}
    test_positions = {}
    for i in range(6):
        cache = np.zeros((2, 1, max_seq_len, H, D), dtype=np.float32)
        cache[:, :, :136, :, :] = np.random.randn(2, 1, 136, H, D).astype(np.float32)
        test_caches[f'cache{i}'] = cache
        test_positions[f'position{i}'] = np.array([136.0], dtype=np.float32)

    outputs = coreml_model.predict({
        'sequence': test_seq,
        'bos_emb': test_bos,
        **test_caches,
        **test_positions,
    })

    print(f"Output keys: {list(outputs.keys())}")
    for k, v in outputs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    convert_flowlm_step()
