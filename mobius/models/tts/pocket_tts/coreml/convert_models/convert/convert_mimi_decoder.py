"""Convert Mimi streaming decoder to CoreML.

NOTE: The Mimi decoder uses in-place state mutations (state[:] = ...) in its
streaming convolution layers (StreamingConv1d, StreamingConvTranspose1d).
coremltools cannot convert these in-place operations directly.

The existing mimi_decoder.mlpackage was converted using a custom traceable
wrapper that rewrites all streaming ops as functional (returning new tensors
instead of mutating in place). This requires rewriting the forward pass of:
  - StreamingConv1d.forward()       (conv.py)
  - StreamingConvTranspose1d.forward()  (conv.py)
  - MimiTransformerLayer attention cache updates  (mimi_transformer.py)

The model has 26 streaming state tensors (see traceable_mimi_decoder.py for
the full list) and produces 1920 audio samples per frame at 24kHz.

To regenerate mimi_decoder.mlpackage:
1. Create a functional TraceableMimiDecoder that avoids all in-place ops
2. Trace with sequence_length=256 for attention caches
3. Convert with compute_precision=FLOAT32, target=macOS15

Input:  latent [1, 512, 1]  +  26 state tensors
Output: audio  [1, 1, 1920] +  26 updated state tensors
"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))  # for: from traceable_* import ...


def convert():
    """Reference conversion — requires functional Mimi wrapper (see docstring)."""
    import torch
    import numpy as np
    import coremltools as ct
    from pocket_tts import TTSModel
    from pocket_tts.modules.stateful_module import init_states

    print("Loading model...")
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    # Show the state structure for reference
    print("\nMimi decoder streaming state:")
    state = init_states(model.mimi.decoder, batch_size=1, sequence_length=256)
    state.update(
        init_states(model.mimi.decoder_transformer, batch_size=1, sequence_length=256)
    )
    if hasattr(model.mimi, "upsample"):
        state.update(
            init_states(model.mimi.upsample, batch_size=1, sequence_length=256)
        )

    total_params = 0
    for mod_name, mod_state in state.items():
        for key, tensor in mod_state.items():
            total_params += tensor.numel()
            print(f"  {mod_name}.{key}: {list(tensor.shape)}")

    print(f"\nTotal state elements: {total_params:,}")
    print(f"State tensors: {sum(len(s) for s in state.values())}")

    print(
        "\nERROR: Direct conversion not supported due to in-place state mutations."
    )
    print("The existing mimi_decoder.mlpackage uses a custom functional wrapper.")
    print("See this file's docstring for details on how to regenerate it.")
    return None


if __name__ == "__main__":
    convert()
