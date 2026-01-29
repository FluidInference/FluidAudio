"""Traceable Mimi streaming decoder for CoreML conversion.

Flattens the stateful Mimi decoder into explicit input/output tensors
so it can be traced with torch.jit.trace and converted to CoreML.

Input:  latent [1, 512, 1] + 26 state tensors
Output: audio [1, 1, 1920] + 26 updated state tensors
"""
import torch
import torch.nn as nn


# Ordered list of (state_name, shape) for the 26 Mimi streaming state tensors.
# Must match the manifest.json order used by the Swift loader.
MIMI_STATE_SPEC = [
    ("upsample_partial", [1, 512, 16]),
    ("attn0_cache", [2, 1, 8, 256, 64]),
    ("attn0_offset", [1]),
    ("attn0_end_offset", [1]),
    ("attn1_cache", [2, 1, 8, 256, 64]),
    ("attn1_offset", [1]),
    ("attn1_end_offset", [1]),
    ("conv0_prev", [1, 512, 6]),
    ("conv0_first", [1]),
    ("convtr0_partial", [1, 256, 6]),
    ("res0_conv0_prev", [1, 256, 2]),
    ("res0_conv0_first", [1]),
    ("res0_conv1_prev", [1, 128, 0]),
    ("res0_conv1_first", [1]),
    ("convtr1_partial", [1, 128, 5]),
    ("res1_conv0_prev", [1, 128, 2]),
    ("res1_conv0_first", [1]),
    ("res1_conv1_prev", [1, 64, 0]),
    ("res1_conv1_first", [1]),
    ("convtr2_partial", [1, 64, 4]),
    ("res2_conv0_prev", [1, 64, 2]),
    ("res2_conv0_first", [1]),
    ("res2_conv1_prev", [1, 32, 0]),
    ("res2_conv1_first", [1]),
    ("conv_final_prev", [1, 64, 2]),
    ("conv_final_first", [1]),
]


class TraceableMimiDecoder(nn.Module):
    """Wrapper that exposes Mimi's streaming state as flat tensor I/O."""

    def __init__(self, mimi_model):
        super().__init__()
        self.mimi = mimi_model

        # Build the mapping from flat state list to nested model_state dict.
        # init_states() returns {module_name: {key: tensor}}.
        from pocket_tts.modules.stateful_module import init_states
        self._nested_state = init_states(self.mimi.decoder, batch_size=1, sequence_length=1)
        # Also add decoder_transformer and upsample states
        self._nested_state.update(
            init_states(self.mimi.decoder_transformer, batch_size=1, sequence_length=1)
        )
        if hasattr(self.mimi, 'upsample'):
            self._nested_state.update(
                init_states(self.mimi.upsample, batch_size=1, sequence_length=1)
            )

    @classmethod
    def from_tts_model(cls, tts_model) -> "TraceableMimiDecoder":
        return cls(tts_model.mimi)

    def _pack_state(self, flat_tensors: tuple) -> dict:
        """Convert flat tensor tuple into nested model_state dict."""
        state = {}
        # Rebuild nested structure from init_states template
        idx = 0
        for module_name, module_state in self._nested_state.items():
            state[module_name] = {}
            for key in module_state:
                state[module_name][key] = flat_tensors[idx]
                idx += 1
        return state

    def _unpack_state(self, state: dict) -> tuple:
        """Extract flat tensor tuple from nested model_state dict."""
        tensors = []
        for module_name, module_state in self._nested_state.items():
            for key in module_state:
                tensors.append(state[module_name][key])
        return tuple(tensors)

    def forward(self, latent, *state_tensors):
        """
        Args:
            latent: [1, 512, 1] quantized latent frame
            *state_tensors: 26 flat state tensors

        Returns:
            audio: [1, 1, 1920] decoded audio frame
            *updated_states: 26 updated state tensors
        """
        model_state = self._pack_state(state_tensors)
        audio = self.mimi.decode_from_latent(latent, model_state)
        updated = self._unpack_state(model_state)
        return (audio,) + updated


def test_traceable_mimi():
    import sys
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    sys.path.insert(0, _project_dir)

    from pocket_tts import TTSModel
    from pocket_tts.modules.stateful_module import init_states

    print("Loading model...")
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Creating traceable Mimi decoder...")
    traceable = TraceableMimiDecoder.from_tts_model(model)
    traceable.eval()

    # Build initial state
    print("Building initial state...")
    state = init_states(model.mimi.decoder, batch_size=1, sequence_length=1)
    state.update(init_states(model.mimi.decoder_transformer, batch_size=1, sequence_length=1))
    if hasattr(model.mimi, 'upsample'):
        state.update(init_states(model.mimi.upsample, batch_size=1, sequence_length=1))

    flat_state = traceable._unpack_state(state)
    print(f"State tensors: {len(flat_state)}")
    for i, t in enumerate(flat_state):
        print(f"  [{i}] shape={list(t.shape)}")

    # Test forward pass
    print("\nTesting forward pass...")
    latent = torch.randn(1, 512, 1)
    with torch.no_grad():
        outputs = traceable(latent, *flat_state)

    audio = outputs[0]
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
    print(f"Updated state tensors: {len(outputs) - 1}")
    print("Done!")


if __name__ == "__main__":
    test_traceable_mimi()
