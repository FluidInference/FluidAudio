"""Traceable Flow Decoder with variable time step for LSD decoding."""
import torch
import torch.nn as nn
from typing import Tuple


class TraceableFlowDecoder(nn.Module):
    """Flow decoder that takes time interval as input for LSD decoding.

    For LSD (Lagrangian Self Distillation) with N steps:
    - Start with noise z_0
    - For i in [0, 1, ..., N-1]:
        s = i / N      (start time)
        t = (i+1) / N  (end time)
        velocity = flow_net(transformer_out, s, t, z_i)
        z_{i+1} = z_i + velocity * (1/N)
    - Final latent = z_N
    """

    def __init__(self, flow_net, ldim: int = 32):
        super().__init__()
        self.flow_net = flow_net
        self.ldim = ldim

    @classmethod
    def from_flowlm(cls, flow_lm) -> "TraceableFlowDecoder":
        return cls(flow_lm.flow_net, flow_lm.ldim)

    def forward(
        self,
        transformer_out: torch.Tensor,  # [B, 1024]
        latent: torch.Tensor,  # [B, 32] current latent estimate
        s: torch.Tensor,  # [B, 1] start time of interval
        t: torch.Tensor,  # [B, 1] end time of interval
    ) -> torch.Tensor:
        """Single flow step.

        Args:
            transformer_out: Conditioning from FlowLM backbone [B, 1024]
            latent: Current latent estimate [B, 32]
            s: Start time [B, 1], range [0, 1)
            t: End time [B, 1], range (0, 1]

        Returns:
            velocity: Flow direction [B, 32]
        """
        # Get flow direction with both time endpoints
        velocity = self.flow_net(transformer_out, s, t, latent)

        return velocity


def test_traceable_flow_decoder():
    print("Loading model...")
    import sys
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    sys.path.insert(0, _project_dir)

    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Creating traceable flow decoder...")
    flow_decoder = TraceableFlowDecoder.from_flowlm(model.flow_lm)
    flow_decoder.eval()

    print("Testing forward pass...")
    transformer_out = torch.randn(1, 1024)
    latent = torch.randn(1, 32)
    s = torch.tensor([[0.0]])
    t = torch.tensor([[0.125]])

    with torch.no_grad():
        velocity = flow_decoder(transformer_out, latent, s, t)

    print(f"Velocity shape: {velocity.shape}")
    print(f"Velocity range: [{velocity.min().item():.4f}, {velocity.max().item():.4f}]")

    print("\nTesting full LSD decoding...")
    num_steps = 8
    latent = torch.randn(1, 32)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        s = torch.tensor([[step * dt]])
        t = torch.tensor([[(step + 1) * dt]])
        with torch.no_grad():
            velocity = flow_decoder(transformer_out, latent, s, t)
        latent = latent + velocity * dt
        print(f"  Step {step}: s={s.item():.3f}, t={t.item():.3f}, latent range [{latent.min().item():.4f}, {latent.max().item():.4f}]")

    print(f"\nFinal latent shape: {latent.shape}")
    print("Done!")


if __name__ == "__main__":
    test_traceable_flow_decoder()
