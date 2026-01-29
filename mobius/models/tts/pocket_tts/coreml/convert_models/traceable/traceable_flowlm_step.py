"""Traceable FlowLM step for CoreML - audio frame generation only.

This is the "step" model used after text/voice conditioning is in the KV cache.
It does NOT concatenate text embeddings - it just processes the audio latent frames.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
import math
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...


class TraceableFlowLMStep(nn.Module):
    """FlowLM backbone for step-by-step generation (no text concatenation).

    Use this after text/voice conditioning is already in the KV cache.
    """

    def __init__(self, max_seq_len: int = 200):
        super().__init__()

        self.num_layers = 6
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = 64
        self.max_seq_len = max_seq_len
        self.rope_max_period = 10000.0

        # Input projection (ldim=32 -> dim=1024)
        self.input_linear = nn.Linear(32, 1024, bias=False)

        # Transformer layers
        for i in range(self.num_layers):
            # Attention
            setattr(self, f'attn{i}_in_proj', nn.Linear(1024, 3 * 1024, bias=False))
            setattr(self, f'attn{i}_out_proj', nn.Linear(1024, 1024, bias=False))

            # Norms
            setattr(self, f'norm{i}_1', nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f'norm{i}_2', nn.LayerNorm(1024, eps=1e-5))

            # FFN
            hidden_dim = 4096
            setattr(self, f'linear{i}_1', nn.Linear(1024, hidden_dim, bias=False))
            setattr(self, f'linear{i}_2', nn.Linear(hidden_dim, 1024, bias=False))

        # Output norm
        self.out_norm = nn.LayerNorm(1024, eps=1e-5)

        # EOS prediction
        self.out_eos = nn.Linear(1024, 1)

    @classmethod
    def from_flowlm(cls, flow_lm_model, max_seq_len: int = 200) -> "TraceableFlowLMStep":
        """Create traceable step model from original FlowLM model."""
        wrapper = cls(max_seq_len)

        # Copy input linear
        wrapper.input_linear.weight.data.copy_(flow_lm_model.input_linear.weight.data)

        # Copy transformer layers
        for i, layer in enumerate(flow_lm_model.transformer.layers):
            # Attention
            getattr(wrapper, f'attn{i}_in_proj').weight.data.copy_(layer.self_attn.in_proj.weight.data)
            getattr(wrapper, f'attn{i}_out_proj').weight.data.copy_(layer.self_attn.out_proj.weight.data)

            # Norms
            getattr(wrapper, f'norm{i}_1').weight.data.copy_(layer.norm1.weight.data)
            getattr(wrapper, f'norm{i}_1').bias.data.copy_(layer.norm1.bias.data)
            getattr(wrapper, f'norm{i}_2').weight.data.copy_(layer.norm2.weight.data)
            getattr(wrapper, f'norm{i}_2').bias.data.copy_(layer.norm2.bias.data)

            # FFN
            getattr(wrapper, f'linear{i}_1').weight.data.copy_(layer.linear1.weight.data)
            getattr(wrapper, f'linear{i}_2').weight.data.copy_(layer.linear2.weight.data)

        # Output norm
        wrapper.out_norm.weight.data.copy_(flow_lm_model.out_norm.weight.data)
        wrapper.out_norm.bias.data.copy_(flow_lm_model.out_norm.bias.data)

        # EOS
        wrapper.out_eos.weight.data.copy_(flow_lm_model.out_eos.weight.data)
        wrapper.out_eos.bias.data.copy_(flow_lm_model.out_eos.bias.data)

        return wrapper

    def _apply_rope_tensor(self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings with tensor offset.

        Uses interleaved pairs: (q[..., 0], q[..., 1]), (q[..., 2], q[..., 3]), etc.
        """
        B, T, H, D = q.shape
        Bk, Tk, Hk, Dk = k.shape
        D_float = float(self.head_dim)
        half_d = self.head_dim // 2

        # Compute RoPE frequencies
        ds = torch.arange(half_d, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.rope_max_period) * 2.0 / D_float))

        # Position indices
        ts = torch.arange(T, device=q.device, dtype=torch.float32)
        offset_f = offset.float() if offset.dtype != torch.float32 else offset
        ts = ts + offset_f.view(B, 1)
        ts = ts.view(B, T, 1, 1)

        # View as interleaved pairs
        q_complex = q.view(B, T, H, half_d, 2)
        k_complex = k.view(Bk, Tk, Hk, half_d, 2)

        # Extract real and imaginary parts
        qr = q_complex[..., 0].float()
        qi = q_complex[..., 1].float()
        kr = k_complex[..., 0].float()
        ki = k_complex[..., 1].float()

        # Compute rotation angles
        rotr = torch.cos(freqs * ts)
        roti = torch.sin(freqs * ts)

        # Apply complex rotation
        qor = qr * rotr - qi * roti
        qoi = qr * roti + qi * rotr
        kor = kr * rotr - ki * roti
        koi = kr * roti + ki * rotr

        # Stack back to original shape
        dtype = q.dtype
        qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
        ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

        return qo.view(B, T, H, D), ko.view(Bk, Tk, Hk, Dk)

    def _streaming_attention(
        self,
        x: torch.Tensor,
        in_proj: nn.Linear,
        out_proj: nn.Linear,
        cache: torch.Tensor,
        position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streaming attention with fixed-size KV cache."""
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        max_len = cache.shape[2]
        max_len_float = float(max_len)

        pos_float = position.float() if position.dtype != torch.float32 else position

        # Project to Q, K, V
        qkv = in_proj(x).reshape(B, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply RoPE
        q, k = self._apply_rope_tensor(q, k, pos_float)

        # Update cache
        new_cache = cache.clone()
        write_base_float = pos_float.view(B, 1, 1, 1)
        write_offsets_float = torch.arange(T, device=x.device, dtype=torch.float32).view(1, T, 1, 1)
        write_indices_float = write_base_float + write_offsets_float
        write_indices_float = write_indices_float - torch.floor(write_indices_float / max_len_float) * max_len_float
        write_indices = write_indices_float.long().expand(B, T, H, D)

        new_cache[0] = new_cache[0].scatter(1, write_indices, k)
        new_cache[1] = new_cache[1].scatter(1, write_indices, v)

        # Full cache attention
        keys = new_cache[0]
        values = new_cache[1]

        # Replace NaN with 0 for attention (NaN * 0 weight = NaN, but 0 * 0 = 0)
        # The mask will ensure these positions don't contribute to the output
        keys = torch.where(torch.isnan(keys), torch.zeros_like(keys), keys)
        values = torch.where(torch.isnan(values), torch.zeros_like(values), values)

        q = q.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Create attention mask
        q_offsets = torch.arange(T, device=x.device, dtype=torch.float32).view(1, T, 1)
        q_positions = pos_float.view(B, 1, 1) + q_offsets
        k_positions = torch.arange(max_len, device=x.device, dtype=torch.float32).view(1, 1, max_len)

        valid_len = pos_float.view(B, 1, 1) + float(T)
        valid_mask = k_positions < valid_len
        causal_mask = k_positions <= q_positions

        attn_mask = valid_mask & causal_mask
        attn_mask = attn_mask.unsqueeze(1)

        # Manual attention (avoids scaled_dot_product_attention op for iOS 17 compat)
        scale = 1.0 / (q.shape[-1] ** 0.5)
        attn_weights = torch.matmul(q, keys.transpose(-2, -1)) * scale
        attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output = out_proj(attn_output)

        new_position = pos_float + float(T)

        return output, new_cache, new_position

    def forward(
        self,
        sequence: torch.Tensor,  # [B, T, 32] input latents
        bos_emb: torch.Tensor,  # [32] BOS embedding
        cache0: torch.Tensor, position0: torch.Tensor,
        cache1: torch.Tensor, position1: torch.Tensor,
        cache2: torch.Tensor, position2: torch.Tensor,
        cache3: torch.Tensor, position3: torch.Tensor,
        cache4: torch.Tensor, position4: torch.Tensor,
        cache5: torch.Tensor, position5: torch.Tensor,
    ):
        """Forward pass for step generation.

        Args:
            sequence: [B, T, 32] input latents (NaN for BOS)
            bos_emb: [32] BOS embedding
            cache0-5: [2, B, max_seq_len, 16, 64] KV caches (pre-filled with text/voice)
            position0-5: [B] current positions

        Returns:
            transformer_out: [B, T, 1024]
            is_eos: [B, T, 1]
            new_cache0-5, new_position0-5
        """
        # Replace NaN values with BOS embedding
        sequence = torch.where(torch.isnan(sequence), bos_emb, sequence)

        # Project input (NO text concatenation - text is already in KV cache)
        x = self.input_linear(sequence)  # [B, T, 1024]

        caches = [cache0, cache1, cache2, cache3, cache4, cache5]
        positions = [position0, position1, position2, position3, position4, position5]
        new_caches = []
        new_positions = []

        # Transformer layers
        for i in range(self.num_layers):
            residual = x
            x_norm = getattr(self, f'norm{i}_1')(x)
            attn_out, new_cache, new_pos = self._streaming_attention(
                x_norm,
                getattr(self, f'attn{i}_in_proj'),
                getattr(self, f'attn{i}_out_proj'),
                caches[i],
                positions[i]
            )
            x = residual + attn_out

            residual = x
            x_norm = getattr(self, f'norm{i}_2')(x)
            ffn_out = getattr(self, f'linear{i}_2')(F.gelu(getattr(self, f'linear{i}_1')(x_norm)))
            x = residual + ffn_out

            new_caches.append(new_cache)
            new_positions.append(new_pos)

        # Output
        x = self.out_norm(x)
        is_eos = self.out_eos(x)

        return (
            x,
            is_eos,
            new_caches[0], new_positions[0],
            new_caches[1], new_positions[1],
            new_caches[2], new_positions[2],
            new_caches[3], new_positions[3],
            new_caches[4], new_positions[4],
            new_caches[5], new_positions[5],
        )


def test_traceable_step():
    """Test the step model matches original FlowLM."""
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Creating traceable step model...")
    step_model = TraceableFlowLMStep.from_flowlm(model.flow_lm, max_seq_len=200)
    step_model.eval()

    # Initialize with voice state
    print("\nLoading voice state...")
    voice_state = model.get_state_for_audio_prompt("alba")
    model._expand_kv_cache(voice_state, sequence_length=200)

    # Process text tokens first to fill KV cache
    print("Processing text tokens...")
    from pocket_tts.models.tts_model import prepare_text_prompt
    prepared_text, _ = prepare_text_prompt("Hello world")
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=text_tokens)

    # Extract KV cache from voice state
    print("Extracting KV cache...")
    caches = []
    positions = []

    for i in range(6):
        key = f'transformer.layers.{i}.self_attn'
        layer_state = voice_state[key]

        # Extract cache [2, B, max_len, H, D]
        cache = layer_state['cache']  # Already [2, B, max_len, H, D]
        caches.append(cache)

        # Position is the LENGTH of current_end (number of elements processed)
        current_end = layer_state['current_end']
        pos = torch.tensor([float(len(current_end))])
        positions.append(pos)

    print(f"Position: {positions[0].item()}")

    # Test forward pass
    print("\nTesting forward pass...")
    bos_emb = model.flow_lm.bos_emb.data
    sequence = torch.full((1, 1, 32), float('nan'))

    with torch.no_grad():
        outputs = step_model(
            sequence, bos_emb,
            caches[0], positions[0],
            caches[1], positions[1],
            caches[2], positions[2],
            caches[3], positions[3],
            caches[4], positions[4],
            caches[5], positions[5],
        )

    transformer_out = outputs[0]
    is_eos = outputs[1]
    print(f"Transformer output shape: {transformer_out.shape}")
    print(f"Transformer output range: [{transformer_out.min():.4f}, {transformer_out.max():.4f}]")
    print(f"EOS output: {is_eos.item():.4f}")
    print("Done!")


if __name__ == "__main__":
    test_traceable_step()
