"""Traceable conditioning step for CoreML - fills KV cache one token at a time.

Unlike the FlowLM step model (which takes 32d latents via input_linear),
this takes 1024d conditioning embeddings directly (text or voice).
No BOS handling, no EOS output - just fills the KV cache.
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


class TraceableCondStep(nn.Module):
    """Processes one conditioning token through the transformer, updating KV cache.

    Input: conditioning [1, 1, 1024] (pre-embedded text or voice token)
    Output: updated KV caches and positions
    """

    def __init__(self, max_seq_len: int = 200):
        super().__init__()
        self.num_layers = 6
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = 64
        self.max_seq_len = max_seq_len
        self.rope_max_period = 10000.0

        for i in range(self.num_layers):
            setattr(self, f'attn{i}_in_proj', nn.Linear(1024, 3 * 1024, bias=False))
            setattr(self, f'attn{i}_out_proj', nn.Linear(1024, 1024, bias=False))
            setattr(self, f'norm{i}_1', nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f'norm{i}_2', nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f'linear{i}_1', nn.Linear(1024, 4096, bias=False))
            setattr(self, f'linear{i}_2', nn.Linear(4096, 1024, bias=False))

        self.out_norm = nn.LayerNorm(1024, eps=1e-5)

    @classmethod
    def from_flowlm(cls, flow_lm_model, max_seq_len: int = 200) -> "TraceableCondStep":
        wrapper = cls(max_seq_len)

        # Copy transformer layers (same weights as step model, no input_linear)
        for i, layer in enumerate(flow_lm_model.transformer.layers):
            getattr(wrapper, f'attn{i}_in_proj').weight.data.copy_(layer.self_attn.in_proj.weight.data)
            getattr(wrapper, f'attn{i}_out_proj').weight.data.copy_(layer.self_attn.out_proj.weight.data)
            getattr(wrapper, f'norm{i}_1').weight.data.copy_(layer.norm1.weight.data)
            getattr(wrapper, f'norm{i}_1').bias.data.copy_(layer.norm1.bias.data)
            getattr(wrapper, f'norm{i}_2').weight.data.copy_(layer.norm2.weight.data)
            getattr(wrapper, f'norm{i}_2').bias.data.copy_(layer.norm2.bias.data)
            getattr(wrapper, f'linear{i}_1').weight.data.copy_(layer.linear1.weight.data)
            getattr(wrapper, f'linear{i}_2').weight.data.copy_(layer.linear2.weight.data)

        wrapper.out_norm.weight.data.copy_(flow_lm_model.out_norm.weight.data)
        wrapper.out_norm.bias.data.copy_(flow_lm_model.out_norm.bias.data)

        return wrapper

    def _apply_rope_tensor(self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings with tensor offset.

        Uses interleaved pairs: (q[..., 0], q[..., 1]), (q[..., 2], q[..., 3]), etc.
        Identical to TraceableFlowLMStep._apply_rope_tensor.
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
        """Streaming attention with fixed-size KV cache.

        Identical to TraceableFlowLMStep._streaming_attention.
        """
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
        conditioning: torch.Tensor,  # [B, 1, 1024] pre-embedded conditioning token
        cache0: torch.Tensor, position0: torch.Tensor,
        cache1: torch.Tensor, position1: torch.Tensor,
        cache2: torch.Tensor, position2: torch.Tensor,
        cache3: torch.Tensor, position3: torch.Tensor,
        cache4: torch.Tensor, position4: torch.Tensor,
        cache5: torch.Tensor, position5: torch.Tensor,
    ):
        """Process one conditioning token through transformer.

        No input_linear (conditioning is already 1024d).
        No BOS handling. No EOS output.
        """
        x = conditioning  # [B, 1, 1024]

        caches = [cache0, cache1, cache2, cache3, cache4, cache5]
        positions = [position0, position1, position2, position3, position4, position5]
        new_caches = []
        new_positions = []

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

        return (
            new_caches[0], new_positions[0],
            new_caches[1], new_positions[1],
            new_caches[2], new_positions[2],
            new_caches[3], new_positions[3],
            new_caches[4], new_positions[4],
            new_caches[5], new_positions[5],
        )


if __name__ == "__main__":
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    cond_step = TraceableCondStep.from_flowlm(model.flow_lm, max_seq_len=200)
    cond_step.eval()
    print(f"Created conditioning step model")

    # Test
    cond = torch.randn(1, 1, 1024)
    cache = torch.full((2, 1, 200, 16, 64), float('nan'))
    pos = torch.zeros(1)
    with torch.no_grad():
        out = cond_step(cond, cache, pos, cache.clone(), pos.clone(),
                        cache.clone(), pos.clone(), cache.clone(), pos.clone(),
                        cache.clone(), pos.clone(), cache.clone(), pos.clone())
    print(f"Output: {len(out)} tensors")
    print(f"New position: {out[1].item()}")
