"""Pure CoreML TTS generation — zero PyTorch dependency.

Dependencies: numpy, sentencepiece, safetensors, coremltools, scipy
NO torch import anywhere.

Pipeline:
1. Text prep (string ops)
2. Tokenize (sentencepiece)
3. Embed text (numpy lookup)
4. Load voice (safetensors)
5. KV cache prefill (CoreML backbone)
6. Autoregressive generation (CoreML step + flow_decoder + mimi)
"""
import os
import re
import numpy as np
import sentencepiece as sp
import coremltools as ct
import scipy.io.wavfile as wavfile
from safetensors.numpy import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONST_DIR = os.path.join(SCRIPT_DIR, "constants")

# Backbone model output key names (mapped from step model outputs)
# Conditioning step model output key names
COND_CACHE_KEYS = [
    "new_cache_1_internal_tensor_assign_2",
    "new_cache_3_internal_tensor_assign_2",
    "new_cache_5_internal_tensor_assign_2",
    "new_cache_7_internal_tensor_assign_2",
    "new_cache_9_internal_tensor_assign_2",
    "new_cache_internal_tensor_assign_2",
]
COND_POS_KEYS = [
    "var_445", "var_864", "var_1283", "var_1702", "var_2121", "var_2365",
]

# Generation step model output key names
STEP_CACHE_KEYS = [
    "new_cache_1_internal_tensor_assign_2",
    "new_cache_3_internal_tensor_assign_2",
    "new_cache_5_internal_tensor_assign_2",
    "new_cache_7_internal_tensor_assign_2",
    "new_cache_9_internal_tensor_assign_2",
    "new_cache_internal_tensor_assign_2",
]
STEP_POS_KEYS = [
    "var_458", "var_877", "var_1296", "var_1715", "var_2134", "var_2553",
]


def prepare_text_prompt(text: str):
    """Normalize text for TTS (pure string ops, no PyTorch)."""
    text = text.strip()
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'  +', ' ', text)

    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]

    if text and text[-1] not in '.!?':
        text += '.'

    word_count = len(text.split())
    if word_count < 5:
        text = ' ' * 8 + text
        frames_after_eos = 3
    else:
        frames_after_eos = 1

    return text, frames_after_eos


def generate_v4(text: str, voice: str = "alba", output_path: str = "coreml_v4.wav", seed: int = 42):
    """Generate audio using pure CoreML — no PyTorch."""
    print(f"Text: '{text}'")
    print(f"Voice: {voice}")
    print(f"Seed: {seed}")

    # 1. Text preparation
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    print(f"Prepared: '{prepared_text}' (frames_after_eos={frames_after_eos})")

    # 2. Tokenize
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.load(os.path.join(CONST_DIR, "tokenizer.model"))
    token_ids = tokenizer.encode(prepared_text)
    print(f"Tokens: {len(token_ids)} ids")

    # 3. Embed text (numpy lookup)
    embed_table = np.load(os.path.join(CONST_DIR, "text_embed_table.npy"))
    text_emb = embed_table[token_ids]  # [T_text, 1024]
    text_emb = text_emb[np.newaxis, :, :]  # [1, T_text, 1024]
    print(f"Text embeddings: {text_emb.shape}")

    # 4. Load voice conditioning
    voice_path = os.path.join(CONST_DIR, f"{voice}.safetensors")
    voice_data = load_file(voice_path)
    voice_emb = voice_data['audio_prompt']  # [1, V, 1024]
    print(f"Voice embeddings: {voice_emb.shape}")

    # 5. Load constants
    bos_emb = np.load(os.path.join(CONST_DIR, "bos_emb.npy"))
    emb_mean = np.load(os.path.join(CONST_DIR, "emb_mean.npy"))
    emb_std = np.load(os.path.join(CONST_DIR, "emb_std.npy"))
    q_weight = np.load(os.path.join(CONST_DIR, "quantizer_weight.npy"))  # [512, 32, 1]
    mimi_state_npz = dict(np.load(os.path.join(CONST_DIR, "mimi_init_state.npz")))

    # 6. Load CoreML models
    print("\nLoading CoreML models...")
    coreml_cond = ct.models.MLModel(
        os.path.join(SCRIPT_DIR, 'cond_step.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_step = ct.models.MLModel(
        os.path.join(SCRIPT_DIR, 'flowlm_step.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_flow = ct.models.MLModel(
        os.path.join(SCRIPT_DIR, 'flow_decoder.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_mimi = ct.models.MLModel(
        os.path.join(SCRIPT_DIR, 'mimi_decoder.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    # 7. Combine conditioning: voice first, then text (matches original model order)
    combined = np.concatenate([voice_emb, text_emb], axis=1)  # [1, V+T, 1024]
    cond_len = combined.shape[1]
    print(f"Conditioning: {cond_len} tokens (voice={voice_emb.shape[1]}, text={text_emb.shape[1]})")

    # 8. Initialize empty KV caches
    caches = {}
    positions = {}
    for i in range(6):
        caches[f'cache{i}'] = np.zeros((2, 1, 512, 16, 64), dtype=np.float32)
        positions[f'position{i}'] = np.array([0.0], dtype=np.float32)

    # 9. Prefill: process each conditioning token one at a time (no padding needed)
    print(f"Prefilling KV cache ({cond_len} tokens)...")
    for tok_idx in range(cond_len):
        cond_token = combined[:, tok_idx:tok_idx + 1, :]  # [1, 1, 1024]
        cond_inputs = {
            'conditioning': cond_token.astype(np.float32),
            **caches,
            **positions,
        }
        cond_out = coreml_cond.predict(cond_inputs)

        for i in range(6):
            caches[f'cache{i}'] = cond_out[COND_CACHE_KEYS[i]]
            positions[f'position{i}'] = cond_out[COND_POS_KEYS[i]]

    start_pos = positions['position0'][0]
    print(f"KV cache filled to position: {start_pos}")

    # 10. Autoregressive generation loop (pure CoreML)
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    print(f"\nGenerating (max {max_gen_len} frames)...")

    np.random.seed(seed)

    audio_chunks = []
    eos_step = None
    sequence = np.full((1, 1, 32), float('nan'), dtype=np.float32)
    num_lsd_steps = 8
    dt = 1.0 / num_lsd_steps
    temp = 0.7

    # Initialize Mimi state
    coreml_mimi_state = {}
    for k, v in mimi_state_npz.items():
        coreml_mimi_state[k] = v.astype(np.float32)
    # Add offset scalars
    coreml_mimi_state.setdefault('attn0_offset', np.array([0.0], dtype=np.float32))
    coreml_mimi_state.setdefault('attn0_end_offset', np.array([0.0], dtype=np.float32))
    coreml_mimi_state.setdefault('attn1_offset', np.array([0.0], dtype=np.float32))
    coreml_mimi_state.setdefault('attn1_end_offset', np.array([0.0], dtype=np.float32))

    for step in range(max_gen_len):
        # Step model
        step_inputs = {
            'sequence': sequence,
            'bos_emb': bos_emb,
            **caches,
            **positions,
        }
        step_out = coreml_step.predict(step_inputs)

        transformer_out = step_out['input']  # [1, 1, 1024]
        eos_logit = step_out['var_2582']  # [1, 1, 1]

        # Update caches/positions
        for i in range(6):
            caches[f'cache{i}'] = step_out[STEP_CACHE_KEYS[i]]
            positions[f'position{i}'] = step_out[STEP_POS_KEYS[i]]

        # EOS check
        is_eos = eos_logit.flatten()[0] > -4.0
        if is_eos and eos_step is None:
            eos_step = step
            print(f"  EOS at step {step}")
        if eos_step is not None and step >= eos_step + frames_after_eos:
            break

        # Flow decode (LSD 8 steps)
        transformer_out_flat = transformer_out.reshape(1, 1024)
        latent = np.random.randn(1, 32).astype(np.float32) * (temp ** 0.5)

        for lsd_step in range(num_lsd_steps):
            s_np = np.array([[lsd_step * dt]], dtype=np.float32)
            t_np = np.array([[(lsd_step + 1) * dt]], dtype=np.float32)
            flow_out = coreml_flow.predict({
                'transformer_out': transformer_out_flat,
                'latent': latent,
                's': s_np,
                't': t_np,
            })
            velocity = list(flow_out.values())[0]
            latent = latent + velocity * dt

        # Denormalize + quantize
        latent_denorm = latent * emb_std.reshape(1, -1) + emb_mean.reshape(1, -1)
        # q_weight is [512, 32, 1] (1D conv kernel), squeeze to [512, 32]
        w = q_weight.squeeze(-1)  # [512, 32]
        quantized = np.dot(latent_denorm, w.T).reshape(1, 512, 1)

        # Mimi decode
        mimi_inputs = {'latent': quantized.astype(np.float32), **coreml_mimi_state}
        mimi_out = coreml_mimi.predict(mimi_inputs)

        audio_frame = mimi_out['var_1445']
        audio_chunks.append(audio_frame)

        # Update Mimi state
        coreml_mimi_state['upsample_partial'] = mimi_out['y_end_1']
        coreml_mimi_state['attn0_cache'] = mimi_out['new_cache_1_internal_tensor_assign_2']
        coreml_mimi_state['attn0_offset'] = mimi_out['var_402']
        coreml_mimi_state['attn0_end_offset'] = mimi_out['new_end_offset_1']
        coreml_mimi_state['attn1_cache'] = mimi_out['new_cache_internal_tensor_assign_2']
        coreml_mimi_state['attn1_offset'] = mimi_out['var_825']
        coreml_mimi_state['attn1_end_offset'] = mimi_out['new_end_offset']
        coreml_mimi_state['conv0_prev'] = mimi_out['var_998']
        coreml_mimi_state['conv0_first'] = mimi_out['var_1006']
        coreml_mimi_state['convtr0_partial'] = mimi_out['var_1048']
        coreml_mimi_state['res0_conv0_prev'] = mimi_out['var_1105']
        coreml_mimi_state['res0_conv0_first'] = mimi_out['var_1113']
        coreml_mimi_state['res0_conv1_prev'] = mimi_out['cast_13']
        coreml_mimi_state['res0_conv1_first'] = mimi_out['var_1134']
        coreml_mimi_state['convtr1_partial'] = mimi_out['var_1178']
        coreml_mimi_state['res1_conv0_prev'] = mimi_out['var_1235']
        coreml_mimi_state['res1_conv0_first'] = mimi_out['var_1243']
        coreml_mimi_state['res1_conv1_prev'] = mimi_out['cast_18']
        coreml_mimi_state['res1_conv1_first'] = mimi_out['var_1264']
        coreml_mimi_state['convtr2_partial'] = mimi_out['var_1308']
        coreml_mimi_state['res2_conv0_prev'] = mimi_out['var_1365']
        coreml_mimi_state['res2_conv0_first'] = mimi_out['var_1373']
        coreml_mimi_state['res2_conv1_prev'] = mimi_out['cast_23']
        coreml_mimi_state['res2_conv1_first'] = mimi_out['var_1394']
        coreml_mimi_state['conv_final_prev'] = mimi_out['var_1450']
        coreml_mimi_state['conv_final_first'] = mimi_out['var_1458']

        # Update sequence for next step
        sequence = latent.reshape(1, 1, 32)

        if step % 20 == 0:
            print(f"  Step {step}...")

    print(f"Generated {len(audio_chunks)} frames")

    # Concatenate and save
    audio = np.concatenate(audio_chunks, axis=-1)
    audio = audio[0, 0]
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9

    sample_rate = 24000
    wavfile.write(output_path, sample_rate, (audio * 32767).astype(np.int16))

    print(f"\nSaved to {output_path}")
    print(f"Duration: {len(audio) / sample_rate:.2f}s")
    return output_path


if __name__ == "__main__":
    generate_v4(
        "Hello, this is pure CoreML text to speech generation.",
        voice="alba",
        output_path="coreml_v4.wav",
    )
