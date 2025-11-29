import torch
import numpy as np
import os

# Path to voice file
# Usually in ~/.cache/fluidaudio/Models/kokoro/voices/zf_xiaobei.pt
# Or wherever KPipeline loads it from.
# KPipeline downloads to HF cache usually, or local.

# Let's try to find it or load it via KPipeline
from kokoro import KPipeline

try:
    pipeline = KPipeline(lang_code='z')
    voice = pipeline.load_voice("zf_xiaobei")
    # voice is a tensor or numpy array
    print(f"Voice type: {type(voice)}")
    if isinstance(voice, torch.Tensor):
        voice = voice.numpy()
    
    print(f"Shape: {voice.shape}")
    print(f"Mean: {np.mean(voice):.6f}")
    print(f"Std: {np.std(voice):.6f}")
    # Average across the first dimension (510)
    # voice shape: (510, 1, 256)
    # We want (256,)
    if len(voice.shape) == 3:
        mean_voice = np.mean(voice, axis=0).flatten()
    else:
        mean_voice = voice.flatten()

    l2_norm = np.linalg.norm(mean_voice)
    print(f"Averaged Voice Shape: {mean_voice.shape}")
    print(f"Averaged Voice L2 Norm: {l2_norm:.6f}")
    
    # Also check the first vector
    first_voice = voice[0].flatten()
    l2_first = np.linalg.norm(first_voice)
    print(f"First Voice L2 Norm: {l2_first:.6f}")
    
    # Save to file for Swift to load?
    # np.save("zf_xiaobei_mean.npy", mean_voice)
    
except Exception as e:
    print(f"Error: {e}")
