import coremltools as ct
import numpy as np
import torch
import json
from pathlib import Path

def test_streaming_models():
    model_dir = Path("Models/ParakeetEOU/Streaming")
    
    # Load metadata
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    print("Metadata loaded.")
    print(f"Cache channel size: {metadata['cache_channel_size']}")
    print(f"Cache time size: {metadata['cache_time_size']}")
    
    # Load models
    print("Loading models...")
    pre_encode = ct.models.MLModel(str(model_dir / "pre_encode.mlpackage"))
    conformer = ct.models.MLModel(str(model_dir / "conformer_streaming.mlpackage"))
    
    # Create dummy input
    # Mel chunk: [1, 128, 9] (chunk_size=8 + 1 overlap?)
    # Wait, chunk size in encoder frames is 8.
    # Mel frames needed = 8 * 4 + 9 (pre_cache) = 41?
    # Let's use a small chunk.
    
    mel_dim = 128
    chunk_len = 128
    mel = np.random.rand(1, mel_dim, chunk_len).astype(np.float32)
    mel_len = np.array([chunk_len], dtype=np.int32)
    
    print("Testing PreEncode...")
    pre_out = pre_encode.predict({
        "mel": mel,
        "mel_length": mel_len
    })
    encoded = pre_out["pre_encoded"]
    print(f"PreEncode output shape: {encoded.shape}")
    
    # Test Conformer
    print("Testing Conformer...")
    
    # Initialize caches
    num_layers = metadata["num_layers"]
    hidden_dim = metadata["hidden_dim"]
    cache_channel_size = metadata["cache_channel_size"]
    cache_time_size = metadata["cache_time_size"]
    
    cache_last_channel = np.zeros((num_layers, 1, cache_channel_size, hidden_dim), dtype=np.float32)
    cache_last_time = np.zeros((num_layers, 1, hidden_dim, cache_time_size), dtype=np.float32)
    cache_last_channel_len = np.zeros((1,), dtype=np.int32)
    
    # Conformer input
    # encoded is [1, T', hidden_dim]
    # We need to pass it as pre_encoded
    
    conf_out = conformer.predict({
        "pre_encoded": encoded,
        "pre_encoded_length": pre_out["pre_encoded_length"],
        "cache_last_channel": cache_last_channel,
        "cache_last_time": cache_last_time,
        "cache_last_channel_len": cache_last_channel_len
    })
    
    print(f"Conformer output shape: {conf_out['encoder'].shape}")
    print(f"New cache channel shape: {conf_out['new_cache_channel'].shape}")
    print(f"New cache time shape: {conf_out['new_cache_time'].shape}")
    
    print("Verification successful!")

if __name__ == "__main__":
    test_streaming_models()
