import coremltools as ct
import numpy as np
import torch
import librosa

def test_single_encoder():
    print("Loading streaming_encoder.mlpackage...")
    model_path = "Models/ParakeetEOU/Streaming/streaming_encoder.mlpackage"
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    
    print("Model loaded. Inputs:")
    for inp in model.get_spec().description.input:
        print(f"  {inp.name}: {inp.type.multiArrayType.shape} ({inp.type.multiArrayType.dataType})")
        
    # Create dummy inputs
    # Mel: [1, 128, 128] (B, D, T)
    mel = np.random.randn(1, 128, 128).astype(np.float32)
    mel_length = np.array([128], dtype=np.int32)
    
    # Caches
    # Need to match shapes from export
    # cache_last_channel: [17, 1, 70, 512]
    cache_last_channel = np.zeros((17, 1, 70, 512), dtype=np.float32)
    # cache_last_time: [17, 1, 512, 8]
    cache_last_time = np.zeros((17, 1, 512, 8), dtype=np.float32)
    # cache_last_channel_len: [1]
    cache_last_channel_len = np.zeros((1,), dtype=np.int32)
    
    inputs = {
        "mel": mel,
        "mel_length": mel_length,
        "cache_last_channel": cache_last_channel,
        "cache_last_time": cache_last_time,
        "cache_last_channel_len": cache_last_channel_len
    }
    
    print("\nRunning prediction...")
    outputs = model.predict(inputs)
    
    print("\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
        
    print("\nSuccess! Single Streaming Encoder is working.")

if __name__ == "__main__":
    test_single_encoder()
