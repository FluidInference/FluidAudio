import coremltools as ct
import numpy as np
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
import json
from pathlib import Path

def test_full_streaming():
    # Paths
    model_dir = Path("Models/ParakeetEOU/Streaming")
    audio_file = "she_sells_seashells_16k.wav"
    
    # Load NeMo model (Reference)
    print("Loading NeMo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu")
    asr_model.eval()
    preprocessor = asr_model.preprocessor
    encoder = asr_model.encoder
    
    # Load CoreML models
    print("Loading CoreML models...")
    pre_encode_ml = ct.models.MLModel(str(model_dir / "pre_encode.mlpackage"))
    conformer_ml = ct.models.MLModel(str(model_dir / "conformer_streaming.mlpackage"))
    
    # Load metadata
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    # Load Audio
    print(f"Loading {audio_file}...")
    if not Path(audio_file).exists():
        # Generate dummy audio if file doesn't exist
        print("Audio file not found, generating dummy sine wave...")
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    else:
        audio, sr = sf.read(audio_file)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    
    # Configuration
    # CoreML pre_encode expects 128 frames
    chunk_frames = 128
    # 128 frames * 10ms stride = 1.28s (approx, depends on window size/stride)
    # NeMo preprocessor: window_stride=0.01s (10ms)
    # So 128 frames comes from approx 1.28s of audio.
    # Samples needed = (frames - 1) * stride + window_size
    # But simpler: just feed enough samples to get >= 128 frames.
    # 16000 * 1.28 = 20480 samples.
    chunk_samples = 20480 
    
    print(f"Processing in chunks of {chunk_samples} samples ({chunk_frames} Mel frames)...")
    
    # Initialize NeMo Cache
    # We need to use the FULL encoder's cache init
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(batch_size=1)
    
    # Initialize CoreML Cache
    num_layers = metadata["num_layers"]
    hidden_dim = metadata["hidden_dim"]
    cache_channel_size = metadata["cache_channel_size"]
    cache_time_size = metadata["cache_time_size"]
    
    ml_cache_channel = np.zeros((num_layers, 1, cache_channel_size, hidden_dim), dtype=np.float32)
    ml_cache_time = np.zeros((num_layers, 1, hidden_dim, cache_time_size), dtype=np.float32)
    ml_cache_len = np.zeros((1,), dtype=np.int32)
    
    # Loop
    offset = 0
    chunk_idx = 0
    
    nemo_outputs = []
    ml_outputs = []
    
    while offset < len(audio):
        chunk_idx += 1
        end = offset + chunk_samples
        chunk = audio[offset:end]
        
        # Pad last chunk
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
        chunk_len = torch.tensor([len(chunk)], dtype=torch.int32)
        
        # 1. Preprocess (Mel Spec)
        # We use NeMo preprocessor for both to ensure identical input
        with torch.no_grad():
            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)
        
        # Ensure exactly 128 frames for CoreML
        # Mel shape: [1, 128, T]
        if mel.shape[2] < chunk_frames:
            # Pad
            mel = torch.nn.functional.pad(mel, (0, chunk_frames - mel.shape[2]))
        elif mel.shape[2] > chunk_frames:
            # Crop
            mel = mel[:, :, :chunk_frames]
        
        mel_np = mel.numpy().astype(np.float32)
        mel_len_np = np.array([chunk_frames], dtype=np.int32)
        
        print(f"Chunk {chunk_idx}: Mel shape {mel.shape}")
        
        # 2. NeMo Inference (Reference)
        # We call the FULL encoder cache_aware_stream_step
        # It handles pre_encode + conformer internally
        with torch.no_grad():
            nemo_out, _, cache_last_channel, cache_last_time, cache_last_channel_len = encoder.cache_aware_stream_step(
                processed_signal=mel,
                processed_signal_length=torch.tensor([chunk_frames]),
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len
            )
        nemo_outputs.append(nemo_out.numpy())
        
        # 3. CoreML Inference
        # Step A: Pre-Encode
        pre_out = pre_encode_ml.predict({
            "mel": mel_np,
            "mel_length": mel_len_np
        })
        pre_encoded = pre_out["pre_encoded"]
        pre_encoded_len = pre_out["pre_encoded_length"]
        
        # Step B: Conformer
        conf_out = conformer_ml.predict({
            "pre_encoded": pre_encoded,
            "pre_encoded_length": pre_encoded_len,
            "cache_last_channel": ml_cache_channel,
            "cache_last_time": ml_cache_time,
            "cache_last_channel_len": ml_cache_len
        })
        
        ml_encoded = conf_out["encoder"]
        ml_cache_channel = conf_out["new_cache_channel"]
        ml_cache_time = conf_out["new_cache_time"]
        ml_cache_len = conf_out["new_cache_len"]
        
        ml_outputs.append(ml_encoded)
        
        # Compare Chunk
        nemo_chunk = nemo_out.numpy()
        ml_chunk = ml_encoded
        
        # NeMo output is [1, D, T] (channel-first)
        # CoreML output is [1, D, T] (channel-first) ? 
        # Wait, ConformerStackWrapper returns [B, hidden, T_out] (channel-first)
        # Let's check shapes
        print(f"  NeMo: {nemo_chunk.shape}")
        print(f"  CoreML: {ml_chunk.shape}")
        
        diff = np.abs(nemo_chunk - ml_chunk)
        print(f"  Max Diff: {diff.max():.6f}")
        print(f"  Mean Diff: {diff.mean():.6f}")
        
        offset += chunk_samples

    print("\nDone.")

if __name__ == "__main__":
    test_full_streaming()
