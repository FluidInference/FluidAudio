
import torch
import nemo.collections.asr as nemo_asr
import numpy as np
import librosa
import jiwer
import typer
from pathlib import Path

def main(
    audio_file: str = typer.Option(..., help="Path to audio file"),
    reference_text: str = typer.Option(..., help="Reference transcript"),
    model_id: str = typer.Option("nvidia/parakeet_realtime_eou_120m-v1", help="Model ID"),
    chunk_ms: int = typer.Option(160, help="Chunk size in milliseconds"),
):
    print(f"Loading model: {model_id}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()
    
    # Setup streaming config
    encoder = asr_model.encoder
    if encoder.streaming_cfg is None:
        encoder.setup_streaming_params()
    
    # Load audio
    print(f"Loading audio: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Calculate chunk size in samples
    chunk_samples = int(chunk_ms / 1000 * 16000)
    print(f"Chunk size: {chunk_ms}ms ({chunk_samples} samples)")
    
    # Initialize cache
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(batch_size=1)
    
    # Preprocessor (we'll run it on the full audio for simplicity, or chunk by chunk?)
    # To be strictly streaming, we should run preprocessor on chunks.
    # But NeMo's preprocessor usually expects a batch.
    # Let's try feeding raw audio chunks to the preprocessor if possible, 
    # or just slice the pre-computed mel features to simulate streaming input.
    # Slicing pre-computed mel is safer to test the ENCODER specifically.
    
    print("Pre-computing Mel Spectrogram (simulating perfect preprocessing)...")
    # Run preprocessor on full audio
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    audio_len = torch.tensor([audio.shape[0]], dtype=torch.long)
    
    processed_signal, processed_signal_len = asr_model.preprocessor(
        input_signal=audio_tensor, length=audio_len
    )
    # processed_signal shape: [1, 80, T]
    
    total_frames = processed_signal.shape[2]
    print(f"Total Mel Frames: {total_frames}")
    
    # Calculate frames per chunk
    # 160ms chunk = 1600 samples. 
    # Mel window stride is usually 10ms. So ~16 frames.
    # Let's calculate based on ratio
    frames_per_chunk = int(chunk_ms / 10) # Approx
    
    print(f"Frames per chunk: {frames_per_chunk}")
    
    print("\n--- Starting Streaming ---")
    for i in range(0, total_frames, frames_per_chunk):
        end = min(i + frames_per_chunk, total_frames)
        chunk_mel = processed_signal[:, :, i:end]
        chunk_len = torch.tensor([chunk_mel.shape[2]], dtype=torch.int32)
        
        if chunk_mel.shape[2] == 0:
            break
            
        with torch.no_grad():
            # Run Encoder
            out = encoder.cache_aware_stream_step(
                processed_signal=chunk_mel,
                processed_signal_length=chunk_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len
            )
            
            encoded_signal = out[0]
            encoded_len = out[1]
            cache_last_channel = out[2]
            cache_last_time = out[3]
            cache_last_channel_len = out[4]
            
            # Check statistics
            mean = encoded_signal.mean().item()
            std = encoded_signal.std().item()
            max_val = encoded_signal.max().item()
            
            print(f"Chunk {i//frames_per_chunk}: Input Frames={chunk_len.item()}, Encoded Len={encoded_len.item()}, Mean={mean:.4f}, Std={std:.4f}, Max={max_val:.4f}")
            
            if encoded_len.item() > 0:
                 # If we get valid encoded frames, that's a good sign.
                 pass

if __name__ == "__main__":
    typer.run(main)
