import torch
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
from pathlib import Path

def export_nemo_encoder_output():
    audio_file = "mobius/models/stt/parakeet-tdt-v2-0.6b/coreml/audio/yc_first_minute_16k.wav"
    output_dir = Path("ReferenceOutputs/EncoderDebug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    print("Loading NeMo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu")
    asr_model.eval()
    
    print(f"Subsampling Factor: {asr_model.encoder.subsampling_factor}")
    
    # Disable dither
    if hasattr(asr_model.preprocessor, "featurizer"):
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        
    # Set streaming params to match Swift
    # chunk_size=16
    # shift_size=15
    asr_model.encoder.setup_streaming_params(chunk_size=16, shift_size=15)
    
    # Force set directly to be sure
    asr_model.encoder.streaming_cfg.chunk_size = 16
    asr_model.encoder.streaming_cfg.shift_size = 15
    
    print(f"NeMo Streaming Config: {asr_model.encoder.streaming_cfg}")
        
    # Swift Configuration
    chunk_frames = 128
    shift_frames = 120
    hop_length = 160 # 10ms
    chunk_samples = chunk_frames * hop_length # 21600
    shift_samples = shift_frames * hop_length # 19200
    
    print(f"Chunk Samples: {chunk_samples}")
    print(f"Shift Samples: {shift_samples}")
    
    # Load Audio
    print(f"Loading audio {audio_file}...")
    audio, sr = sf.read(audio_file, dtype="float32")
    if sr != 16000:
        print(f"Warning: Sample rate is {sr}, expected 16000")
        
    # Initialize Cache
    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(batch_size=1)
    
    # Process Chunks
    buffer = audio.tolist()
    step = 0
    
    while len(buffer) >= chunk_samples:
        chunk = buffer[:chunk_samples]
        
        # Convert to tensor [1, T]
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
        chunk_len = torch.tensor([len(chunk)], dtype=torch.int32)
        
        with torch.no_grad():
            # 1. Preprocess (Audio -> Mel)
            # We need to manually preprocess because cache_aware_stream_step expects processed signal (Mel)
            # if we want to be explicit, OR we can let it handle it.
            # In single_streaming_encoder_wrapper.py, we saw it calls cache_aware_stream_step with bypass_pre_encode=False.
            # This means it expects AUDIO if the encoder includes pre_encode?
            # Wait, cache_aware_stream_step in ConformerEncoder usually expects Mel features.
            # But SingleStreamingEncoderWrapper passes `audio_signal` (Mel?) to it?
            
            # Let's check SingleStreamingEncoderWrapper again.
            # It takes `mel` input.
            # So cache_aware_stream_step expects Mel.
            
            # So we must run preprocessor first.
            processed_signal, processed_signal_len = asr_model.preprocessor(
                input_signal=chunk_tensor, length=chunk_len
            )
            
            # processed_signal is [B, D, T_mel]
            # Verify shape
            if step == 0:
                print(f"Step 0 Mel Shape: {processed_signal.shape}")
                # Should be [1, 128, 137] or similar
            
            # Slice to chunk_frames (136)
            processed_signal = processed_signal[:, :, :chunk_frames]
            processed_signal_len = torch.clamp(processed_signal_len, max=chunk_frames)
            
            if step == 0:
                print(f"Step 0 Sliced Mel Shape: {processed_signal.shape}")
            
            # 2. Run Encoder
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = asr_model.encoder.cache_aware_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                bypass_pre_encode=False # Perform pre-encode (subsampling)
            )
            
            # Save Output
            output_path = output_dir / f"nemo_encoder_step_{step}.npy"
            np.save(output_path, encoded.cpu().numpy())
            
            # Log
            print(f"Step {step}: Saved {output_path} (Shape: {encoded.shape})")
            
            # Shift Buffer
            buffer = buffer[shift_samples:]
            step += 1
            
            if step >= 5: # Only save first 5 steps
                break

if __name__ == "__main__":
    export_nemo_encoder_output()
