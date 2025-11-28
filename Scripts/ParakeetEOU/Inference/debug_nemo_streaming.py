import torch
import soundfile as sf
import numpy as np
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

print("DEBUG SCRIPT STARTED")

def debug_streaming_config(
    audio_path: str,
    model_id: str = "nvidia/parakeet_realtime_eou_120m-v1"
):
    print(f"\n{'='*60}")
    print(f"Debugging NeMo Streaming Configuration")
    print(f"{'='*60}")
    
    # Load model
    print("Loading NeMo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_id, map_location="cpu"
    )
    asr_model.eval()
    
    encoder = asr_model.encoder
    
    # Print current streaming config
    print("\n--- Current Streaming Config ---")
    if hasattr(encoder, 'streaming_cfg'):
        print(encoder.streaming_cfg)
    else:
        print("No streaming_cfg found on encoder!")
        
    # Experiment 1: Try to set a simpler streaming config
    # Based on FastConformer defaults or common settings
    print("\n--- Experiment 1: Setting Explicit Streaming Config ---")
    
    # Try 160ms chunk (16 frames)
    # Subsampling is 8x. 16 frames input -> 2 frames output?
    # Or is chunk_size in output frames?
    # NeMo docs say chunk_size is in "steps" (subsampled frames).
    
    # Let's try to set a config that matches what we think it should be
    # chunk_size=16 (160ms if 10ms stride?) No, 16 steps * 8 * 10ms = 1280ms?
    # Let's check the default again: chunk_size=[9, 16]
    
    # Let's try to run inference with the DEFAULT config first on a small chunk
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    
    # Take 1.28s chunk
    chunk_samples = int(1.28 * sr)
    chunk = audio[:chunk_samples]
    chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
    chunk_len = torch.tensor([len(chunk)], dtype=torch.int32)
    
    # Preprocess
    with torch.no_grad():
        mel, mel_len = asr_model.preprocessor(
            input_signal=chunk_tensor,
            length=chunk_len
        )
    
    print(f"\nMel shape: {mel.shape}")
    
    # Initialize cache
    cache_last_channel, cache_last_time, cache_len = encoder.get_initial_cache_state(1)
    
    # Run step
    try:
        with torch.no_grad():
            outputs = encoder.cache_aware_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_len,
            )
        enc_out = outputs[0]
        print(f"Default Config Output Shape: {enc_out.shape}")
        print(f"Default Config Output Mean: {enc_out.mean().item():.4f}")
        print(f"Default Config Output Std: {enc_out.std().item():.4f}")
        
        # Decode
        decoder = asr_model.decoder
        joint = asr_model.joint
        blank_id = int(decoder.blank_idx)
        vocab = asr_model.tokenizer.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        
        # Simple greedy decode of this frame
        # (Copying simplified logic)
        h = torch.zeros(int(decoder.pred_rnn_layers), 1, int(decoder.pred_hidden))
        c = torch.zeros(int(decoder.pred_rnn_layers), 1, int(decoder.pred_hidden))
        
        # Just check first frame
        enc_frame = enc_out[:, :, 0:1]
        targets = torch.tensor([[blank_id]], dtype=torch.int64)
        target_len = torch.tensor([1], dtype=torch.int64)
        with torch.no_grad():
            dec_out, _, _ = decoder(targets=targets, target_length=target_len, states=[h, c])
            joint_out = joint.joint(enc_frame.transpose(1, 2), dec_out[:, :, :1].transpose(1, 2))
            logits = joint_out.squeeze()
            token_id = logits.argmax().item()
            print(f"Predicted Token ID: {token_id} ({id_to_token.get(token_id, '???')})")
            
    except Exception as e:
        print(f"Default Config Failed: {e}")

    # Experiment 3: Multi-chunk streaming with EXACTLY 128 frames
    print("\n--- Experiment 3: Multi-chunk Streaming (128 frames / 1280ms) ---")
    
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    
    print(f"Audio loaded: {len(audio)} samples, SR: {sr}")
    
    # Chunk size: 1280ms = 1.28s
    # Samples: 1.28 * 16000 = 20480
    chunk_samples = 20480
    print(f"Chunk samples: {chunk_samples}")
    
    # Initialize cache
    cache_last_channel, cache_last_time, cache_len = encoder.get_initial_cache_state(1)
    
    # Initialize decoder state
    decoder = asr_model.decoder
    joint = asr_model.joint
    blank_id = int(decoder.blank_idx)
    vocab = asr_model.tokenizer.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    h = torch.zeros(int(decoder.pred_rnn_layers), 1, int(decoder.pred_hidden))
    c = torch.zeros(int(decoder.pred_rnn_layers), 1, int(decoder.pred_hidden))
    decoder_state = (h, c)
    
    # Try 160ms chunks
    # 160ms * 16000 = 2560 samples
    chunk_samples = 2560
    print(f"Testing with 160ms chunks ({chunk_samples} samples)")
    
    num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    all_tokens = []
    
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        
        # Pad if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
        chunk_len = torch.tensor([len(chunk)], dtype=torch.int32)
        
        # Preprocess
        with torch.no_grad():
            mel, mel_len = asr_model.preprocessor(
                input_signal=chunk_tensor,
                length=chunk_len
            )
        
        # Run encoder
        with torch.no_grad():
            outputs = encoder.cache_aware_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_len,
            )
        
        enc_out = outputs[0]
        cache_last_channel = outputs[2]
        cache_last_time = outputs[3]
        cache_len = outputs[4]
        
        # Decode (Greedy)
        chunk_tokens = []
        time_steps = enc_out.shape[2]
        
        for t in range(time_steps):
            enc_frame = enc_out[:, :, t:t+1]
            current_token = blank_id if not chunk_tokens else chunk_tokens[-1]
            
            # Max symbols per step
            for _ in range(5):
                targets = torch.tensor([[current_token]], dtype=torch.int64)
                target_len = torch.tensor([1], dtype=torch.int64)
                with torch.no_grad():
                    dec_out, _, (h, c) = decoder(targets=targets, target_length=target_len, states=[h, c])
                    joint_out = joint.joint(enc_frame.transpose(1, 2), dec_out[:, :, :1].transpose(1, 2))
                    logits = joint_out.squeeze()
                    token_id = logits.argmax().item()
                
                if token_id == blank_id:
                    break
                
                chunk_tokens.append(token_id)
                current_token = token_id
        
        all_tokens.extend(chunk_tokens)
        
        # Print chunk text
        chunk_text = ""
        for tid in chunk_tokens:
            tstr = id_to_token.get(tid, '')
            if tstr.startswith(' '): chunk_text += " " + tstr[1:]
            else: chunk_text += tstr
        print(f"Chunk {i+1}: '{chunk_text}' (Mel: {mel.shape}, Enc: {enc_out.shape})")

    # Final text
    final_text = ""
    for tid in all_tokens:
        tstr = id_to_token.get(tid, '')
        if tstr.startswith(' '): final_text += " " + tstr[1:]
        else: final_text += tstr
    print(f"\nFinal Text: '{final_text}'")
