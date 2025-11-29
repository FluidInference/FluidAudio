#!/usr/bin/env python3
"""Full PyTorch streaming inference with decoder/joint to compare with CoreML."""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path

import nemo.collections.asr as nemo_asr


def greedy_decode_streaming(
    encoder_output, 
    encoder_length,
    decoder_model,
    joint_model,
    decoder_state,
    blank_id,
    eos_id=None,
    max_symbols_per_step=10
):
    """Simplified greedy decoding for streaming (processes encoder output incrementally)."""
    
    batch_size = encoder_output.shape[0]
    time_steps = encoder_output.shape[2]
    
    tokens = []
    
    # Decoder hidden state (h, c) from previous chunk or zeros
    h, c = decoder_state
    
    for t in range(time_steps):
        # Get encoder frame [B, D, 1]
        enc_frame = encoder_output[:, :, t:t+1]
        
        # Start with blank or last token
        current_token = blank_id if not tokens else tokens[-1]
        
        symbols_this_frame = 0
        while symbols_this_frame < max_symbols_per_step:
            # Run decoder
            targets = torch.tensor([[current_token]], dtype=torch.int64)
            target_len = torch.tensor([1], dtype=torch.int64)
            
            with torch.no_grad():
                dec_out, _, (h, c) = decoder_model(
                    targets=targets,
                    target_length=target_len,
                    states=[h, c]
                )
            
            # Run joint [B, D_enc, 1] + [B, D_dec, 1] -> [B, vocab]
            with torch.no_grad():
                # Joint network expects specific input format
                # Use positional arguments as keyword arguments might vary
                joint_out = joint_model.joint(
                    enc_frame.transpose(1, 2),  # [B, 1, D_enc]
                    dec_out[:, :, :1].transpose(1, 2),  # [B, 1, D_dec]
                )
            
            # Get prediction [B, 1, 1, vocab] -> [vocab]
            logits = joint_out.squeeze(0).squeeze(0).squeeze(0)  # [vocab]
            token_id = logits.argmax().item()
            
            if token_id == blank_id:
                break  # Move to next frame
            
            tokens.append(token_id)
            current_token = token_id
            symbols_this_frame += 1
            
            if eos_id is not None and token_id == eos_id:
                break
    
    return tokens, (h, c)


def test_full_streaming_inference(
    audio_path: str,
    chunk_ms: int = 320,
    model_id: str = "nvidia/parakeet_realtime_eou_120m-v1"
):
    """Run complete streaming inference including decoder and joint."""
    
    print(f"\n{'='*60}")
    print(f"Full PyTorch Streaming Inference (with Decoder/Joint)")
    print(f"{'='*60}")
    print(f"Audio: {audio_path}")
    print(f"Chunk size: {chunk_ms}ms\n")
    
    # Load model
    print("Loading NeMo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_id, map_location="cpu"
    )
    asr_model.eval()
    
    # Get components
    encoder = asr_model.encoder
    decoder = asr_model.decoder
    joint = asr_model.joint
    
    # Enable RNNT export mode
    decoder._rnnt_export = True
    
    # Get config
    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    chunk_samples = int(chunk_ms / 1000 * sample_rate)
    blank_id = int(decoder.blank_idx)
    vocab = asr_model.tokenizer.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Check for EOU token
    eou_id = vocab.get('<EOU>', None)
    
    print(f"Vocab size: {len(vocab)}, Blank ID: {blank_id}, EOU ID: {eou_id}")

    # Dump vocab to file for comparison
    import json
    with open("vocab_pytorch.json", "w") as f:
        # vocab is token -> id
        # we want id -> token
        id_to_token_str = {str(v): k for k, v in vocab.items()}
        json.dump(id_to_token_str, f, indent=2)
    print("Dumped vocabulary to vocab_pytorch.json")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    if sr != sample_rate:
        raise ValueError(f"Audio sample rate {sr} != model rate {sample_rate}")
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    print(f"Audio: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
    
    # Initialize cache state
    cache_last_channel, cache_last_time, cache_len = encoder.get_initial_cache_state(1)
    
    # Initialize decoder state
    decoder_hidden_size = int(decoder.pred_hidden)
    decoder_layers = int(decoder.pred_rnn_layers)
    h = torch.zeros(decoder_layers, 1, decoder_hidden_size)
    c = torch.zeros(decoder_layers, 1, decoder_hidden_size)
    decoder_state = (h, c)
    
    # Audio buffer for continuous preprocessing
    buffer_size_seconds = 4.0
    buffer_samples = int(buffer_size_seconds * sample_rate)
    audio_buffer = np.zeros(buffer_samples, dtype=np.float32)
    
    # Process chunks
    num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    print(f"Processing {num_chunks} chunks with buffering...\n")
    
    all_tokens = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, len(audio))
        chunk = audio[start_idx:end_idx]
        
        # Pad last chunk
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
        # Update buffer
        audio_buffer = np.roll(audio_buffer, -len(chunk))
        audio_buffer[-len(chunk):] = chunk
        
        # Use entire buffer for preprocessing
        buffer_tensor = torch.from_numpy(audio_buffer).unsqueeze(0).float()
        buffer_len = torch.tensor([len(audio_buffer)], dtype=torch.int32)
        
        # Preprocessor
        with torch.no_grad():
            mel, mel_len = asr_model.preprocessor(
                input_signal=buffer_tensor,
                length=buffer_len
            )
            
        # Extract features for the NEW chunk
        # Stride is usually 10ms (0.01s)
        stride_ms = 10
        # Calculate expected frames for this chunk
        # For 1280ms -> 128 frames. CoreML expects 129.
        # For 2500ms -> 250 frames.
        
        # We should extract frames corresponding to the chunk duration
        # Plus maybe 1 frame for overlap/safety?
        # NeMo uses: int(chunk_len / stride)
        
        extract_frames = int(chunk_ms / stride_ms)
        if chunk_ms == 1280:
             extract_frames = 129 # Special case for our CoreML model
        
        total_frames = mel.shape[2]
        
        if total_frames >= extract_frames:
            mel_chunk = mel[:, :, -extract_frames:]
            mel_chunk_len = torch.tensor([extract_frames], dtype=torch.int32)
        else:
            mel_chunk = mel
            mel_chunk_len = torch.tensor([total_frames], dtype=torch.int32)
            
        # Streaming encoder
        with torch.no_grad():
            outputs = encoder.cache_aware_stream_step(
                processed_signal=mel_chunk,
                processed_signal_length=mel_chunk_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_len,
            )
        
        enc_out = outputs[0]  # [B, hidden, T]
        enc_len = outputs[1]
        cache_last_channel = outputs[2]
        cache_last_time = outputs[3]
        cache_len = outputs[4]
        
        # Decode this chunk
        chunk_tokens, decoder_state = greedy_decode_streaming(
            enc_out, enc_len.item(),
            decoder, joint, decoder_state,
            blank_id, eou_id
        )
        
        all_tokens.extend(chunk_tokens)
        
        # Convert tokens to text
        chunk_text = ""
        for token_id in chunk_tokens:
            token_str = id_to_token.get(token_id, f"<{token_id}>")
            if token_str.startswith('▁'):
                chunk_text += " " + token_str[1:]
            else:
                chunk_text += token_str
        
        print(f"Chunk {i+1}/{num_chunks}: "
              f"enc_frames={enc_len.item()}, "
              f"tokens={len(chunk_tokens)}, "
              f"text=\"{chunk_text.strip()}\"")
    
    # Final text
    final_text = ""
    for token_id in all_tokens:
        token_str = id_to_token.get(token_id, f"<{token_id}>")
        if token_str.startswith('▁'):
            final_text += " " + token_str[1:]
        else:
            final_text += token_str
    
    print(f"\n{'='*60}")
    print(f"Final Result:")
    print(f"{'='*60}")
    print(f"Text: \"{final_text.strip()}\"")
    print(f"Tokens: {len(all_tokens)}")
    print(f"Token IDs: {all_tokens[:20]}{'...' if len(all_tokens) > 20 else ''}")
    
    return all_tokens, final_text.strip()


if __name__ == "__main__":
    audio_path = "jfk.wav"
    
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        exit(1)
    
    # Test with 1280ms chunks (matching Swift implementation)
    print("Testing with 1280ms chunks (matching Swift implementation):")
    tokens, text = test_full_streaming_inference(audio_path, chunk_ms=1280)
    
    # Test with full audio (batch simulation)
    print("\n\nTesting with full audio (2500ms):")
    tokens3, text3 = test_full_streaming_inference(audio_path, chunk_ms=1280)

    # Test with 160ms chunks (NVIDIA recommendation)
    print("\n\nTesting with 160ms chunks (NVIDIA recommendation):")
    tokens4, text4 = test_full_streaming_inference(audio_path, chunk_ms=160)

    # Test with 720ms chunks (Possible config value)
    print("\n\nTesting with 720ms chunks (Possible config value):")
    tokens5, text5 = test_full_streaming_inference(audio_path, chunk_ms=720)
