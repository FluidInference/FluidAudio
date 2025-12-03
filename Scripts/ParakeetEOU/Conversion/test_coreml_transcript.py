import coremltools as ct
import numpy as np
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
import json
from pathlib import Path

def test_coreml_transcript():
    # Paths
    model_dir = Path("Models/ParakeetEOU/Streaming")
    audio_file = "mobius/models/stt/parakeet-tdt-v2-0.6b/coreml/audio/yc_first_minute_16k.wav"
    
    # Load Metadata
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
        
    blank_id = metadata["blank_id"]
    vocab_size = metadata["vocab_size"]
    
    # Load NeMo Tokenizer (for decoding IDs)
    print("Loading NeMo tokenizer...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu")
    tokenizer = asr_model.tokenizer
    
    # Load CoreML models
    print("Loading CoreML models...")
    # pre_encode_ml = ct.models.MLModel(str(model_dir / "pre_encode.mlpackage"))
    # conformer_ml = ct.models.MLModel(str(model_dir / "conformer_streaming.mlpackage"))
    streaming_encoder_ml = ct.models.MLModel(str(model_dir / "streaming_encoder.mlpackage"))
    decoder_ml = ct.models.MLModel(str(model_dir / "decoder.mlpackage"))
    joint_ml = ct.models.MLModel(str(model_dir / "joint_decision.mlpackage"))
    
    # Load Audio
    print(f"Loading {audio_file}...")
    audio, sample_rate = sf.read(audio_file)
    audio = audio.astype(np.float32)
    
    # Add dither to match NeMo training/inference conditions
    # NeMo default dither is often 1e-5
    # dither_amount = 1e-5
    # audio += np.random.normal(0, dither_amount, audio.shape).astype(np.float32)
    
    # Normalize if needed (NeMo usually normalizes)
    # But sf.read is already -1..1 usually.
    
    # Create Buffer
    # We need to feed audio to buffer.
    # But wait, streaming_buffer.append_audio_file reads file again!
    # We need to append TENSOR/ARRAY.
    # CacheAwareStreamingAudioBuffer.append_audio_signal(audio_signal)
    
    # If stereo, convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    # --- Step 1: Streaming Encoder ---
    print("Running Streaming Encoder...")
    
    # Setup Streaming Params on NeMo model to ensure buffer works correctly
    # We try to target ~128 Mel frames.
    # Previous default chunk_size=16 gave 25 Mel frames.
    # We try chunk_size=80.
    if asr_model.encoder.streaming_cfg is None:
        asr_model.encoder.setup_streaming_params(chunk_size=16, shift_size=15)
    else:
        asr_model.encoder.setup_streaming_params(chunk_size=16, shift_size=15)
        
    # Override Streaming Config
    # chunk_size=16 (output) -> 135 input frames (approx, produces 16 output frames)
    # shift_size=15 (output) -> 120 input frames (advancement)
    # valid_out_len=15 (output)
    asr_model.encoder.streaming_cfg.chunk_size = 135
    asr_model.encoder.streaming_cfg.shift_size = 120
    valid_out_len = 15
        
    # Initialize Buffer
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    
    
    # Pass numpy array [T] (or [C, T]?)
    # preprocess_audio expects numpy array and converts to tensor [1, T]
    _ = streaming_buffer.append_audio(audio, stream_id=-1)
    
    # Init Cache
    num_layers = metadata["num_layers"]
    hidden_dim = metadata["hidden_dim"]
    cache_channel_size = metadata["cache_channel_size"]
    cache_time_size = metadata["cache_time_size"]
    
    ml_cache_channel = np.zeros((num_layers, 1, cache_channel_size, hidden_dim), dtype=np.float32)
    ml_cache_time = np.zeros((num_layers, 1, hidden_dim, cache_time_size), dtype=np.float32)
    ml_cache_len = np.zeros((1,), dtype=np.int32)
    
    print(f"Streaming Config: {asr_model.encoder.streaming_cfg}")
    
    all_encoded = []
    chunk_frames = 135
    
    for i, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
        # chunk_audio is [B, D, T] (Mel)
        mel = chunk_audio
        current_len = mel.shape[2]
        # print(f"Chunk {i}: Mel shape: {mel.shape}")
        
        # Expect 135 frames
        if current_len != chunk_frames:
            if current_len < chunk_frames:
                mel = torch.nn.functional.pad(mel, (0, chunk_frames - current_len))
            elif current_len > chunk_frames:
                mel = mel[:, :, :chunk_frames]
                
        mel_np = mel.numpy().astype(np.float32)
        mel_len_np = np.array([chunk_frames], dtype=np.int32)
        
        enc_out = streaming_encoder_ml.predict({
            "mel": mel_np,
            "mel_length": mel_len_np,
            "cache_last_channel": ml_cache_channel,
            "cache_last_time": ml_cache_time,
            "cache_last_channel_len": ml_cache_len
        })
        
        ml_encoded = enc_out["encoded"] # [1, D, T]
        
        # Slice to valid_out_len
        if ml_encoded.shape[2] > valid_out_len:
            # print(f"DEBUG: Slicing encoded from {ml_encoded.shape[2]} to {valid_out_len}")
            ml_encoded = ml_encoded[:, :, :valid_out_len]
            
        print(f"Step {i}: encoded mean={np.mean(ml_encoded)}, std={np.std(ml_encoded)}")
        ml_cache_channel = enc_out["new_cache_last_channel"]
        ml_cache_time = enc_out["new_cache_last_time"]
        ml_cache_len = enc_out["new_cache_last_channel_len"]
        print(f"Step {i}: cache_len={ml_cache_len}")
        
        all_encoded.append(ml_encoded)
        
    full_encoded = np.concatenate(all_encoded, axis=2) # [1, D, Total_T]
    print(f"Full Encoded Shape: {full_encoded.shape}")
    
    # --- Step 2: RNNT Decoding ---
    print("Running RNNT Decoding...")
    
    # Init Decoder State
    decoder_layers = metadata["decoder_layers"]
    decoder_hidden = metadata["decoder_hidden"]
    
    h_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
    c_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
    
    # Start Token (Blank)
    last_token = blank_id
    
    predicted_ids = []
    
    # Find EOU ID
    eou_id = None
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "get_vocab"):
        vocab = tokenizer.tokenizer.get_vocab()
        for token, idx in vocab.items():
            if "<EOU>" in token.upper() or "eou" in token.lower():
                eou_id = idx
                print(f"Found EOU Token: {token} -> {eou_id}")
                break
    
    # Loop over time steps
    T = full_encoded.shape[2]
    
    for t in range(T):
        # Encoder step: [1, D, 1]
        enc_step = full_encoded[:, :, t:t+1]
        
        # Max symbols per step (prevent infinite loop)
        max_symbols = 10
        symbols_added = 0
        
        while symbols_added < max_symbols:
            # Predict Decoder Step
            # Input: targets=[[last_token]], target_length=[1], h_in, c_in
            dec_in = {
                "targets": np.array([[last_token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "h_in": h_state,
                "c_in": c_state
            }
            
            dec_out = decoder_ml.predict(dec_in)
            
            decoder_step = dec_out["decoder"] # [1, D, U]
            # print(f"DEBUG: decoder_step shape: {decoder_step.shape}")
            
            # Joint expects [1, D, 1]
            if len(decoder_step.shape) == 3 and decoder_step.shape[2] > 1:
                 # print(f"DEBUG: Slicing decoder_step to [:, :, -1:]")
                 decoder_step = decoder_step[:, :, -1:]
            
            # Predict Joint
            joint_in = {
                "encoder_step": enc_step,
                "decoder_step": decoder_step
            }
            
            joint_out = joint_ml.predict(joint_in)
            
            token_id = int(joint_out["token_id"].item())
            token_prob = float(joint_out["token_prob"].item())
            
            if token_id == blank_id:
                # Blank: move to next time step
                # print(f"    Step {t}: Blank ({token_prob:.4f})")
                break
            else:
                print(f"    Step {t} Inner {symbols_added}: Token {token_id} ({token_prob:.4f})")
                
                # Debug IDs
                # print(f"DEBUG: token_id={token_id}, eou_id={eou_id}, blank_id={blank_id}")
                
                is_eou = False
                if eou_id is not None and token_id == eou_id:
                    is_eou = True
                elif token_id == 1024: # Hardcoded check based on observation
                    is_eou = True
                    
                if is_eou:
                    print(f"    Step {t}: EOU DETECTED! Resetting Decoder State.")
                    # Reset Decoder State
                    h_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
                    c_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
                    last_token = blank_id
                    
                    # Do not append EOU token to transcript.
                    # Maybe append a space or newline?
                    # predicted_ids.append(token_id) 
                    
                    # Break inner loop to move to next time step
                    break
                
                # Non-blank: emit token, update state, stay at same time step
                predicted_ids.append(token_id)
                last_token = token_id
                
                # Update Decoder State
                h_state = dec_out["h_out"]
                c_state = dec_out["c_out"]
                
                symbols_added += 1
    
    # Decode IDs to Text
    text = tokenizer.ids_to_text(predicted_ids)
    print(f"\nCoreML Transcript: {text}")
    
    # Compare with Reference
    ref_path = Path("ReferenceOutputs/nemo_transcript.txt")
    if ref_path.exists():
        with open(ref_path) as f:
            ref_text = f.read().strip()
        print(f"Reference Transcript: {ref_text}")
        
        if text == ref_text:
            print("✓ MATCH!")
        else:
            print("✗ MISMATCH")

if __name__ == "__main__":
    test_coreml_transcript()
