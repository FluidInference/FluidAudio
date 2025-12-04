import coremltools as ct
import numpy as np
import torch
import soundfile as sf
import json
from pathlib import Path
import sentencepiece as spm

def test_pure_coreml():
    # Paths
    model_dir = Path("Models/ParakeetEOU/Streaming")
    audio_file = "/Users/kikow/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean/2961/960/2961-960-0005.flac"
    
    # Load Metadata
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
        
    # tokenizer_meta = metadata["tokenizer"]
    blank_id = metadata["blank_id"]
    vocab_size = metadata["vocab_size"]
    
    # Load SentencePiece Tokenizer
    print("Loading SentencePiece tokenizer...")
    sp_model_path = model_dir / "tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    
    # Find EOU ID
    eou_id = 1024 # Hardcoded or find from vocab
    # Verify EOU
    # eou_token = sp.id_to_piece(eou_id)
    # print(f"EOU Token: {eou_token}")
    
    # Load CoreML models
    print("Loading CoreML models...")
    # Note: Export script names it parakeet_eou_preprocessor.mlpackage
    preprocessor_ml = ct.models.MLModel(str(model_dir / "parakeet_eou_preprocessor.mlpackage"))
    streaming_encoder_ml = ct.models.MLModel(str(model_dir / "streaming_encoder.mlpackage"))
    decoder_ml = ct.models.MLModel(str(model_dir / "decoder.mlpackage"))
    joint_ml = ct.models.MLModel(str(model_dir / "joint_decision.mlpackage"))
    
    # Load Audio
    print(f"Loading {audio_file}...")
    audio, sample_rate = sf.read(audio_file)
    audio = audio.astype(np.float32)
    
    # If stereo, convert to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
        
    # --- Streaming Parameters ---
    # Encoder Input: 135 frames
    # Encoder Shift: 120 frames
    # Preprocessor Hop Length: 160 (default for Parakeet/Conformer usually)
    # We can check asr_model.preprocessor.featurizer.hop_length
    hop_length = 160
    # if hasattr(asr_model, "preprocessor") and hasattr(asr_model.preprocessor, "featurizer"):
    #     hop_length = asr_model.preprocessor.featurizer.hop_length
    print(f"Hop Length: {hop_length}")
    
    chunk_frames = 135
    shift_frames = 120
    valid_out_len = 15
    
    chunk_samples = chunk_frames * hop_length
    shift_samples = shift_frames * hop_length
    
    print(f"Chunk Samples: {chunk_samples} ({chunk_frames} frames)")
    print(f"Shift Samples: {shift_samples} ({shift_frames} frames)")
    
    # Init Cache
    # Hardcoded for Parakeet 120m (Wait, error said 17 layers?)
    # Maybe it includes some other layers or the model is deeper?
    num_layers = 17 
    hidden_dim = 512 
    cache_channel_size = 70 
    cache_time_size = 8 
    
    # if hasattr(asr_model, "encoder"):
    #    pass # Skip dynamic lookup to avoid config errors
    
    ml_cache_channel = np.zeros((num_layers, 1, cache_channel_size, hidden_dim), dtype=np.float32)
    ml_cache_time = np.zeros((num_layers, 1, hidden_dim, cache_time_size), dtype=np.float32)
    ml_cache_len = np.zeros((1,), dtype=np.int32)
    
    # Init Decoder State
    # metadata["decoder_layers"] is now at top level
    decoder_layers = metadata.get("decoder_layers", 1)
    decoder_hidden = metadata.get("decoder_hidden", 640)
    
    h_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
    c_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
    last_token = blank_id
    predicted_ids = []
    
    # Find EOU ID
    eou_id = 1024
    # if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "get_vocab"):
    #     vocab = tokenizer.tokenizer.get_vocab()
    #     for token, idx in vocab.items():
    #         if "<EOU>" in token.upper() or "eou" in token.lower():
    #             eou_id = idx
    #             print(f"Found EOU Token: {token} -> {eou_id}")
    #             break
    
    # Streaming Loop
    total_samples = len(audio)
    current_sample = 0
    step = 0
    
    while current_sample < total_samples:
        # Extract Chunk
        end_sample = current_sample + chunk_samples
        chunk_audio = audio[current_sample:end_sample]
        
        # Pad if needed (last chunk)
        if len(chunk_audio) < chunk_samples:
            pad_len = chunk_samples - len(chunk_audio)
            chunk_audio = np.pad(chunk_audio, (0, pad_len), mode='constant')
            
        # Add Dither (Optional but recommended for stability based on previous tests)
        dither_amount = 1e-5
        chunk_audio += np.random.normal(0, dither_amount, chunk_audio.shape).astype(np.float32)
        
        # Preprocess (CoreML)
        # Input: audio_signal [1, samples], audio_length [1]
        audio_signal = chunk_audio.reshape(1, -1)
        audio_length = np.array([len(chunk_audio)], dtype=np.int32)
        
        prep_out = preprocessor_ml.predict({
            "audio_signal": audio_signal,
            "audio_length": audio_length
        })
        
        mel = prep_out["mel"] # [1, D, T]
        mel_length = prep_out["mel_length"]
        
        # Slice Mel to expected chunk size if needed
        if mel.shape[2] > chunk_frames:
            mel = mel[:, :, :chunk_frames]
            mel_length = np.array([chunk_frames], dtype=np.int32)
        elif mel.shape[2] < chunk_frames:
            # Pad if too short?
            pad_len = chunk_frames - mel.shape[2]
            mel = np.pad(mel, ((0,0), (0,0), (0, pad_len)), mode='constant')
            mel_length = np.array([chunk_frames], dtype=np.int32)
        
        # Run Encoder
        enc_out = streaming_encoder_ml.predict({
            "mel": mel,
            "mel_length": mel_length, # Or fixed 135?
            "cache_last_channel": ml_cache_channel,
            "cache_last_time": ml_cache_time,
            "cache_last_channel_len": ml_cache_len
        })
        
        ml_encoded = enc_out["encoded"]
        
        # Slice Output
        if ml_encoded.shape[2] > valid_out_len:
            ml_encoded = ml_encoded[:, :, :valid_out_len]
            
        # Update Cache
        ml_cache_channel = enc_out["new_cache_last_channel"]
        ml_cache_time = enc_out["new_cache_last_time"]
        ml_cache_len = enc_out["new_cache_last_channel_len"]
        
        # Decode (RNNT)
        T_enc = ml_encoded.shape[2]
        for t in range(T_enc):
            enc_step = ml_encoded[:, :, t:t+1]
            
            max_symbols = 10
            symbols_added = 0
            
            while symbols_added < max_symbols:
                dec_in = {
                    "targets": np.array([[last_token]], dtype=np.int32),
                    "target_length": np.array([1], dtype=np.int32),
                    "h_in": h_state,
                    "c_in": c_state
                }
                dec_out = decoder_ml.predict(dec_in)
                decoder_step = dec_out["decoder"]
                if len(decoder_step.shape) == 3 and decoder_step.shape[2] > 1:
                     decoder_step = decoder_step[:, :, -1:]
                
                joint_in = {
                    "encoder_step": enc_step,
                    "decoder_step": decoder_step
                }
                joint_out = joint_ml.predict(joint_in)
                
                token_id = int(joint_out["token_id"].item())
                
                if token_id == blank_id:
                    break
                else:
                    is_eou = False
                    if eou_id is not None and token_id == eou_id:
                        is_eou = True
                    elif token_id == 1024:
                        is_eou = True
                        
                    if is_eou:
                        print(f"Step {step}: EOU DETECTED! Resetting Decoder State.")
                        h_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
                        c_state = np.zeros((decoder_layers, 1, decoder_hidden), dtype=np.float32)
                        last_token = blank_id
                        break
                    
                    predicted_ids.append(token_id)
                    last_token = token_id
                    h_state = dec_out["h_out"]
                    c_state = dec_out["c_out"]
                    symbols_added += 1
        
        # Advance Window
        current_sample += shift_samples
        step += 1
        
    # Final Transcript
    # Final Transcript
    text = sp.decode(predicted_ids)
    print(f"\nPure CoreML Transcript: {text}")

if __name__ == "__main__":
    test_pure_coreml()
