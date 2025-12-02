import argparse
import torch
import torchaudio
import coremltools as ct
import numpy as np
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from pathlib import Path
import jiwer
import time

def load_manifest(dataset_path, subset='test-clean', max_files=None):
    subset_dir = Path(dataset_path) / subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {subset_dir}")
    
    flac_files = list(subset_dir.rglob('*.flac'))
    if not flac_files:
        raise FileNotFoundError(f"No FLAC files found in {subset_dir}")
    
    # Sort for determinism
    flac_files = sorted(flac_files)
    
    entries = []
    for flac_path in flac_files:
        if max_files and len(entries) >= max_files:
            break
            
        speaker_id = flac_path.parent.parent.name
        chapter_id = flac_path.parent.name
        trans_file = flac_path.parent / f"{speaker_id}-{chapter_id}.trans.txt"
        
        if trans_file.exists():
            utterance_id = flac_path.stem
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2 and parts[0] == utterance_id:
                        entries.append({
                            'audio_filepath': str(flac_path),
                            'text': parts[1],
                            'duration': 0
                        })
                        break
    print(f"Loaded {len(entries)} entries from {subset_dir}")
    return entries

def run_coreml_pipeline(coreml_encoder, coreml_decoder, coreml_joint, pytorch_model, audio_path):
    # 1. Load Audio
    try:
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        audio_tensor = audio
        audio_len = torch.tensor([audio.shape[1]], dtype=torch.long)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return {'hypothesis': "", 'audio_length': 0}

    # 2. Setup Streaming Params & Buffer
    # Use chunk_size=4 to match PyTorch success (approx 320ms compute, 410ms input)
    pytorch_model.encoder.setup_streaming_params(chunk_size=4, shift_size=4)
    
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=pytorch_model,
        online_normalization=False,
        pad_and_drop_preencoded=False
    )
    streaming_buffer.append_audio_file(audio_path, stream_id=-1)
    
    # 3. CoreML True Streaming Loop
    # Initialize CoreML Cache (Encoder)
    num_layers = 17
    cache_last_channel = np.zeros((num_layers, 1, 70, 512), dtype=np.float32)
    cache_last_time = np.zeros((num_layers, 1, 512, 8), dtype=np.float32)
    cache_last_channel_len = np.zeros((1,), dtype=np.int32)
    
    # Initialize Decoder State
    h_state = np.zeros((1, 1, 640), dtype=np.float32)
    c_state = np.zeros((1, 1, 640), dtype=np.float32)
    
    blank_token = 1026 # Parakeet blank
    last_token = blank_token
    
    hypothesis_tokens = []
    max_symbols_per_step = 10
    
    fixed_chunk_frames = 41 # Matches export for chunk_size=4
    
    for chunk_audio, chunk_len in streaming_buffer:
        # --- Encoder Step ---
        # chunk_audio: [1, 128, T]
        T_curr = chunk_audio.shape[2]
        
        if T_curr < fixed_chunk_frames:
            pad_amt = fixed_chunk_frames - T_curr
            padding = torch.full((1, 128, pad_amt), -16.0)
            chunk_audio = torch.cat([chunk_audio, padding], dim=2)
        elif T_curr > fixed_chunk_frames:
             chunk_audio = chunk_audio[:, :, :fixed_chunk_frames]
             
        chunk_mel_input = chunk_audio.numpy()
        mel_len_input = np.array([fixed_chunk_frames], dtype=np.int32)

        inputs = {
            "mel": chunk_mel_input,
            "mel_length": mel_len_input,
            "cache_last_channel": cache_last_channel,
            "cache_last_time": cache_last_time,
            "cache_last_channel_len": cache_last_channel_len
        }
        
        outputs = coreml_encoder.predict(inputs)
        
        cache_last_channel = outputs["cache_last_channel_out"]
        cache_last_time = outputs["cache_last_time_out"]
        cache_last_channel_len = outputs["cache_last_channel_len_out"]
        
        enc_out = outputs["encoder"] # [1, 512, 4]
        
        # --- Decoder Step (Immediate) ---
        T_enc = enc_out.shape[2]
        
        for t in range(T_enc):
            enc_t = enc_out[:, :, t:t+1] # [1, 512, 1]
            
            # Initialize Decoder Output (Cache)
            targets = np.array([[last_token]], dtype=np.int32)
            target_length = np.array([1], dtype=np.int32)
            
            dec_inputs = {
                "targets": targets,
                "target_length": target_length,
                "h_in": h_state,
                "c_in": c_state
            }
            
            dec_outputs = coreml_decoder.predict(dec_inputs)
            decoder_step = dec_outputs["decoder_output"]
            h_state_next = dec_outputs["h_out"]
            c_state_next = dec_outputs["c_out"]
            
            symbols_added = 0
            while symbols_added < max_symbols_per_step:
                joint_inputs = {
                    "encoder_output": enc_t,
                    "decoder_output": decoder_step
                }
                
                joint_outputs = coreml_joint.predict(joint_inputs)
                
                logits = joint_outputs["logits"]
                token_id = int(np.argmax(logits))
                
                if token_id == blank_token:
                    break
                
                # EOU Check (1024)
                if token_id == 1024:
                    # Reset State
                    h_state = np.zeros((1, 1, 640), dtype=np.float32)
                    c_state = np.zeros((1, 1, 640), dtype=np.float32)
                    last_token = blank_token
                    break
                
                else:
                    hypothesis_tokens.append(token_id)
                    last_token = token_id
                    symbols_added += 1
                    
                    h_state = h_state_next
                    c_state = c_state_next
                    
                    targets = np.array([[last_token]], dtype=np.int32)
                    dec_inputs = {
                        "targets": targets,
                        "target_length": target_length,
                        "h_in": h_state,
                        "c_in": c_state
                    }
                    dec_outputs = coreml_decoder.predict(dec_inputs)
                    decoder_step = dec_outputs["decoder_output"]
                    h_state_next = dec_outputs["h_out"]
                    c_state_next = dec_outputs["c_out"]
    
    # Decode tokens
    vocab_size = pytorch_model.tokenizer.vocab_size
    valid_tokens = [t for t in hypothesis_tokens if t < vocab_size]
    
    if len(valid_tokens) != len(hypothesis_tokens):
        print(f"Filtered {len(hypothesis_tokens) - len(valid_tokens)} invalid tokens (>= {vocab_size})")
        
    if not valid_tokens:
        return {
            'hypothesis': "",
            'audio_length': audio.shape[1] / 16000
        }
        
    hypothesis = pytorch_model.decoding.decode_tokens_to_str([valid_tokens])[0]
    hypothesis = hypothesis.replace("<EOU>", "").strip()
    
    return {
        'hypothesis': hypothesis,
        'audio_length': audio.shape[1] / 16000
    }

    return {
        'hypothesis': hypothesis,
        'audio_length': audio.shape[1] / 16000
    }

def run_pytorch_streaming_pipeline(pytorch_model, audio_path):
    # 1. Load Audio
    try:
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        audio_tensor = audio
        audio_len = torch.tensor([audio.shape[1]], dtype=torch.long)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return {'hypothesis': "", 'audio_length': 0}

    # 2. Preprocessor
    with torch.no_grad():
        processed_signal, processed_signal_len = pytorch_model.preprocessor(
            input_signal=audio_tensor, length=audio_len
        )
    
    # 3. Streaming Loop
    total_frames = processed_signal.shape[2]
    chunk_frames = 32 # Match CoreML
    
    # Initialize Cache
    num_layers = 17
    cache_last_channel = torch.zeros(num_layers, 1, 70, 512)
    cache_last_time = torch.zeros(num_layers, 1, 512, 8)
    cache_last_channel_len = torch.zeros(1, dtype=torch.long)
    
    # Initialize Decoder State
    decoder_state = None
    last_token = torch.tensor([[1026]], dtype=torch.long) # Blank token
    
    final_hyp_tokens = []
    
    for i in range(0, total_frames, chunk_frames):
        end = min(i + chunk_frames, total_frames)
        chunk_mel = processed_signal[:, :, i:end] # [1, D, T]
        
        # Pad to chunk_frames if needed
        if chunk_mel.shape[2] < chunk_frames:
            pad_amt = chunk_frames - chunk_mel.shape[2]
            chunk_mel = torch.nn.functional.pad(chunk_mel, (0, pad_amt))
            
        chunk_len = torch.tensor([chunk_mel.shape[2]], dtype=torch.long)
        
        with torch.no_grad():
            # 1. Encoder Step
            (
                enc_out,
                enc_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len
            ) = pytorch_model.encoder.forward_internal(
                audio_signal=chunk_mel,
                length=chunk_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len
            )
            
            # enc_out: [B, D, T_out] -> [1, 512, T_out]
            # Transpose to [B, T_out, D] for Joint
            enc_out = enc_out.transpose(1, 2)
            
            # 2. Greedy Decoding Loop (Symbol Loop)
            # For each acoustic frame t
            for t in range(enc_out.shape[1]):
                f_t = enc_out[:, t:t+1, :] # [1, 1, 512]
                
                # Project Encoder (Joint.enc)
                # pytorch_model.joint.enc is the Linear layer
                # Or use pytorch_model.joint(enc_out, dec_out) which does projection internally?
                # Standard RNNTJoint: forward(f, g) -> res -> joint_net
                # But we need to loop over symbols u.
                
                # Pre-project encoder for this frame
                f_t_proj = pytorch_model.joint.enc(f_t) # [1, 1, 640]
                
                # Limit max symbols per frame (e.g. 10) to prevent infinite loops
                max_symbols = 10
                symbols_added = 0
                
                while symbols_added < max_symbols:
                    # Decoder Step
                    # decoder.forward(targets, lengths, states)
                    # targets: [B, 1] (last token)
                    g, _, decoder_state = pytorch_model.decoder.forward(
                        targets=last_token,
                        target_length=torch.tensor([1]),
                        states=decoder_state
                    )
                    
                    # g: [B, 640, U+1?] -> [1, 640, 2]
                    # We want the last step output
                    g = g[:, :, -1:] # [1, 640, 1]
                    g = g.transpose(1, 2) # [1, 1, 640]
                    
                    # Project Decoder (Joint.pred)
                    g_proj = pytorch_model.joint.pred(g) # [1, 1, 640]
                    
                    # Joint
                    # joint_net(f + g)
                    # Note: f_t_proj and g_proj are [1, 1, 640]
                    # We broadcast? They are same shape here.
                    out = pytorch_model.joint.joint_net(f_t_proj + g_proj) # [1, 1, 1027]
                    
                    # Argmax
                    k = out.argmax(dim=-1) # [1, 1]
                    pred_token = k.item()
                    
                    if pred_token == 1026: # Blank
                        break
                    else:
                        final_hyp_tokens.append(pred_token)
                        last_token = k # Update last token
                        # decoder_state is already updated by forward()
                        # But wait! If we predict a symbol, we advance decoder state.
                        # If we predict blank, we DO NOT advance decoder state?
                        # In standard RNNT:
                        # If blank: advance t (next acoustic frame), keep u (decoder state).
                        # If symbol: advance u (update decoder state), keep t (same acoustic frame).
                        
                        # My decoder.forward call UPDATED the state.
                        # If I predict blank, I should DISCARD the new state?
                        # YES!
                        # But wait, decoder.forward takes the *previous* token/state and produces the *current* embedding/state.
                        # The state returned is the state AFTER processing `last_token`.
                        # This state is what we use to predict the NEXT token.
                        # So if we predict a symbol, we KEEP this state and use it for the next step.
                        # If we predict blank, we KEEP the *previous* state (before this forward)?
                        # No, the state corresponds to the *current* position `u`.
                        # The `g` vector corresponds to `h_u`.
                        # `f_t` corresponds to `h_t`.
                        # `Joint(h_t, h_u)` produces prob of `y_{u+1}` or `blank`.
                        
                        # If `blank`: we move to `t+1`. We stay at `u`. State `h_u` is unchanged.
                        # If `symbol`: we move to `u+1`. We update `h_u` to `h_{u+1}`. We stay at `t`.
                        
                        # So:
                        # 1. We have `decoder_state` (corresponding to `u`).
                        # 2. We compute `g` from `last_token` and `decoder_state`.
                        #    Wait, `decoder.forward` usually takes `last_token` and `previous_state` and returns `current_embedding` and `new_state`.
                        #    So `g` is `P(u)`. `decoder_state` is `State(u)`.
                        #    Actually, for LSTM, `forward` does one step.
                        
                        # Let's verify:
                        # `g, _, new_state = decoder(last_token, state)`
                        # `out = joint(f, g)`
                        # If `out` -> Symbol:
                        #    We accept `new_state` as the current state.
                        #    We update `last_token` to Symbol.
                        #    We loop again (same `t`).
                        # If `out` -> Blank:
                        #    We discard `new_state`.
                        #    We keep `state` (old).
                        #    We break loop (next `t`).
                        
                        # BUT, `decoder.forward` is expensive. We don't want to re-compute it if we stay at `u`.
                        # But we only stay at `u` if we predict Blank, which means we move to next `t`.
                        # For the next `t`, we need `g` (which depends on `u`).
                        # So we should cache `g` and `state`?
                        # Yes.
                        
                        # Correct Logic:
                        # Initialize `decoder_state = None`.
                        # Initialize `last_token = Blank`.
                        # Compute `g, _, next_decoder_state = decoder(last_token, decoder_state)` ONCE.
                        # `g_proj = joint.pred(g)`
                        
                        # Loop t:
                        #   `f_t_proj = ...`
                        #   Loop u:
                        #     `logits = joint(f_t_proj + g_proj)`
                        #     `k = argmax`
                        #     If k == Blank:
                        #        break (advance t)
                        #     Else:
                        #        Append k.
                        #        `last_token = k`
                        #        `decoder_state = next_decoder_state` (Accept the state transition)
                        #        # Compute NEXT g and state
                        #        `g, _, next_decoder_state = decoder(last_token, decoder_state)`
                        #        `g_proj = joint.pred(g)`
                        
                        # This looks correct.
                        # But I need to initialize `g` and `next_decoder_state` before the loop.
                        
                        pass
    
    # Refined Logic Implementation
    
    # Initialize Decoder
    # First step: Feed Blank/SOS to get initial g and state
    # Note: Parakeet uses Blank (1026) as SOS? Or does it rely on zero state?
    # Usually we feed SOS. Let's assume 1026 is SOS.
    
    last_token = torch.tensor([[1026]], dtype=torch.long)
    decoder_state = None
    
    # Pre-compute initial g
    g, _, next_decoder_state = pytorch_model.decoder.forward(
        targets=last_token,
        target_length=torch.tensor([1]),
        states=decoder_state
    )
    # g: [1, 640, 2] -> Slice last
    g = g[:, :, -1:] # [1, 640, 1]
    g = g.transpose(1, 2) # [1, 1, 640]
    
    g_proj = pytorch_model.joint.pred(g)
    
    # Update decoder_state to next_decoder_state?
    # No, `next_decoder_state` is the state AFTER processing `last_token`.
    # This is the state we need for the NEXT step if we emit a token.
    # Wait, `g` is the embedding used for prediction.
    # So `g` and `next_decoder_state` go together.
    # We hold `g` and `next_decoder_state` as "Current Decoder Output".
    # If we emit a symbol, we use `next_decoder_state` as the input for the NEXT decoder step.
    
    # Let's call the holding variables `current_g_proj` and `candidate_state`.
    current_g_proj = g_proj
    candidate_state = next_decoder_state
    
    # Current state input to decoder (for next step)
    # Actually, `decoder.forward` takes `states`.
    # If we emit a symbol, the `states` for the NEXT call should be `candidate_state`.
    # So we need to track `current_state_for_input`.
    # Initially `None`.
    # After first call, `candidate_state` is the state after SOS.
    
    # Wait, if we emit a symbol, we call decoder with that symbol and `candidate_state`.
    # So `candidate_state` IS the state we maintain.
    
    # Let's verify:
    # 1. Start: `last_token=SOS`, `state=None`.
    # 2. `g, _, state = decoder(SOS, None)`.
    # 3. `g` is used to predict first token.
    # 4. `joint(f, g)`.
    # 5. If `k` (symbol):
    #    `last_token = k`.
    #    `g, _, state = decoder(k, state)`.
    #    Loop.
    # 6. If `Blank`:
    #    Keep `g` and `state` as is.
    #    Advance `f`.
    
    # Yes, this is correct.
    
    # So,
def run_pytorch_streaming_pipeline(pytorch_model, audio_path):
    import librosa
    # Load audio
    audio, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Preprocessing
    processed_signal, processed_signal_length = pytorch_model.preprocessor(
        input_signal=torch.tensor([audio]),
        length=torch.tensor([len(audio)])
    )
    
    # Switch to Greedy Strategy (if not already)
    if pytorch_model.decoding.cfg.strategy != 'greedy':
        print("Switching to 'greedy' decoding strategy for streaming...")
        from omegaconf import OmegaConf
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
        
        new_cfg = OmegaConf.create({
            'strategy': 'greedy',
            'greedy': {'max_symbols': 10},
            'preserve_alignments': True,
            'compute_timestamps': False
        })
        
        pytorch_model.decoding = RNNTBPEDecoding(
            decoding_cfg=new_cfg,
            decoder=pytorch_model.decoder,
            joint=pytorch_model.joint,
            tokenizer=pytorch_model.tokenizer
        )
    
    # Streaming Loop
    total_frames = processed_signal.shape[2]
    chunk_frames = 32 
    
    # Initialize Cache
    num_layers = 17
    cache_last_channel = torch.zeros(num_layers, 1, 70, 512)
    cache_last_time = torch.zeros(num_layers, 1, 512, 8)
    cache_last_channel_len = torch.zeros(1, dtype=torch.long)
    
    previous_hypotheses = None
    previous_pred_out = None
    
    final_hyp = ""
    
    for i in range(0, total_frames, chunk_frames):
        end = min(i + chunk_frames, total_frames)
        chunk_mel = processed_signal[:, :, i:end] # [1, D, T]
        
        # Pad to chunk_frames if needed
        if chunk_mel.shape[2] < chunk_frames:
            pad_amt = chunk_frames - chunk_mel.shape[2]
            chunk_mel = torch.nn.functional.pad(chunk_mel, (0, pad_amt))
            
        chunk_len = torch.tensor([chunk_mel.shape[2]], dtype=torch.long)
        
        with torch.no_grad():
            # Native Streaming Step
            (
                greedy_predictions,
                all_hyp_text,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                best_hyp_list, # This is the Hypothesis list
            ) = pytorch_model.conformer_stream_step(
                processed_signal=chunk_mel,
                processed_signal_length=chunk_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=previous_pred_out
            )
            
            # Update previous_hypotheses for next step
            previous_hypotheses = best_hyp_list
            
            # Extract text from best_hyp
            current_hyp_obj = None
            if best_hyp_list:
                if isinstance(best_hyp_list, list):
                    current_hyp_obj = best_hyp_list[0]
                else:
                    current_hyp_obj = best_hyp_list

            # Check for EOU (1024)
            is_eou = False
            if current_hyp_obj:
                if hasattr(current_hyp_obj, 'y_sequence'):
                    # y_sequence might be list or tensor
                    y_seq = current_hyp_obj.y_sequence
                    if isinstance(y_seq, list):
                        if 1024 in y_seq:
                            is_eou = True
                    elif torch.is_tensor(y_seq):
                        if (y_seq == 1024).any():
                            is_eou = True
            
            if is_eou:
                # EOU detected
                # Append current segment text to final_hyp
                if current_hyp_obj and hasattr(current_hyp_obj, 'text'):
                    final_hyp += current_hyp_obj.text + " "
                
                # Reset state for next segment
                previous_hypotheses = None
                print("DEBUG: EOU detected, resetting previous_hypotheses")
            else:
                # Not EOU, just update final_hyp with current segment text (temporarily)
                # We can't just append, we need to store it.
                # But since we return final_hyp at the end, we need to combine committed text + current segment.
                pass

    # End of loop
    # Append any remaining text from the last segment
    if previous_hypotheses:
        last_hyp_list = previous_hypotheses
        if isinstance(last_hyp_list, list):
            last_hyp_obj = last_hyp_list[0]
        else:
            last_hyp_obj = last_hyp_list
            
        if last_hyp_obj and hasattr(last_hyp_obj, 'text'):
            final_hyp += last_hyp_obj.text

    # Strip <eou>
    final_hyp = final_hyp.replace("<eou>", "").replace("<EOU>", "").strip()

    return {
        'hypothesis': final_hyp,
        'audio_length': audio.shape[0] / 16000
    }

def run_pytorch_pipeline(pytorch_model, audio_path):
    # Pure PyTorch inference using transcribe() (Offline)
    try:
        # Try positional argument for paths2audio_files
        hypotheses = pytorch_model.transcribe([audio_path], batch_size=1, verbose=False)
        
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        
        hypothesis = hypotheses[0]
        
        # Hypothesis object has 'text' attribute?
        if hasattr(hypothesis, 'text'):
            hypothesis = hypothesis.text
        
        # Strip <eou> and <EOU>
        if isinstance(hypothesis, str):
            hypothesis = hypothesis.replace("<eou>", "").replace("<EOU>", "").strip()
        
        return {
            'hypothesis': hypothesis,
            'audio_length': 0 # Placeholder
        }
        
    except Exception as e:
        print(f"Error running PyTorch pipeline on {audio_path}: {e}")
        return {'hypothesis': "", 'audio_length': 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/Users/kikow/Library/Caches/fluidaudio/LibriSpeech/LibriSpeech', help='Path to LibriSpeech')
    parser.add_argument('--subset', default='test-clean', help='Subset to test')
    parser.add_argument('--max-files', type=int, default=100, help='Number of files to process')
    
    # Default paths based on file list
    parser.add_argument('--coreml-encoder', default='streaming_encoder_320ms.mlpackage')
    parser.add_argument('--coreml-decoder', default='parakeet_decoder.mlpackage')
    parser.add_argument('--coreml-joint', default='parakeet_joint.mlpackage')
    
    parser.add_argument('--pytorch-model', default='nvidia/parakeet_realtime_eou_120m-v1')
    parser.add_argument('--coreml-preprocessor', default='preprocessor_160ms.mlpackage')
    parser.add_argument('--hybrid', action='store_true', help='Use Hybrid mode (CoreML Encoder + PyTorch Decoder)')
    parser.add_argument('--pytorch-only', action='store_true', help='Use pure PyTorch model (Offline)')
    parser.add_argument('--pytorch-streaming', action='store_true', help='Use pure PyTorch model (Streaming Simulation)')
    args = parser.parse_args()
    
    print(f"Loading PyTorch model: {args.pytorch_model}")
    pytorch_model = nemo_asr.models.ASRModel.from_pretrained(args.pytorch_model, map_location="cpu")
    pytorch_model.eval()
    
    # Only load CoreML if not pytorch-only or pytorch-streaming
    coreml_encoder = None
    coreml_decoder = None
    coreml_joint = None
    coreml_preprocessor = None
    
    if not args.pytorch_only and not args.pytorch_streaming:
        print(f"Loading CoreML Encoder: {args.coreml_encoder}")
        coreml_encoder = ct.models.MLModel(args.coreml_encoder)
        
        if args.hybrid:
            print(f"Loading CoreML Preprocessor: {args.coreml_preprocessor}")
            try:
                coreml_preprocessor = ct.models.MLModel(args.coreml_preprocessor)
            except Exception as e:
                print(f"Failed to load CoreML Preprocessor: {e}")
                print("Falling back to PyTorch Preprocessor")
        
        if not args.hybrid:
            print(f"Loading CoreML Decoder: {args.coreml_decoder}")
            coreml_decoder = ct.models.MLModel(args.coreml_decoder)
            
            print(f"Loading CoreML Joint: {args.coreml_joint}")
            coreml_joint = ct.models.MLModel(args.coreml_joint)
    elif args.pytorch_streaming:
        print("Running in PYTORCH-STREAMING mode")
    else:
        print("Running in PYTORCH-ONLY (Offline) mode")
    
    entries = load_manifest(args.dataset, args.subset, args.max_files)
    
    total_wer = 0
    count = 0
    start_time = time.time()
    
    print(f"Starting Benchmark on {len(entries)} files...")
    
    for i, entry in enumerate(entries):
        try:
            if args.pytorch_streaming:
                result = run_pytorch_streaming_pipeline(pytorch_model, entry['audio_filepath'])
            elif args.pytorch_only:
                result = run_pytorch_pipeline(pytorch_model, entry['audio_filepath'])
            elif args.hybrid:
                result = run_hybrid_pipeline(coreml_encoder, pytorch_model, entry['audio_filepath'], coreml_preprocessor)
            else:
                result = run_coreml_pipeline(coreml_encoder, coreml_decoder, coreml_joint, pytorch_model, entry['audio_filepath'])
            
            ref = entry['text'].lower()
            hyp = result['hypothesis'].lower()
            
            wer = jiwer.wer(ref, hyp)
            total_wer += wer
            count += 1
            
            print(f"[{i+1}/{len(entries)}] {Path(entry['audio_filepath']).name} | WER: {wer:.2%} | Ref: '{ref}' | Hyp: '{hyp}'")
        except Exception as e:
            print(f"[{i+1}/{len(entries)}] Failed: {e}")
            import traceback
            traceback.print_exc()
            
    if count > 0:
        avg_wer = total_wer / count
        print(f"\nAverage WER over {count} files: {avg_wer:.2%}")
    else:
        print("\nNo files processed successfully.")

def run_hybrid_pipeline(coreml_encoder, pytorch_model, audio_path, coreml_preprocessor=None):
    # 1. Load Audio
    try:
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        audio_tensor = audio
        audio_len = torch.tensor([audio.shape[1]], dtype=torch.long)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return {'hypothesis': "", 'audio_length': 0}

    # 2. Preprocessor
    if coreml_preprocessor:
        # CoreML Preprocessor
        # Input: input_signal (1, N)
        # Output: mel (1, 128, T)
        audio_np = audio.numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.reshape(1, -1) # Ensure (1, N)
            
        inputs = {
            "input_signal": audio_np,
            "length": np.array([audio_np.shape[1]], dtype=np.float32)
        }
        out = coreml_preprocessor.predict(inputs)
        processed_signal = torch.from_numpy(out["mel"]) # (1, 128, T)
        # CoreML might return (1, 1, 128, T) or similar?
        if processed_signal.ndim == 4:
             processed_signal = processed_signal.squeeze(0)
             
        # Check shape
        # PyTorch expects (1, 128, T)
    else:
        # PyTorch Preprocessor
        with torch.no_grad():
            processed_signal, processed_signal_len = pytorch_model.preprocessor(
                input_signal=audio_tensor, length=audio_len
            )
    
    # 3. CoreML Encoder Loop
    total_frames = processed_signal.shape[2]
    
    # Initialize CoreML Cache
    num_layers = 17
    cache_last_channel = np.zeros((num_layers, 1, 70, 512), dtype=np.float32)
    cache_last_time = np.zeros((num_layers, 1, 512, 8), dtype=np.float32)
    cache_last_channel_len = np.zeros((1,), dtype=np.int32)
    
    accumulated_encoder_output = []
    
    fixed_chunk_size = 32 
    chunk_frames = 32
    
    for i in range(0, total_frames, chunk_frames):
        end = min(i + chunk_frames, total_frames)
        chunk_mel = processed_signal[:, :, i:end].numpy() # [1, 128, T]
        
        current_chunk_len = chunk_mel.shape[2]
        
        # Pad if needed
        if current_chunk_len < fixed_chunk_size:
            pad_amt = fixed_chunk_size - current_chunk_len
            padding = np.full((1, 128, pad_amt), -16.0, dtype=np.float32)
            chunk_mel_input = np.concatenate([chunk_mel, padding], axis=2)
            mel_len_input = np.array([fixed_chunk_size], dtype=np.int32)
        else:
            chunk_mel_input = chunk_mel
            mel_len_input = np.array([fixed_chunk_size], dtype=np.int32)

        inputs = {
            "mel": chunk_mel_input,
            "mel_length": mel_len_input,
            "cache_last_channel": cache_last_channel,
            "cache_last_time": cache_last_time,
            "cache_last_channel_len": cache_last_channel_len
        }
        
        outputs = coreml_encoder.predict(inputs)
        
        cache_last_channel = outputs["cache_last_channel_out"]
        cache_last_time = outputs["cache_last_time_out"]
        cache_last_channel_len = outputs["cache_last_channel_len_out"]
        
        enc_out = outputs["encoder"]
        # enc_len = outputs["encoder_length"] # Always 3?
        
        accumulated_encoder_output.append(enc_out)

    if not accumulated_encoder_output:
        return {'hypothesis': "", 'audio_length': audio.shape[1] / 16000}

    # Concatenate Encoder Outputs: [1, 512, T]
    encoder_output = np.concatenate(accumulated_encoder_output, axis=2)
    
    # 4. PyTorch Decoding
    # Convert to Tensor
    encoder_output_tensor = torch.from_numpy(encoder_output) # [1, 512, T]
    # Transpose to [B, D, T] (It is already)
    
    # We need to pass valid length.
    # Estimate from original audio length?
    # Or just use full T.
    # Parakeet subsampling is 4x? 8x?
    # 320ms (32 frames) -> 3 frames.
    # 32 / 3 = 10.6?
    # Actually, let's trust the decoder to handle padding or just pass full length.
    encoded_lengths = torch.tensor([encoder_output.shape[2]], dtype=torch.long)
    
    with torch.no_grad():
        # Use greedy decoding
        # rnnt_decoder_predictions_tensor expects (B, D, T)
        hypotheses = pytorch_model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoder_output_tensor,
            encoded_lengths=encoded_lengths,
            return_hypotheses=True
        )
        
    hypothesis = hypotheses[0].text
    
    return {
        'hypothesis': hypothesis,
        'audio_length': audio.shape[1] / 16000
    }

if __name__ == "__main__":
    main()
