import torch
import nemo.collections.asr as nemo_asr
import numpy as np
from pathlib import Path
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

def extract_transcriptions(hyps):
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions

def generate_reference():
    audio_file = "mobius/models/stt/parakeet-tdt-v2-0.6b/coreml/audio/yc_first_minute_16k.wav"
    output_dir = Path("ReferenceOutputs")
    output_dir.mkdir(exist_ok=True)
    
    # Load Model
    print("Loading NeMo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu")
    asr_model.eval()
    
    # Disable dither and padding for deterministic results
    if hasattr(asr_model.preprocessor, "featurizer"):
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
    
    # Setup Streaming Params
    # chunk_size=16 (default)
    # shift_size=15 -> 1 frame overlap/drop
    asr_model.encoder.setup_streaming_params(chunk_size=16, shift_size=15)
        
    asr_model.encoder.streaming_cfg.chunk_size = 135
    asr_model.encoder.streaming_cfg.shift_size = 128
        
    # Disable timestamps to avoid NeMo bug with dict merging
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    decoding_cfg = RNNTDecodingConfig(fused_batch_size=-1, compute_timestamps=False)
    if hasattr(asr_model, 'change_decoding_strategy'):
        asr_model.change_decoding_strategy(decoding_cfg)
        
    print(f"Streaming Config: {asr_model.encoder.streaming_cfg}")

    # Initialize Buffer
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=False, # Default for this model?
        pad_and_drop_preencoded=False,
    )
    
    # Append Audio
    print(f"Processing {audio_file}...")
    _ = streaming_buffer.append_audio_file(audio_file, stream_id=-1)
    
    # Run Streaming Inference
    print("Running streaming inference...")
    
    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(batch_size=1)
    previous_hypotheses = None
    pred_out_stream = None
    all_streaming_tran = []
    all_encoded = []
    
    streaming_buffer_iter = iter(streaming_buffer)
    
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        with torch.no_grad():
            # chunk_audio is already features (B, D, T)
            
            # Manual stream step to get encoded output
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = asr_model.encoder.cache_aware_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                drop_extra_pre_encoded=calc_drop_extra_pre_encoded(asr_model, step_num, False),
                bypass_pre_encode=False,
            )
            
            # Run Decoder/Joint
            best_hyp = asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
                partial_hypotheses=previous_hypotheses,
            )
            
            # Extract text
            # best_hyp is list of Hypothesis.
            # We only have batch=1.
            hyp = best_hyp[0]
            y_sequence = hyp.y_sequence
            
            # Check for EOU in y_sequence
            # EOU ID is likely 1024 or 1025.
            # We can check if any token is EOU.
            # But rnnt_decoder_predictions_tensor returns the WHOLE sequence for the chunk?
            # Or just the new tokens?
            # It returns the sequence of tokens emitted in this step.
            
            # If EOU is in the sequence, we should reset.
            # But wait, rnnt_decoder_predictions_tensor updates state internally?
            # No, we pass `partial_hypotheses` (which contains state?).
            # Actually, `rnnt_decoder_predictions_tensor` returns `best_hyp`.
            # `best_hyp` has `score`, `y_sequence`, `dec_state`, `last_token`?
            # NeMo's Hypothesis class: y_sequence, score, length, dec_state, last_token (maybe?)
            
            # If we want to reset, we need to clear `previous_hypotheses`.
            # `previous_hypotheses` is passed to next step.
            
            # Let's check if EOU (1024 or 1025) is in y_sequence.
            # We need to know EOU ID.
            # We can find it from tokenizer.
            # But let's assume 1024/1025.
            
            eou_detected = False
            clean_y_sequence = []
            for token in y_sequence:
                if token == 1024 or token == 1025:
                    eou_detected = True
                    print(f"Step {step_num}: EOU DETECTED (Token {token})! Resetting State.")
                    break # Stop processing this sequence
                clean_y_sequence.append(token)
            
            # Append text
            if clean_y_sequence:
                text = asr_model.tokenizer.ids_to_text(clean_y_sequence)
                final_text += text
                print(f"Step {step_num}: {text}")
            
            if eou_detected:
                # Reset State!
                # previous_hypotheses = None
                # Or re-initialize it?
                # rnnt_decoder_predictions_tensor handles None by initializing default state.
                previous_hypotheses = None
            else:
                previous_hypotheses = best_hyp
            
            # Log stats
            import numpy as np
            enc_np = encoded.cpu().numpy()
            print(f"Step {step_num}: encoded mean={np.mean(enc_np)}, std={np.std(enc_np)}")
            # Decode text
            # Already done above

            if pred_out_stream is not None:
                # pred_out_stream might be a list or tensor. 
                # If it's a list (from previous steps?), we might just want the current chunk?
                # Actually, conformer_stream_step returns the *accumulated* output if keep_all_outputs is True?
                # No, it returns the current step output usually.
                # Let's just save it.
                pass 

    # final_text is now accumulated inside the loop
    print(f"Transcript: {final_text}")
    
    with open(output_dir / "nemo_transcript.txt", "w") as f:
        f.write(final_text)
    print(f"Saved {output_dir / 'nemo_transcript.txt'}")

    # Standard Transcription for comparison
    print("\nRunning standard transcription...")
    standard_text = asr_model.transcribe([audio_file])[0]
    print(f"Standard Transcript: {standard_text}")

if __name__ == "__main__":
    generate_reference()
