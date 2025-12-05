import torch
import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np
import jiwer
import glob
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json
from open_asr_normalizer import EnglishTextNormalizer

def benchmark_nemo_pure():
    # 1. Load NeMo Model
    model_id = "nvidia/parakeet_realtime_eou_120m-v1"
    print(f"Loading NeMo model {model_id}...")
    if Path(model_id).exists():
        asr_model = nemo_asr.models.ASRModel.restore_from(model_id, map_location="cpu")
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()

    # --- Monkey-Patch Fix for forward_internal ---
    import random
    from omegaconf import ListConfig
    import torch.nn as nn
    import types

    def fixed_forward_internal(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        # Fix: Use size(2) (Time) instead of size(1) (Dim) for [B, D, T] input
        # This function expects [B, D, T] input and transposes it to [B, T, D].
        # print(f"DEBUG: fixed_forward_internal running. Cache shape: {cache_last_channel.shape if cache_last_channel is not None else 'None'}")

        
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_next_layer, cache_last_time_next_layer) = audio_signal
                
                # FIX 1: Fix self-attention cache (cache_last_channel)
                # If returned cache is too small (collapsed), append to old cache
                expected_cache_len = self.streaming_cfg.last_channel_cache_size
                if cache_last_channel_next_layer.size(1) < expected_cache_len:
                    # Concatenate old + new along Time dimension (dim 1)
                    new_cache = torch.cat([cache_last_channel_cur, cache_last_channel_next_layer], dim=1)
                    if new_cache.size(1) > expected_cache_len:
                        new_cache = new_cache[:, -expected_cache_len:, :]
                    cache_last_channel_next_layer = new_cache

                # FIX 2: Fix convolution cache (cache_last_time)
                # Use previous cache size as expected size
                expected_time_cache_len = cache_last_time_cur.size(2)
                if cache_last_time_next_layer.size(2) < expected_time_cache_len:
                    # Concatenate old + new along Time dimension (dim 2)
                    new_time_cache = torch.cat([cache_last_time_cur, cache_last_time_next_layer], dim=2)
                    if new_time_cache.size(2) > expected_time_cache_len:
                        new_time_cache = new_time_cache[:, :, -expected_time_cache_len:]
                    cache_last_time_next_layer = new_time_cache

                cache_last_channel_next.append(cache_last_channel_next_layer)
                cache_last_time_next.append(cache_last_time_next_layer)

            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                if should_drop:
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length

    asr_model.encoder.forward_internal = types.MethodType(fixed_forward_internal, asr_model.encoder)
    # ---------------------------------------------

    
    # 2. Dataset
    dataset_path = "/Users/kikow/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean"
    print(f"Scanning {dataset_path}...")
    files = glob.glob(f"{dataset_path}/**/*.flac", recursive=True)
    # Run on all files
    files = sorted(files)
    print(f"Found {len(files)} files.")
    
    # Load Transcripts
    transcripts = {}
    for f in files:
        p = Path(f)
        trans_file = list(p.parent.glob("*.trans.txt"))[0]
        with open(trans_file) as tf:
            for line in tf:
                parts = line.strip().split()
                file_id = parts[0]
                text = " ".join(parts[1:])
                if file_id == p.stem:
                    transcripts[f] = text
                    break
    
    # 3. Benchmark Loop (Simulated Streaming)
    results = []
    
    # NeMo's streaming config
    # The model is trained with specific chunk size/shift.
    # We should inspect the config.
    cfg = asr_model.cfg
    print(f"Model Config: {cfg.encoder.get('chunk_size', 'N/A')}")
    
    # For Parakeet EOU, it's usually 1.28s chunk (160ms shift? No, 80ms shift usually for conformer, but streaming shift is different)
    # Let's use the built-in streaming inference if available, or simulate it.
    # The safest way to "verify our streaming mode" is to use the buffered inference provided by NeMo examples.
    
    # However, for a direct comparison, let's try to use the `transcribe` method with buffering if possible,
    # OR manually loop like we do in Swift.
    
    # Let's use the manual loop to be explicit.
    
    # Manual Streaming Loop
    print("Running streaming benchmark...")
    
    # Get initial cache state
    # We need to know the signature of encoder.forward
    # Usually: audio_signal, audio_signal_length, cache_last_channel, cache_last_time, cache_last_channel_len
    




    # Text Normalization (Mirroring Swift)
    
    # Initialize Normalizer
    normalizer = EnglishTextNormalizer()

    def normalize_text(text):
        norm = normalizer(text)
        # Remove trailing 'eou' (common artifact in this model's tokenizer)
        if norm.endswith("eou"):
            norm = norm[:-3].strip()
        return norm

    # Process in batches to avoid losing all progress on crash
    batch_size = 50
    results = []
    
    print(f"Processing {len(files)} files in batches of {batch_size}...")
    
    # Offline Benchmark (Disabled for debugging)
    # for i in range(0, len(files), batch_size):
    #     pass

    # Offline Benchmark skipped for speed
    # if results:
    #     avg_wer = np.mean(results)
    #     print(f"Average WER (Offline): {avg_wer*100:.2f}%")
    
    # If we really want streaming verification, we need the loop.
    # But let's see if offline works first.


    # ... (Previous code) ...

    # 4. Streaming Benchmark (Buffered Encoder)
    print("\n=== Running Streaming Benchmark (160ms Chunks) ===")
    
    # Setup Streaming Params
    # 160ms = 16 mel frames. Subsampling = 4.
    # Chunk = 16/4 = 4 encoder steps.
    # Shift = 80ms = 8 mel frames = 2 encoder steps.
    asr_model.encoder.setup_streaming_params(chunk_size=4, shift_size=2)
    print(f"Streaming Config: {asr_model.encoder.streaming_cfg}")
    
    streaming_results = []
    
    # Disable dither for consistency
    if hasattr(asr_model.preprocessor, 'featurizer'):
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        
    print(f"Processing {len(files)} files in batches of {batch_size}...")
    
    print(f"Processing {len(files)} files in batches of {batch_size}...")
    
    start_idx = 0
    print(f"DEBUG: files length: {len(files)}")
    print(f"DEBUG: start_idx: {start_idx}")
    
    for i in range(start_idx, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        print(f"DEBUG: Processing batch {i} with {len(batch_files)} files")
        sys.stdout.flush()
        
        try:
            # We process files one by one for streaming simulation
            for idx, audio_file in enumerate(batch_files):
                print(f"Processing {i+idx}: {audio_file}...", flush=True)
                # 1. Load Audio
                audio, sr = sf.read(audio_file)
                audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0) # [1, T]
                audio_len = torch.tensor([audio.shape[1]], dtype=torch.long)
                
                # 2. Compute Full Mel (Simplification)
                # In real streaming, we'd chunk audio. Here we chunk Mel to test Encoder.
                processed_signal, processed_signal_len = asr_model.preprocessor(
                    input_signal=audio, length=audio_len
                )
                # processed_signal: [1, D, T]
                
                # 3. Init Cache
                cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(batch_size=1)
                
                # 4. Streaming Loop
                mel = processed_signal
                T = mel.shape[2]
                
                chunk_size_mel = 16 # 160ms
                shift_size_mel = 8  # 80ms
                
                encoded_outputs = []
                
                # We need to handle the loop carefully.
                # NeMo's cache_aware_stream_step expects [B, D, T_chunk]
                
                current_idx = 0
                while current_idx < T:
                    # Extract chunk
                    # For correct convolution, we might need overlap?
                    # cache_aware_stream_step handles overlap via cache.
                    # We just feed new frames?
                    # Wait, setup_streaming_params sets the model to expect specific chunk sizes.
                    # If we feed 'chunk_size' (16 frames), it produces 'chunk_size' (4 frames) output?
                    # Let's try feeding exactly 16 frames.
                    
                    end_idx = min(current_idx + chunk_size_mel, T)
                    chunk = mel[:, :, current_idx:end_idx]
                    chunk_len = torch.tensor([chunk.shape[2]], dtype=torch.long)
                    
                    # Pad if last chunk is too small?
                    # NeMo might handle it, or we might need to pad.
                    if chunk.shape[2] < chunk_size_mel:
                        pad_amt = chunk_size_mel - chunk.shape[2]
                        chunk = torch.nn.functional.pad(chunk, (0, pad_amt))
                        chunk_len = torch.tensor([chunk_size_mel], dtype=torch.long)
                        
                    # Encode
                    out = asr_model.encoder.cache_aware_stream_step(
                        processed_signal=chunk,
                        processed_signal_length=chunk_len,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len
                    )
                    
                    # Unpack
                    encoded_chunk = out[0] # [1, D, T_out]
                    # out[1] is len
                    cache_last_channel = out[2]
                    cache_last_time = out[3]
                    cache_last_channel_len = out[4]
                    
                    encoded_outputs.append(encoded_chunk)
                    
                    # Shift
                    # If we use shift_size=2 (8 mel frames), we advance by 8 frames.
                    # But we fed 16 frames.
                    # Does cache_aware_stream_step consume 'shift' amount or 'chunk' amount?
                    # Usually it consumes 'chunk' amount of *new* data if we manage overlap manually.
                    # But here we rely on NeMo's internal management.
                    # If we configured shift=2, chunk=4 (encoder steps).
                    # That implies 50% overlap in encoder output?
                    # No, usually streaming means: Input N frames, Output M frames, Update State.
                    # Next Input N frames...
                    # If we want contiguous output, we should advance by the amount of *valid* output * subsampling?
                    
                    # Let's assume we feed non-overlapping chunks of 'shift_size' (8 frames)?
                    # No, convolution needs context.
                    # Let's feed 'chunk_size' (16 frames) and advance by 'shift_size' (8 frames).
                    
                    current_idx += shift_size_mel
                    
                    # Break if we passed the end
                    if current_idx >= T:
                        break
                
                # 5. Concatenate
                if encoded_outputs:
                    full_encoded = torch.cat(encoded_outputs, dim=2)
                    full_encoded_len = torch.tensor([full_encoded.shape[2]], dtype=torch.long)
                else:
                    print(f"DEBUG: No encoded outputs for {audio_file}")
                    continue
                    
                    # 6. Decode
                    # Use the standard greedy decoding on the full sequence
                    best_hyp = asr_model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=full_encoded,
                        encoded_lengths=full_encoded_len,
                        return_hypotheses=True
                    )
                    
                    text = best_hyp[0].text
                    
                    # 7. Calculate WER
                    ref = transcripts[audio_file]
                    norm_ref = normalize_text(ref)
                    norm_hyp = normalize_text(text)
                    
                    wer = jiwer.wer(norm_ref, norm_hyp)
                    streaming_results.append(wer)
                    
                    if wer > 0 and len(streaming_results) <= 5:
                        print(f"Streaming Mismatch in {audio_file}:")
                        print(f"Ref: {norm_ref}")
                        print(f"Hyp: {norm_hyp}")
                        print(f"WER: {wer}")
                        print("-" * 20)

        except Exception as e:
            print(f"Error in streaming batch {i}: {e}")
            continue
            
        # Interim
        if (i // batch_size) % 5 == 0:
            avg = np.mean(streaming_results) if streaming_results else 0
            print(f"Streaming Processed {len(streaming_results)}/{len(files)} | WER: {avg*100:.2f}%")

    if streaming_results:
        avg_wer = np.mean(streaming_results)
        print(f"Final Average WER (Streaming 160ms): {avg_wer*100:.2f}%")
    else:
        print("No streaming results.")

if __name__ == "__main__":
    benchmark_nemo_pure()
