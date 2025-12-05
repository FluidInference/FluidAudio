import torch
import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np

import inspect

def repro_crash():
    model_id = "nvidia/parakeet_realtime_eou_120m-v1"
    print(f"Loading NeMo model {model_id}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    
    # Print source of cache_aware_stream_step
    try:
        print("Source of cache_aware_stream_step:")
        print(inspect.getsource(asr_model.encoder.cache_aware_stream_step))
    except Exception as e:
        print(f"Could not get source: {e}")

    asr_model.eval()
    with open("encoder_structure.txt", "w") as f:
        f.write(str(asr_model.encoder))



    # Monkey-patch forward_internal to debug
    original_forward_internal = asr_model.encoder.forward_internal
    
    print(f"Attention Class: {asr_model.encoder.layers[0].self_attn.__class__}")
    try:
        print("Attention Source (update_cache):")
        print(inspect.getsource(asr_model.encoder.layers[0].self_attn.update_cache))
    except:
        print("Could not get update_cache source")


    
    import random
    from omegaconf import ListConfig
    import torch.nn as nn

    # Monkey-patch forward_internal with FULL implementation and DEBUG prints
    def debug_forward_internal(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        print(f"DEBUG: forward_internal input: {audio_signal.shape}")
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal = torch.transpose(audio_signal, 1, 2)
        print(f"DEBUG: after transpose: {audio_signal.shape}")

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            # self.streaming_cfg is set by setup_streaming_cfg(), called in the init
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        print(f"DEBUG: max_audio_length (initial): {max_audio_length}")
        
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

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )
        print(f"DEBUG: att_mask shape: {att_mask.shape if att_mask is not None else 'None'}")

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
                # print(f"DEBUG: layer {lth} cache: {cache_last_channel_cur.shape}")
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
                
                print(f"DEBUG: Layer {lth}, cache_cur: {cache_last_channel_cur.shape}, next: {cache_last_channel_next_layer.shape}")
                print(f"DEBUG: Layer {lth}, time_cache_cur: {cache_last_time_cur.shape}, next_time: {cache_last_time_next_layer.shape}")
                
                # FIX: If returned cache is too small (collapsed), append to old cache
                # This handles cases where the layer returns only new frames instead of the full updated cache
                expected_cache_len = self.streaming_cfg.last_channel_cache_size
                # Check Time dimension (dim 1)
                if cache_last_channel_next_layer.size(1) < expected_cache_len:
                    print(f"DEBUG: Layer {lth} returned small cache {cache_last_channel_next_layer.shape}. Concatenating...")
                    # Concatenate old + new along Time dimension (dim 1)
                    # cache_last_channel_cur is [B, T, D]
                    new_cache = torch.cat([cache_last_channel_cur, cache_last_channel_next_layer], dim=1)
                    # Slice to keep fixed size
                    if new_cache.size(1) > expected_cache_len:
                        new_cache = new_cache[:, -expected_cache_len:, :]
                    cache_last_channel_next_layer = new_cache
                
                # FIX 2: Also fix time cache (convolution cache)
                # cache_last_time is [B, D, T]
                expected_time_cache_len = cache_last_time_cur.size(2)
                if cache_last_time_next_layer.size(2) < expected_time_cache_len:
                    print(f"DEBUG: Layer {lth} returned small time cache {cache_last_time_next_layer.shape}. Concatenating...")
                    new_time_cache = torch.cat([cache_last_time_cur, cache_last_time_next_layer], dim=2)
                    if new_time_cache.size(2) > expected_time_cache_len:
                        new_time_cache = new_time_cache[:, :, -expected_time_cache_len:]
                    cache_last_time_next_layer = new_time_cache

                cache_last_channel_next.append(cache_last_channel_next_layer)
                cache_last_time_next.append(cache_last_time_next_layer)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                print(f"DEBUG: reduction at layer {lth}")
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                print(f"DEBUG: max_audio_length (after reduction): {max_audio_length}")
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )
                print(f"DEBUG: att_mask shape (after reduction): {att_mask.shape if att_mask is not None else 'None'}")

            # saving tensors if required for interctc loss
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
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

    import types
    asr_model.encoder.forward_internal = types.MethodType(debug_forward_internal, asr_model.encoder)
    
    audio_file = "/Users/kikow/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean/8463/294825/8463-294825-0008.flac"
    print(f"Processing {audio_file}...")
    
    # Force manual config update
    # from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingConfig
    
    # Let's try to set it directly.
    asr_model.encoder.setup_streaming_params(chunk_size=4, shift_size=2)
    
    # Manually override if setup failed to give what we want
    # We want chunk_size to be 4.
    # The printed config showed chunk_size=[25, 32].
    
    # Let's inspect what 'chunk_size=4' implies.
    # If I pass 4, I expect it to be 4.
    
    # Let's try to force it.
    # We need to know the structure. It seems to be a dataclass or struct.
    # Let's try to set the property if it's mutable.
    try:
        asr_model.encoder.streaming_cfg.chunk_size = [4, 4] 
        asr_model.encoder.streaming_cfg.shift_size = [2, 2]
    except:
        print("Could not set streaming_cfg attributes directly.")

    print(f"Updated Streaming Config (Post-Force): {asr_model.encoder.streaming_cfg}")
    print(f"last_channel_cache_size: {asr_model.encoder.streaming_cfg.last_channel_cache_size}")

    
    audio, sr = sf.read(audio_file)
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    audio_len = torch.tensor([audio.shape[1]], dtype=torch.long)
    
    processed_signal, processed_signal_len = asr_model.preprocessor(
        input_signal=audio, length=audio_len
    )
    
    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(batch_size=1)
    
    mel = processed_signal
    T = mel.shape[2]
    chunk_size_mel = 16
    shift_size_mel = 8
    
    current_idx = 0
    step = 0
    while current_idx < T:
        end_idx = min(current_idx + chunk_size_mel, T)
        chunk = mel[:, :, current_idx:end_idx]
        
        if chunk.shape[2] < chunk_size_mel:
            pad_amt = chunk_size_mel - chunk.shape[2]
            chunk = torch.nn.functional.pad(chunk, (0, pad_amt))
        
        # chunk_len = torch.tensor([4], dtype=torch.long) # Hardcoded encoder steps
        chunk_len = torch.tensor([chunk.shape[2]], dtype=torch.long)
        
        print(f"Step {step}: chunk={chunk.shape}, chunk_len={chunk_len}, cache_channel={cache_last_channel.shape}, cache_len={cache_last_channel_len}")
        
        # Pass [B, D, T] directly now that forward_internal is fixed
        # chunk_transposed = chunk.transpose(1, 2)
        
        try:
            out = asr_model.encoder.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len
            )
            
            cache_last_channel = out[2]
            cache_last_time = out[3]
            cache_last_channel_len = out[4]
            
        except Exception as e:
            print(f"CRASH at step {step}: {e}")
            import traceback
            traceback.print_exc()
            # Print detailed shapes if possible
            break
            
        current_idx += shift_size_mel
        step += 1

if __name__ == "__main__":
    repro_crash()
