class SingleStreamingEncoderWrapper(torch.nn.Module):
    """
    Combines Pre-Encode and Conformer into a SINGLE streaming model.
    
    Inputs:
      - audio_signal: [B, T_audio] (Raw Audio)
      - audio_length: [B]
      - cache_last_channel: [B, D, C, T_cache]
      - cache_last_time: [B, D, T_cache, D_time]
      - cache_last_channel_len: [B, D]
      
    Outputs:
      - encoded: [B, D, T_enc]
      - encoded_len: [B]
      - new_cache_last_channel
      - new_cache_last_time
      - new_cache_last_channel_len
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        audio_signal,
        audio_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
    ):
        # 1. Run Pre-Encode (Audio -> Mel -> Subsampled Features)
        # Note: We rely on the client to provide the "Convolution Context" (overlapping audio).
        # So we just run the standard pre_encode.
        
        # We call the internal _pre_encode method of the ConformerEncoder (or similar)
        # But wait, cache_aware_stream_step handles pre_encode if bypass_pre_encode=False (default).
        # So we can just call cache_aware_stream_step directly with the RAW audio!
        
        outputs = self.encoder.cache_aware_stream_step(
            processed_signal=audio_signal,
            processed_signal_length=audio_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=False # IMPORTANT: Do NOT bypass. Let it do Mel + Subsampling.
        )
        
        # outputs: (encoded, encoded_len, new_cache_last_channel, new_cache_last_time, new_cache_last_channel_len)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
