import torch
import nemo.collections.asr as nemo_asr

print("Loading model...")
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet_realtime_eou_120m-v1')
model.eval()

encoder = model.encoder

print("\n=== Encoder Streaming Configuration ===")
if hasattr(encoder, 'streaming_cfg') and encoder.streaming_cfg is not None:
    cfg = encoder.streaming_cfg
    print(f"Streaming config type: {type(cfg)}")
    print(f"Config attributes: {dir(cfg)}")
    print()
    
    # Print all config attributes
    for attr in dir(cfg):
        if not attr.startswith('_'):
            value = getattr(cfg, attr, None)
            if not callable(value):
                print(f"  {attr}: {value}")
    
    print("\n=== Key Parameters ===")
    if hasattr(cfg, 'chunk_size'):
        print(f"chunk_size: {cfg.chunk_size}")
    if hasattr(cfg, 'left_chunks'):
        print(f"left_chunks (context): {cfg.left_chunks}")
    if hasattr(cfg, 'cache_drop_size'):
        print(f"cache_drop_size: {cfg.cache_drop_size}")
    if hasattr(cfg, 'drop_extra_pre_encoded'):
        print(f"drop_extra_pre_encoded: {cfg.drop_extra_pre_encoded}")
    if hasattr(cfg, 'last_channel_cache_size'):
        print(f"last_channel_cache_size: {cfg.last_channel_cache_size}")
    if hasattr(cfg, 'last_time_cache_size'):
        print(f"last_time_cache_size: {cfg.last_time_cache_size}")

print("\n=== Encoder Architecture ===")
print(f"Encoder type: {type(encoder)}")
print(f"Encoder class: {encoder.__class__.__name__}")

# Check for streaming methods
if hasattr(encoder, 'cache_aware_stream_step'):
    import inspect
    sig = inspect.signature(encoder.cache_aware_stream_step)
    print(f"\ncache_aware_stream_step signature:")
    print(f"  {sig}")
    
    # Get docstring
    if encoder.cache_aware_stream_step.__doc__:
        print(f"\nDocstring:")
        print(encoder.cache_aware_stream_step.__doc__[:500])

# Check if there's any built-in overlap mechanism
print("\n=== Testing Different Chunk Sizes ===")
# Create test chunks
for chunk_frames in [16, 32, 100]:
    mel = torch.randn(1, 128, chunk_frames)
    mel_len = torch.tensor([chunk_frames], dtype=torch.int32)
    
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(batch_size=1)
    
    with torch.no_grad():
        outputs = encoder.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_len,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
    
    enc_out = outputs[0]
    enc_len = outputs[1]
    
    print(f"Input: {chunk_frames} frames -> Output: {enc_len.item()} frames (shape: {enc_out.shape})")
