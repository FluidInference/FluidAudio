
import nemo.collections.asr as nemo_asr
import torch

model_id = "nvidia/parakeet_realtime_eou_120m-v1"
print(f"Loading {model_id}...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")

print("\n=== Model Config ===")
if hasattr(asr_model.encoder, 'streaming_cfg'):
    print(f"Streaming Config: {asr_model.encoder.streaming_cfg}")
else:
    print("No streaming_cfg found on encoder")

if hasattr(asr_model.encoder, 'subsampling_factor'):
    print(f"Subsampling Factor: {asr_model.encoder.subsampling_factor}")
else:
    print("No subsampling_factor found on encoder")

print(f"\nPreprocessor Config:")
print(asr_model.cfg.preprocessor)
