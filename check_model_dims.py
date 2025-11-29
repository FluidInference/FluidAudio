import torch
import nemo.collections.asr as nemo_asr

print("Loading model...")
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet_realtime_eou_120m-v1')
model.eval()

# Create dummy audio
audio = torch.randn(1, 16000)  # 1 second of audio
audio_len = torch.tensor([16000])

print("\nRunning preprocessor...")
with torch.no_grad():
    mel, mel_len = model.preprocessor(input_signal=audio, length=audio_len)

print(f"\nPreprocessor output shape: {mel.shape}")
print(f"Mel features (channels): {mel.shape[1]}")
print(f"Mel frames: {mel.shape[2]}")
print(f"Mel length: {mel_len.item()}")

print(f"\nEncoder expected input dim: {model.encoder.d_model if hasattr(model.encoder, 'd_model') else 'unknown'}")
