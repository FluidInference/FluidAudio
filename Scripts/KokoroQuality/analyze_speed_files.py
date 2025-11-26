import soundfile as sf
import os

files = [
    "wav_compare/swift_zh_ipa_0.9.wav",
    "wav_compare/swift_zh_ipa_0.95.wav",
    "wav_compare/swift_zh_ipa_0.75.wav",
    "wav_compare/swift_zh_ipa_final.wav" # Default (1.0)
]

for f in files:
    if os.path.exists(f):
        data, samplerate = sf.read(f)
        duration = len(data) / samplerate
        print(f"File: {f}")
        print(f"  Duration: {duration:.4f}s")
    else:
        print(f"File not found: {f}")
