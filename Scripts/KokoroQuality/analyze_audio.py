import soundfile as sf
import os

files = [
    "wav_compare/swift_zh_ipa_spaced_slow.wav",
    "wav_compare/ref_zh.wav"
]

for f in files:
    if os.path.exists(f):
        data, samplerate = sf.read(f)
        duration = len(data) / samplerate
        print(f"File: {f}")
        print(f"  Sample Rate: {samplerate}")
        print(f"  Samples: {len(data)}")
        print(f"  Duration: {duration:.4f}s")
    else:
        print(f"File not found: {f}")
