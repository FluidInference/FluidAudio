import os
import soundfile as sf
from kokoro import KPipeline
import torch

# Constants
OUTPUT_DIR = "wav_compare"
PROMPT = "我觉得学好英语是一件很有必要的事！"
VOICE = "zf_xiaobei" # Chinese voice (zf_003 not found on HF)

def generate_ref_wav():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize pipeline for Chinese
    # lang_code='z' for Chinese
    print(f"Initializing KPipeline for Chinese (lang_code='z')...")
    pipeline = KPipeline(lang_code='z') 
    
    print(f"Generating ref for prompt: {PROMPT}")
    print(f"Voice: {VOICE}")
    
    total_audio = []
    
    # Iterate over the generator yielded by pipeline()
    for i_chunk, (gs, ps, audio) in enumerate(pipeline(PROMPT, voice=VOICE, speed=1)):
        print(f"  Chunk {i_chunk}:")
        print(f"    Text: '{gs}'")
        print(f"    Phonemes: '{ps}'")
        
        if audio is not None:
            total_audio.append(audio)
    
    if total_audio:
        final_audio = torch.cat(total_audio, dim=0)
        # Save to file
        filename = "ref_zh.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        sf.write(filepath, final_audio.cpu().numpy(), 24000)
        print(f"Saved {filepath}")
    else:
        print("No audio generated!")

if __name__ == "__main__":
    generate_ref_wav()
