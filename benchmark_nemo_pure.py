import torch
import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np
from pathlib import Path
import glob
import jiwer
from tqdm import tqdm
import json

def benchmark_nemo_pure():
    # 1. Load NeMo Model
    model_id = "nvidia/parakeet_realtime_eou_120m-v1"
    print(f"Loading NeMo model {model_id}...")
    if Path(model_id).exists():
        asr_model = nemo_asr.models.ASRModel.restore_from(model_id, map_location="cpu")
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location="cpu")
    asr_model.eval()
    
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
    
    for audio_file in tqdm(files):
        audio, sr = sf.read(audio_file)
        audio = audio.astype(np.float32)
        
        # 1. Init Cache
        cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(batch_size=1)
        
        # Decoder State
        # RNNT Decoder state: (batch, layers, hidden)?
        # NeMo RNNT decoder usually handles state internally if not stateless?
        # But for streaming we might need to manage it if we use `predict` step by step.
        # However, `asr_model.decoder` is usually an LSTM or similar.
        # Let's check if we can use `asr_model.decoder.predict(..., state=...)`.
        
        # Actually, let's try to use the `StreamingFeatureBufferer` + `transcribe` if possible?
        # No, let's stick to the loop.
        
        # Decoder init
        # We need to know the blank id
        blank_id = asr_model.decoder.blank_idx if hasattr(asr_model.decoder, 'blank_idx') else asr_model.tokenizer.vocab_size
        
        # State for decoder (LSTM)
        # We might need to inspect the model to know the state shape.
        # Or we can just rely on the fact that `asr_model.decoder` might be stateless (if it's just embedding + projection)?
        # No, RNNT decoder is usually LSTM/RNN.
        
        # Let's assume we can just run the encoder in streaming mode and then use greedy decoding on the output?
        # No, RNNT decoding is coupled.
        
        # This is getting complicated to implement from scratch without seeing the code.
        # BUT, we have `benchmark_hybrid_encoder.py` which had the logic for CoreML.
        # The NeMo model should have similar inputs/outputs.
        
        # Let's try to use `asr_model.transcribe` with `batch_size=1` but fix the arguments.
        # It might be `paths2audio_files` is for CTC.
        # Try `audio_files`?
        
        # Let's try to inspect `asr_model.transcribe` arguments via help() or dir() in a small script?
        # Or just try `paths2audio_files` -> `paths2audio_files` (list).
        # The error said `unexpected keyword argument`.
        
        # Maybe `asr_model.transcribe(paths2audio_files=[...])` is correct for some models but not this one?
        # Let's try `asr_model.transcribe([audio_file])` (positional).
        
        pass



    # Text Normalization (Mirroring Swift)
    import re
    def normalize_text(text):
        text = text.lower()
        # Abbreviations
        abbreviations = {
            "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor",
            "prof": "professor", "st": "saint", "jr": "junior", "sr": "senior",
            "esq": "esquire", "capt": "captain", "gov": "governor",
            "ald": "alderman", "gen": "general", "sen": "senator",
            "rep": "representative", "pres": "president", "rev": "reverend",
            "hon": "honorable", "asst": "assistant", "assoc": "associate",
            "lt": "lieutenant", "col": "colonel", "vs": "versus",
            "inc": "incorporated", "ltd": "limited", "co": "company",
            "am": "a m", "pm": "p m", "ad": "ad", "bc": "bc"
        }
        for abbrev, expansion in abbreviations.items():
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            text = re.sub(pattern, expansion, text)
        
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        
        # Remove trailing 'eou' (common artifact in this model's tokenizer)
        if text.endswith("eou"):
            text = text[:-3]
        
        return text.strip()

    # Process in batches to avoid losing all progress on crash
    batch_size = 50
    results = []
    
    print(f"Processing {len(files)} files in batches of {batch_size}...")
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        try:
            # Transcribe batch
            hypotheses = asr_model.transcribe(batch_files, batch_size=1, verbose=False)
            
            for j, hyp in enumerate(hypotheses):
                # NeMo RNNT transcribe returns list of Hypothesis objects
                text = hyp.text
                
                # Index in global list
                global_idx = i + j
                ref = transcripts[files[global_idx]]
                
                norm_ref = normalize_text(ref)
                norm_hyp = normalize_text(text)
                
                wer = jiwer.wer(norm_ref, norm_hyp)
                results.append(wer)
                
                if wer > 0 and len(results) <= 10:
                    print(f"Mismatch in {files[global_idx]}:")
                    print(f"Ref: {norm_ref}")
                    print(f"Hyp: {norm_hyp}")
                    print(f"WER: {wer}")
                    print("-" * 20)
                    
        except Exception as e:
            print(f"Error processing batch {i}-{i+batch_size}: {e}")
            continue
            
        # Print interim progress
        if (i // batch_size) % 5 == 0:
            current_avg = np.mean(results) if results else 0
            print(f"Processed {len(results)}/{len(files)} | Current WER: {current_avg*100:.2f}%")

    if results:
        avg_wer = np.mean(results)
        print(f"Average WER (Offline): {avg_wer*100:.2f}%")
    
    # If we really want streaming verification, we need the loop.
    # But let's see if offline works first.


if __name__ == "__main__":
    benchmark_nemo_pure()
