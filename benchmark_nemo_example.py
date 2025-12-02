import torch
import soundfile as sf
import librosa
import numpy as np
import logging
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_decoding_strategy(asr_model, strategy='greedy'):
    """
    Sets up the decoding strategy.
    Adapted from NeMo example, but with fallback for RNNTBPEDecoding.
    """
    print(f"Setting up decoding strategy: {strategy}")
    
    # Create a config for the desired strategy
    # The example uses cfg.rnnt_decoding, we'll create a minimal one
    decoding_cfg = OmegaConf.create({
        'strategy': strategy,
        'greedy': {'max_symbols': 10}, # Standard greedy params
        'fused_batch_size': -1,
        'compute_timestamps': False, # Disable for stability
        'preserve_alignments': False
    })

    if hasattr(asr_model, 'change_decoding_strategy'):
        try:
            asr_model.change_decoding_strategy(decoding_cfg)
            print("Successfully changed decoding strategy via change_decoding_strategy")
            return
        except Exception as e:
            print(f"Standard change_decoding_strategy failed: {e}")
            print("Attempting manual replacement...")

    # Manual replacement fallback (Required for Parakeet EOU)
    if hasattr(asr_model, 'decoding') and isinstance(asr_model.decoding, RNNTBPEDecoding):
        new_decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg,
            decoder=asr_model.decoder,
            joint=asr_model.joint,
            tokenizer=asr_model.tokenizer
        )
        asr_model.decoding = new_decoding
        print("Successfully replaced decoding strategy manually.")
    else:
        print("Could not change decoding strategy.")

def perform_streaming(asr_model, streaming_buffer, device):
    """
    Performs streaming inference using conformer_stream_step.
    Follows the NeMo example structure.
    """
    # Get initial cache state
    # Note: The example uses batch_size from buffer, we assume 1 for simplicity here
    batch_size = 1
    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )
    
    # Move cache to device
    if cache_last_channel is not None:
        cache_last_channel = cache_last_channel.to(device)
        cache_last_time = cache_last_time.to(device)
        cache_last_channel_len = cache_last_channel_len.to(device)

    previous_hypotheses = None
    previous_pred_out = None
    
    final_transcription = ""
    
    print("Starting streaming loop...")
    
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
        chunk_audio = chunk_audio.to(device)
        chunk_lengths = chunk_lengths.to(device)
        
        print(f"Step {step_num}: chunk_audio shape: {chunk_audio.shape}")
        
        # conformer_stream_step
        with torch.no_grad():
            (
                greedy_predictions,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                best_hyp_list,
            ) = asr_model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False, # We don't need to keep all outputs for now
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=previous_pred_out,
                return_transcription=True
            )
        
        # Update state for next step
        previous_hypotheses = best_hyp_list
        
        # Extract text and handle EOU (The "Complex" Part)
        current_hyp = best_hyp_list[0] if isinstance(best_hyp_list, list) else best_hyp_list
        
        # Check for EOU (1024)
        is_eou = False
        if hasattr(current_hyp, 'y_sequence'):
             y_seq = current_hyp.y_sequence
             if isinstance(y_seq, list) and 1024 in y_seq:
                 is_eou = True
             elif torch.is_tensor(y_seq) and (y_seq == 1024).any():
                 is_eou = True
        
        if is_eou:
            # FIX: Reset decoder state on EOU
            previous_hypotheses = None
            if hasattr(current_hyp, 'text'):
                final_transcription += current_hyp.text + " "
        
        # Note: If not EOU, we don't append text yet because it's partial.
        # The example accumulates `transcribed_texts` but that might be for the whole batch/history?
        # In strict streaming, we usually only commit on EOU or stability.
        # For this demo, we'll just print partials.
        
        # print(f"Step {step_num}: {current_hyp.text if hasattr(current_hyp, 'text') else ''}")

    # Append final bit
    if previous_hypotheses:
        last_hyp = previous_hypotheses[0] if isinstance(previous_hypotheses, list) else previous_hypotheses
        if hasattr(last_hyp, 'text'):
            final_transcription += last_hyp.text

    return final_transcription.replace("<eou>", "").strip()

import argparse
import jiwer
from pathlib import Path

def load_manifest(dataset_path, subset='test-clean', max_files=None):
    subset_dir = Path(dataset_path) / subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {subset_dir}")
    
    flac_files = list(subset_dir.rglob('*.flac'))
    if not flac_files:
        raise FileNotFoundError(f"No FLAC files found in {subset_dir}")
    
    # Sort for determinism
    flac_files = sorted(flac_files)
    
    entries = []
    for flac_path in flac_files:
        if max_files and len(entries) >= max_files:
            break
            
        speaker_id = flac_path.parent.parent.name
        chapter_id = flac_path.parent.name
        trans_file = flac_path.parent / f"{speaker_id}-{chapter_id}.trans.txt"
        
        if trans_file.exists():
            utterance_id = flac_path.stem
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2 and parts[0] == utterance_id:
                        entries.append({
                            'audio_filepath': str(flac_path),
                            'text': parts[1],
                            'duration': 0
                        })
                        break
    print(f"Loaded {len(entries)} entries from {subset_dir}")
    return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-files', type=int, default=100)
    args = parser.parse_args()

    model_id = "nvidia/parakeet_realtime_eou_120m-v1"
    dataset_path = "/Users/kikow/Library/Caches/fluidaudio/LibriSpeech/LibriSpeech"
    
    device = torch.device("cpu") # Force CPU for now
    
    print(f"Loading model: {model_id}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location=device)
    model.eval()
    
    # 1. Setup Decoding Strategy (Crucial Step)
    setup_decoding_strategy(model, strategy='greedy')
    
    # 2. Setup Streaming Params
    model.encoder.setup_streaming_params(chunk_size=4, shift_size=4)
    print(f"Updated Streaming Config: {model.encoder.streaming_cfg}")
    
    # Load Data
    entries = load_manifest(dataset_path, max_files=args.max_files)
    
    total_wer = 0
    count = 0
    
    print(f"Starting Benchmark on {len(entries)} files...")
    
    for i, entry in enumerate(entries):
        audio_file = entry['audio_filepath']
        ref_text = entry['text'].lower()
        
        # Create buffer per file (clean state)
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=model,
            online_normalization=False, 
            pad_and_drop_preencoded=False
        )
        
        streaming_buffer.append_audio_file(audio_file, stream_id=-1)
        
        # 3. Perform Streaming
        hyp_text = perform_streaming(model, streaming_buffer, device)
        
        # Calculate WER
        wer = jiwer.wer(ref_text, hyp_text)
        total_wer += wer
        count += 1
        
        print(f"[{i+1}/{len(entries)}] {Path(audio_file).name} | WER: {wer*100:.2f}% | Ref: '{ref_text}' | Hyp: '{hyp_text}'")

    avg_wer = total_wer / count if count > 0 else 0
    print(f"\nAverage WER over {count} files: {avg_wer*100:.2f}%")

if __name__ == "__main__":
    main()
