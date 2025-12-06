#!/usr/bin/env python3
"""
Test the exported CoreML streaming encoder against NeMo.
"""

import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import typer


def main(
    model_path: str = typer.Option(
        "Models/ParakeetEOU/Streaming/streaming_encoder.mlpackage",
        help="Path to CoreML model"
    ),
    audio_file: str = typer.Option(
        "she_sells_seashells_16k.wav",
        help="Audio file to test"
    ),
):
    """Test CoreML streaming encoder against NeMo."""
    import nemo.collections.asr as nemo_asr
    import soundfile as sf

    # Load NeMo model
    typer.echo("Loading NeMo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet_realtime_eou_120m-v1",
        map_location="cpu"
    )
    asr_model.eval()

    encoder = asr_model.encoder
    preprocessor = asr_model.preprocessor

    # Load CoreML model
    typer.echo(f"Loading CoreML model from {model_path}...")
    coreml_model = ct.models.MLModel(model_path)

    # Load config
    config_path = Path(model_path).parent / "streaming_encoder_config.json"
    with open(config_path) as f:
        config = json.load(f)

    typer.echo(f"Config: {config}")

    # Load audio
    typer.echo(f"Loading audio: {audio_file}...")
    audio, sr = sf.read(audio_file)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    typer.echo(f"Audio: {len(audio)/sr:.2f}s at {sr}Hz")

    # Initialize caches
    num_layers = config["num_layers"]
    cache_channel_size = config["cache_channel_size"]
    cache_time_size = config["cache_time_size"]
    hidden_dim = config["hidden_dim"]
    mel_frames = config["mel_frames"]

    # NeMo cache
    cache_last_channel, cache_last_time, cache_len = encoder.get_initial_cache_state(batch_size=1)

    # CoreML cache (same shapes)
    coreml_cache_channel = np.zeros((num_layers, 1, cache_channel_size, hidden_dim), dtype=np.float32)
    coreml_cache_time = np.zeros((num_layers, 1, hidden_dim, cache_time_size), dtype=np.float32)
    coreml_cache_len = np.array([0], dtype=np.int32)

    # Process in chunks
    chunk_samples = int(sr * config["chunk_ms"] / 1000)
    typer.echo(f"\nProcessing in {config['chunk_ms']}ms chunks ({chunk_samples} samples)...")

    nemo_outputs = []
    coreml_outputs = []
    offset = 0
    chunk_num = 0

    while offset < len(audio):
        chunk_num += 1
        end = min(offset + chunk_samples, len(audio))
        chunk = audio[offset:end]
        actual_len = len(chunk)

        # Pad if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
        chunk_len = torch.tensor([actual_len], dtype=torch.int32)

        # Preprocessor
        with torch.no_grad():
            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

        typer.echo(f"\nChunk {chunk_num}: offset={offset}, mel_shape={mel.shape}")

        # Pad/truncate mel to expected size
        mel_np = mel.numpy()
        if mel_np.shape[2] < mel_frames:
            mel_np = np.pad(mel_np, ((0, 0), (0, 0), (0, mel_frames - mel_np.shape[2])), mode='constant')
        elif mel_np.shape[2] > mel_frames:
            mel_np = mel_np[:, :, :mel_frames]

        mel_len_np = np.array([min(mel_len.item(), mel_frames)], dtype=np.int32)

        # NeMo streaming encoder
        with torch.no_grad():
            nemo_enc, nemo_enc_len, cache_last_channel, cache_last_time, cache_len = \
                encoder.cache_aware_stream_step(
                    processed_signal=mel,
                    processed_signal_length=mel_len,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_len,
                    keep_all_outputs=True,
                )

        typer.echo(f"  NeMo: enc_shape={nemo_enc.shape}, enc_len={nemo_enc_len.item()}")
        nemo_outputs.append(nemo_enc)

        # CoreML streaming encoder
        coreml_input = {
            "mel": mel_np,
            "mel_length": mel_len_np,
            "cache_last_channel": coreml_cache_channel,
            "cache_last_time": coreml_cache_time,
            "cache_last_channel_len": coreml_cache_len,
        }

        coreml_out = coreml_model.predict(coreml_input)

        coreml_enc = coreml_out["encoder"]
        coreml_enc_len = coreml_out["encoder_length"]
        coreml_cache_channel = coreml_out["cache_last_channel_out"]
        coreml_cache_time = coreml_out["cache_last_time_out"]
        coreml_cache_len = coreml_out["cache_last_channel_len_out"]

        typer.echo(f"  CoreML: enc_shape={coreml_enc.shape}, enc_len={coreml_enc_len}")
        coreml_outputs.append(coreml_enc)

        # Compare outputs
        nemo_np = nemo_enc.numpy()

        # Handle different output lengths
        min_len = min(nemo_np.shape[2], coreml_enc.shape[2])
        diff = np.abs(nemo_np[:, :, :min_len] - coreml_enc[:, :, :min_len])
        max_diff = diff.max()
        mean_diff = diff.mean()

        typer.echo(f"  Diff: max={max_diff:.6f}, mean={mean_diff:.6f}")

        offset += chunk_samples

    # Final comparison
    typer.echo("\n" + "="*60)
    typer.echo("SUMMARY")
    typer.echo("="*60)

    nemo_full = torch.cat(nemo_outputs, dim=2).numpy()
    coreml_full = np.concatenate(coreml_outputs, axis=2)

    typer.echo(f"NeMo full output: {nemo_full.shape}")
    typer.echo(f"CoreML full output: {coreml_full.shape}")

    min_t = min(nemo_full.shape[2], coreml_full.shape[2])
    final_diff = np.abs(nemo_full[:, :, :min_t] - coreml_full[:, :, :min_t])
    typer.echo(f"Final diff: max={final_diff.max():.6f}, mean={final_diff.mean():.6f}")

    if final_diff.max() < 0.01:
        typer.echo("\n✓ CoreML model matches NeMo output!")
    else:
        typer.echo("\n✗ Significant difference between CoreML and NeMo")


if __name__ == "__main__":
    typer.run(main)
