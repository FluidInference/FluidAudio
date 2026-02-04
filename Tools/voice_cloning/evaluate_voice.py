#!/usr/bin/env python3
"""Evaluate voice cloning quality using speaker embeddings.

Compares a reference voice sample with synthesized TTS output using
neural speaker embeddings - the gold standard for speaker verification.

Requirements:
    pip install resemblyzer

Usage:
    python evaluate_voice.py reference.wav synthesized.wav
    python evaluate_voice.py reference.wav synthesized.wav --plot
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Resemblyzer expects 16kHz


def load_audio(path: Path) -> np.ndarray:
    """Load audio and resample to 16kHz for Resemblyzer."""
    try:
        import librosa
        audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return audio
    except ImportError:
        from scipy.io import wavfile
        from scipy import signal
        sr, audio = wavfile.read(str(path))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = signal.resample(audio, num_samples)
        return audio.astype(np.float32)


def compute_speaker_similarity(ref_audio: np.ndarray, syn_audio: np.ndarray) -> float:
    """Compute speaker similarity using Resemblyzer embeddings."""
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder()

    ref_emb = encoder.embed_utterance(ref_audio)
    syn_emb = encoder.embed_utterance(syn_audio)

    # Cosine similarity
    similarity = np.dot(ref_emb, syn_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(syn_emb))
    return float(similarity)


def evaluate_voice_cloning(
    reference_path: Path,
    synthesized_path: Path,
    plot: bool = False
) -> dict:
    """Evaluate voice cloning quality using speaker embeddings."""
    logger.info(f"Reference:   {reference_path}")
    logger.info(f"Synthesized: {synthesized_path}")
    logger.info("")

    # Load audio
    ref_audio = load_audio(reference_path)
    syn_audio = load_audio(synthesized_path)

    logger.info(f"Reference duration:   {len(ref_audio) / SAMPLE_RATE:.2f}s")
    logger.info(f"Synthesized duration: {len(syn_audio) / SAMPLE_RATE:.2f}s")
    logger.info("")

    # Compute speaker similarity
    logger.info("Computing speaker similarity...")
    similarity = compute_speaker_similarity(ref_audio, syn_audio)

    metrics = {'speaker_similarity': similarity}

    logger.info("")
    logger.info(f"  Speaker Similarity: {similarity:.4f}")

    # Quality interpretation
    if similarity >= 0.85:
        quality = "Excellent"
    elif similarity >= 0.75:
        quality = "Good"
    elif similarity >= 0.65:
        quality = "Fair"
    else:
        quality = "Poor"

    metrics['quality'] = quality
    logger.info(f"  Quality:            {quality}")

    # Plot if requested
    if plot:
        plot_embeddings(ref_audio, syn_audio, reference_path.stem, synthesized_path.stem)

    return metrics


def plot_embeddings(ref_audio: np.ndarray, syn_audio: np.ndarray,
                    ref_name: str, syn_name: str):
    """Visualize speaker embeddings."""
    try:
        import matplotlib.pyplot as plt
        from resemblyzer import VoiceEncoder
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    encoder = VoiceEncoder()

    # Get embeddings
    ref_emb = encoder.embed_utterance(ref_audio)
    syn_emb = encoder.embed_utterance(syn_audio)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Embedding comparison
    axes[0].bar(range(len(ref_emb)), ref_emb, alpha=0.7, label=f'Reference: {ref_name}')
    axes[0].bar(range(len(syn_emb)), syn_emb, alpha=0.7, label=f'Synthesized: {syn_name}')
    axes[0].set_xlabel('Embedding Dimension')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Speaker Embeddings')
    axes[0].legend()

    # Similarity heatmap
    similarity_matrix = np.outer(ref_emb, syn_emb)
    im = axes[1].imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
    axes[1].set_xlabel('Synthesized Embedding')
    axes[1].set_ylabel('Reference Embedding')
    axes[1].set_title('Embedding Correlation')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('speaker_comparison.png', dpi=150)
    logger.info("\nSaved comparison plot to: speaker_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate voice cloning using speaker embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speaker Similarity Thresholds:
  0.85+  Excellent - Very close voice match
  0.75+  Good      - Clearly same speaker
  0.65+  Fair      - Some similarity
  <0.65  Poor      - Different speaker characteristics

Requirements:
  pip install resemblyzer

Examples:
  python evaluate_voice.py original_speaker.wav tts_output.wav
  python evaluate_voice.py reference.wav synthesized.wav --plot
"""
    )
    parser.add_argument("reference", type=Path, help="Reference voice audio file")
    parser.add_argument("synthesized", type=Path, help="Synthesized TTS audio file")
    parser.add_argument("--plot", action="store_true", help="Show embedding comparison plots")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")

    args = parser.parse_args()

    if not args.reference.exists():
        logger.error(f"Reference file not found: {args.reference}")
        sys.exit(1)
    if not args.synthesized.exists():
        logger.error(f"Synthesized file not found: {args.synthesized}")
        sys.exit(1)

    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        logger.error("resemblyzer not installed. Run: pip install resemblyzer")
        sys.exit(1)

    metrics = evaluate_voice_cloning(args.reference, args.synthesized, plot=args.plot)

    if args.json:
        import json
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
