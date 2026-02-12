#!/usr/bin/env python3
"""Evaluate voice cloning quality using spectral similarity.

Compares a reference voice sample with synthesized TTS output using
mel-spectrogram cosine similarity - no neural network required.

Requirements:
    pip install librosa numpy scipy

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

SAMPLE_RATE = 24000  # PocketTTS native sample rate


def load_audio(path: Path) -> np.ndarray:
    """Load audio and resample to target sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
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


def compute_mel_spectrogram(audio: np.ndarray, n_mels: int = 80, n_fft: int = 1024,
                            hop_length: int = 256) -> np.ndarray:
    """Compute mel spectrogram."""
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
        return librosa.power_to_db(mel, ref=np.max)
    except ImportError:
        # Fallback using scipy
        from scipy import signal
        from scipy.fftpack import dct

        # Simple STFT
        _, _, Sxx = signal.spectrogram(audio, fs=SAMPLE_RATE, nperseg=n_fft,
                                        noverlap=n_fft - hop_length)
        # Approximate mel scaling (simplified)
        mel_basis = np.zeros((n_mels, Sxx.shape[0]))
        for i in range(n_mels):
            center = int(Sxx.shape[0] * (i + 1) / (n_mels + 1))
            width = max(1, Sxx.shape[0] // (n_mels * 2))
            mel_basis[i, max(0, center-width):min(Sxx.shape[0], center+width)] = 1
        mel_basis = mel_basis / (mel_basis.sum(axis=1, keepdims=True) + 1e-8)
        mel = np.dot(mel_basis, Sxx)
        return 10 * np.log10(mel + 1e-10)


def compute_mfcc(audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Compute MFCCs."""
    try:
        import librosa
        return librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
    except ImportError:
        mel = compute_mel_spectrogram(audio)
        from scipy.fftpack import dct
        return dct(mel, type=2, axis=0, norm='ortho')[:n_mfcc]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    # Truncate to same length
    min_len = min(len(a_flat), len(b_flat))
    a_flat = a_flat[:min_len]
    b_flat = b_flat[:min_len]

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def compute_spectral_similarity(ref_audio: np.ndarray, syn_audio: np.ndarray) -> dict:
    """Compute spectral similarity metrics."""
    # Compute mel spectrograms
    ref_mel = compute_mel_spectrogram(ref_audio)
    syn_mel = compute_mel_spectrogram(syn_audio)

    # Compute mean mel vectors (voice timbre signature)
    ref_mel_mean = ref_mel.mean(axis=1)
    syn_mel_mean = syn_mel.mean(axis=1)
    mel_similarity = cosine_similarity(ref_mel_mean, syn_mel_mean)

    # Compute MFCCs
    ref_mfcc = compute_mfcc(ref_audio)
    syn_mfcc = compute_mfcc(syn_audio)

    # MFCC mean (captures voice characteristics)
    ref_mfcc_mean = ref_mfcc.mean(axis=1)
    syn_mfcc_mean = syn_mfcc.mean(axis=1)
    mfcc_similarity = cosine_similarity(ref_mfcc_mean, syn_mfcc_mean)

    # MFCC std (captures dynamics)
    ref_mfcc_std = ref_mfcc.std(axis=1)
    syn_mfcc_std = syn_mfcc.std(axis=1)
    mfcc_std_similarity = cosine_similarity(ref_mfcc_std, syn_mfcc_std)

    return {
        'mel_similarity': mel_similarity,
        'mfcc_similarity': mfcc_similarity,
        'mfcc_std_similarity': mfcc_std_similarity,
    }


def evaluate_voice_cloning(
    reference_path: Path,
    synthesized_path: Path,
    plot: bool = False
) -> dict:
    """Evaluate voice cloning quality using spectral similarity."""
    logger.info(f"Reference:   {reference_path}")
    logger.info(f"Synthesized: {synthesized_path}")
    logger.info("")

    # Load audio
    ref_audio = load_audio(reference_path)
    syn_audio = load_audio(synthesized_path)

    logger.info(f"Reference duration:   {len(ref_audio) / SAMPLE_RATE:.2f}s")
    logger.info(f"Synthesized duration: {len(syn_audio) / SAMPLE_RATE:.2f}s")
    logger.info("")

    # Compute spectral similarity
    logger.info("Computing spectral similarity...")
    metrics = compute_spectral_similarity(ref_audio, syn_audio)

    # Combined score (weighted average)
    combined = (
        0.4 * metrics['mel_similarity'] +
        0.4 * metrics['mfcc_similarity'] +
        0.2 * metrics['mfcc_std_similarity']
    )
    metrics['combined_similarity'] = combined

    logger.info("")
    logger.info(f"  Mel Similarity:      {metrics['mel_similarity']:.4f}")
    logger.info(f"  MFCC Similarity:     {metrics['mfcc_similarity']:.4f}")
    logger.info(f"  MFCC Std Similarity: {metrics['mfcc_std_similarity']:.4f}")
    logger.info(f"  Combined Score:      {combined:.4f}")

    # Quality interpretation
    if combined >= 0.90:
        quality = "Excellent"
    elif combined >= 0.80:
        quality = "Good"
    elif combined >= 0.70:
        quality = "Fair"
    else:
        quality = "Poor"

    metrics['quality'] = quality
    logger.info(f"  Quality:             {quality}")

    # Plot if requested
    if plot:
        plot_spectrograms(ref_audio, syn_audio, reference_path.stem, synthesized_path.stem)

    return metrics


def plot_spectrograms(ref_audio: np.ndarray, syn_audio: np.ndarray,
                      ref_name: str, syn_name: str):
    """Visualize mel spectrograms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    ref_mel = compute_mel_spectrogram(ref_audio)
    syn_mel = compute_mel_spectrogram(syn_audio)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Reference mel spectrogram
    im0 = axes[0, 0].imshow(ref_mel, aspect='auto', origin='lower', cmap='magma')
    axes[0, 0].set_title(f'Reference: {ref_name}')
    axes[0, 0].set_ylabel('Mel bin')
    plt.colorbar(im0, ax=axes[0, 0], format='%+2.0f dB')

    # Synthesized mel spectrogram
    im1 = axes[0, 1].imshow(syn_mel, aspect='auto', origin='lower', cmap='magma')
    axes[0, 1].set_title(f'Synthesized: {syn_name}')
    axes[0, 1].set_ylabel('Mel bin')
    plt.colorbar(im1, ax=axes[0, 1], format='%+2.0f dB')

    # Mean mel comparison
    ref_mel_mean = ref_mel.mean(axis=1)
    syn_mel_mean = syn_mel.mean(axis=1)
    axes[1, 0].plot(ref_mel_mean, label='Reference', alpha=0.8)
    axes[1, 0].plot(syn_mel_mean, label='Synthesized', alpha=0.8)
    axes[1, 0].set_xlabel('Mel bin')
    axes[1, 0].set_ylabel('Mean energy (dB)')
    axes[1, 0].set_title('Mean Mel Spectrum (Voice Timbre)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # MFCC comparison
    ref_mfcc = compute_mfcc(ref_audio).mean(axis=1)
    syn_mfcc = compute_mfcc(syn_audio).mean(axis=1)
    x = np.arange(len(ref_mfcc))
    width = 0.35
    axes[1, 1].bar(x - width/2, ref_mfcc, width, label='Reference', alpha=0.8)
    axes[1, 1].bar(x + width/2, syn_mfcc, width, label='Synthesized', alpha=0.8)
    axes[1, 1].set_xlabel('MFCC coefficient')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Mean MFCCs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectral_comparison.png', dpi=150)
    logger.info("\nSaved comparison plot to: spectral_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate voice cloning using spectral similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Spectral Similarity Thresholds:
  0.90+  Excellent - Very close spectral match
  0.80+  Good      - Similar voice characteristics
  0.70+  Fair      - Some similarity
  <0.70  Poor      - Different spectral characteristics

Metrics:
  - Mel Similarity: Cosine similarity of mean mel spectrum (timbre)
  - MFCC Similarity: Cosine similarity of mean MFCCs (voice characteristics)
  - MFCC Std Similarity: Similarity of MFCC dynamics

Requirements:
  pip install librosa numpy
  # Or minimal: pip install scipy numpy

Examples:
  python evaluate_voice.py original_speaker.wav tts_output.wav
  python evaluate_voice.py reference.wav synthesized.wav --plot
"""
    )
    parser.add_argument("reference", type=Path, help="Reference voice audio file")
    parser.add_argument("synthesized", type=Path, help="Synthesized TTS audio file")
    parser.add_argument("--plot", action="store_true", help="Show spectrogram comparison plots")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")

    args = parser.parse_args()

    if not args.reference.exists():
        logger.error(f"Reference file not found: {args.reference}")
        sys.exit(1)
    if not args.synthesized.exists():
        logger.error(f"Synthesized file not found: {args.synthesized}")
        sys.exit(1)

    metrics = evaluate_voice_cloning(args.reference, args.synthesized, plot=args.plot)

    if args.json:
        import json
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
