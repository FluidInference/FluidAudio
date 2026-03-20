import Foundation

/// Available TTS synthesis backends.
public enum TtsBackend: Sendable {
    /// Kokoro 82M — phoneme-based, multi-voice, chunk-oriented synthesis.
    case kokoro
    /// PocketTTS — flow-matching language model, autoregressive streaming synthesis.
    case pocketTts
    /// VoxCPM 1.5 — diffusion autoregressive TTS, 44.1kHz bilingual EN/ZH with voice cloning.
    case voxCpm
}
