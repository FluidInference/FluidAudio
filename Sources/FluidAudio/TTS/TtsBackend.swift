import Foundation

/// Available TTS synthesis backends.
public enum TtsBackend: Sendable {
    /// Kokoro 82M — phoneme-based, multi-voice, chunk-oriented synthesis.
    case kokoro
    /// PocketTTS — flow-matching language model, autoregressive streaming synthesis.
    case pocketTts
    /// Qwen3-TTS — large language model-based multilingual TTS (English, Chinese).
    case qwen3Tts
}
