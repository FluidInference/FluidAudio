import Foundation

/// KittenTTS model variant selector.
public enum KittenTtsVariant: String, CaseIterable, Sendable {
    /// KittenTTS Nano — 15M params, distilled from Kokoro-82M.
    case nano
    /// KittenTTS Mini — 80M params, StyleTTS2 with speed control.
    case mini
}

/// Available TTS synthesis backends.
public enum TtsBackend: Sendable {
    /// Kokoro 82M — phoneme-based, multi-voice, chunk-oriented synthesis.
    case kokoro
    /// PocketTTS — flow-matching language model, autoregressive streaming synthesis.
    case pocketTts
    /// KittenTTS — single-shot StyleTTS2-based synthesis (Nano 15M / Mini 80M).
    case kittenTts(KittenTtsVariant)
}
