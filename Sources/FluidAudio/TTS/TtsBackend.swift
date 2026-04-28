import Foundation

/// Available TTS synthesis backends.
public enum TtsBackend: Sendable {
    /// Kokoro 82M — phoneme-based, multi-voice, chunk-oriented synthesis.
    case kokoro
    /// PocketTTS — flow-matching language model, autoregressive streaming synthesis.
    case pocketTts
    /// CosyVoice3 — Mandarin zero-shot voice cloning via Qwen2 LM + Flow CFM + HiFT.
    ///
    /// > Note: **Experimental / beta.** End-to-end synthesis is currently
    /// > slow (RTFx < 1.0 typical on Apple Silicon). Cause is partially
    /// > in the Flow CFM stage which must run fp32 on CPU/GPU (fp16 + ANE
    /// > produces NaNs through fused `layer_norm`) and partially in HiFT
    /// > sinegen ops that fall back to CPU. May be a model issue, may be
    /// > recoverable via better conversion — treat as preliminary.
    case cosyvoice3
    /// laishere/kokoro 7-stage CoreML chain (ALBERT → PostAlbert → Alignment →
    /// Prosody → Noise → Vocoder → Tail) with per-stage ANE/GPU assignment.
    case kokoroAne
}
