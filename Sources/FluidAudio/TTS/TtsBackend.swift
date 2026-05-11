import Foundation

/// Available TTS synthesis backends.
public enum TtsBackend: Sendable {
    /// PocketTTS — flow-matching language model, autoregressive streaming synthesis.
    case pocketTts
    /// laishere/kokoro 7-stage CoreML chain (ALBERT → PostAlbert → Alignment →
    /// Prosody → Noise → Vocoder → Tail) with per-stage ANE/GPU assignment.
    case kokoroAne
    /// StyleTTS2 (LibriTTS, iteration_3) — 8-stage CoreML pipeline:
    /// `text_encoder → bert → ref_encoder → fused_diffusion_sampler →
    /// duration_predictor → fused_f0n_har_source → decoder_pre →
    /// decoder_upsample`. Reference-audio-driven style; 24 kHz mono output.
    ///
    /// > Note: Phonemization mirrors Kokoro — Misaki preprocessed lexicon
    /// > (`us_lexicon_cache.json`) lookup first, BART G2P CoreML
    /// > (`G2PEncoder.mlmodelc` / `G2PDecoder.mlmodelc`) for OOV English
    /// > words. Misaki's 5-char ASCII diphthong shorthand
    /// > (`A O I Y W` → `eɪ oʊ aɪ ɔɪ aʊ`) is expanded before encoding so
    /// > the output matches the espeak IPA StyleTTS2 was trained on.
    /// > Callers with their own espeak-compatible phonemizer can bypass
    /// > the entire stack via `StyleTTS2Manager.synthesize(ipa:...)`.
    case styletts2
}
