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
    /// Supertonic-3 v1.7.3 — 4-stage CoreML pipeline:
    /// `text_encoder → duration_predictor → vector_estimator (8-step
    /// flow-matching diffusion with CFG) → vocoder`. Multilingual
    /// (31 languages), 44.1 kHz mono output. Voice styling via per-voice
    /// JSON (`style_ttl` / `style_dp` tensors).
    case supertonic3
    /// LuxTTS (ZipVoice-Distill) — zero-shot voice cloning conditioned on a
    /// short prompt clip + transcript. 3-stage CoreML pipeline
    /// (`TextEncoder → FmDecoder ×4 anchor-Euler steps → fixed-shape Vocos
    /// vocoder`), 48 kHz mono output.
    ///
    /// English text input runs through the bundled espeak-parity G2P
    /// (`LuxTtsG2p`); pre-phonemized espeak IPA is accepted via
    /// `LuxTtsManager.synthesize(phonemes:...)`.
    case luxtts
}
