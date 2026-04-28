import Foundation

/// Assembled frontend output fed to `CosyVoice3Synthesizer`.
///
/// This is the intermediate state between the text frontend and the
/// LLM/Flow/HiFT pipeline — every entry point produces one of these:
///
/// - **Phase 2 (production):** `CosyVoice3TtsManager.synthesize(text:promptAssets:)`
///   builds an in-memory instance from a tokenized + embedded `lm_input_embeds`
///   plus the caller's `CosyVoice3PromptAssets`.
/// - **Phase 1 (parity harness):** `load(from:)` reads a `.safetensors` produced
///   by `mobius/models/tts/cosyvoice3/coreml/verify/export_swift_fixture.py`
///   so the Swift synthesizer can be diffed against a Python golden run.
///
/// Both paths converge on `CosyVoice3Synthesizer.synthesize(fixture:)`.
public struct CosyVoice3FrontendFixture: Sendable {
    /// LLM prefill `inputs_embeds` — shape `[1, tPre, 896]` fp32.
    public let lmInputEmbeds: [Float]
    public let tPre: Int

    /// Speech-token prompt fed to Flow's `token_total` prefix.
    public let promptSpeechIds: [Int32]

    /// Prompt mel — shape `[1, promptMelFrames, 80]` fp32.
    public let promptMel: [Float]
    public let promptMelFrames: Int

    /// Speaker embedding — shape `[1, 192]` fp32.
    public let spkEmbedding: [Float]

    /// Python-captured decoded token stream (used for seeded parity playback).
    public let decodedTokens: [Int32]

    public let seed: Int32
    public let numPromptMel: Int
    public let audioLengthSamples: Int

    public static func load(from url: URL) throws -> CosyVoice3FrontendFixture {
        let file = try SafetensorsFile(url: url)

        let lmInfo = try file.info("lm_input_embeds")
        guard
            lmInfo.dtype == .f32,
            lmInfo.shape.count == 3,
            lmInfo.shape[0] == 1,
            lmInfo.shape[2] == CosyVoice3Constants.embedDim
        else {
            throw CosyVoice3Error.invalidFixture(
                "lm_input_embeds expects [1, t_pre, 896] fp32, got shape=\(lmInfo.shape) dtype=\(lmInfo.dtype.rawValue)"
            )
        }
        let lmInputEmbeds = try file.asFloat32("lm_input_embeds")
        let tPre = lmInfo.shape[1]
        guard tPre > 0 && tPre <= CosyVoice3Constants.prefillLength else {
            throw CosyVoice3Error.prefillTooLong(tPre)
        }

        let promptIdsInfo = try file.info("llm_prompt_speech_ids")
        guard
            promptIdsInfo.shape.count == 2,
            promptIdsInfo.shape[0] == 1
        else {
            throw CosyVoice3Error.invalidFixture(
                "llm_prompt_speech_ids expects [1, N], got \(promptIdsInfo.shape)")
        }
        let promptSpeechIds = try file.asInt32("llm_prompt_speech_ids")

        let promptMelInfo = try file.info("prompt_mel")
        guard
            promptMelInfo.dtype == .f32,
            promptMelInfo.shape.count == 3,
            promptMelInfo.shape[0] == 1,
            promptMelInfo.shape[2] == CosyVoice3Constants.melBins
        else {
            throw CosyVoice3Error.invalidFixture(
                "prompt_mel expects [1, frames, 80] fp32, got \(promptMelInfo.shape)")
        }
        let promptMel = try file.asFloat32("prompt_mel")
        let promptMelFrames = promptMelInfo.shape[1]

        let spkInfo = try file.info("spk_embedding")
        guard
            spkInfo.dtype == .f32,
            spkInfo.shape == [1, CosyVoice3Constants.speakerEmbeddingDim]
        else {
            throw CosyVoice3Error.invalidFixture(
                "spk_embedding expects [1, 192] fp32, got \(spkInfo.shape)")
        }
        let spkEmbedding = try file.asFloat32("spk_embedding")

        let decodedTokens = try file.asInt32("decoded_tokens")
        let seedValue = try file.asInt32("seed").first ?? 0

        let numPromptMel = try file.asInt("num_prompt_mel")
        let audioLengthSamples = try file.asInt("audio_length_samples")

        return CosyVoice3FrontendFixture(
            lmInputEmbeds: lmInputEmbeds,
            tPre: tPre,
            promptSpeechIds: promptSpeechIds,
            promptMel: promptMel,
            promptMelFrames: promptMelFrames,
            spkEmbedding: spkEmbedding,
            decodedTokens: decodedTokens,
            seed: seedValue,
            numPromptMel: numPromptMel,
            audioLengthSamples: audioLengthSamples)
    }
}
