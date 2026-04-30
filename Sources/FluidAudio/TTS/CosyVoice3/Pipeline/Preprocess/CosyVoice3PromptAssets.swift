import Foundation

/// Zero-shot prompt assets bundled alongside CosyVoice3 inference.
///
/// Phase 2 keeps SpeechTokenizer and CAMPPlus Python-side: `llmPromptSpeechIds`
/// and `spkEmbedding` are precomputed from a reference prompt WAV and shipped
/// as a single safetensors file with a JSON sidecar carrying the prompt text.
/// A later phase will regenerate these on-device once the SpeechTokenizer and
/// CAMPPlus DSPs + CoreML bindings land.
///
/// The shipping layout mirrors what
/// `verify/export_swift_fixture.py` produces, so the Phase 1 fixture doubles
/// as a valid prompt-assets bundle:
///
/// ```
/// <bundle>.safetensors
///     llm_prompt_speech_ids  int32   [1, N_speech]
///     prompt_mel             float32 [1, 2*N_speech, 80]
///     spk_embedding          float32 [1, 192]
///     (any other tensors are ignored)
/// <bundle>.json
///     { "prompt_text": "...", "tts_text": "..." }
/// ```
public struct CosyVoice3PromptAssets: Sendable {

    /// Prompt text seed. MUST contain `<|endofprompt|>` (id 151646).
    public let promptText: String

    /// Discrete speech token prefix fed to Flow (`token_total[:, :N_speech]`)
    /// AND used to build the LLM prefill embed table.
    public let promptSpeechIds: [Int32]

    /// Mel frames computed from the prompt WAV (`[1, 2*N_speech, 80]` fp32).
    /// Flattened row-major `[frames * 80]`; `promptMelFrames` is the frame count.
    public let promptMel: [Float]
    public let promptMelFrames: Int

    /// CAMPPlus speaker embedding for the prompt voice (`[1, 192]` fp32).
    public let spkEmbedding: [Float]

    public init(
        promptText: String,
        promptSpeechIds: [Int32],
        promptMel: [Float],
        promptMelFrames: Int,
        spkEmbedding: [Float]
    ) {
        self.promptText = promptText
        self.promptSpeechIds = promptSpeechIds
        self.promptMel = promptMel
        self.promptMelFrames = promptMelFrames
        self.spkEmbedding = spkEmbedding
    }

    /// Load from `<bundle>.safetensors` + `<bundle>.json` sidecar.
    ///
    /// - Parameter url: URL to the `.safetensors` file. The sidecar is expected
    ///   next to it with the same basename and `.json` extension.
    public static func load(from url: URL) throws -> CosyVoice3PromptAssets {
        let file = try SafetensorsFile(url: url)

        let idsInfo = try file.info("llm_prompt_speech_ids")
        guard idsInfo.shape.count == 2, idsInfo.shape[0] == 1 else {
            throw CosyVoice3Error.invalidFixture(
                "llm_prompt_speech_ids expects [1, N], got \(idsInfo.shape)")
        }
        let promptSpeechIds = try file.asInt32("llm_prompt_speech_ids")

        let melInfo = try file.info("prompt_mel")
        guard
            melInfo.dtype == .f32,
            melInfo.shape.count == 3,
            melInfo.shape[0] == 1,
            melInfo.shape[2] == CosyVoice3Constants.melBins
        else {
            throw CosyVoice3Error.invalidFixture(
                "prompt_mel expects [1, frames, 80] fp32, got \(melInfo.shape)")
        }
        let promptMel = try file.asFloat32("prompt_mel")
        let promptMelFrames = melInfo.shape[1]

        let spkInfo = try file.info("spk_embedding")
        guard
            spkInfo.dtype == .f32,
            spkInfo.shape == [1, CosyVoice3Constants.speakerEmbeddingDim]
        else {
            throw CosyVoice3Error.invalidFixture(
                "spk_embedding expects [1, 192] fp32, got \(spkInfo.shape)")
        }
        let spkEmbedding = try file.asFloat32("spk_embedding")

        let sidecarURL = url.deletingPathExtension().appendingPathExtension("json")
        guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
            throw CosyVoice3Error.invalidFixture(
                "prompt sidecar JSON not found next to \(url.lastPathComponent) — expected \(sidecarURL.lastPathComponent)"
            )
        }
        struct Sidecar: Decodable { let prompt_text: String }
        let sidecar: Sidecar
        do {
            sidecar = try JSONDecoder().decode(
                Sidecar.self, from: try Data(contentsOf: sidecarURL))
        } catch {
            throw CosyVoice3Error.invalidFixture(
                "failed to decode \(sidecarURL.lastPathComponent): \(error)")
        }

        return CosyVoice3PromptAssets(
            promptText: sidecar.prompt_text,
            promptSpeechIds: promptSpeechIds,
            promptMel: promptMel,
            promptMelFrames: promptMelFrames,
            spkEmbedding: spkEmbedding)
    }
}
