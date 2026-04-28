@preconcurrency import CoreML
import Foundation

/// Phase 2 text frontend. Turns raw (prompt_text, tts_text) + prompt speech ids
/// into the three tensors the LLM-Prefill stage needs:
///   - `lm_input_embeds` [1, T_pre, 896] fp32
///   - `t_pre`
///   - The concatenated text token ids (for Python-side debugging parity).
///
/// Mirrors `src/text_frontend.build_frontend_inputs` but only the text path;
/// CAMPPlus speaker embedding and SpeechTokenizer prompt ids remain
/// Python-computed and shipped via `CosyVoice3PromptAssets` (see
/// `CosyVoice3TtsManager` Phase 2 API).
public final class CosyVoice3TextFrontend {

    public struct Assembled: Sendable {
        public let lmInputEmbeds: MLMultiArray  // [1, T_pre, 896] fp32
        public let tPre: Int
        public let textTokenIds: [Int32]  // prompt + tts concatenated
    }

    private let tokenizer: Qwen2BpeTokenizer
    private let embeddings: CosyVoice3TextEmbeddings

    public init(tokenizer: Qwen2BpeTokenizer, embeddings: CosyVoice3TextEmbeddings) {
        self.tokenizer = tokenizer
        self.embeddings = embeddings
    }

    /// Tokenize `prompt_text + tts_text`, look up text embeddings, concatenate
    /// with sos / task_id / prompt_speech_ids speech embeddings, and return
    /// the assembled LLM-Prefill input.
    ///
    /// - Note: `promptText` MUST contain the `<|endofprompt|>` token
    ///   (id 151646). The Python pipeline asserts this in
    ///   `cosyvoice/llm.py:478`.
    public func assemble(
        promptText: String,
        ttsText: String,
        promptSpeechIds: [Int32]
    ) throws -> Assembled {
        let promptIds = tokenizer.encode(promptText)
        let ttsIds = tokenizer.encode(ttsText)
        // Python asserts 151646 is present somewhere in the combined token
        // stream. Enforce here to avoid silent parity breakage.
        let endOfPrompt: Int32 = 151_646
        guard promptIds.contains(endOfPrompt) || ttsIds.contains(endOfPrompt) else {
            throw CosyVoice3Error.invalidShape(
                "<|endofprompt|> (id 151646) not present in promptText or ttsText")
        }
        let combined = promptIds + ttsIds

        let (embeds, tPre) = try embeddings.assembleLmInput(
            textTokenIds: combined,
            promptSpeechIds: promptSpeechIds)
        guard tPre <= CosyVoice3Constants.prefillLength else {
            throw CosyVoice3Error.invalidShape(
                "assembled T_pre=\(tPre) exceeds LLM-Prefill length \(CosyVoice3Constants.prefillLength)"
            )
        }
        return Assembled(lmInputEmbeds: embeds, tPre: tPre, textTokenIds: combined)
    }
}
