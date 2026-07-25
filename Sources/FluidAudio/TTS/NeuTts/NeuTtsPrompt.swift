import Foundation

/// Prompt construction for NeuTTS-2E, mirroring upstream
/// `neutts.NeuTTS._apply_chat_template` (BPE input format):
///
///     <|TEXT_PROMPT_START|> {ref_text ids} [<|EMOTION|>] {text ids}
///     <|TEXT_PROMPT_END|> <|SPEECH_GENERATION_START|> {ref speech-code ids}
///
/// Generation then samples `<|speech_N|>` ids until `<|SPEECH_GENERATION_END|>`.
struct NeuTtsPrompt {

    /// Pre-encoded speaker reference (codes + transcript), from `samples/<name>.json`.
    struct SpeakerReference: Codable {
        let codes: [Int]
        let text: String
    }

    enum PromptError: Error, LocalizedError {
        case unknownSpeaker(String)
        case unknownEmotion(String)
        case missingSpecialToken(String)
        case promptTooLong(Int, limit: Int)

        var errorDescription: String? {
            switch self {
            case .unknownSpeaker(let s):
                return "Unknown speaker '\(s)'. Available: \(NeuTtsConstants.speakers.joined(separator: ", "))"
            case .unknownEmotion(let e):
                return "Unknown emotion '\(e)'. Available: \(NeuTtsConstants.emotions.joined(separator: ", "))"
            case .missingSpecialToken(let t):
                return "Special token \(t) missing from tokenizer vocabulary"
            case .promptTooLong(let n, let limit):
                return "Prompt is \(n) tokens; prefill window is \(limit). Use shorter text."
            }
        }
    }

    /// NFKC + curly-quote normalization, mirroring upstream `_normalize_text`.
    static func normalize(_ text: String) -> String {
        let mapped =
            text
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
        return mapped.precomposedStringWithCompatibilityMapping
    }

    /// Full prompt token ids for (text, speaker, emotion).
    static func buildIds(
        tokenizer: NeuTtsBpeTokenizer,
        text: String,
        reference: SpeakerReference,
        emotion: String
    ) throws -> [Int] {
        guard NeuTtsConstants.emotions.contains(emotion) else {
            throw PromptError.unknownEmotion(emotion)
        }

        func special(_ name: String) throws -> Int {
            guard let id = tokenizer.tokenId(name) else {
                throw PromptError.missingSpecialToken(name)
            }
            return id
        }

        let refText = normalize(reference.text)
        let mainText = normalize(text)

        let inputIds: [Int]
        if emotion == "neutral" {
            // Single-pass encode so BPE resolves the boundary the same way
            // upstream does.
            inputIds = tokenizer.encode("\(refText) \(mainText)")
        } else {
            let emotionId = try special("<|\(emotion.uppercased())|>")
            inputIds = tokenizer.encode(refText) + [emotionId] + tokenizer.encode(mainText)
        }

        let speechBase = try special("<|speech_0|>")
        let codeIds = reference.codes.map { speechBase + $0 }

        let ids =
            [try special("<|TEXT_PROMPT_START|>")]
            + inputIds
            + [try special("<|TEXT_PROMPT_END|>")]
            + [try special("<|SPEECH_GENERATION_START|>")]
            + codeIds

        guard ids.count <= NeuTtsConstants.prefillLength else {
            throw PromptError.promptTooLong(ids.count, limit: NeuTtsConstants.prefillLength)
        }
        return ids
    }

    /// Map generated ids back to NeuCodec code indices (speech token ids are
    /// contiguous starting at `<|speech_0|>`).
    static func extractCodes(tokenizer: NeuTtsBpeTokenizer, ids: [Int]) throws -> [Int] {
        guard let base = tokenizer.tokenId("<|speech_0|>") else {
            throw PromptError.missingSpecialToken("<|speech_0|>")
        }
        return ids.compactMap { id in
            let code = id - base
            return (0..<NeuTtsConstants.codebookSize).contains(code) ? code : nil
        }
    }
}
