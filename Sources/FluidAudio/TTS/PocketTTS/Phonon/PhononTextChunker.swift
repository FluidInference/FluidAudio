#if TTS
import Foundation

/// The tokenizer operations the Phonon chunker needs: the same encode and decode pair as the
/// `ptts::Tokenizer` trait.
public protocol PhononChunkTokenizer {
    func encode(_ text: String) throws -> [Int]
    func decode(_ tokens: [Int]) throws -> String
}

/// Sentence-aligned chunking for Phonon TTS, ported from `split_into_best_sentences` in
/// `ptts/src/tts_model.rs`.
///
/// The text is prepared (``PhononTextNormalizer/prepareTextPrompt(_:)``) and tokenized, split
/// after runs of end-of-sentence tokens (`.`, `!`, `...`, `?`), and whole sentences are grouped
/// greedily into chunks of at most `maxTokens` tokens. A single sentence longer than that stays
/// one chunk. Upstream normalizes the whole input before chunking; ``chunk(_:tokenizer:maxTokens:)``
/// does both.
public enum PhononTextChunker {

    /// Default token budget per chunk (`MAX_TOKENS_PER_CHUNK`).
    public static let maxTokensPerChunk = 50

    /// Normalize `text` with ``PhononTextNormalizer`` and split it into chunks, as the Phonon
    /// frontends do before preparing and tokenizing each chunk.
    public static func chunk<T: PhononChunkTokenizer>(
        _ text: String, tokenizer: T, maxTokens: Int = maxTokensPerChunk
    ) throws -> [String] {
        try splitIntoBestSentences(PhononTextNormalizer.normalize(text), tokenizer: tokenizer, maxTokens: maxTokens)
    }

    /// Split already normalized `text` into sentence-aligned chunks of at most `maxTokens`
    /// tokens each (`split_into_best_sentences`).
    public static func splitIntoBestSentences<T: PhononChunkTokenizer>(
        _ text: String, tokenizer: T, maxTokens: Int = maxTokensPerChunk
    ) throws -> [String] {
        let prepared = PhononScalar.trimWhitespace(PhononTextNormalizer.prepareTextPrompt(text).text)
        let tokens = try tokenizer.encode(prepared)

        // End-of-sentence token ids: tokenize ".!...?" and skip the first token, which carries
        // the leading space marker of a SentencePiece model.
        let markerTokens = try tokenizer.encode(".!...?")
        let eosTokens = Set(markerTokens.count > 1 ? Array(markerTokens.dropFirst()) : markerTokens)

        // Sentence boundaries: where a non-EOS token follows one or more EOS tokens.
        var boundaries = [0]
        var previousWasEos = false
        for (index, token) in tokens.enumerated() {
            if eosTokens.contains(token) {
                previousWasEos = true
            } else {
                if previousWasEos {
                    boundaries.append(index)
                }
                previousWasEos = false
            }
        }
        boundaries.append(tokens.count)

        var sentences: [(count: Int, text: String)] = []
        for i in 0..<(boundaries.count - 1) {
            let (start, end) = (boundaries[i], boundaries[i + 1])
            sentences.append((end - start, try tokenizer.decode(Array(tokens[start..<end]))))
        }

        // Greedily group sentences into chunks that stay under maxTokens.
        var chunks: [String] = []
        var current = ""
        var currentCount = 0
        for (count, sentence) in sentences {
            if current.isEmpty {
                current = sentence
                currentCount = count
                continue
            }
            if currentCount + count > maxTokens {
                chunks.append(PhononScalar.trimWhitespace(current))
                current = sentence
                currentCount = count
            } else {
                current += " " + sentence
                currentCount += count
            }
        }
        if !current.isEmpty {
            chunks.append(PhononScalar.trimWhitespace(current))
        }
        return chunks
    }
}
#endif
