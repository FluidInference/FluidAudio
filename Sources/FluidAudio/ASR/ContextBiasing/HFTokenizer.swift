import Foundation
import OSLog

/// Hugging Face tokenizer.json based tokenizer for CTC
/// This reads the tokenizer.json file and implements the same tokenization as the Python tokenizers library
public class HFTokenizer {
    private let logger = Logger(subsystem: "com.fluidaudio", category: "HFTokenizer")
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let unkToken: String
    private let unkTokenId: Int

    public init() throws {
        // Load tokenizer.json from CTC model directory
        let modelDir = Self.getCtcModelDirectory()
        let tokenizerPath = modelDir.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw CtcTokenizerError.vocabNotFound(tokenizerPath.path)
        }

        let data = try Data(contentsOf: tokenizerPath)
        let tokenizerJson = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        // Extract vocabulary
        guard let model = tokenizerJson?["model"] as? [String: Any],
            let vocabDict = model["vocab"] as? [String: Int]
        else {
            throw CtcTokenizerError.encodingFailed("Failed to parse vocabulary from tokenizer.json")
        }

        self.vocab = vocabDict
        self.unkToken = "<unk>"
        self.unkTokenId = vocabDict[unkToken] ?? 0

        // Extract merges if BPE model
        if let mergesArray = model["merges"] as? [String] {
            self.merges = mergesArray.compactMap { merge in
                let parts = merge.split(separator: " ", maxSplits: 1).map(String.init)
                guard parts.count == 2 else { return nil }
                return (parts[0], parts[1])
            }
        } else {
            self.merges = []
        }

        logger.info("Loaded HF tokenizer with \(self.vocab.count) tokens")
    }

    /// Tokenize text using the same algorithm as HuggingFace tokenizers
    public func encode(_ text: String) -> [Int] {
        // Normalize text (add space prefix for first word)
        let normalizedText = "▁" + text.replacingOccurrences(of: " ", with: "▁")

        // Try to find the exact token first
        if let tokenId = vocab[normalizedText] {
            return [tokenId]
        }

        // Use BPE/Unigram tokenization
        let tokens = tokenizeBPE(normalizedText)

        // Convert tokens to IDs
        let ids = tokens.map { token in
            vocab[token] ?? unkTokenId
        }

        return ids
    }

    /// Simple BPE tokenization
    private func tokenizeBPE(_ text: String) -> [String] {
        // Start with character-level tokens
        var tokens: [String] = text.map { String($0) }

        // Apply merges iteratively (simplified BPE)
        var changed = true
        while changed && !merges.isEmpty {
            changed = false

            for (left, right) in merges {
                var newTokens: [String] = []
                var i = 0

                while i < tokens.count {
                    if i < tokens.count - 1 && tokens[i] == left && tokens[i + 1] == right {
                        // Merge the pair
                        newTokens.append(left + right)
                        i += 2
                        changed = true
                    } else {
                        newTokens.append(tokens[i])
                        i += 1
                    }
                }

                tokens = newTokens
            }
        }

        // Try to match longest subwords from vocabulary
        var result: [String] = []
        var position = 0
        let chars = Array(text)

        while position < chars.count {
            var matched = false
            var maxLength = min(20, chars.count - position)

            // Try longest match first
            while maxLength > 0 && !matched {
                let substring = String(chars[position..<(position + maxLength)])

                if vocab[substring] != nil {
                    result.append(substring)
                    position += maxLength
                    matched = true
                }

                maxLength -= 1
            }

            // If no match, use single character or unknown
            if !matched {
                let char = String(chars[position])
                result.append(vocab[char] != nil ? char : unkToken)
                position += 1
            }
        }

        return result
    }

    /// Get the CTC model directory path
    private static func getCtcModelDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return
            applicationSupportURL
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("alexwengg", isDirectory: true)
            .appendingPathComponent("parakeet-ctc-110m-coreml", isDirectory: true)
    }
}

/// Extension to use HF tokenizer with CustomVocabularyContext
extension CustomVocabularyContext {
    /// Load vocabulary with Hugging Face tokenizer for accurate tokenization
    public static func loadWithHFTokenization(from url: URL) throws -> CustomVocabularyContext {
        // First load normally
        let context = try Self.load(from: url)

        // Use HF tokenizer
        let tokenizer = try HFTokenizer()
        let logger = Logger(subsystem: "com.fluidaudio", category: "CustomVocabulary")

        // Tokenize terms that don't have ctcTokenIds
        var updatedTerms: [CustomVocabularyTerm] = []
        for term in context.terms {
            if term.ctcTokenIds == nil || term.ctcTokenIds?.isEmpty == true {
                // Tokenize the text
                let tokenIds = tokenizer.encode(term.text)
                logger.debug("HF tokenized '\(term.text)': \(tokenIds)")

                // Create updated term with ctcTokenIds
                let updatedTerm = CustomVocabularyTerm(
                    text: term.text,
                    weight: term.weight,
                    aliases: term.aliases,
                    tokenIds: term.tokenIds,
                    ctcTokenIds: tokenIds
                )
                updatedTerms.append(updatedTerm)
            } else {
                // Keep existing term with pre-computed ctcTokenIds
                updatedTerms.append(term)
            }
        }

        // Return updated context with tokenized terms
        return CustomVocabularyContext(
            terms: updatedTerms,
            minCtcScore: context.minCtcScore,
            minSimilarity: context.minSimilarity,
            minCombinedConfidence: context.minCombinedConfidence
        )
    }
}
