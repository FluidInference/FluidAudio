import Foundation
import OSLog

/// CTC tokenizer wrapper for automatic vocabulary tokenization
public class CtcTokenizer {
    private let logger = Logger(subsystem: "com.fluidaudio", category: "CtcTokenizer")
    private let vocabPath: URL
    private let vocabulary: [Int: String]

    public init() throws {
        // Get the CTC model directory
        let modelDir = Self.getCtcModelDirectory()
        self.vocabPath = modelDir.appendingPathComponent("vocab.json")

        // Load vocabulary for tokenization
        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw CtcTokenizerError.vocabNotFound(vocabPath.path)
        }

        let data = try Data(contentsOf: vocabPath)
        let vocabDict = try JSONDecoder().decode([String: String].self, from: data)

        // Convert string keys to Int
        var vocab: [Int: String] = [:]
        for (key, value) in vocabDict {
            if let intKey = Int(key) {
                vocab[intKey] = value
            }
        }
        self.vocabulary = vocab
        logger.info("Loaded CTC vocabulary with \(vocab.count) tokens")
    }

    /// Tokenize text into CTC token IDs
    /// This is a simplified tokenizer that matches subwords/characters from the vocabulary
    public func encode(_ text: String) -> [Int] {
        // Normalize text to lowercase (CTC models typically use lowercase)
        let normalizedText = text.lowercased()
        var result: [Int] = []
        var position = normalizedText.startIndex

        while position < normalizedText.endIndex {
            var matched = false
            var matchLength = min(20, normalizedText.distance(from: position, to: normalizedText.endIndex))

            // Try to match longest possible subword first
            while matchLength > 0 {
                let endPos = normalizedText.index(position, offsetBy: matchLength)
                let substring = String(normalizedText[position..<endPos])

                // Check with space prefix (SentencePiece style)
                let withSpace = position == normalizedText.startIndex ? "▁" + substring : substring

                // Find token ID for this substring
                if let tokenId = findTokenId(for: withSpace) {
                    result.append(tokenId)
                    position = endPos
                    matched = true
                    break
                } else if let tokenId = findTokenId(for: substring) {
                    result.append(tokenId)
                    position = endPos
                    matched = true
                    break
                }

                matchLength -= 1
            }

            // If no match found, try single character
            if !matched {
                let char = String(normalizedText[position])
                if let tokenId = findTokenId(for: char) {
                    result.append(tokenId)
                } else {
                    // Unknown character, use <unk> token (typically 0)
                    result.append(0)
                    logger.debug("Unknown character '\(char)' in text '\(text)'")
                }
                position = normalizedText.index(after: position)
            }
        }

        return result
    }

    /// Find token ID for a given string in the vocabulary
    private func findTokenId(for token: String) -> Int? {
        for (id, vocabToken) in vocabulary {
            if vocabToken == token {
                return id
            }
        }
        return nil
    }

    /// Get the CTC model directory path
    private static func getCtcModelDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return applicationSupportURL
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("ctckit-pro", isDirectory: true)
            .appendingPathComponent("parakeet-tdt_ctc-110m", isDirectory: true)
    }
}

/// Errors for CTC tokenizer
public enum CtcTokenizerError: Error {
    case vocabNotFound(String)
    case encodingFailed(String)

    public var localizedDescription: String {
        switch self {
        case .vocabNotFound(let path):
            return "CTC vocabulary not found at: \(path)"
        case .encodingFailed(let reason):
            return "Tokenization failed: \(reason)"
        }
    }
}

/// Extension to CustomVocabularyContext to add automatic CTC tokenization
extension CustomVocabularyContext {
    /// Load vocabulary with automatic CTC tokenization
    public static func loadWithCtcTokenization(from url: URL) throws -> CustomVocabularyContext {
        // First load normally
        var context = try Self.load(from: url)

        // Initialize CTC tokenizer
        let tokenizer = try CtcTokenizer()
        let logger = Logger(subsystem: "com.fluidaudio", category: "CustomVocabulary")

        // Tokenize terms that don't have ctcTokenIds
        var updatedTerms: [CustomVocabularyTerm] = []
        for term in context.terms {
            if term.ctcTokenIds == nil || term.ctcTokenIds?.isEmpty == true {
                // Tokenize the text
                let tokenIds = tokenizer.encode(term.text)
                logger.debug("Auto-tokenized '\(term.text)': \(tokenIds)")

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
            alpha: context.alpha,
            contextScore: context.contextScore,
            depthScaling: context.depthScaling,
            scorePerPhrase: context.scorePerPhrase,
            minCtcScore: context.minCtcScore,
            minSimilarity: context.minSimilarity,
            minCombinedConfidence: context.minCombinedConfidence
        )
    }
}