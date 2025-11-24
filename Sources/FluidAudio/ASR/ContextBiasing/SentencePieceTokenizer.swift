import Foundation
import OSLog
import SentencePieceWrapper

/// SentencePiece-based CTC tokenizer for accurate vocabulary tokenization
public class SentencePieceCtcTokenizer {
    private let logger = Logger(subsystem: "com.fluidaudio", category: "SentencePieceCtcTokenizer")
    private let modelPath: URL
    private let processor: SentencePieceProcessor?

    public init() throws {
        // Get the CTC model directory
        let modelDir = Self.getCtcModelDirectory()

        // Prefer the CTC tokenizer model name, fallback to legacy spm.model
        let preferredModelPath = modelDir.appendingPathComponent("ctc_tokenizer.model")
        let legacyModelPath = modelDir.appendingPathComponent("spm.model")
        let spModelPath: URL
        if FileManager.default.fileExists(atPath: preferredModelPath.path) {
            spModelPath = preferredModelPath
        } else {
            spModelPath = legacyModelPath
        }

        // Check if SentencePiece model exists
        if FileManager.default.fileExists(atPath: spModelPath.path) {
            self.modelPath = spModelPath
            self.processor = spModelPath.path.withCString { path -> SentencePieceProcessor? in
                sentencepiece_create(path)
            }

            if processor != nil {
                logger.info("Loaded SentencePiece model from \(spModelPath.path)")
            } else {
                logger.warning("Failed to load SentencePiece model, falling back to simple tokenizer")
            }
        } else {
            // No SentencePiece model, will use fallback
            self.modelPath = modelDir
            self.processor = nil
            logger.info("No SentencePiece model found, using fallback tokenizer")
        }
    }

    deinit {
        if let processor = processor {
            sentencepiece_destroy(processor)
        }
    }

    /// Tokenize text into CTC token IDs using SentencePiece
    public func encode(_ text: String) -> [Int] {
        // If we have a SentencePiece processor, use it
        if let processor = processor {
            var idsPtr: UnsafeMutablePointer<Int32>?
            let count = text.withCString { textPtr -> Int32 in
                sentencepiece_encode_as_ids(processor, textPtr, &idsPtr)
            }

            guard count > 0, let ids = idsPtr else {
                logger.debug("SentencePiece encoding failed for '\(text)', using fallback")
                return fallbackEncode(text)
            }

            var result: [Int] = []
            for i in 0..<Int(count) {
                result.append(Int(ids[i]))
            }

            sentencepiece_free_ids(ids)
            return result
        } else {
            // Use fallback tokenizer
            return fallbackEncode(text)
        }
    }

    /// Fallback tokenizer when SentencePiece is not available
    /// This uses the same logic as the original CtcTokenizer
    private func fallbackEncode(_ text: String) -> [Int] {
        // Use the existing CtcTokenizer implementation
        do {
            let fallbackTokenizer = try CtcTokenizer()
            return fallbackTokenizer.encode(text)
        } catch {
            logger.error("Fallback tokenizer failed: \(error)")
            return []
        }
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

/// Extension to use SentencePiece tokenizer with CustomVocabularyContext
extension CustomVocabularyContext {
    /// Load vocabulary with SentencePiece CTC tokenization
    public static func loadWithSentencePieceTokenization(from url: URL) throws -> CustomVocabularyContext {
        // First load normally
        let context = try Self.load(from: url)

        // Try to use SentencePiece tokenizer, fall back to simple tokenizer if needed
        let tokenizer: SentencePieceCtcTokenizer
        do {
            tokenizer = try SentencePieceCtcTokenizer()
        } catch {
            // If SentencePiece fails, use the simple tokenizer
            return try loadWithCtcTokenization(from: url)
        }

        let logger = Logger(subsystem: "com.fluidaudio", category: "CustomVocabulary")

        // Tokenize terms that don't have ctcTokenIds
        var updatedTerms: [CustomVocabularyTerm] = []
        for term in context.terms {
            if term.ctcTokenIds == nil || term.ctcTokenIds?.isEmpty == true {
                // Tokenize the text using SentencePiece
                let tokenIds = tokenizer.encode(term.text)
                logger.debug("SentencePiece tokenized '\(term.text)': \(tokenIds)")

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
