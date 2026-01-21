import Foundation
import OSLog
@preconcurrency import Tokenizers
import os

/// Type alias to disambiguate from local Tokenizer class
private typealias HFTokenizerProtocol = Tokenizers.Tokenizer

// MARK: - CTC Tokenizer

/// CTC tokenizer using HuggingFace tokenizer.json for accurate BPE tokenization.
/// This provides tokenization matching the original model training.
public final class CtcTokenizer: Sendable {
    private let hfTokenizer: HFTokenizer

    /// Errors that can occur during tokenizer initialization
    public enum Error: Swift.Error, LocalizedError {
        case tokenizerNotFound(URL)
        case missingFile(String, URL)
        case initializationFailed(Swift.Error)
        case applicationSupportNotFound

        public var errorDescription: String? {
            switch self {
            case .tokenizerNotFound(let url):
                return "tokenizer.json not found at \(url.path)"
            case .missingFile(let filename, let folder):
                return "Missing required file '\(filename)' in \(folder.path)"
            case .initializationFailed(let error):
                return "Failed to initialize HuggingFace tokenizer: \(error.localizedDescription)"
            case .applicationSupportNotFound:
                return "Application Support directory not found"
            }
        }
    }

    // MARK: - Async Factory (Recommended)

    /// Load the CTC tokenizer asynchronously from a specific model directory.
    /// This is the recommended API as it avoids blocking.
    ///
    /// - Parameter modelDirectory: Directory containing tokenizer.json
    /// - Returns: Initialized CtcTokenizer
    /// - Throws: `CtcTokenizer.Error` if tokenizer files cannot be loaded
    public static func load(from modelDirectory: URL) async throws -> CtcTokenizer {
        let tokenizerPath = modelDirectory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw Error.tokenizerNotFound(modelDirectory)
        }

        let hfTokenizer = try await HFTokenizer(modelFolder: modelDirectory)
        return CtcTokenizer(hfTokenizer: hfTokenizer)
    }

    /// Load the CTC tokenizer asynchronously using the default 110m model directory.
    ///
    /// - Returns: Initialized CtcTokenizer
    /// - Throws: `CtcTokenizer.Error` if tokenizer files cannot be loaded
    public static func load() async throws -> CtcTokenizer {
        try await load(from: getCtcModelDirectory())
    }

    // MARK: - Private Init

    /// Private initializer used by async factory method
    private init(hfTokenizer: HFTokenizer) {
        self.hfTokenizer = hfTokenizer
    }

    // MARK: - Sync Init (Legacy)

    /// Initialize the CTC tokenizer from a specific model directory.
    /// Loads tokenizer.json from the specified directory.
    ///
    /// - Note: This blocks the calling thread. Prefer `load(from:)` async factory when possible.
    /// - Parameter modelDirectory: Directory containing tokenizer.json
    /// - Throws: `CtcTokenizer.Error` if tokenizer files cannot be loaded
    public init(modelDirectory: URL) throws {
        let tokenizerPath = modelDirectory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw Error.tokenizerNotFound(modelDirectory)
        }

        // Use OSAllocatedUnfairLock for thread-safe async bridging (properly Sendable)
        let resultLock = OSAllocatedUnfairLock<Result<HFTokenizer, Swift.Error>?>(initialState: nil)
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            do {
                let tokenizer = try await HFTokenizer(modelFolder: modelDirectory)
                resultLock.withLock { $0 = .success(tokenizer) }
            } catch {
                resultLock.withLock { $0 = .failure(error) }
            }
            semaphore.signal()
        }

        semaphore.wait()

        let result = resultLock.withLock { $0 }

        switch result {
        case .success(let tokenizer):
            self.hfTokenizer = tokenizer
        case .failure(let error):
            throw Error.initializationFailed(error)
        case .none:
            throw Error.initializationFailed(
                NSError(
                    domain: "CtcTokenizer", code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Async initialization did not complete"])
            )
        }
    }

    /// Initialize the CTC tokenizer using the default 110m model directory.
    /// Convenience initializer for backward compatibility.
    ///
    /// - Note: This blocks the calling thread. Prefer `load()` async factory when possible.
    /// - Throws: `CtcTokenizer.Error` if tokenizer files cannot be loaded
    public convenience init() throws {
        try self.init(modelDirectory: Self.getCtcModelDirectory())
    }

    /// Tokenize text into CTC token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        hfTokenizer.encode(text)
    }

    /// Decode token IDs back to text.
    ///
    /// - Parameter ids: Array of token IDs
    /// - Returns: Decoded text
    public func decode(_ ids: [Int]) -> String {
        hfTokenizer.decode(ids)
    }

    /// Get the token string for a single token ID.
    ///
    /// - Parameter id: Token ID
    /// - Returns: Token string or nil if invalid
    public func idToPiece(_ id: Int) -> String? {
        hfTokenizer.idToToken(id)
    }

    /// Get the CTC model directory path
    private static func getCtcModelDirectory() throws -> URL {
        guard
            let applicationSupportURL = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            throw Error.applicationSupportNotFound
        }
        return
            applicationSupportURL
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("parakeet-ctc-110m-coreml", isDirectory: true)
    }
}

// MARK: - HuggingFace Tokenizer (Private Implementation)

/// HuggingFace tokenizer that loads tokenizer.json directly using swift-transformers.
/// This provides accurate BPE tokenization matching the original model training.
/// Marked Sendable because it's immutable after initialization.
private final class HFTokenizer: Sendable {
    private let tokenizer: any HFTokenizerProtocol

    /// Load tokenizer from a local model folder containing tokenizer.json
    ///
    /// Required files in folder:
    /// - tokenizer.json (main tokenizer data)
    /// - tokenizer_config.json (tokenizer settings)
    ///
    /// - Parameter modelFolder: URL to folder containing tokenizer files
    init(modelFolder: URL) async throws {
        // Verify required files exist
        let tokenizerJsonPath = modelFolder.appendingPathComponent("tokenizer.json")
        let tokenizerConfigPath = modelFolder.appendingPathComponent("tokenizer_config.json")

        guard FileManager.default.fileExists(atPath: tokenizerJsonPath.path) else {
            throw CtcTokenizer.Error.missingFile("tokenizer.json", modelFolder)
        }
        guard FileManager.default.fileExists(atPath: tokenizerConfigPath.path) else {
            throw CtcTokenizer.Error.missingFile("tokenizer_config.json", modelFolder)
        }

        do {
            self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        } catch {
            throw CtcTokenizer.Error.initializationFailed(error)
        }
    }

    // MARK: - Encoding

    /// Encode text to token IDs without special tokens.
    func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    // MARK: - Decoding

    /// Decode token IDs to text, skipping special tokens.
    func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }

    /// Get the token string for a single token ID.
    func idToToken(_ id: Int) -> String? {
        let decoded = tokenizer.decode(tokens: [id], skipSpecialTokens: false)
        return decoded.isEmpty ? nil : decoded
    }
}

// MARK: - CustomVocabularyContext Extension

extension CustomVocabularyContext {
    /// Load vocabulary with CTC tokenization (JSON format)
    public static func loadWithSentencePieceTokenization(from url: URL) throws -> CustomVocabularyContext {
        let context = try Self.load(from: url)
        return try tokenizeContext(context)
    }

    /// Load vocabulary from simple text format with CTC tokenization.
    /// Format: one word per line, optionally "word: alias1, alias2, ..."
    public static func loadFromSimpleFormatWithTokenization(from url: URL) throws -> CustomVocabularyContext {
        let context = try loadFromSimpleFormat(from: url)
        return try tokenizeContext(context)
    }

    /// Add CTC token IDs to terms using the tokenizer.
    /// Also expands aliases into separate CTC detection entries.
    private static func tokenizeContext(_ context: CustomVocabularyContext) throws -> CustomVocabularyContext {
        let tokenizer = try CtcTokenizer()
        let logger = Logger(subsystem: "com.fluidaudio", category: "CustomVocabulary")

        var expandedTerms: [CustomVocabularyTerm] = []
        var tokenizedCount = 0
        var aliasExpansionCount = 0

        for term in context.terms {
            // 1. Add the canonical term with its own CTC tokens
            let canonicalTokenIds = term.ctcTokenIds ?? tokenizer.encode(term.text)
            let canonicalTerm = CustomVocabularyTerm(
                text: term.text,
                weight: term.weight,
                aliases: term.aliases,
                tokenIds: term.tokenIds,
                ctcTokenIds: canonicalTokenIds
            )
            expandedTerms.append(canonicalTerm)

            if term.ctcTokenIds == nil {
                tokenizedCount += 1
                logger.debug("Tokenized '\(term.text)': \(canonicalTokenIds)")
            }

            // 2. Expand aliases: create additional CTC detection entries
            if let aliases = term.aliases {
                for alias in aliases {
                    let aliasTokenIds = tokenizer.encode(alias)
                    let aliasTerm = CustomVocabularyTerm(
                        text: term.text,  // Canonical form for replacement
                        weight: term.weight,
                        aliases: term.aliases,
                        tokenIds: term.tokenIds,
                        ctcTokenIds: aliasTokenIds
                    )
                    expandedTerms.append(aliasTerm)
                    aliasExpansionCount += 1
                    logger.debug("Tokenized alias '\(alias)' -> '\(term.text)': \(aliasTokenIds)")
                }
            }
        }

        if tokenizedCount > 0 || aliasExpansionCount > 0 {
            logger.info(
                "Auto-tokenized \(tokenizedCount) vocabulary terms, expanded \(aliasExpansionCount) aliases")
        }

        return CustomVocabularyContext(
            terms: expandedTerms,
            minCtcScore: context.minCtcScore,
            minSimilarity: context.minSimilarity,
            minCombinedConfidence: context.minCombinedConfidence
        )
    }
}
