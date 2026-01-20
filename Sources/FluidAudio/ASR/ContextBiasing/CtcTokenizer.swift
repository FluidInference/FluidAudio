import Foundation
import OSLog
import Tokenizers

/// Type alias to disambiguate from local Tokenizer class
private typealias HFTokenizerProtocol = Tokenizers.Tokenizer

// MARK: - CTC Tokenizer

/// CTC tokenizer using HuggingFace tokenizer.json for accurate BPE tokenization.
/// This provides tokenization matching the original model training.
public final class CtcTokenizer {
    private let logger = Logger(subsystem: "com.fluidaudio", category: "CtcTokenizer")
    private let hfTokenizer: HFTokenizer

    /// Errors that can occur during tokenizer initialization
    public enum Error: Swift.Error, LocalizedError {
        case tokenizerNotFound(URL)
        case missingFile(String, URL)
        case initializationFailed(Swift.Error)

        public var errorDescription: String? {
            switch self {
            case .tokenizerNotFound(let url):
                return "tokenizer.json not found at \(url.path)"
            case .missingFile(let filename, let folder):
                return "Missing required file '\(filename)' in \(folder.path)"
            case .initializationFailed(let error):
                return "Failed to initialize HuggingFace tokenizer: \(error.localizedDescription)"
            }
        }
    }

    /// Initialize the CTC tokenizer from a specific model directory.
    /// Loads tokenizer.json from the specified directory.
    ///
    /// - Parameter modelDirectory: Directory containing tokenizer.json
    /// - Throws: `CtcTokenizer.Error` if tokenizer files cannot be loaded
    public init(modelDirectory: URL) throws {
        let tokenizerPath = modelDirectory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw Error.tokenizerNotFound(modelDirectory)
        }

        // Load HFTokenizer synchronously by blocking on the async call
        // Use a Sendable box to safely transfer result across concurrency boundary
        let resultBox = ResultBox()
        let semaphore = DispatchSemaphore(value: 0)

        Task { @Sendable in
            do {
                let tokenizer = try await HFTokenizer(modelFolder: modelDirectory)
                resultBox.set(.success(tokenizer))
            } catch {
                resultBox.set(.failure(error))
            }
            semaphore.signal()
        }

        semaphore.wait()

        switch resultBox.result {
        case .success(let tokenizer):
            self.hfTokenizer = tokenizer
            logger.info("Loaded HuggingFace tokenizer from \(modelDirectory.path)")
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

    /// Get vocabulary size.
    /// Returns 0 if vocabulary size cannot be determined.
    public func vocabSize() -> Int {
        hfTokenizer.vocabularySize ?? 0
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
            .appendingPathComponent("parakeet-ctc-110m-coreml", isDirectory: true)
    }
}

// MARK: - HuggingFace Tokenizer (Private Implementation)

/// HuggingFace tokenizer that loads tokenizer.json directly using swift-transformers.
/// This provides accurate BPE tokenization matching the original model training.
private final class HFTokenizer {
    private let logger = Logger(subsystem: "com.fluidaudio", category: "HFTokenizer")
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
            logger.info("Loaded HuggingFace tokenizer from \(modelFolder.path)")
        } catch {
            logger.error("Failed to load tokenizer: \(error.localizedDescription)")
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

    // MARK: - Vocabulary Info

    /// Get vocabulary size if available
    var vocabularySize: Int? {
        // swift-transformers doesn't expose vocab size directly
        nil
    }
}

// MARK: - Sendable Result Box

/// Thread-safe box for passing results across concurrency boundaries.
/// Uses a lock to ensure safe access from multiple threads.
private final class ResultBox: @unchecked Sendable {
    private var _result: Result<HFTokenizer, Swift.Error>?
    private let lock = NSLock()

    var result: Result<HFTokenizer, Swift.Error>? {
        lock.lock()
        defer { lock.unlock() }
        return _result
    }

    func set(_ value: Result<HFTokenizer, Swift.Error>) {
        lock.lock()
        defer { lock.unlock() }
        _result = value
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
