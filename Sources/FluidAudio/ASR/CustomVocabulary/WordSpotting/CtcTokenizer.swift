import Foundation
@preconcurrency import Tokenizers

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

    // MARK: - Async Factory

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

    // MARK: - Encoding/Decoding

    /// Tokenize text into CTC token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        hfTokenizer.encode(text)
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

}
