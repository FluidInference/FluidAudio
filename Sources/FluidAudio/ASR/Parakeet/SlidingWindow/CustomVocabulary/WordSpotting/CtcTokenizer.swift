import Foundation

// MARK: - CTC Tokenizer

/// CTC tokenizer using HuggingFace tokenizer.json for accurate BPE tokenization.
/// This provides tokenization matching the original model training.
public final class CtcTokenizer: Sendable {
    private let bpeTokenizer: BpeTokenizer

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

        do {
            let bpeTokenizer = try BpeTokenizer.load(from: modelDirectory)
            return CtcTokenizer(bpeTokenizer: bpeTokenizer)
        } catch {
            throw Error.initializationFailed(error)
        }
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
    private init(bpeTokenizer: BpeTokenizer) {
        self.bpeTokenizer = bpeTokenizer
    }

    // MARK: - Encoding/Decoding

    /// Tokenize text into CTC token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) -> [Int] {
        bpeTokenizer.encode(text, addSpecialTokens: false)
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
