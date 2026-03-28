import Foundation

/// BPE tokenizer for CTC vocabulary boosting.
/// Only implements encoding - no decoding, chat templates, or other features.
/// Supports the specific tokenizer.json format used by Parakeet models.
public final class BpeTokenizer: Sendable {
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let addedTokens: [String: Int]

    public enum Error: Swift.Error, LocalizedError {
        case fileNotFound(URL)
        case invalidJSON(String)
        case missingField(String)
        case unsupportedTokenizerType(String)

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let url):
                return "tokenizer.json not found at \(url.path)"
            case .invalidJSON(let message):
                return "Invalid JSON: \(message)"
            case .missingField(let field):
                return "Missing required field: \(field)"
            case .unsupportedTokenizerType(let type):
                return "Unsupported tokenizer type: \(type). Only 'BPE' is supported."
            }
        }
    }

    /// Load tokenizer from a folder containing tokenizer.json
    public static func load(from modelFolder: URL) throws -> BpeTokenizer {
        let tokenizerPath = modelFolder.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw Error.fileNotFound(tokenizerPath)
        }

        let data = try Data(contentsOf: tokenizerPath)

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Error.invalidJSON("Root is not a dictionary")
        }

        // Parse model section
        guard let model = json["model"] as? [String: Any] else {
            throw Error.missingField("model")
        }

        guard let modelType = model["type"] as? String else {
            throw Error.missingField("model.type")
        }

        guard modelType == "BPE" else {
            throw Error.unsupportedTokenizerType(modelType)
        }

        // Parse vocabulary: {"token": id, ...}
        guard let vocabDict = model["vocab"] as? [String: Int] else {
            throw Error.missingField("model.vocab")
        }

        // Parse merges: ["a b", "c d", ...]
        guard let mergesArray = model["merges"] as? [String] else {
            throw Error.missingField("model.merges")
        }

        let merges = mergesArray.compactMap { mergeStr -> (String, String)? in
            let parts = mergeStr.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { return nil }
            return (String(parts[0]), String(parts[1]))
        }

        // Parse added_tokens (special tokens like <unk>, <pad>)
        var addedTokensDict: [String: Int] = [:]
        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for token in addedTokens {
                if let content = token["content"] as? String,
                    let id = token["id"] as? Int
                {
                    addedTokensDict[content] = id
                }
            }
        }

        return BpeTokenizer(
            vocab: vocabDict,
            merges: merges,
            addedTokens: addedTokensDict
        )
    }

    private init(vocab: [String: Int], merges: [(String, String)], addedTokens: [String: Int]) {
        self.vocab = vocab
        self.merges = merges
        self.addedTokens = addedTokens
    }

    /// Encode text to token IDs using BPE
    public func encode(_ text: String, addSpecialTokens: Bool = false) -> [Int] {
        // Pre-tokenize: replace spaces with ▁ (sentencepiece style)
        let preprocessed = "▁" + text.replacingOccurrences(of: " ", with: "▁")

        // Split into characters
        var word = preprocessed.map { String($0) }

        // Apply BPE merges iteratively
        while true {
            // Find the highest priority merge (earliest in merges list)
            var bestMerge: (index: Int, mergeIndex: Int)? = nil

            for i in 0..<word.count - 1 {
                let pair = (word[i], word[i + 1])

                // Check if this pair has a merge rule
                if let mergeIndex = merges.firstIndex(where: { $0.0 == pair.0 && $0.1 == pair.1 }) {
                    if bestMerge == nil || mergeIndex < bestMerge!.mergeIndex {
                        bestMerge = (i, mergeIndex)
                    }
                }
            }

            // No more merges possible
            guard let (index, _) = bestMerge else { break }

            // Apply the merge
            let merged = word[index] + word[index + 1]
            word[index] = merged
            word.remove(at: index + 1)
        }

        // Convert tokens to IDs
        return word.compactMap { token -> Int? in
            // Check added tokens first (special tokens)
            if let id = addedTokens[token] {
                return id
            }
            // Then check vocabulary
            if let id = vocab[token] {
                return id
            }
            // Unknown token - return <unk> ID if available
            return addedTokens["<unk>"] ?? vocab["<unk>"] ?? 0
        }
    }
}
