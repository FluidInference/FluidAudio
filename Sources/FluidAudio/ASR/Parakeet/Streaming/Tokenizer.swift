import Foundation

public final class Tokenizer: Sendable {
    private let vocab: [String: String]
    private let idToToken: [Int: String]

    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: String]

        self.vocab = json
        var idToToken: [Int: String] = [:]
        for (key, value) in json {
            if let id = Int(key) {
                idToToken[id] = value
            }
        }
        self.idToToken = idToToken
    }

    public func decode(ids: [Int]) -> String {
        var text = ""
        for id in ids {
            if let token = idToToken[id] {
                text += token
            }
        }
        // Replace SentencePiece word boundary marker with space, then trim
        return text.replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    /// Returns the exact token string from vocab for a token id.
    public func rawToken(for id: Int) -> String? {
        idToToken[id]
    }

    /// Return the raw SentencePiece piece for a given token id, or `nil`
    /// if the id is not in the vocabulary. Used by callers that need
    /// the original piece text (e.g. multilingual lang-tag inspection).
    public func piece(forId id: Int) -> String? {
        return idToToken[id]
    }

    /// Full id → piece mapping. Used by overlap-merge logic that needs to
    /// classify pieces (e.g. `ChunkProcessor.spliceSafeTokenIds`).
    public var vocabulary: [Int: String] {
        idToToken
    }
}
