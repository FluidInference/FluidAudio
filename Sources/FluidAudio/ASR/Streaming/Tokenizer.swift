import Foundation

public class Tokenizer {
    private var vocab: [String: String] = [:]
    private var idToToken: [Int: String] = [:]

    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: String]

        self.vocab = json
        for (key, value) in json {
            if let id = Int(key) {
                self.idToToken[id] = value
            }
        }
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
}
