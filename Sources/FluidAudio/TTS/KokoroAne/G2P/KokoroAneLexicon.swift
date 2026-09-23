import Foundation

/// Word → phoneme lexicon read from a Kokoro lexicon cache
/// (`fr_lexicon_cache.json`, `es_lexicon_cache.json`). The schema is the one
/// `us_lexicon_cache.json` uses — `{lower, caseSensitive}` mapping a word to
/// its phoneme tokens — plus an optional `hAspire` word list for French.
/// Tokens are joined back into strings at load.
struct KokoroAneLexicon: Sendable {
    private let lower: [String: String]
    private let caseSensitive: [String: String]
    private let hAspireWords: Set<String>

    private struct Payload: Decodable {
        let lower: [String: [String]]
        let caseSensitive: [String: [String]]?
        let hAspire: [String]?
    }

    init(contentsOf url: URL) throws {
        let payload: Payload
        do {
            payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        } catch {
            throw KokoroAneError.inputProcessingFailed(
                "Failed to load lexicon cache \(url.lastPathComponent): \(error.localizedDescription)")
        }
        self.init(
            entries: payload.lower.mapValues { $0.joined() },
            caseSensitive: (payload.caseSensitive ?? [:]).mapValues { $0.joined() },
            hAspire: Set(payload.hAspire ?? []))
    }

    init(entries: [String: String], caseSensitive: [String: String] = [:], hAspire: Set<String> = []) {
        lower = entries
        self.caseSensitive = caseSensitive
        hAspireWords = hAspire
    }

    static let empty = KokoroAneLexicon(entries: [:])

    var count: Int { lower.count }

    /// Exact spelling first, then the lowercased form.
    func lookup(_ word: String) -> String? {
        caseSensitive[word] ?? lower[word.lowercased()]
    }

    func contains(_ word: String) -> Bool {
        lookup(word) != nil
    }

    /// French h aspiré (héros): blocks liaison and elision.
    func isHAspire(_ word: String) -> Bool {
        hAspireWords.contains(word.lowercased())
    }
}
