import Foundation

/// Phoneme-token table for LuxTTS (`tokens.txt`, EmiliaTokenizer format:
/// one `{token}\t{id}` pair per line).
///
/// The English token set is per-character espeak IPA (stress marks, length
/// marks and combining diacritics are separate single-scalar tokens), so a
/// raw espeak phoneme string maps to ids scalar-by-scalar. Multi-character
/// entries (the pinyin initial/final tokens for Mandarin) are only reachable
/// through `tokenIds(tokens:)`.
public struct LuxTtsTokenizer: Sendable {

    private static let logger = AppLogger(category: "LuxTtsTokenizer")

    public let tokenToId: [String: Int]
    public let padId: Int
    public var vocabSize: Int { tokenToId.count }

    public init(tokensFileURL: URL) throws {
        let content: String
        do {
            content = try String(contentsOf: tokensFileURL, encoding: .utf8)
        } catch {
            throw LuxTtsError.tokenizerFailed(
                "cannot read \(tokensFileURL.path): \(error.localizedDescription)")
        }

        var map: [String: Int] = [:]
        for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
            // Split on the FIRST tab only: the space token line is a literal
            // space character followed by "\t3".
            guard let tab = line.firstIndex(of: "\t"), let id = Int(line[line.index(after: tab)...])
            else {
                throw LuxTtsError.tokenizerFailed("malformed tokens.txt line: \(line)")
            }
            let token = String(line[..<tab])
            guard map[token] == nil else {
                throw LuxTtsError.tokenizerFailed("duplicate token: \(token)")
            }
            map[token] = id
        }
        guard let pad = map["_"] else {
            throw LuxTtsError.tokenizerFailed("tokens.txt has no pad token '_'")
        }
        self.tokenToId = map
        self.padId = pad
    }

    /// Map an espeak-style IPA phoneme string to token ids, one Unicode
    /// scalar at a time (mirrors upstream: OOV scalars are skipped).
    public func tokenIds(phonemes: String) -> [Int] {
        var ids: [Int] = []
        ids.reserveCapacity(phonemes.unicodeScalars.count)
        var skipped: Set<String> = []
        for scalar in phonemes.unicodeScalars {
            let token = String(scalar)
            if let id = tokenToId[token] {
                ids.append(id)
            } else {
                skipped.insert(token)
            }
        }
        if !skipped.isEmpty {
            Self.logger.warning("Skipped OOV phoneme tokens: \(skipped.sorted().joined(separator: " "))")
        }
        return ids
    }

    /// Map explicit token strings (e.g. pinyin initials/finals) to ids,
    /// skipping OOV entries like upstream.
    public func tokenIds(tokens: [String]) -> [Int] {
        tokens.compactMap { tokenToId[$0] }
    }
}
