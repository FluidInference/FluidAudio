import Foundation

/// 178-token espeak-ng IPA + stress vocabulary shipped as
/// `constants/text_cleaner_vocab.json` alongside the StyleTTS2 mlpackages.
///
/// File schema (from the conversion repo):
/// ```json
/// {
///   "pad_token": "$",
///   "num_tokens": 178,
///   "symbols": [...],
///   "token_to_id": { "$": 0, ... }
/// }
/// ```
///
/// Mirrors the upstream `text_utils.TextCleaner` table at conversion time
/// (yl4579/StyleTTS2). Includes ASCII letters, punctuation, the full IPA
/// inventory used by espeak-ng, and stress/length markers (`ˈ ˌ ː ˑ`).
public struct StyleTTS2Vocab: Sendable {

    /// IPA character → token id (`Character` is an extended grapheme cluster,
    /// which correctly handles the combining vertical line below `◌̩`).
    public let map: [Character: Int32]
    public let padTokenId: Int32

    public init(map: [Character: Int32], padTokenId: Int32) {
        self.map = map
        self.padTokenId = padTokenId
    }

    /// Load from `text_cleaner_vocab.json`.
    public static func load(from url: URL) throws -> StyleTTS2Vocab {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StyleTTS2Error.modelNotFound(url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw StyleTTS2Error.invalidConfiguration(
                "\(url.lastPathComponent): expected top-level JSON object")
        }
        guard let tokenToId = json["token_to_id"] as? [String: Any] else {
            throw StyleTTS2Error.invalidConfiguration(
                "\(url.lastPathComponent): missing or invalid token_to_id field")
        }

        var parsed: [Character: Int32] = [:]
        parsed.reserveCapacity(tokenToId.count)
        for (key, value) in tokenToId {
            // The vocab includes graphemes that span multiple Unicode scalars
            // (e.g. the syllabic-consonant combining mark `◌̩`). Use
            // `Character` so we get the same boundary as Swift's grapheme
            // iteration and don't accidentally split combining marks off
            // their base.
            guard let ch = key.first, key.count == 1 else { continue }
            if let intValue = value as? Int {
                parsed[ch] = Int32(intValue)
            }
        }

        // Resolve the pad token id from the explicit `pad_token` field if
        // present; otherwise fall back to "$" → 0 which is the upstream
        // contract.
        let padId: Int32
        if let padToken = (json["pad_token"] as? String)?.first,
            let id = parsed[padToken]
        {
            padId = id
        } else {
            padId = 0
        }

        return StyleTTS2Vocab(map: parsed, padTokenId: padId)
    }

    /// Encode a phoneme string to token ids. Unknown graphemes are dropped
    /// (matches upstream `text_utils.TextCleaner.__call__`, which logs and
    /// skips unmapped characters rather than failing).
    ///
    /// No BOS/EOS wrapping — StyleTTS2's `text_predictor` is fed the raw
    /// id sequence, padded to the bucket length by the caller.
    public func encode(_ phonemes: String) -> [Int32] {
        // Iterate over Unicode scalars (not Characters) so combining marks
        // like U+0329 (syllabic) and U+0361 (tie bar) are looked up against
        // their own vocab entries instead of being grouped into a grapheme
        // cluster with their base char — which would silently drop both.
        // Mirrors upstream `text_utils.TextCleaner.__call__`, which iterates
        // codepoints.
        var ids: [Int32] = []
        ids.reserveCapacity(phonemes.unicodeScalars.count)
        for scalar in phonemes.unicodeScalars {
            if let id = map[Character(scalar)] {
                ids.append(id)
            }
        }
        return ids
    }
}
