import Foundation

/// IPA phoneme → token id mapping shipped as `vocab.json` alongside the
/// 7 mlmodelc bundles.
///
/// The file is the `vocab` field of Kokoro's HF config, ~177 entries.
public struct KokoroAneVocab: Sendable {

    public let map: [Character: Int32]

    public init(map: [Character: Int32]) {
        self.map = map
    }

    /// Load from a JSON file. The expected format is an object whose keys are
    /// single-character IPA strings and whose values are integers.
    public static func load(from url: URL) throws -> KokoroAneVocab {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KokoroAneError.vocabMissing(url)
        }
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw KokoroAneError.vocabParseFailed(url, "expected top-level JSON object")
        }
        var parsed: [Character: Int32] = [:]
        parsed.reserveCapacity(json.count)
        for (key, value) in json {
            guard let ch = key.first, key.count == 1 else { continue }
            if let intValue = value as? Int {
                parsed[ch] = Int32(intValue)
            }
        }
        return KokoroAneVocab(map: parsed)
    }

    /// Phoneme length as the Python reference counts it (`len(ps)`): Unicode
    /// scalars, so a nasal vowel such as `ɑ̃` is two symbols, not one Character.
    public static func phonemeLength(_ phonemes: String) -> Int {
        phonemes.unicodeScalars.count
    }

    /// Encode an IPA phoneme string into `[BOS, ...ids, EOS]` int32 tokens.
    /// Phonemes not in the vocab are silently dropped (matches the Python
    /// reference: `filter(lambda i: i is not None, map(lambda p: vocab.get(p), ps))`).
    /// Iterates by Unicode scalar like the reference, so combining marks
    /// (the U+0303 nasal tilde in French/Portuguese IPA) map to their own id.
    public func encode(_ phonemes: String) throws -> [Int32] {
        let length = Self.phonemeLength(phonemes)
        if length > KokoroAneConstants.maxPhonemeLength {
            throw KokoroAneError.phonemeSequenceTooLong(length)
        }
        var ids: [Int32] = []
        ids.reserveCapacity(length + 2)
        ids.append(KokoroAneConstants.bosTokenId)
        for scalar in phonemes.unicodeScalars {
            if let id = map[Character(scalar)] {
                ids.append(id)
            }
        }
        ids.append(KokoroAneConstants.eosTokenId)
        return ids
    }
}
