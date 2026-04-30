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

    /// Encode an IPA phoneme string into `[BOS, ...ids, EOS]` int32 tokens.
    /// Phonemes not in the vocab are silently dropped (matches the Python
    /// reference: `filter(lambda i: i is not None, map(lambda p: vocab.get(p), ps))`).
    public func encode(_ phonemes: String) throws -> [Int32] {
        if phonemes.count > KokoroAneConstants.maxPhonemeLength {
            throw KokoroAneError.phonemeSequenceTooLong(phonemes.count)
        }
        var ids: [Int32] = []
        ids.reserveCapacity(phonemes.count + 2)
        ids.append(KokoroAneConstants.bosTokenId)
        for ch in phonemes {
            if let id = map[ch] {
                ids.append(id)
            }
        }
        ids.append(KokoroAneConstants.eosTokenId)
        return ids
    }
}
