import Foundation

#if canImport(Metaphone3)
import Metaphone3

/// Swift wrapper for the Metaphone3 phonetic encoding algorithm
public class Metaphone3Encoder {
    private var handle: Metaphone3Handle?

    /// Whether to encode non-initial vowels (default: true)
    public var encodeVowels: Bool = true {
        didSet {
            if let h = handle {
                metaphone3_set_encode_vowels(h, encodeVowels ? 1 : 0)
            }
        }
    }

    /// Whether to use exact encoding, distinguishing voiced/unvoiced consonants (default: true)
    public var encodeExact: Bool = true {
        didSet {
            if let h = handle {
                metaphone3_set_encode_exact(h, encodeExact ? 1 : 0)
            }
        }
    }

    /// Maximum length of the encoded key (default: 8)
    public var keyLength: UInt16 = 8 {
        didSet {
            if let h = handle {
                _ = metaphone3_set_key_length(h, keyLength)
            }
        }
    }

    /// Result of a Metaphone3 encoding operation
    public struct EncodingResult {
        /// The primary phonetic encoding
        public let metaph: String

        /// The alternate phonetic encoding (may be empty if no alternate exists)
        public let alternateMetaph: String

        /// Whether an alternate encoding exists and differs from the primary
        public var hasAlternate: Bool {
            return !alternateMetaph.isEmpty && alternateMetaph != metaph
        }
    }

    /// Initialize a new Metaphone3 encoder
    public init() {
        handle = metaphone3_create()
        if let h = handle {
            metaphone3_set_encode_vowels(h, 1)
            metaphone3_set_encode_exact(h, 1)
        }
    }

    deinit {
        if let h = handle {
            metaphone3_destroy(h)
        }
    }

    /// Encode a word using the current encoder settings
    /// - Parameter word: The word to encode
    /// - Returns: An EncodingResult containing the primary and alternate encodings
    public func encode(_ word: String) -> EncodingResult {
        guard let h = handle else {
            return EncodingResult(metaph: "", alternateMetaph: "")
        }

        var result = Metaphone3Result()
        word.withCString { cString in
            metaphone3_encode(h, cString, &result)
        }

        let metaph = withUnsafePointer(to: &result.metaph) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 33) { charPtr in
                String(cString: charPtr)
            }
        }

        let alternateMetaph = withUnsafePointer(to: &result.alternateMetaph) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 33) { charPtr in
                String(cString: charPtr)
            }
        }

        return EncodingResult(metaph: metaph, alternateMetaph: alternateMetaph)
    }

    /// Convenience method to encode a word with default settings
    /// - Parameter word: The word to encode
    /// - Returns: An EncodingResult containing the primary and alternate encodings
    public static func encode(_ word: String) -> EncodingResult {
        var result = Metaphone3Result()
        word.withCString { cString in
            metaphone3_encode_word(cString, &result)
        }

        let metaph = withUnsafePointer(to: &result.metaph) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 33) { charPtr in
                String(cString: charPtr)
            }
        }

        let alternateMetaph = withUnsafePointer(to: &result.alternateMetaph) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 33) { charPtr in
                String(cString: charPtr)
            }
        }

        return EncodingResult(metaph: metaph, alternateMetaph: alternateMetaph)
    }

    /// Check if two words have matching phonetic encodings
    /// - Parameters:
    ///   - word1: First word to compare
    ///   - word2: Second word to compare
    /// - Returns: true if any of their encodings match
    public func soundsLike(_ word1: String, _ word2: String) -> Bool {
        let result1 = encode(word1)
        let result2 = encode(word2)

        // Check if primary encodings match
        if result1.metaph == result2.metaph {
            return true
        }

        // Check if any combination of primary/alternate match
        if result1.hasAlternate && result1.alternateMetaph == result2.metaph {
            return true
        }
        if result2.hasAlternate && result1.metaph == result2.alternateMetaph {
            return true
        }
        if result1.hasAlternate && result2.hasAlternate && result1.alternateMetaph == result2.alternateMetaph {
            return true
        }

        return false
    }
}

// MARK: - String Extension for convenience
extension String {
    /// Get the Metaphone3 encoding of this string
    public var metaphone: Metaphone3Encoder.EncodingResult {
        return Metaphone3Encoder.encode(self)
    }

    /// Check if this string sounds like another string using Metaphone3
    public func soundsLike(_ other: String) -> Bool {
        let encoder = Metaphone3Encoder()
        return encoder.soundsLike(self, other)
    }
}

#endif
