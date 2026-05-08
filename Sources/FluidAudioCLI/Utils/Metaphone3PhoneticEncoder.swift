#if METAPHONE3_AVAILABLE
import FluidAudio
import Foundation
import Metaphone3

/// Adapter that exposes the (paid) Metaphone3 phonetic encoder through
/// FluidAudio's `PhoneticEncoder` protocol.
///
/// FluidAudio core has no Metaphone3 dependency. This file is compiled
/// only when `Frameworks/Metaphone3.xcframework` is present at the
/// package root — gated by Package.swift's `METAPHONE3_AVAILABLE`
/// compilation condition. Open-source consumers who do not own a
/// Metaphone3 license will not compile or link this file.
///
/// Use `Metaphone3PhoneticEncoder()` and pass it to
/// `VocabularyRescorer.create(..., phoneticEncoder:)` to enable the
/// rescorer's phonetic-fallback gate.
public struct Metaphone3PhoneticEncoder: PhoneticEncoder {

    public init() {}

    public func encode(_ word: String) -> String {
        let result = call(word)
        return result.metaph
    }

    public func encodeAlternate(_ word: String) -> String? {
        let result = call(word)
        return result.alternateMetaph.isEmpty ? nil : result.alternateMetaph
    }

    /// Wrapper around the static C entry point. The static call creates
    /// and destroys a Metaphone3 handle internally per invocation, so
    /// this struct can be `Sendable` without managing handle lifetime.
    private func call(_ word: String) -> (metaph: String, alternateMetaph: String) {
        var result = Metaphone3Result()
        word.withCString { cString in
            metaphone3_encode_word(cString, &result)
        }
        let primary = withUnsafePointer(to: &result.metaph) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 33) { String(cString: $0) }
        }
        let alternate = withUnsafePointer(to: &result.alternateMetaph) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 33) { String(cString: $0) }
        }
        return (primary, alternate)
    }
}
#endif
