import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
enum PhonemeMapper {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "PhonemeMapper")

    /// Map a sequence of IPA tokens to Kokoro vocabulary tokens, filtering to `allowed`.
    /// Unknown symbols are approximated when possible; otherwise dropped.
    static func mapIPA(_ ipaTokens: [String], allowed: Set<String>) -> [String] {
        var out: [String] = []

        for (i, p) in ipaTokens.enumerated() {
            if allowed.contains(p) {
                out.append(p)
                continue
            }

            // Try pair merge for affricates represented as separate chars
            if i + 1 < ipaTokens.count {
                let pair = p + ipaTokens[i + 1]
                if let mapped = mapSingle(pair, allowed: allowed) {
                    out.append(mapped)
                    continue
                }
            }

            if let mapped = mapSingle(p, allowed: allowed) {
                out.append(mapped)
            }
        }

        return out
    }

    private static func mapSingle(_ raw: String, allowed: Set<String>) -> String? {
        // If stress/length/diacritics are used and in vocab, pass-through
        if allowed.contains(raw) { return raw }

        // Normalize some IPA to approximate Kokoro inventory
        let ipaToKokoro: [String: String] = [
            // Affricates
            "t͡ʃ": "ʧ", "tʃ": "ʧ", "d͡ʒ": "ʤ", "dʒ": "ʤ",
            // Fricatives
            "ʃ": "ʃ", "ʒ": "ʒ", "θ": "θ", "ð": "ð",
            // Approximants / alveolars
            "ɹ": "r", "ɾ": "t", "ɫ": "l",
            // Nasals
            "ŋ": "ŋ",
            // Vowels
            "æ": "æ", "ɑ": "ɑ", "ɒ": "ɑ", "ʌ": "ʌ",
            "ɪ": "ɪ", "i": "i", "ʊ": "ʊ", "u": "u",
            "ə": "ə", "ɚ": "ɚ", "ɝ": "ɝ",
            "ɛ": "ɛ", "e": "e", "o": "o", "ɔ": "ɔ",
            // Diphthongs
            "eɪ": "e", "oʊ": "o", "aɪ": "a", "aʊ": "a", "ɔɪ": "ɔ",
        ]

        if let mapped = ipaToKokoro[raw], allowed.contains(mapped) { return mapped }

        // Simple latin fallback: map ascii letters and digits if they exist
        if raw.count == 1, let ch = raw.unicodeScalars.first {
            if CharacterSet.letters.contains(ch) || CharacterSet.decimalDigits.contains(ch) {
                let s = String(raw)
                if allowed.contains(s) { return s }
            }
        }
        return nil
    }
}
