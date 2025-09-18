import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
enum PhonemeMapper {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "PhonemeMapper")

    /// Map a sequence of IPA tokens to Kokoro vocabulary tokens, filtering to `allowed`.
    /// Unknown symbols are approximated when possible; otherwise dropped.
    static func mapIPA(_ ipaTokens: [String], allowed: Set<String>) -> [String] {
        var out: [String] = []

        for token in ipaTokens {
            let expanded = expand(token, allowed: allowed)
            for part in expanded {
                if allowed.contains(part) {
                    out.append(part)
                    continue
                }

                if let mapped = mapSingle(part, allowed: allowed) {
                    out.append(mapped)
                }
            }
        }

        return out
    }

    private static let toneCharacters = CharacterSet(charactersIn: "0123456789˥˦˧˨˩ːˑ˘˙ˊˋˉˇˌˏˎˍ˔˕ˀˁ˞")

    private static func expand(_ raw: String, allowed: Set<String>) -> [String] {
        let sanitizedScalars = raw.unicodeScalars.filter { !toneCharacters.contains($0) }
        if sanitizedScalars.isEmpty { return [] }

        var sanitized = String(sanitizedScalars)
        if sanitized.isEmpty { return [] }

        sanitized = sanitized.replacingOccurrences(of: "_", with: "")
        sanitized = sanitized.replacingOccurrences(of: "'", with: "")
        sanitized = sanitized.replacingOccurrences(of: "ˈ", with: "")
        sanitized = sanitized.replacingOccurrences(of: "ˌ", with: "")

        // Normalize common affricate spellings from eSpeak output
        let replacements: [(String, String)] = [
            ("tsʰ", "ʦ"),
            ("ts", "ʦ"),
            ("tɕʰ", "ʨ"),
            ("tɕ", "ʨ"),
            ("dʑ", "ʥ"),
            ("ʂʰ", "ʂ"),
            ("ɕʰ", "ɕ"),
        ]
        for (old, new) in replacements {
            sanitized = sanitized.replacingOccurrences(of: old, with: new)
        }

        sanitized = sanitized.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sanitized.isEmpty else { return [] }

        if sanitized.contains(where: { $0.isWhitespace }) {
            return sanitized.split { $0.isWhitespace }.map { String($0) }
        }

        if allowed.contains(sanitized) {
            return [sanitized]
        }

        if sanitized.count > 1 {
            return sanitized.unicodeScalars.map { String($0) }
        }

        return [sanitized]
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
            // Mandarin retroflex and alveolo-palatal approximations
            "ɻ": "r",
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
