#if TTS
import Foundation

/// Rewrites American misaki phonemes from the BART G2P fallback (it only
/// knows the US lexicon) into misaki's British inventory, for words missing
/// from the British lexicon. Misaki itself falls back to eSpeak `en-gb`.
///
/// Measured on the 153,873 words in both misaki lexicons, these rules take
/// the mean edit distance to the British entry from 1.59 to 0.78 phonemes
/// and exact matches from 18 % to 53 %:
///   1. US-only symbols: `O` → `Q`, `ᵻ` → `ɪ`, flaps and glottal stops → `t`.
///   2. Non-rhotic `ɹ`: before a non-vowel (or at the end), `ɜɹ ɑɹ ɔɹ ɛɹ`
///      lengthen (`ɜː` …), `ɪɹ ʊɹ Iɹ Wɹ` become `ɪə` …, `əɹ` → `ə`.
///   3. `æ` → `a` (misaki's British TRAP vowel).
///   4. `ɑ` → `ɒ` unless already long (LOT; PALM words are in the lexicon).
enum MisakiBritishFallback {

    private static let vowels: Set<Character> = [
        "A", "I", "O", "Q", "W", "Y", "a", "æ", "e", "i", "o", "u", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ",
        "ᵊ", "ᵻ", "ː",
    ]
    private static let stressMarks: Set<Character> = ["ˈ", "ˌ"]

    static func fromAmerican(_ phonemes: String) -> String {
        var symbols: [Character] = phonemes.map { ch in
            switch ch {
            case "O": return "Q"
            case "ᵻ": return "ɪ"
            case "ɾ", "T", "ʔ": return "t"
            default: return ch
            }
        }
        var out: [Character] = []
        out.reserveCapacity(symbols.count)
        for (i, ch) in symbols.enumerated() {
            guard ch == "ɹ" else {
                out.append(ch)
                continue
            }
            var j = i + 1
            while j < symbols.count, stressMarks.contains(symbols[j]) { j += 1 }
            if j < symbols.count, vowels.contains(symbols[j]) {
                out.append(ch)  // linking r before a vowel stays
                continue
            }
            switch out.last {
            case "ɜ", "ɑ", "ɔ", "ɛ": out.append("ː")
            case "ɪ", "ʊ", "I", "W": out.append("ə")
            case "ə": break
            default: out.append(ch)
            }
        }
        symbols = out
        for i in symbols.indices {
            if symbols[i] == "æ" {
                symbols[i] = "a"
            } else if symbols[i] == "ɑ", i + 1 >= symbols.count || symbols[i + 1] != "ː" {
                symbols[i] = "ɒ"
            }
        }
        return String(symbols)
    }
}
#endif
