import Foundation

/// Pure-Swift port of `mobius/styletts2/text_utils.py::TextCleaner`.
///
/// Maps each character of an espeak-IPA string to an integer ID using the
/// fixed `pad + punctuation + letters + IPA letters` symbol table that
/// StyleTTS2 trained against. Anything outside the table is silently
/// dropped (mirrors the upstream behaviour: it `print(text)` and skips).
public enum StyleTTS2TextCleaner {

    /// Pad symbol (id 0). The Python orchestrator always inserts a leading
    /// 0 token, which is replicated by `encode(_:)` below.
    public static let padSymbol: Character = "$"

    /// Punctuation symbols (15 chars). Includes ASCII punctuation, the em
    /// dash, ellipsis, ASCII + curly + Spanish quotes, and a literal space.
    public static let punctuation: String = ";:,.!?¡¿—…\"«»\u{201C}\u{201D} "

    /// Latin alphabet (52 chars).
    public static let letters: String =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    /// IPA letters lifted verbatim from `text_utils.py`. The two single-quote
    /// looking characters are U+0027 (apostrophe) framing
    /// U+0329 (combining vertical line below) and the `ᵻ` near-close near-front
    /// unrounded vowel — both are part of the upstream training vocabulary.
    public static let ipaLetters: String =
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'\u{0329}'ᵻ"

    /// Symbol table in the canonical training order.
    public static let symbols: [Character] =
        [padSymbol] + Array(punctuation) + Array(letters) + Array(ipaLetters)

    /// Char → ID lookup. Built once at type-init time.
    public static let dictionary: [Character: Int32] = {
        var dict: [Character: Int32] = [:]
        for (idx, ch) in symbols.enumerated() {
            // The upstream table contains a duplicate apostrophe (the
            // pre-stress mark `'\u{0329}'` actually frames a combining
            // vertical line below with apostrophes on either side). The
            // first occurrence wins — that matches the `dicts[symbols[i]]`
            // assignment loop behaviour where the second write overwrites
            // the first, but for parity we want the *last* write because
            // Python overwrites on collision. Use last-write-wins.
            dict[ch] = Int32(idx)
        }
        return dict
    }()

    /// Encode the given (already-phonemized) string into a list of IDs.
    /// Mirrors `text_utils.TextCleaner.__call__` with the leading-0
    /// insertion from `coreml/inference.py:447`.
    public static func encode(_ phonemes: String, prependPad: Bool = true) -> [Int32] {
        var ids: [Int32] = []
        ids.reserveCapacity(phonemes.count + (prependPad ? 1 : 0))
        if prependPad {
            ids.append(0)
        }
        for ch in phonemes {
            if let id = dictionary[ch] {
                ids.append(id)
            }
            // Silently drop unknown characters — same as the upstream
            // `print(text)` + `continue` branch.
        }
        return ids
    }

    /// Total symbol count (used by tests + as a sanity check on shape).
    public static var vocabularySize: Int { symbols.count }
}
