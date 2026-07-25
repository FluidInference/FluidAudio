import Foundation

/// Inflect v2 symbol table + token encoder.
///
/// Inflect uses the keithito/Tacotron symbol set (`runtime/text/symbols.py`):
/// a pad, punctuation, ASCII letters, then the full IPA inventory. Token IDs
/// are indices into that list. The four source literals are reproduced
/// verbatim; concatenating them and enumerating Unicode *scalars* reproduces
/// Python's `[_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)`
/// exactly (confirmed byte-identical against the shipped `symbols.py`).
///
/// Two parity details that a naive port gets wrong:
///   1. Iterate `unicodeScalars`, not `Character`s. A syllabic consonant like
///      `n̩` (n + U+0329) is one grapheme but two symbols in Python — the
///      combining mark has its own id.
///   2. The apostrophe `'` appears twice in `_letters_ipa`; Python's
///      `{s: i for i, s in enumerate(symbols)}` keeps the *last* index, so the
///      scalar→id map overwrites rather than skipping duplicates.
enum InflectSymbols {

    /// keithito literals, verbatim from `runtime/text/symbols.py`.
    private static let pad = "_"
    private static let punctuation = ";:,.!?¡¿—…\"«»“” "
    private static let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    private static let lettersIPA =
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    /// Ordered symbol list; `count == 178`. Index == token id.
    static let symbols: [Unicode.Scalar] = {
        var out: [Unicode.Scalar] = []
        for literal in [pad, punctuation, letters, lettersIPA] {
            out.append(contentsOf: literal.unicodeScalars)
        }
        return out
    }()

    /// Scalar → token id. Later occurrences win (matches Python dict build).
    private static let scalarToID: [Unicode.Scalar: Int32] = {
        var map: [Unicode.Scalar: Int32] = [:]
        for (index, scalar) in symbols.enumerated() {
            map[scalar] = Int32(index)
        }
        return map
    }()

    /// Pad / blank token id (`_`), interspersed between phonemes.
    static let blankID: Int32 = 0

    /// Map a phoneme string to token ids, dropping scalars outside the symbol
    /// set (mirrors `cleaned_text_to_sequence`'s `if s in _symbol_to_id`).
    static func sequence(for phonemes: String) -> [Int32] {
        var ids: [Int32] = []
        ids.reserveCapacity(phonemes.unicodeScalars.count)
        for scalar in phonemes.unicodeScalars {
            if let id = scalarToID[scalar] {
                ids.append(id)
            }
        }
        return ids
    }

    /// Intersperse `blankID` around every token: `commons.intersperse(seq, 0)`
    /// produces `[0, t0, 0, t1, 0, …, tn, 0]` (length `2·count + 1`).
    static func intersperse(_ sequence: [Int32]) -> [Int32] {
        guard !sequence.isEmpty else { return [blankID] }
        var out: [Int32] = [blankID]
        out.reserveCapacity(sequence.count * 2 + 1)
        for token in sequence {
            out.append(token)
            out.append(blankID)
        }
        return out
    }

    /// Full text→token pipeline: `intersperse(cleaned_text_to_sequence(ipa))`.
    static func encode(_ phonemes: String) -> [Int32] {
        intersperse(sequence(for: phonemes))
    }
}
