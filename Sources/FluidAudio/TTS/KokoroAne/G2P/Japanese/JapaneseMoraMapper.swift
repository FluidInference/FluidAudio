import Foundation

/// Converts OpenJTalk katakana moras to Kokoro's Japanese IPA alphabet.
///
/// The table follows Misaki's Cutlet frontend (MIT), which is the frontend
/// used for the Japanese Kokoro training/benchmark input. OpenJTalk supplies
/// the contextual reading; this type deliberately does not emit OpenJTalk's
/// pitch annotations because `_`, `^`, and `-` are absent from ANE-ja's vocab.
enum JapaneseMoraMapper {
    static func phonemize(_ moras: [String]) throws -> String {
        var output = ""
        for index in moras.indices {
            let mora = moras[index]
            if mora == "ン" {
                output += nasal(before: moras[safe: index + 1].flatMap { table[$0] })
                continue
            }
            if mora == "ッ" {
                output += "ʔ"
                continue
            }
            if mora == "ー" {
                output += "ː"
                continue
            }
            guard let mapped = table[mora] else {
                throw KokoroAneError.inputProcessingFailed(
                    "OpenJTalk emitted unsupported Japanese mora '\(mora)'.")
            }

            // VOICEVOX represents a long-vowel mark as a repeated vowel mora
            // (`キョー` -> `キョ`, `オ`). Kokoro/Misaki represents it as `ː`.
            if mapped.count == 1, let vowel = mapped.first, vowels.contains(vowel),
                output.last == vowel || (vowel == "ɯ" && output.last == "ɨ")
            {
                output += "ː"
            } else {
                output += mapped
            }
        }
        return output
    }

    private static func nasal(before next: String?) -> String {
        guard let next else { return "ɴ" }
        if next.hasPrefix("m") || next.hasPrefix("p") || next.hasPrefix("b") { return "m" }
        if next.hasPrefix("k") || next.hasPrefix("ɡ") { return "ŋ" }
        if next.hasPrefix("ɲ") || next.hasPrefix("ʨ") || next.hasPrefix("ʥ") { return "ɲ" }
        if ["n", "t", "d", "ɾ", "z", "ʣ"].contains(where: next.hasPrefix) { return "n" }
        return "ɴ"
    }

    private static let vowels: Set<Character> = ["a", "i", "ɨ", "ɯ", "e", "o"]

    // Adapted from hexgrad/misaki `misaki/cutlet.py` (MIT).
    private static let table: [String: String] = [
        "ァ": "a", "ア": "a", "ィ": "i", "イ": "i", "ゥ": "ɯ", "ウ": "ɯ", "ェ": "e", "エ": "e",
        "ォ": "o", "オ": "o", "カ": "ka", "ガ": "ɡa", "キ": "kʲi", "ギ": "ɡʲi", "ク": "kɯ", "グ": "ɡɯ",
        "ケ": "ke", "ゲ": "ɡe", "コ": "ko", "ゴ": "ɡo", "サ": "sa", "ザ": "ʣa", "シ": "ɕi", "ジ": "ʥi",
        "ス": "sɨ", "ズ": "zɨ", "セ": "se", "ゼ": "ʣe", "ソ": "so", "ゾ": "ʣo", "タ": "ta", "ダ": "da",
        "チ": "ʨi", "ヂ": "ʥi", "ツ": "ʦɨ", "ヅ": "zɨ", "テ": "te", "デ": "de", "ト": "to", "ド": "do",
        "ナ": "na", "ニ": "ɲi", "ヌ": "nɯ", "ネ": "ne", "ノ": "no", "ハ": "ha", "バ": "ba", "パ": "pa",
        "ヒ": "çi", "ビ": "bʲi", "ピ": "pʲi", "フ": "ɸɯ", "ブ": "bɯ", "プ": "pɯ", "ヘ": "he", "ベ": "be",
        "ペ": "pe", "ホ": "ho", "ボ": "bo", "ポ": "po", "マ": "ma", "ミ": "mʲi", "ム": "mɯ", "メ": "me",
        "モ": "mo", "ャ": "ja", "ヤ": "ja", "ュ": "jɯ", "ユ": "jɯ", "ョ": "jo", "ヨ": "jo", "ラ": "ɾa",
        "リ": "ɾʲi", "ル": "ɾɯ", "レ": "ɾe", "ロ": "ɾo", "ヮ": "βa", "ワ": "βa", "ヰ": "i", "ヱ": "e",
        "ヲ": "o", "ヴ": "vɯ", "ヵ": "ka", "ヶ": "ke", "ヷ": "va", "ヸ": "vʲi", "ヹ": "ve", "ヺ": "vo",

        "イェ": "je", "ウィ": "βi", "ウェ": "βe", "ウォ": "βo", "キェ": "kʲe", "キャ": "kʲa", "キュ": "kʲɨ",
        "キョ": "kʲo", "ギャ": "ɡʲa", "ギュ": "ɡʲɨ", "ギョ": "ɡʲo", "クァ": "kᵝa", "クィ": "kᵝi",
        "クェ": "kᵝe", "クォ": "kᵝo", "グァ": "ɡᵝa", "グィ": "ɡᵝi", "グェ": "ɡᵝe", "グォ": "ɡᵝo",
        "シェ": "ɕe", "シャ": "ɕa", "シュ": "ɕɨ", "ショ": "ɕo", "ジェ": "ʥe", "ジャ": "ʥa", "ジュ": "ʥɨ",
        "ジョ": "ʥo", "チェ": "ʨe", "チャ": "ʨa", "チュ": "ʨɨ", "チョ": "ʨo", "ヂャ": "ʥa", "ヂュ": "ʥɨ",
        "ヂョ": "ʥo", "ツァ": "ʦa", "ツィ": "ʦʲi", "ツェ": "ʦe", "ツォ": "ʦo", "ティ": "tʲi", "テュ": "tʲɨ",
        "ディ": "dʲi", "デュ": "dʲɨ", "トゥ": "tɯ", "ドゥ": "dɯ", "ニェ": "ɲe", "ニャ": "ɲa", "ニュ": "ɲɨ",
        "ニョ": "ɲo", "ヒェ": "çe", "ヒャ": "ça", "ヒュ": "çɨ", "ヒョ": "ço", "ビャ": "bʲa", "ビュ": "bʲɨ",
        "ビョ": "bʲo", "ピャ": "pʲa", "ピュ": "pʲɨ", "ピョ": "pʲo", "ファ": "ɸa", "フィ": "ɸʲi", "フェ": "ɸe",
        "フォ": "ɸo", "フュ": "ɸʲɨ", "フョ": "ɸʲo", "ミャ": "mʲa", "ミュ": "mʲɨ", "ミョ": "mʲo",
        "リャ": "ɾʲa", "リュ": "ɾʲɨ", "リョ": "ɾʲo", "ヴァ": "va", "ヴィ": "vʲi", "ヴェ": "ve", "ヴォ": "vo",
        "ヴュ": "bʲɨ", "ヴョ": "bʲo",
    ]
}

extension Collection {
    fileprivate subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
