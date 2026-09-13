import Foundation

/// Port of Misaki's Cutlet frontend (`misaki/cutlet.py`, MIT, adapted from
/// polm/cutlet), the text → IPA path Kokoro's Japanese voices were trained on.
///
/// Pipeline: normalization (NFKC, half-width katakana, digit runs read as
/// kana) → MeCab tokens → regrouping of tokens that form a dictionary word →
/// per-word kana → IPA with Cutlet's context rules → Cutlet's spacing.
enum JapaneseCutlet {
    struct Word {
        var surface: String
        var hira: String
        var charType: Int
    }

    private struct Token {
        var surface: String
        var space: Bool
    }

    static func phonemize(_ text: String, tokenizer: JapaneseTokenizer, words dictionaryWords: Set<String>) -> String {
        guard !text.isEmpty else { return "" }
        let normalized = normalize(text)
        let words = tokenizer.tokenize(normalized).map { token -> Word in
            let reading = token.pron.isEmpty ? (token.kana.isEmpty ? token.surface : token.kana) : token.pron
            return Word(
                surface: token.surface, hira: katakanaToHiragana(reading),
                charType: token.charType == 7 || !token.isUnknown ? 6 : token.charType)
        }
        let tokens = romajiTokens(words, dictionaryWords: dictionaryWords)
        var out = tokens.map { $0.surface + ($0.space ? " " : "") }.joined()
        out = out.split(whereSeparator: { $0 == " " }).joined(separator: " ")
        out = out.replacingOccurrences(of: "(", with: "«").replacingOccurrences(of: ")", with: "»")
        return removingSpacesAroundGlottalStop(out)
    }

    // MARK: - Normalization

    static func normalize(_ text: String) -> String {
        var result = ""
        // Wave dash before a digit reads as a range: から.
        var characters = Array(text)
        for i in characters.indices where (characters[i] == "〜" || characters[i] == "～") {
            if i + 1 < characters.count, characters[i + 1].isASCII, characters[i + 1].isNumber {
                characters[i] = "か"
                characters.insert("ら", at: i + 1)
            }
        }
        for character in characters {
            if let mapped = katakanaPhoneticExtensions[character] {
                result.append(mapped)
            } else {
                result.append(character)
            }
        }
        // NFKC folds full-width alphanumerics to ASCII and half-width katakana
        // (including split dakuten) to full-width, as mojimoji did for Cutlet.
        result = result.precomposedStringWithCompatibilityMapping
        var out = ""
        var digits = ""
        for character in result {
            if character.isASCII, character.isNumber {
                digits.append(character)
            } else {
                if !digits.isEmpty {
                    out += " " + JapaneseNumberReader.hiragana(digits)
                    digits = ""
                }
                out.append(character)
            }
        }
        if !digits.isEmpty { out += " " + JapaneseNumberReader.hiragana(digits) }
        return out
    }

    static func katakanaToHiragana(_ text: String) -> String {
        String(
            String.UnicodeScalarView(
                text.unicodeScalars.map { scalar in
                    (0x30A1...0x30F6).contains(scalar.value) ? Unicode.Scalar(scalar.value - 0x60)! : scalar
                }))
    }

    // MARK: - Tokens

    private static func romajiTokens(_ input: [Word], dictionaryWords: Set<String>) -> [Token] {
        // Regroup consecutive tokens of one character class that together form
        // a dictionary word (MeCab splits 日本語 into 日本 + 語).
        var groups: [[Word]] = []
        var i = 0
        while i < input.count {
            var z = i + 1
            while z < input.count, input[z].charType == input[i].charType { z += 1 }
            var end: Int? = nil
            var j = z
            while j > i {
                if dictionaryWords.contains(input[i..<j].map(\.surface).joined()) {
                    end = j
                    break
                }
                j -= 1
            }
            if let end {
                groups.append(Array(input[i..<end]))
                i = end
            } else {
                groups.append([input[i]])
                i += 1
            }
        }
        let words = groups.map { group in
            Word(surface: group.map(\.surface).joined(), hira: group.map(\.hira).joined(), charType: group[0].charType)
        }
        var out: [Token] = []
        for word in words {
            let roma = romajiWord(word)
            var token = Token(surface: roma, space: false)
            let surface = word.surface
            if ["「", "『", "«"].contains(surface) || ["(", "["].contains(roma) {
                if !out.isEmpty { out[out.count - 1].space = true }
            } else if ["」", "』", "»"].contains(surface) || ["]", ")", ".", ",", "?", "!", ":"].contains(roma) {
                if !out.isEmpty { out[out.count - 1].space = false }
                token.space = true
            } else if roma == " " {
                token.space = false
            } else {
                token.space = true
            }
            out.append(token)
        }
        for index in out.indices {
            out[index].surface = out[index].surface.replacingOccurrences(of: "っ", with: "")
        }
        return out
    }

    private static func romajiWord(_ word: Word) -> String {
        let surface = word.surface
        if surface.allSatisfy({ $0.isASCII }) { return surface }
        if word.charType == 3 {  // symbol
            return surface.map { kanaTable[String($0)] ?? String($0) }.joined()
        }
        if word.charType != 6 { return "" }
        let hira = Array(word.hira)
        var out = ""
        for (index, character) in hira.enumerated() {
            let previous = index > 0 ? hira[index - 1] : nil
            let next = index + 1 < hira.count ? hira[index + 1] : nil
            out += singleMapping(previous: previous, current: character, next: next)
        }
        return out
    }

    private static func singleMapping(previous: Character?, current: Character, next: Character?) -> String {
        if odoriji.contains(current) {
            if current == "ゝ" || current == "ヽ" { return previous.map(String.init) ?? "" }
            if current == "ゞ" || current == "ヾ" {
                guard let previous, let voiced = addDakuten(previous) else { return "" }
                return kanaTable[String(voiced)] ?? ""
            }
            return ""
        }
        if let previous, let digraph = kanaTable[String(previous) + String(current)] { return digraph }
        if let next, kanaTable[String(current) + String(next)] != nil { return "" }
        if let next, sutegana.contains(next) {
            if current == "っ" { return "" }
            let base = kanaTable[String(current)] ?? ""
            return String(base.dropLast()) + (kanaTable[String(next)] ?? "")
        }
        if sutegana.contains(current) { return "" }
        if current == "ー" { return "ː" }
        if current == "っ" { return "ʔ" }
        if current == "ん" {
            if let next, let following = kanaTable[String(next)], let first = following.first {
                if "mpb".contains(first) { return "m" }
                if "kɡ".contains(first) { return "ŋ" }
                if following.hasPrefix("ɲ") || following.hasPrefix("ʨ") || following.hasPrefix("ʥ") { return "ɲ" }
                if "ntdɾz".contains(first) { return "n" }
            }
            return "ɴ"
        }
        return kanaTable[String(current)] ?? ""
    }

    private static func removingSpacesAroundGlottalStop(_ text: String) -> String {
        // Cutlet: drop a space before ʔ unless it follows punctuation, and a
        // space after ʔ unless a quote opens.
        var result = ""
        let characters = Array(text)
        for (index, character) in characters.enumerated() {
            guard character == " " else {
                result.append(character)
                continue
            }
            let before = index > 0 ? characters[index - 1] : nil
            let after = index + 1 < characters.count ? characters[index + 1] : nil
            if after == "ʔ", let before, !"!\",.:;?»—…”".contains(before) { continue }
            if before == "ʔ", after.map({ !"\"«“".contains($0) }) ?? true { continue }
            result.append(character)
        }
        return result
    }

    private static func addDakuten(_ kana: Character) -> Character? {
        let plain = Array("かきくけこさしすせそたちつてとはひふへほ")
        let voiced = Array("がぎぐげござじずぜぞだぢづでどばびぶべぼ")
        guard let index = plain.firstIndex(of: kana) else { return nil }
        return voiced[index]
    }

    private static let sutegana: Set<Character> = ["ゃ", "ゅ", "ょ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ"]
    private static let odoriji: Set<Character> = ["〃", "々", "ゝ", "ゞ", "ヽ"]

    private static let katakanaPhoneticExtensions: [Character: Character] = [
        "ㇰ": "ク", "ㇱ": "シ", "ㇲ": "ス", "ㇳ": "ト", "ㇴ": "ヌ", "ㇵ": "ハ", "ㇶ": "ヒ", "ㇷ": "フ", "ㇸ": "ヘ",
        "ㇹ": "ホ", "ㇺ": "ム", "ㇻ": "ラ", "ㇼ": "リ", "ㇽ": "ル", "ㇾ": "レ", "ㇿ": "ロ",
    ]

    // Cutlet's HEPBURN table (misaki/cutlet.py), hiragana-keyed.
    static let kanaTable: [String: String] = [
        "ぁ": "a", "あ": "a", "ぃ": "i", "い": "i", "ぅ": "ɯ", "う": "ɯ", "ぇ": "e", "え": "e", "ぉ": "o", "お": "o",
        "か": "ka", "が": "ɡa", "き": "kʲi", "ぎ": "ɡʲi", "く": "kɯ", "ぐ": "ɡɯ", "け": "ke", "げ": "ɡe", "こ": "ko",
        "ご": "ɡo", "さ": "sa", "ざ": "ʣa", "し": "ɕi", "じ": "ʥi", "す": "sɨ", "ず": "zɨ", "せ": "se", "ぜ": "ʣe",
        "そ": "so", "ぞ": "ʣo", "た": "ta", "だ": "da", "ち": "ʨi", "ぢ": "ʥi", "つ": "ʦɨ", "づ": "zɨ", "て": "te",
        "で": "de", "と": "to", "ど": "do", "な": "na", "に": "ɲi", "ぬ": "nɯ", "ね": "ne", "の": "no", "は": "ha",
        "ば": "ba", "ぱ": "pa", "ひ": "çi", "び": "bʲi", "ぴ": "pʲi", "ふ": "ɸɯ", "ぶ": "bɯ", "ぷ": "pɯ", "へ": "he",
        "べ": "be", "ぺ": "pe", "ほ": "ho", "ぼ": "bo", "ぽ": "po", "ま": "ma", "み": "mʲi", "む": "mɯ", "め": "me",
        "も": "mo", "ゃ": "ja", "や": "ja", "ゅ": "jɯ", "ゆ": "jɯ", "ょ": "jo", "よ": "jo", "ら": "ɾa", "り": "ɾʲi",
        "る": "ɾɯ", "れ": "ɾe", "ろ": "ɾo", "ゎ": "βa", "わ": "βa", "ゐ": "i", "ゑ": "e", "を": "o", "ゔ": "vɯ",
        "ゕ": "ka", "ゖ": "ke", "ヷ": "va", "ヸ": "vʲi", "ヹ": "ve", "ヺ": "vo",

        "いぇ": "je", "うぃ": "βi", "うぇ": "βe", "うぉ": "βo", "きぇ": "kʲe", "きゃ": "kʲa", "きゅ": "kʲɨ",
        "きょ": "kʲo", "ぎゃ": "ɡʲa", "ぎゅ": "ɡʲɨ", "ぎょ": "ɡʲo", "くぁ": "kᵝa", "くぃ": "kᵝi", "くぇ": "kᵝe",
        "くぉ": "kᵝo", "ぐぁ": "ɡᵝa", "ぐぃ": "ɡᵝi", "ぐぇ": "ɡᵝe", "ぐぉ": "ɡᵝo", "しぇ": "ɕe", "しゃ": "ɕa",
        "しゅ": "ɕɨ", "しょ": "ɕo", "じぇ": "ʥe", "じゃ": "ʥa", "じゅ": "ʥɨ", "じょ": "ʥo", "ちぇ": "ʨe",
        "ちゃ": "ʨa", "ちゅ": "ʨɨ", "ちょ": "ʨo", "ぢゃ": "ʥa", "ぢゅ": "ʥɨ", "ぢょ": "ʥo", "つぁ": "ʦa",
        "つぃ": "ʦʲi", "つぇ": "ʦe", "つぉ": "ʦo", "てぃ": "tʲi", "てゅ": "tʲɨ", "でぃ": "dʲi", "でゅ": "dʲɨ",
        "とぅ": "tɯ", "どぅ": "dɯ", "にぇ": "ɲe", "にゃ": "ɲa", "にゅ": "ɲɨ", "にょ": "ɲo", "ひぇ": "çe",
        "ひゃ": "ça", "ひゅ": "çɨ", "ひょ": "ço", "びゃ": "bʲa", "びゅ": "bʲɨ", "びょ": "bʲo", "ぴゃ": "pʲa",
        "ぴゅ": "pʲɨ", "ぴょ": "pʲo", "ふぁ": "ɸa", "ふぃ": "ɸʲi", "ふぇ": "ɸe", "ふぉ": "ɸo", "ふゅ": "ɸʲɨ",
        "ふょ": "ɸʲo", "みゃ": "mʲa", "みゅ": "mʲɨ", "みょ": "mʲo", "りゃ": "ɾʲa", "りゅ": "ɾʲɨ", "りょ": "ɾʲo",
        "ゔぁ": "va", "ゔぃ": "vʲi", "ゔぇ": "ve", "ゔぉ": "vo", "ゔゅ": "bʲɨ", "ゔょ": "bʲo",

        "。": ".", "、": ",", "？": "?", "！": "!", "「": "“", "」": "”", "『": "“", "』": "”", "：": ":", "；": ";",
        "（": "(", "）": ")", "《": "(", "》": ")", "【": "[", "】": "]", "・": " ", "，": ",", "～": "—", "〜": "—",
        "—": "—", "«": "“", "»": "”", "゚": "", "゙": "",
    ]
}

/// Port of Misaki's `num2kana.Convert(..., 'hiragana')` (MIT, from
/// Greatdane/Convert-Numbers-to-Japanese): reads a digit string as hiragana.
/// Numbers of ten or more digits, which the original rejects, are read digit
/// by digit.
enum JapaneseNumberReader {
    private static let table: [String: String] = [
        ".": "てん", "0": "ゼロ", "1": "いち", "2": "に", "3": "さん", "4": "よん", "5": "ご", "6": "ろく", "7": "なな",
        "8": "はち", "9": "きゅう", "10": "じゅう", "100": "ひゃく", "1000": "せん", "10000": "まん", "100000000": "おく",
        "300": "さんびゃく", "600": "ろっぴゃく", "800": "はっぴゃく", "3000": "さんぜん", "8000": "はっせん",
        "01000": "いっせん",
    ]

    static func hiragana(_ digits: String) -> String {
        var number = digits.replacingOccurrences(of: ",", with: "")
        guard !number.isEmpty else { return "" }
        if number.count > 9 { return number.map { table[String($0)] ?? "" }.joined() }
        while number.count > 1, number.first == "0" { number.removeFirst() }
        return convert(Array(number)).replacingOccurrences(of: " ", with: "")
    }

    private static func one(_ d: Character) -> String { table[String(d)] ?? "" }

    private static func two(_ n: [Character]) -> String {
        if n[0] == "0" { return one(n[1]) }
        if n == ["1", "0"] { return table["10"]! }
        if n[0] == "1" { return table["10"]! + " " + one(n[1]) }
        if n[1] == "0" { return one(n[0]) + " " + table["10"]! }
        return [one(n[0]), table["10"]!, one(n[1])].joined(separator: " ")
    }

    private static func three(_ n: [Character]) -> String {
        var parts: [String] = []
        switch n[0] {
        case "1": parts.append(table["100"]!)
        case "3": parts.append(table["300"]!)
        case "6": parts.append(table["600"]!)
        case "8": parts.append(table["800"]!)
        default:
            parts.append(one(n[0]))
            parts.append(table["100"]!)
        }
        if !(n[1] == "0" && n[2] == "0") {
            parts.append(n[1] == "0" ? one(n[2]) : two(Array(n[1...])))
        }
        return parts.joined(separator: " ")
    }

    private static func four(_ input: [Character], standalone: Bool) -> String {
        var n = input
        if n == ["0", "0", "0", "0"] { return "" }
        while n.first == "0" { n.removeFirst() }
        switch n.count {
        case 1: return one(n[0])
        case 2: return two(n)
        case 3: return three(n)
        default: break
        }
        var parts: [String] = []
        switch n[0] {
        case "1": parts.append(table[standalone ? "1000" : "01000"]!)
        case "3": parts.append(table["3000"]!)
        case "8": parts.append(table["8000"]!)
        default:
            parts.append(one(n[0]))
            parts.append(table["1000"]!)
        }
        if !(n[1] == "0" && n[2] == "0" && n[3] == "0") {
            parts.append(n[1] == "0" ? two(Array(n[2...])) : three(Array(n[1...])))
        }
        return parts.joined(separator: " ")
    }

    private static func long(_ n: [Character]) -> String {
        var parts: [String] = []
        let head = Array(n.dropLast(4))
        switch head.count {
        case 1:
            parts.append(one(head[0]))
            parts.append(table["10000"]!)
        case 2:
            parts.append(two(head))
            parts.append(table["10000"]!)
        case 3:
            parts.append(three(head))
            parts.append(table["10000"]!)
        case 4:
            parts.append(four(head, standalone: false))
            parts.append(table["10000"]!)
        default:  // 5 digits before the last four: X億 + 万 block
            parts.append(one(head[0]))
            parts.append(table["100000000"]!)
            parts.append(four(Array(head[1...]), standalone: false))
            if head[1...] != ["0", "0", "0", "0"] { parts.append(table["10000"]!) }
        }
        parts.append(four(Array(n.suffix(4)), standalone: false))
        return parts.joined(separator: " ")
    }

    private static func convert(_ n: [Character]) -> String {
        switch n.count {
        case 1: return one(n[0])
        case 2: return two(n)
        case 3: return three(n)
        case 4: return four(n, standalone: true)
        default: return long(n)
        }
    }
}
