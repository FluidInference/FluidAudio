import Foundation

/// Rule-based Spanish grapheme-to-phoneme frontend for the Kokoro ANE Spanish
/// variant.
///
/// Kokoro's Spanish voices were trained on espeak-ng `es` (Castilian) IPA
/// after Misaki's `EspeakG2P` post-processing, so this frontend reproduces
/// those conventions rather than a textbook transcription: θ for c/z,
/// diphthong ligatures (`ai` → `I`, `ei` → `A`, `au` → `W`), the stress mark
/// placed before the nucleus vowel, alternating secondary stress, unstressed
/// function words, and phrase-level b/d/g lenition. Spanish spelling is
/// regular enough that no lexicon is needed.
///
/// Input is expected to be normalized already (digits verbalized); digits
/// that remain are dropped.
enum SpanishG2P {

    /// Text → Kokoro IPA.
    static func phonemize(_ text: String) -> String {
        let normalized = text.precomposedStringWithCanonicalMapping
            .replacingOccurrences(of: "«", with: "“")
            .replacingOccurrences(of: "»", with: "”")
            .replacingOccurrences(of: "'", with: " ")
        let tokens = tokenize(normalized)

        var items: [Item] = []
        for (index, token) in tokens.enumerated() {
            guard token.isWord else {
                items.append(Item(kind: .punctuation, text: token.text, stress: .primary))
                continue
            }
            let lower = token.text.lowercased()
            var stress: Stress =
                unstressedWords.contains(lower) ? .none : (secondaryWords.contains(lower) ? .secondary : .primary)
            // Function words at a phrase end regain full stress ("lo que.").
            let next = index + 1 < tokens.count ? tokens[index + 1].text : "."
            if stress != .primary, pauseMarks.contains(next) {
                stress = .primary
            }
            items.append(Item(kind: isAcronym(token.text) ? .acronym : .word, text: lower, stress: stress))
        }

        var output: [(isWord: Bool, text: String)] = []
        var previous: Unicode.Scalar?  // last phone of the phrase so far; nil after a pause
        for item in items {
            if item.kind == .punctuation {
                output.append((false, item.text))
                if pauseMarks.contains(item.text) { previous = nil }
                continue
            }
            var phonemes: String
            switch item.kind {
            case .acronym: phonemes = spell(item.text)
            default: phonemes = item.text == "y" ? "i" : phonemizeWord(item.text, stress: item.stress)
            }
            phonemes = applyAllophones(phonemes, previous: previous)

            let bare = stripStress(phonemes)
            if previous != nil, let last = output.last, last.isWord, let first = bare.unicodeScalars.first {
                // Cross-word sandhi on the previous word's final consonant.
                var prior = last.text
                if prior.hasSuffix("n"), first == "b" || first == "p" {
                    prior = String(prior.dropLast()) + "m"
                } else if prior.hasSuffix("n"), first == "ɡ" || first == "x" {
                    prior = String(prior.dropLast()) + "ŋ"
                } else if prior.hasSuffix("d"), "aeiou".unicodeScalars.contains(first) {
                    prior = String(prior.dropLast()) + "ð"
                }
                output[output.count - 1].text = prior
            }
            output.append((true, phonemes))
            if let last = bare.unicodeScalars.last { previous = last }
        }
        return join(output)
    }

    // MARK: - Word level

    /// One orthographic word (lowercased) → IPA with stress marks.
    static func phonemizeWord(_ word: String, stress: Stress) -> String {
        let w = word.precomposedStringWithCanonicalMapping.lowercased()
        if stress == .primary, let fixed = lexicon[w] { return fixed }
        // -mente adverbs carry two primary stresses: the stem's and mˈente.
        if stress == .primary, w.count > 6, w.hasSuffix("mente") {
            return phonemizeWord(String(w.dropLast(5)), stress: .primary) + "mˈente"
        }
        var segments = segment(Array(foldForeign(w)))
        guard let syllables = syllabify(&segments) else {
            return segments.map(\.phone).joined()
        }
        let stressed = stressedSyllable(word: w, segments: segments, syllables: syllables)

        var out = ""
        for (index, syllable) in syllables.enumerated() {
            let onset = syllable.onset.map { segments[$0].phone }.joined()
            let (glides, nucleusCore) = nucleus(syllable, segments)
            var core = nucleusCore
            let coda = syllable.coda.map { segments[$0].phone }.joined()
            var mark = ""
            if index == stressed {
                mark = stress.mark
                // Stressed e before a nasal coda + consonant opens to ɛ (ˈɛntɾe).
                if stress == .primary, core == "e", let firstCoda = syllable.coda.first,
                    segments[firstCoda].phone == "n",
                    syllable.coda.count >= 2
                        || (index + 1 < syllables.count && !syllables[index + 1].onset.isEmpty)
                {
                    core = "ɛ"
                }
            } else if stress == .primary, index < stressed - 1, index % 2 == 0 {
                mark = "ˌ"
            }
            out += onset + glides + mark + core + coda
        }
        return out
    }

    // MARK: - Segmentation

    enum SegmentKind { case consonant, vowel, glide }

    struct Segment {
        var phone: String
        var kind: SegmentKind
        var accented: Bool
    }

    struct Syllable {
        var onset: [Int]
        var nucleus: [Int]
        var coda: [Int]
    }

    /// Orthography → phone segments. `i`/`u` start as glide candidates and are
    /// resolved against their neighbours in ``syllabify(_:)``.
    static func segment(_ w: [Character]) -> [Segment] {
        var out: [Segment] = []
        var i = 0
        let n = w.count
        func next(_ k: Int = 1) -> Character? { i + k < n ? w[i + k] : nil }
        func consonant(_ phone: String) { out.append(Segment(phone: phone, kind: .consonant, accented: false)) }

        while i < n {
            let c = w[i]
            switch c {
            case "a", "e", "o", "á", "é", "ó":
                out.append(Segment(phone: String(stripAccent(c)), kind: .vowel, accented: "áéó".contains(c)))
            case "i", "í", "u", "ú", "ü":
                let accented = c == "í" || c == "ú"
                out.append(Segment(phone: String(stripAccent(c)), kind: accented ? .vowel : .glide, accented: accented))
            case "y":
                if i + 1 >= n, let last = out.last, last.phone == "u" {
                    // muy → mˈuj
                    out[out.count - 1].kind = .vowel
                    consonant("j")
                } else if let nx = next(), vowelLetters.contains(nx) {
                    consonant("ʝ")
                } else {
                    out.append(Segment(phone: "i", kind: .glide, accented: false))
                }
            case "c":
                if next() == "h" {
                    consonant("ʧ")
                    i += 1
                } else if let nx = next(), "eiéí".contains(nx) {
                    consonant("θ")
                } else {
                    consonant("k")
                }
            case "q":
                consonant("k")
                if next() == "u" { i += 1 }
            case "g":
                if i + 1 == n, let last = out.last, last.phone == "n" {
                    out[out.count - 1].phone = "ŋ"  // Hong Kong
                } else if let nx = next(), "eiéí".contains(nx) {
                    consonant("x")
                } else if next() == "u", let nx2 = next(2), "eiéí".contains(nx2) {
                    consonant("ɡ")
                    i += 1
                } else {
                    consonant("ɡ")
                }
            case "j":
                consonant("x")
            case "h":
                // Silent, but it still separates vowels (prohibido → pɾoiβˈiðo).
                if i > 0, next() != nil { consonant("") }
            case "l":
                if next() == "l" {
                    consonant("ʎ")
                    i += 1
                } else {
                    consonant("l")
                }
            case "r":
                if next() == "r" {
                    consonant("r")
                    i += 1
                } else if i == 0 || (out.last.map { ["n", "l", "s"].contains($0.phone) } ?? false) {
                    consonant("r")
                } else {
                    consonant("ɾ")
                }
            case "ñ": consonant("ɲ")
            case "v": consonant("b")
            case "z": consonant("θ")
            case "x":
                consonant("k")
                consonant("s")
            default:
                consonant(String(c))
            }
            i += 1
        }
        return out
    }

    /// Resolve glides, then split into syllables (maximal onsets limited to
    /// Spanish obstruent + liquid clusters). Returns nil for vowelless input.
    static func syllabify(_ segs: inout [Segment]) -> [Syllable]? {
        let n = segs.count
        // A glide candidate with no vocalic neighbour is a vowel.
        for k in 0..<n where segs[k].kind == .glide {
            let prevVocalic = k > 0 && segs[k - 1].kind != .consonant
            let nextVocalic = k + 1 < n && segs[k + 1].kind != .consonant
            if !prevVocalic && !nextVocalic { segs[k].kind = .vowel }
        }
        // Hiatus after a word-initial liquid, a C+liquid cluster, or a trill
        // (cliente, riesgo, luego, carruaje).
        for k in 0..<n where segs[k].kind == .glide && k > 0 && k + 1 < n && segs[k + 1].kind == .vowel {
            let liquid = segs[k - 1].phone
            guard liquid == "l" || liquid == "r" || liquid == "ɾ" else { continue }
            if k - 1 == 0 || segs[k - 2].kind == .consonant || liquid == "r" {
                segs[k].kind = .vowel
            }
        }
        // iu / ui: the second is the nucleus.
        if n > 1 {
            for k in 0..<(n - 1) where segs[k].kind == .glide && segs[k + 1].kind == .glide {
                segs[k + 1].kind = .vowel
            }
        }

        // Nucleus spans: vocalic runs, split between two full vowels (hiatus).
        var spans: [[Int]] = []
        var k = 0
        while k < n {
            guard segs[k].kind != .consonant else {
                k += 1
                continue
            }
            var group: [Int] = []
            while k < n, segs[k].kind != .consonant {
                if segs[k].kind == .vowel, group.contains(where: { segs[$0].kind == .vowel }) {
                    spans.append(group)
                    group = []
                }
                group.append(k)
                k += 1
            }
            spans.append(group)
        }
        guard !spans.isEmpty else { return nil }

        var syllables: [Syllable] = []
        var previousEnd = -1
        for (index, span) in spans.enumerated() {
            let cluster = Array((previousEnd + 1)..<span[0])
            var onset = cluster
            if index > 0 {
                var coda: [Int] = []
                if cluster.count >= 2 {
                    let pair = segs[cluster[cluster.count - 2]].phone + segs[cluster[cluster.count - 1]].phone
                    let split = onsetClusters.contains(pair) ? cluster.count - 2 : cluster.count - 1
                    onset = Array(cluster[split...])
                    coda = Array(cluster[..<split])
                }
                syllables[syllables.count - 1].coda = coda
            }
            syllables.append(Syllable(onset: onset, nucleus: span, coda: []))
            previousEnd = span[span.count - 1]
        }
        syllables[syllables.count - 1].coda = Array((previousEnd + 1)..<n)
        return syllables
    }

    /// Written accent wins; otherwise penultimate after a vowel, n or s, else final.
    static func stressedSyllable(word: String, segments: [Segment], syllables: [Syllable]) -> Int {
        if let accented = syllables.firstIndex(where: { $0.nucleus.contains { segments[$0].accented } }) {
            return accented
        }
        guard syllables.count > 1 else { return 0 }
        if let last = word.last, "aeiouns".contains(last) {
            return syllables.count - 2
        }
        return syllables.count - 1
    }

    /// Render a nucleus as (leading glides, stressable core). Falling
    /// diphthongs use Misaki's espeak ligatures.
    static func nucleus(_ syllable: Syllable, _ segs: [Segment]) -> (String, String) {
        let parts = syllable.nucleus.map { segs[$0] }
        let rendered = parts.map { $0.kind == .glide ? ($0.phone == "i" ? "j" : "w") : $0.phone }
        if parts.count >= 2, parts[parts.count - 1].kind == .glide, parts[parts.count - 2].kind == .vowel,
            let tie = fallingDiphthongs[parts[parts.count - 2].phone + parts[parts.count - 1].phone]
        {
            return (rendered.dropLast(2).joined(), tie)
        }
        let vowelIndex = parts.firstIndex { $0.kind == .vowel } ?? (parts.count - 1)
        return (rendered[..<vowelIndex].joined(), rendered[vowelIndex...].joined())
    }

    // MARK: - Phrase level

    /// b/d/g lenite to β/ð/ɣ except after a pause or a nasal; a coda ɡ and a
    /// word-final d stay stops. Also nasal place assimilation and the
    /// espeak `pt` → `pːt` length mark.
    static func applyAllophones(_ phonemes: String, previous: Unicode.Scalar?) -> String {
        let scalars = Array(phonemes.unicodeScalars)
        var out = ""
        var last = previous
        for (index, scalar) in scalars.enumerated() {
            if scalar == "ˈ" || scalar == "ˌ" {
                out.unicodeScalars.append(scalar)
                continue
            }
            let rest = scalars[(index + 1)...].filter { $0 != "ˈ" && $0 != "ˌ" }
            let next = rest.first
            let afterPauseOrNasal = last == nil || nasals.contains(last!)
            var phone = String(scalar)
            switch scalar {
            case "p" where next == "t":
                phone = "pː"
            case "b":
                if !(afterPauseOrNasal || next == "t") { phone = "β" }
            case "d":
                if !afterPauseOrNasal, !rest.isEmpty { phone = "ð" }
            case "ɡ":
                let coda = next.map { !vocalicPhones.contains($0) && !"ɾlwjr".unicodeScalars.contains($0) } ?? true
                if !(afterPauseOrNasal || coda) { phone = "ɣ" }
            case "n":
                if let next, "bpf".unicodeScalars.contains(next) {
                    phone = "m"
                } else if next == "ɡ" || next == "x" {
                    phone = "ŋ"
                }
            default:
                break
            }
            out += phone
            last = phone.unicodeScalars.last
        }
        return out
    }

    static func spell(_ acronym: String) -> String {
        let names = acronym.lowercased().compactMap { letterNames[$0] }.map { phonemizeWord($0, stress: .primary) }
        guard let final = names.last else { return "" }
        return names.dropLast().map { $0.replacingOccurrences(of: "ˈ", with: "ˌ") }.joined() + final
    }

    static func isAcronym(_ token: String) -> Bool {
        let isUpper = token == token.uppercased() && token != token.lowercased()
        if token.count >= 2, token.count <= 5, isUpper { return true }
        return token.count == 1 && !"aeiouy".contains(token.lowercased())
    }

    // MARK: - Helpers

    enum Stress {
        case primary, secondary, none
        var mark: String {
            switch self {
            case .primary: return "ˈ"
            case .secondary: return "ˌ"
            case .none: return ""
            }
        }
    }

    private enum ItemKind { case word, acronym, punctuation }

    private struct Item {
        var kind: ItemKind
        var text: String
        var stress: Stress
    }

    struct Token {
        var text: String
        var isWord: Bool
    }

    /// Letter runs are words; every other non-space, non-digit character is a
    /// punctuation token of its own.
    static func tokenize(_ text: String) -> [Token] {
        var tokens: [Token] = []
        var word = ""
        for ch in text {
            if ch.isLetter {
                word.append(ch)
                continue
            }
            if !word.isEmpty {
                tokens.append(Token(text: word, isWord: true))
                word = ""
            }
            if !ch.isWhitespace, !ch.isNumber, ch != "_" {
                tokens.append(Token(text: String(ch), isWord: false))
            }
        }
        if !word.isEmpty { tokens.append(Token(text: word, isWord: true)) }
        return tokens
    }

    private static func join(_ items: [(isWord: Bool, text: String)]) -> String {
        var s = ""
        for item in items {
            if !item.isWord, attachLeft.contains(item.text) {
                while s.last == " " { s.removeLast() }
                s += item.text + " "
            } else if !item.isWord, attachRight.contains(item.text) {
                s += item.text
            } else {
                s += item.text + " "
            }
        }
        return s.trimmingCharacters(in: .whitespaces)
    }

    private static func stripStress(_ s: String) -> String {
        String(String.UnicodeScalarView(s.unicodeScalars.filter { $0 != "ˈ" && $0 != "ˌ" }))
    }

    private static func stripAccent(_ c: Character) -> Character {
        switch c {
        case "á": return "a"
        case "é": return "e"
        case "í": return "i"
        case "ó": return "o"
        case "ú", "ü": return "u"
        default: return c
        }
    }

    /// Letters outside the Spanish alphabet lose their diacritics (ğ → g) or
    /// are dropped when they have no Spanish base letter.
    static func foldForeign(_ word: String) -> String {
        var out = ""
        for ch in word {
            if spanishLetters.contains(ch) {
                out.append(ch)
                continue
            }
            let base = String(ch).decomposedStringWithCanonicalMapping.unicodeScalars
                .filter { !$0.properties.isDiacritic }
            let folded = String(String.UnicodeScalarView(base))
            if !folded.isEmpty, folded.allSatisfy({ spanishLetters.contains($0) }) {
                out += folded
            }
        }
        return out
    }

    // MARK: - Tables

    /// Words espeak-ng reads without stress in running text.
    static let unstressedWords: Set<String> = [
        "de", "la", "el", "en", "que", "y", "los", "a", "las", "se", "del", "con", "su", "por", "o", "al", "sus",
        "si", "lo", "le", "sin", "tras", "e", "quien", "u", "nos", "mis", "me", "te", "mi", "tu", "tus", "les",
        "os",
    ]

    /// Words espeak-ng reads with secondary stress in running text.
    static let secondaryWords: Set<String> = [
        "para", "como", "entre", "pero", "cuando", "sobre", "desde", "hacia", "hasta", "quienes", "bajo",
        "aunque", "donde", "mientras", "porque", "nuestros", "nuestra", "nuestro", "nuestras", "ante", "cuanto",
    ]

    static let lexicon: [String: String] = ["ser": "sˈer"]

    static let letterNames: [Character: String] = [
        "a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "efe", "g": "ge", "h": "ache", "i": "i",
        "j": "jota", "k": "ka", "l": "ele", "m": "eme", "n": "ene", "ñ": "eñe", "o": "o", "p": "pe", "q": "cu",
        "r": "ere", "s": "ese", "t": "te", "u": "u", "v": "uve", "w": "uvedoble", "x": "equis", "y": "igriega",
        "z": "zeta",
    ]

    static let fallingDiphthongs: [String: String] = [
        "ai": "I", "au": "W", "ei": "A", "oi": "oɪ", "eu": "eʊ", "ui": "uj", "ou": "oʊ",
    ]

    static let onsetClusters: Set<String> = [
        "pɾ", "bɾ", "tɾ", "dɾ", "kɾ", "ɡɾ", "fɾ", "pl", "bl", "kl", "ɡl", "fl", "tl",
    ]

    static let pauseMarks: Set<String> = [",", ".", "!", "?", ";", ":", "—", "…", "(", ")", "«", "»", "\"", "“", "”"]
    private static let attachLeft: Set<String> = [",", ".", "!", "?", ";", ":", "…", ")", "»", "”"]
    private static let attachRight: Set<String> = ["¿", "¡", "(", "«", "“"]

    private static let vowelLetters: Set<Character> = ["a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ü"]
    private static let spanishLetters = Set("abcdefghijklmnopqrstuvwxyzáéíóúüñ")
    private static let nasals = Set("mnɲŋ".unicodeScalars)
    private static let vocalicPhones = Set("aeiouɛIAWOɪʊ".unicodeScalars)
}
