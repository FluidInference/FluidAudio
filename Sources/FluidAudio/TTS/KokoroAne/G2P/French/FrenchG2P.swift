import Foundation

/// Text frontend for the Kokoro ANE French variant.
///
/// Word pronunciations come from `fr_lexicon_cache.json` (ipa-dict `fr_FR`
/// vocabulary with espeak-ng `fr-fr` pronunciations, the IPA Kokoro's French
/// voice was trained on). Words it does not list go through the CharsiuG2P
/// CoreML model (``MultilingualG2PModel``) and are rewritten into the same
/// conventions — see ``FrenchPhonology``.
actor FrenchG2P {
    typealias Fallback = @Sendable (String) async throws -> String?

    private let lexicon: KokoroAneLexicon
    private let fallback: Fallback
    /// Fallback results, including misses (stored as ""), so each unknown word
    /// runs the autoregressive model once per session.
    private var fallbackCache: [String: String] = [:]

    init(lexicon: KokoroAneLexicon, fallback: @escaping Fallback) {
        self.lexicon = lexicon
        self.fallback = fallback
    }

    func phonemize(_ text: String) async throws -> String {
        let lexicon = self.lexicon
        var misses: [String] = []
        _ = FrenchPhonology.phonemize(text, isLexiconEntry: lexicon.contains) { word in
            let hit = FrenchPhonology.lookup(word, lexicon: lexicon)
            if hit == nil { misses.append(word) }
            return hit
        }
        // Apostrophe/hyphen compounds resolve through their parts instead.
        for word in misses where fallbackCache[word] == nil && word.allSatisfy(\.isLetter) {
            fallbackCache[word] = try await fallback(word) ?? ""
        }
        let cache = fallbackCache
        return FrenchPhonology.phonemize(text, isLexiconEntry: lexicon.contains) { word in
            FrenchPhonology.lookup(word, lexicon: lexicon)
                ?? cache[word].flatMap { $0.isEmpty ? nil : .raw($0) }
        }
    }
}

/// Sentence-level French phonology in espeak-ng `fr-fr` conventions (after
/// Misaki's `EspeakG2P` post-processing): stress on the last full vowel of
/// each content word, unstressed clitics and function words, elision,
/// liaison, and the vowel-quality / schwa choices where ipa-dict and espeak
/// disagree. Pure and synchronous; word pronunciations come from `lookup`.
enum FrenchPhonology {

    enum Stress { case primary, secondary, none }

    /// Where a word's pronunciation came from.
    enum Pronunciation: Equatable {
        /// Lexicon cache entry: already espeak-style, with citation stress.
        case espeak(String, hAspire: Bool)
        /// ipa-dict-style IPA (citation overrides, CharsiuG2P output) that
        /// still needs ``mapToEspeak(_:word:)`` and stress.
        case raw(String)
    }

    /// Text → Kokoro IPA. `lookup` resolves a lowercase word;
    /// `isLexiconEntry` says whether a token containing an apostrophe is
    /// listed whole (aujourd'hui, c'est).
    static func phonemize(
        _ text: String,
        isLexiconEntry: (String) -> Bool,
        lookup: (String) -> Pronunciation?
    ) -> String {
        let normalized = text.precomposedStringWithCanonicalMapping
            .replacingOccurrences(of: "’", with: "'")
            .replacingOccurrences(of: "«", with: "“")
            .replacingOccurrences(of: "»", with: "”")
        let tokens = tokenize(normalized)

        struct Item {
            var isWord: Bool
            var phonemes: String
            var word: String = ""
            var hAspire = false
            var spaceBefore = false
        }

        var items: [Item] = []
        var skip = 0
        for (index, token) in tokens.enumerated() {
            if skip > 0 {
                skip -= 1
                continue
            }
            guard token.isWord else {
                items.append(Item(isWord: false, phonemes: token.text, spaceBefore: token.spaceBefore))
                continue
            }
            if let (length, phonemes) = matchPhrase(tokens, at: index) {
                let last = tokens[index + length - 1].text.lowercased()
                items.append(Item(isWord: true, phonemes: phonemes, word: last, spaceBefore: token.spaceBefore))
                skip = length - 1
                continue
            }
            var lower = token.text.lowercased()
            let trailingApostrophe = lower.hasSuffix("'")
            while lower.hasSuffix("'") { lower.removeLast() }
            if isSpelledAcronym(token.text) {
                items.append(Item(isWord: true, phonemes: spell(lower), word: lower, spaceBefore: token.spaceBefore))
                continue
            }
            var prefix = ""
            var core = lower
            if trailingApostrophe, let elided = elisionPrefixes[lower] {
                // A clitic cut off by a quote or bracket (qu'« une ») keeps
                // only its consonant, as espeak reads it.
                prefix = elided
                core = ""
            } else if lower.contains("'"), !isLexiconEntry(lower) {
                let parts = lower.split(separator: "'", maxSplits: 1, omittingEmptySubsequences: false)
                if let elided = elisionPrefixes[String(parts[0])] {
                    prefix = elided
                    core = parts.count > 1 ? String(parts[1]) : ""
                }
            }
            let classWord =
                core.contains("'")
                ? String(core.split(separator: "'", maxSplits: 1, omittingEmptySubsequences: false).last ?? "")
                : core
            var stress: Stress =
                unstressedWords.contains(classWord)
                ? .none : (secondaryWords.contains(classWord) ? .secondary : .primary)
            let next = index + 1 < tokens.count ? tokens[index + 1].text : "."
            if stress != .primary, pauseMarks.contains(next) {
                stress = .primary
            }

            var phonemes: String
            var hAspire = false
            if core.contains("-"), stress == .primary {
                // Hyphenated compounds: every part carries its own stress
                // (États-Unis → etˈazynˈi), with liaison between parts.
                let parts = core.split(separator: "-").map(String.init)
                let resolved = parts.map { wordPhonemes($0, stress: .primary, lookup: lookup) }
                hAspire = resolved.first?.hAspire ?? false
                phonemes = ""
                for (j, part) in parts.enumerated() {
                    var piece = resolved[j].phonemes
                    if j + 1 < parts.count, !resolved[j + 1].hAspire, startsWithVowel(resolved[j + 1].phonemes) {
                        piece = applyLiaison(part, piece)
                    }
                    phonemes += piece
                }
            } else {
                let resolved =
                    core.isEmpty
                    ? (phonemes: "", hAspire: false) : wordPhonemes(core, stress: stress, lookup: lookup)
                phonemes = resolved.phonemes
                hAspire = resolved.hAspire
            }
            if !prefix.isEmpty {
                phonemes = mapToEspeak(prefix, word: "") + phonemes
            }
            items.append(
                Item(
                    isWord: true, phonemes: phonemes, word: classWord, hAspire: hAspire,
                    spaceBefore: token.spaceBefore))
        }

        var out: [(isWord: Bool, text: String, spaceBefore: Bool)] = []
        for (k, item) in items.enumerated() {
            guard item.isWord else {
                out.append((false, item.phonemes, item.spaceBefore))
                continue
            }
            var phonemes = item.phonemes
            if k + 1 < items.count, items[k + 1].isWord, !items[k + 1].hAspire,
                startsWithVowel(items[k + 1].phonemes)
            {
                phonemes = applyLiaison(item.word, phonemes)  // elided too: d'un ami → dœ̃n, l'on a → lɔ̃n
            }
            out.append((true, phonemes, item.spaceBefore))
        }
        return join(out)
    }

    /// Initialisms without a vowel letter (SNCF, TGV) and lone capitals are
    /// spelled out; pronounceable ones (ONU, OTAN) are read as words.
    static func isSpelledAcronym(_ token: String) -> Bool {
        guard token == token.uppercased(), token != token.lowercased() else { return false }
        if token.count == 1 {
            // À, É, Ô at a sentence start are words, not letters.
            guard let letter = token.lowercased().first, letterNames[letter] != nil else { return false }
            return token != "A" && token != "Y"
        }
        return token.count <= 5 && !token.contains(where: { "AEIOUYÀÂÉÈÊËÎÏÔÛÙÜ".contains($0) })
    }

    /// Letter names with secondary stress, primary on the last (ˌɛsˌɛnsˌeˈɛf).
    static func spell(_ acronym: String) -> String {
        let names = acronym.compactMap { letterNames[$0] }
        guard let final = names.last else { return "" }
        return names.dropLast().map { addStress($0, .secondary) }.joined() + addStress(final, .primary)
    }

    /// Fixed expressions espeak reads as one unit (tout le monde → tulmˈɔ̃d).
    /// Returns the number of word tokens consumed and their phonemes.
    static func matchPhrase(_ tokens: [Token], at index: Int) -> (Int, String)? {
        for (words, phonemes) in phrases {
            guard index + words.count <= tokens.count else { continue }
            let window = tokens[index..<(index + words.count)]
            if window.allSatisfy(\.isWord),
                zip(window, words).allSatisfy({ $0.text.lowercased() == $1 })
            {
                return (words.count, phonemes)
            }
        }
        return nil
    }

    static let phrases: [([String], String)] = [
        (["tout", "le", "monde"], "tulmˈɔ̃d"),
        (["parce", "que"], "paʁskˌə"),
        (["est-ce", "que"], "ɛs kə"),
        (["c'est-à-dire"], "sɛtadˈiʁ"),
        (["etc"], "ɛtseteʁˈa"),
    ]

    /// Override → lexicon-cache lookup used by ``FrenchG2P``.
    static func lookup(_ word: String, lexicon: KokoroAneLexicon) -> Pronunciation? {
        if let override = citationOverrides[word] { return .raw(override) }
        return lexicon.lookup(word).map { .espeak($0, hAspire: lexicon.isHAspire(word)) }
    }

    /// One word → espeak-style phonemes with `stress` applied, and whether it
    /// starts with an h aspiré, which blocks liaison. Lexicon entries keep
    /// espeak's own stress placement when the word is fully stressed.
    static func wordPhonemes(
        _ word: String, stress: Stress, lookup: (String) -> Pronunciation?
    ) -> (phonemes: String, hAspire: Bool) {
        switch lookup(word) {
        case .espeak(let phonemes, let hAspire):
            if stress == .primary, phonemes.contains("ˈ") { return (phonemes, hAspire) }
            return (addStress(stripStress(phonemes), stress), hAspire)
        case .raw(let raw):
            return (addStress(mapToEspeak(raw, word: word), stress), raw.hasPrefix("ʼ"))
        case nil:
            // Compounds resolve through their parts (États-Unis, presqu'île).
            guard word.contains("-") || word.contains("'") else { return ("", false) }
            let joined = word.split(whereSeparator: { $0 == "-" || $0 == "'" }).map {
                wordPhonemes(String($0), stress: .none, lookup: lookup).phonemes
            }
            return (addStress(joined.joined(), stress), false)
        }
    }

    static func stripStress(_ phonemes: String) -> String {
        String(String.UnicodeScalarView(phonemes.unicodeScalars.filter { $0 != "ˈ" && $0 != "ˌ" }))
    }

    // MARK: - ipa-dict → espeak conventions

    static func mapToEspeak(_ raw: String, word: String) -> String {
        var p = raw.replacingOccurrences(of: "g", with: "ɡ")
            .replacingOccurrences(of: "ɥ", with: "y")
            .replacingOccurrences(of: "ʼ", with: "")
            .replacingOccurrences(of: "‿", with: "")
        let w = Array(word)

        // espeak keeps orthographic b before a voiceless consonant (obtenir).
        if matches(word, #"b[tsc]"#) {
            p = replaceFirst(in: p, pattern: #"p(?=[tsk])"#, with: "b")
        }

        var s = Array(p.unicodeScalars)
        // Word-initial C + e + C schwa that ipa-dict dropped (semaine, chemin).
        if s.count >= 2, w.count >= 3, w[1] == "e", !"aeiouy".contains(w[0]), !"aeiouysxz".contains(w[2]),
            w.count == 3 || w[2] != w[3], consonants.contains(s[0]), consonants.contains(s[1]),
            !onsetSeconds.contains(s[1])
        {
            s.insert("ə", at: 1)
        }
        p = String(String.UnicodeScalarView(s))

        // e → ɛ before a spelled double consonant (excellent, processus).
        for match in allMatches(word, #"e(ll|ss|tt|nn|mm|rr)"#) {
            let phone = match == "rr" ? "ʁ" : String(match.prefix(1))
            if let range = p.range(of: "e" + phone) {
                p.replaceSubrange(range, with: "ɛ" + phone)
            }
        }

        // Mid vowels: espeak closes ɔ in open non-final syllables and uses ø
        // for eu outside the final syllable.
        s = Array(p.unicodeScalars)
        let last = lastVowelIndex(s)
        let hasDoubleO = matches(word, #"o(mm|nn|pp|ll|tt|rr|ff|cc)"#)
        for i in s.indices {
            let next: Unicode.Scalar? = i + 1 < s.count ? s[i + 1] : nil
            if s[i] == "ɔ", next != "\u{0303}", i < last {
                let closed =
                    s.count > i + 2 && next.map(consonants.contains) == true && consonants.contains(s[i + 2])
                    && !onsetSeconds.contains(s[i + 2])
                if !closed, !hasDoubleO { s[i] = "o" }
            } else if s[i] == "œ", next != "\u{0303}" {
                let beforeLabialOrT = next.map { "vptb".unicodeScalars.contains($0) } ?? false
                if i < last || (beforeLabialOrT && !word.contains("œ")) { s[i] = "ø" }
            }
        }

        // Word-internal schwa between single consonants after a vowel (devenu → dəvny).
        var k = 2
        while k < s.count - 2 {
            if s[k] == "ə", consonants.contains(s[k - 1]), vowels.contains(s[k - 2]), consonants.contains(s[k + 1]),
                vowels.contains(s[k + 2])
            {
                s.remove(at: k)
            }
            k += 1
        }
        p = String(String.UnicodeScalarView(s))

        // ɲ is written nj except word-finally after i/o (ligne, Pologne).
        p = replaceAll(in: p, pattern: #"ɲ(?!$)"#, with: "nj")
        p = replaceAll(in: p, pattern: #"(?<=[aeɛ])ɲ$"#, with: "nj")
        // C + liquid + i + j + V: espeak drops the glide (oubliez, client).
        p = replaceAll(in: p, pattern: #"(?<=[bdfɡkpstv][lʁ])ij(?=[aeiouyøœɛɔɑə])"#, with: "i")
        // ô / û in the final syllable are long (contrôle, sûr).
        if matches(word, #"[ôû][^aeiouyéèêàâîôû]*e?s?$"#) {
            s = Array(p.unicodeScalars)
            let lastVowel = lastVowelIndex(s)
            if lastVowel >= 0, s[lastVowel] == "o" || s[lastVowel] == "y" {
                s.insert("ː", at: lastVowel + 1)
                p = String(String.UnicodeScalarView(s))
            }
        }
        return p
    }

    // MARK: - Stress

    /// Index of the last vowel scalar, skipping a final schwa when an earlier
    /// vowel exists; -1 when there is none.
    static func lastVowelIndex(_ s: [Unicode.Scalar]) -> Int {
        guard let lastIndex = s.lastIndex(where: vowels.contains) else { return -1 }
        if s[lastIndex] == "ə", let earlier = s[..<lastIndex].lastIndex(where: vowels.contains) {
            return earlier
        }
        return lastIndex
    }

    static func addStress(_ phonemes: String, _ stress: Stress) -> String {
        guard stress != .none else { return phonemes }
        var s = Array(phonemes.unicodeScalars)
        let index = lastVowelIndex(s)
        guard index >= 0 else { return phonemes }
        s.insert(stress == .primary ? "ˈ" : "ˌ", at: index)
        return String(String.UnicodeScalarView(s))
    }

    // MARK: - Liaison

    /// `phonemes` with espeak-ng's liaison onto a following vowel-initial
    /// word applied: a linking consonant appended (lez‿, ɡʁɑ̃t‿), a spoken
    /// final s voiced (six ans → siz), or unchanged.
    static func applyLiaison(_ word: String, _ phonemes: String) -> String {
        let word = word.replacingOccurrences(of: "'", with: "")
        guard let letter = word.last,
            let last = phonemes.unicodeScalars.last(where: { $0 != "ˈ" && $0 != "ˌ" })
        else { return phonemes }
        switch letter {
        case "s", "x", "z":
            // Final s already spoken: numerals voice it (six ans → siz‿ɑ̃),
            // plurals in -es still link (classes entières → klasz‿), words
            // whose s is lexical keep it (os, bus, sens, index).
            if last == "s" || last == "z" {
                if voicedFinalS.contains(word) { return String(phonemes.dropLast()) + "z" }
                return word.hasSuffix("es") ? phonemes + "z" : phonemes
            }
            if sLiaisonWords.contains(word) { return phonemes + "z" }
            if matches(word, #"(is|us|és)$"#) { return phonemes }  // participles: connus, mis
            return phonemes + "z"
        case "n":
            return nasalLiaisonWords.contains(word) && last == "\u{0303}" ? phonemes + "n" : phonemes
        case "t", "d":
            // Verb -ent (peuvent, mettent → mɛtt‿) links even after a spoken
            // t; adverbs and nouns in -ent/-ment end in a vowel and do not.
            if word.hasSuffix("ent"), consonants.contains(last) { return phonemes + "t" }
            guard last != "t", last != "d" else { return phonemes }  // already spoken: sept, huit
            if tLiaisonWords.contains(word) || matches(word, #"(ait|aient|eut|ont)$"#) { return phonemes + "t" }
            return phonemes
        case "f":
            return word == "neuf" && last == "f" ? String(phonemes.dropLast()) + "v" : phonemes  // neuf ans
        case "p":
            return (word == "trop" || word == "beaucoup") && last != "p" ? phonemes + "p" : phonemes
        default:
            return phonemes
        }
    }

    static func startsWithVowel(_ phonemes: String) -> Bool {
        guard let first = phonemes.unicodeScalars.first(where: { $0 != "ˈ" && $0 != "ˌ" }) else { return false }
        return vowels.contains(first) || first == "j" || first == "w"
    }

    // MARK: - Tokens

    struct Token {
        var text: String
        var isWord: Bool
        /// Whitespace preceded the token in the input; espeak mirrors it
        /// (« bonjour » vs «civilisation», objectif : …).
        var spaceBefore = false
    }

    /// Words are letter runs joined by internal apostrophes or hyphens
    /// (l'homme, États-Unis), with an optional trailing apostrophe (jusqu').
    /// Any other non-space, non-digit character is its own punctuation token.
    static func tokenize(_ text: String) -> [Token] {
        let chars = Array(text)
        var tokens: [Token] = []
        var i = 0
        var sawSpace = false
        while i < chars.count {
            let ch = chars[i]
            if ch.isWhitespace {
                sawSpace = true
                i += 1
                continue
            }
            if ch.isLetter {
                var j = i
                while j < chars.count, chars[j].isLetter { j += 1 }
                while j + 1 < chars.count, chars[j] == "'" || chars[j] == "-", chars[j + 1].isLetter {
                    j += 1
                    while j < chars.count, chars[j].isLetter { j += 1 }
                }
                if j < chars.count, chars[j] == "'" { j += 1 }
                tokens.append(Token(text: String(chars[i..<j]), isWord: true, spaceBefore: sawSpace))
                sawSpace = false
                i = j
                continue
            }
            if !ch.isNumber, ch != "_" {
                tokens.append(Token(text: String(ch), isWord: false, spaceBefore: sawSpace))
                sawSpace = false
            }
            i += 1
        }
        return tokens
    }

    /// Join with the input's spacing. Adjacent words always get a space; an
    /// empty item (unpronounceable word) passes its leading space on.
    private static func join(_ items: [(isWord: Bool, text: String, spaceBefore: Bool)]) -> String {
        var s = ""
        var pendingSpace = false
        var lastWasWord = false
        for item in items {
            pendingSpace = pendingSpace || item.spaceBefore
            if item.text.isEmpty { continue }
            if !s.isEmpty, pendingSpace || (item.isWord && lastWasWord) { s += " " }
            s += item.text
            pendingSpace = false
            lastWasWord = item.isWord
        }
        return s
    }

    // MARK: - Regex helpers

    private static func regex(_ pattern: String) -> NSRegularExpression? {
        try? NSRegularExpression(pattern: pattern)
    }

    private static func matches(_ text: String, _ pattern: String) -> Bool {
        regex(pattern)?.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    /// Capture group 1 of every match, in order.
    private static func allMatches(_ text: String, _ pattern: String) -> [String] {
        guard let re = regex(pattern) else { return [] }
        return re.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
            Range($0.range(at: 1), in: text).map { String(text[$0]) }
        }
    }

    private static func replaceFirst(in text: String, pattern: String, with template: String) -> String {
        guard let re = regex(pattern),
            let match = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
            let range = Range(match.range, in: text)
        else { return text }
        return text.replacingCharacters(in: range, with: template)
    }

    private static func replaceAll(in text: String, pattern: String, with template: String) -> String {
        guard let re = regex(pattern) else { return text }
        return re.stringByReplacingMatches(
            in: text, range: NSRange(text.startIndex..., in: text), withTemplate: template)
    }

    // MARK: - Tables

    /// Words espeak-ng reads without stress in running text.
    static let unstressedWords: Set<String> = [
        "de", "les", "la", "et", "le", "des", "à", "un", "est", "du", "en", "une", "que", "qui", "a", "il", "pour",
        "dans", "plus", "sur", "au", "ce", "ou", "se", "qu", "leur", "aux", "par", "comme", "mais", "autres", "vous",
        "ne", "tout", "y", "ils", "on", "leurs", "peu", "tous", "son", "elle", "cette", "car", "ces", "sa", "autre",
        "elles", "toute", "ses", "chaque", "sans", "sous", "nous", "entre", "très", "nos", "dont", "où", "vos", "tel",
        "quand", "lors", "vers", "eu", "toutes", "tels", "près", "non", "celles", "furent", "me", "ceux", "notre",
        "quel", "cet", "telle", "quelle", "je", "ni", "mes", "chez", "ça", "quant", "dès", "votre", "faut", "sommes",
        "l", "d", "j", "m", "n", "s", "t", "c", "jusqu", "lorsqu", "puisqu", "te", "tu", "mon", "ma", "ton", "ta",
        "pas", "contre",
    ]

    /// Words espeak-ng reads with secondary stress in running text.
    static let secondaryWords: Set<String> = [
        "avec", "aussi", "plusieurs", "cela", "ainsi", "beaucoup", "après", "plupart", "quelques", "encore", "soit",
        "assez", "parmi", "toujours", "afin", "presque", "alors", "avant", "mieux", "durant", "tandis", "rien",
        "depuis", "comment", "quelque", "néanmoins", "jamais", "loin", "autour", "toutefois", "lorsque", "puisque",
        "lui",
    ]

    /// Running-text forms where ipa-dict's first variant is another reading
    /// (est: ɛst "east" vs ɛ "is").
    static let citationOverrides: [String: String] = [
        "est": "ɛ", "plus": "ply", "tous": "tu", "un": "œ̃", "six": "sis", "dix": "dis", "fils": "fis", "y": "i",
        "os": "ɔs", "ai": "e", "es": "ɛ", "as": "a", "eu": "y", "eus": "y", "eut": "y", "août": "ut", "ces": "se",
        "pays": "pɛi",
    ]

    static let letterNames: [Character: String] = [
        "a": "a", "b": "be", "c": "se", "d": "de", "e": "ə", "f": "ɛf", "g": "ʒe", "h": "aʃ", "i": "i", "j": "ʒi",
        "k": "ka", "l": "ɛl", "m": "ɛm", "n": "ɛn", "o": "o", "p": "pe", "q": "ky", "r": "ɛʁ", "s": "ɛs", "t": "te",
        "u": "y", "v": "ve", "w": "dubləve", "x": "iks", "y": "iɡʁɛk", "z": "zɛd",
    ]

    static let elisionPrefixes: [String: String] = [
        "jusqu": "ʒysk", "lorsqu": "lɔʁsk", "puisqu": "pɥisk", "quoiqu": "kwak", "qu": "k", "l": "l", "d": "d",
        "j": "ʒ", "m": "m", "n": "n", "s": "s", "t": "t", "c": "s",
    ]

    static let nasalLiaisonWords: Set<String> = [
        "en", "on", "un", "bien", "son", "mon", "ton", "aucun", "rien", "bon", "certain", "plein", "lun",
    ]
    static let tLiaisonWords: Set<String> = [
        "tout", "sont", "ont", "fait", "peut", "doit", "vont", "font", "dont", "quand", "grand", "petit", "avant",
        "pendant", "devant", "était", "est", "soit", "furent", "sept", "huit", "vingt", "fort", "trop",
    ]
    /// Words whose spoken final s voices to z in liaison.
    static let voicedFinalS: Set<String> = ["six", "dix"]
    static let sLiaisonWords: Set<String> = [
        "plus", "nous", "vous", "sous", "dans", "pas", "mais", "très", "jamais", "trois", "moins", "puis", "depuis",
        "alors", "toujours", "après", "dès", "chez", "assez",
    ]

    static let vowels = Set("aeiouyøœɛɔɑəɜ".unicodeScalars)
    static let consonants = Set("bdfɡklmnpstvzʃʒʁjwɲŋ".unicodeScalars)
    /// Second members of onset clusters, which keep the previous syllable open.
    static let onsetSeconds = Set("ʁlj".unicodeScalars)

    static let pauseMarks: Set<String> = [",", ".", "!", "?", ";", ":", "—", "…", "(", ")", "«", "»", "\"", "“", "”"]
}
