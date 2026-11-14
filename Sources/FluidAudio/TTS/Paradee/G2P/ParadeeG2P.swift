#if TTS
@preconcurrency import CoreML
import Foundation

/// Paradee's own grapheme-to-phoneme frontend, without eSpeak NG (GPL-3.0):
/// Hindi, French and Spanish text → the eSpeak-style phonemes Paradee v1.3 was
/// trained on, and English words missing from the Misaki lexicon → American or
/// British phonemes. A port of upstream `paradee/g2p.py` (package 1.3.1,
/// Apache-2.0), built from three parts in `v1.3/g2p/`:
///
/// - `ParadeeG2P.mlmodelc`: a 0.76M-parameter CTC Transformer that reads one
///   word plus the last letter of the word before and the first letter of the
///   word after (enough for French liaison and Spanish b/d/g softening);
/// - word lists in `paradee_g2p.json` (the words the model gets wrong, and the
///   3,000 most common words of each language), some keyed by those letters;
/// - a table of number pieces per language (``ParadeeG2PNumbers``).
///
/// Upstream measures 0.80 % (Hindi), 0.95 % (French) and 0.59 % (Spanish) of
/// phoneme characters different from eSpeak. Text is processed by Unicode
/// scalar, as the Python reference does by code point.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
struct ParadeeG2P {

    /// Letters of one word the model reads (upstream `_MAX_WORD`).
    static let maxWordLength = 60
    /// Rows per prediction (the model's batch upper bound).
    static let maxBatch = 64

    let data: ParadeeG2PData
    let model: MLModel

    // MARK: - Sentences

    /// One sentence of Hindi, French or Spanish → phonemes, one space between
    /// words, punctuation kept (upstream `G2P.__call__`).
    func phonemize(_ text: String, language: ParadeeLanguage) throws -> String {
        let scalars = Self.prepare(text, language: language, percent: Self.percentWords[language] ?? "")
        let pieces = Self.splitWhitespace(scalars)
        let found = pieces.map(Self.word)
        let contexts = Self.contexts(found)
        let table = data.numbers[language.rawValue] ?? [:]

        enum Part {
            case text(String)
            case job(Int)
        }
        var jobs: [(word: String, before: String, after: String)] = []
        var plan: [(lead: String, parts: [Part]?, trail: String)] = []
        for (index, raw) in pieces.enumerated() {
            guard let word = found[index], let context = contexts[index] else {
                plan.append((Self.string(raw), nil, ""))
                continue
            }
            let core = word.core
            var parts: [(isNumber: Bool, piece: [Unicode.Scalar])] = []
            for run in Self.digitRuns(core) {
                if run.isDigits {
                    parts.append((true, run.scalars))
                } else {
                    let piece = Self.strip(run.scalars, characters: "-.,/:")
                    if piece.contains(where: Self.isAlphanumeric) { parts.append((false, piece)) }
                }
            }
            var ids: [Part] = []
            for part in parts {
                // Upstream compares by value, so a repeated piece counts as first / last too.
                let first = part.piece == parts[0].piece
                let last = part.piece == parts[parts.count - 1].piece
                if part.isNumber {
                    if part.piece.count < 10 {
                        let before = first ? context.before : "#"
                        ids.append(
                            .text(
                                ParadeeG2PNumbers.read(
                                    Self.value(part.piece), language: language, table: table, before: before)))
                    } else {  // a long string of digits is read digit by digit
                        let digits = part.piece.map {
                            ParadeeG2PNumbers.read(Self.value([$0]), language: language, table: table)
                        }
                        ids.append(.text(digits.joined(separator: " ")))
                    }
                } else {
                    jobs.append((Self.string(part.piece), first ? context.before : "#", last ? context.after : "#"))
                    ids.append(.job(jobs.count - 1))
                }
            }
            plan.append(
                (Self.string(Array(raw[..<word.start])), ids, Self.string(Array(raw[(word.start + core.count)...]))))
        }

        let resolved = try words(language: language, jobs)
        let out = plan.map { lead, parts, trail -> String in
            guard let parts else { return lead }
            let spoken = parts.map { part -> String in
                switch part {
                case .text(let text): return text
                case .job(let index): return resolved[index]
                }
            }
            return lead + spoken.joined(separator: " ") + trail
        }
        return out.joined(separator: " ").replacingOccurrences(of: "-", with: "")
    }

    // MARK: - Words

    /// Phonemes for each (word, letter before, letter after): the word lists
    /// first (in context, then plain), then the model (upstream `G2P.words`).
    func words(language: ParadeeLanguage, _ items: [(word: String, before: String, after: String)]) throws -> [String] {
        let lexicon = data.lexicon[language.rawValue]
        var out: [String?] = items.map { item in
            if let inContext = lexicon?.inContext[ParadeeG2PData.contextKey(item.word, item.before, item.after)],
                !inContext.isEmpty
            {
                return inContext
            }
            return lexicon?.words[item.word]
        }
        let todo = out.indices.filter { out[$0] == nil }
        let predicted = try predict(todo.map { (language, items[$0].before, items[$0].word, items[$0].after) })
        for (index, phonemes) in zip(todo, predicted) {
            out[index] = phonemes
        }
        return out.map { $0 ?? "" }
    }

    /// The phonemes of an English word missing from the Misaki lexicon, as
    /// upstream `_EnglishFallback` gives misaki: the English word list, else
    /// the model with no neighbouring letters. `nil` for a word with digits
    /// (upstream hands those back to misaki) or without letters.
    func englishWord(_ word: String, british: Bool) throws -> String? {
        let scalars = Array(word.precomposedStringWithCanonicalMapping.unicodeScalars)
        guard !scalars.contains(where: Self.isDecimal) else { return nil }
        let core = Self.strip(scalars, characters: "-.,/:'")
        guard core.contains(where: Self.isAlphanumeric) else { return nil }
        let language: ParadeeLanguage = british ? .britishEnglish : .americanEnglish
        let key = Self.string(core)
        if let listed = data.englishLowercased[language.rawValue]?[key.lowercased()] { return listed }
        return try predict([(language, "#", key, "#")]).first
    }

    // MARK: - Model

    /// Phonemes from the model alone for each (language, letter before, word,
    /// letter after), batched (upstream `G2P.model`). The float model's output
    /// for a row does not depend on the other rows or on padding.
    func predict(_ items: [(ParadeeLanguage, String, String, String)]) throws -> [String] {
        guard !items.isEmpty else { return [] }
        let rows = items.map { language, before, word, after in data.row(language, before, word, after) }
        var out: [String] = []
        out.reserveCapacity(rows.count)
        var start = 0
        while start < rows.count {
            let batch = Array(rows[start..<min(start + Self.maxBatch, rows.count)])
            out += try predictBatch(batch)
            start += batch.count
        }
        return out
    }

    private func predictBatch(_ rows: [[Int32]]) throws -> [String] {
        let width = rows.map(\.count).max() ?? 0
        let shape = [rows.count, width].map(NSNumber.init)
        do {
            let chars = try MLMultiArray(shape: shape, dataType: .int32)
            let pad = try MLMultiArray(shape: shape, dataType: .int32)
            let charPtr = chars.dataPointer.bindMemory(to: Int32.self, capacity: rows.count * width)
            let padPtr = pad.dataPointer.bindMemory(to: Int32.self, capacity: rows.count * width)
            for (r, row) in rows.enumerated() {
                for t in 0..<width {
                    charPtr[r * width + t] = t < row.count ? row[t] : 0
                    padPtr[r * width + t] = t < row.count ? 0 : 1
                }
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "chars": MLFeatureValue(multiArray: chars), "pad": MLFeatureValue(multiArray: pad),
            ])
            guard let best = try model.prediction(from: input).featureValue(for: "best")?.multiArrayValue else {
                throw ParadeeError.predictionFailed("G2P output 'best' missing")
            }
            let strides = best.strides.map(\.intValue)
            let rowStride = strides[0]
            let stepStride = strides[1]
            return try rows.enumerated().map { r, row in
                var steps: [Int32] = []
                steps.reserveCapacity(3 * row.count)
                for t in 0..<(3 * row.count) {
                    steps.append(try Self.int32(best, offset: r * rowStride + t * stepStride))
                }
                return data.decode(steps)
            }
        } catch let error as ParadeeError {
            throw error
        } catch {
            throw ParadeeError.predictionFailed("G2P: \(error)")
        }
    }

    private static func int32(_ array: MLMultiArray, offset: Int) throws -> Int32 {
        switch array.dataType {
        case .int32: return array.dataPointer.bindMemory(to: Int32.self, capacity: offset + 1)[offset]
        case .float32: return Int32(array.dataPointer.bindMemory(to: Float.self, capacity: offset + 1)[offset])
        default: throw ParadeeError.predictionFailed("G2P output 'best' has data type \(array.dataType.rawValue)")
        }
    }

    // MARK: - Text (upstream `__call__` preprocessing, `_words`, `_contexts`)

    static let percentWords: [ParadeeLanguage: String] = [.hindi: "प्रतिशत", .french: "pour cent", .spanish: "por ciento"]
    private static let zeroWidth: Set<Unicode.Scalar> = ["\u{200B}", "\u{200C}", "\u{200D}", "\u{FEFF}"]

    /// Devanagari digits → ASCII, `«»` → `“”`, zero-width characters removed,
    /// a space after `।`/`॥` when text follows, French digit groups joined
    /// (`12 000`), and `%` spelled out.
    static func prepare(_ text: String, language: ParadeeLanguage, percent: String) -> [Unicode.Scalar] {
        var s: [Unicode.Scalar] = []
        for scalar in text.precomposedStringWithCanonicalMapping.unicodeScalars {
            switch scalar.value {
            case 0x0966...0x096F: s.append(Unicode.Scalar(UInt8(scalar.value - 0x0966) + 0x30))
            case 0x00AB: s.append("\u{201C}")
            case 0x00BB: s.append("\u{201D}")
            default: if !zeroWidth.contains(scalar) { s.append(scalar) }
            }
        }
        var spaced: [Unicode.Scalar] = []
        for (i, scalar) in s.enumerated() {
            spaced.append(scalar)
            if scalar == "।" || scalar == "॥", i + 1 < s.count, !isSpace(s[i + 1]) { spaced.append(" ") }
        }
        s = spaced
        if language == .french {  // (?<=\d)[   ](?=\d{3}\b)
            var joined: [Unicode.Scalar] = []
            for (i, scalar) in s.enumerated() {
                let separator = scalar == " " || scalar == "\u{00A0}" || scalar == "\u{202F}"
                if separator, i > 0, isDecimal(s[i - 1]), i + 3 < s.count, s[(i + 1)...(i + 3)].allSatisfy(isDecimal),
                    i + 4 == s.count || !isWordCharacter(s[i + 4])
                {
                    continue
                }
                joined.append(scalar)
            }
            s = joined
        }
        // (\d)\s?% -> "\1 <percent>"
        var out: [Unicode.Scalar] = []
        var i = 0
        while i < s.count {
            if isDecimal(s[i]) {
                if i + 1 < s.count, s[i + 1] == "%" {
                    out.append(s[i])
                    out += (" " + percent).unicodeScalars
                    i += 2
                    continue
                }
                if i + 2 < s.count, isSpace(s[i + 1]), s[i + 2] == "%" {
                    out.append(s[i])
                    out += (" " + percent).unicodeScalars
                    i += 3
                    continue
                }
            }
            out.append(s[i])
            i += 1
        }
        return out
    }

    /// Python `str.split()`: runs of non-whitespace.
    static func splitWhitespace(_ s: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        var out: [[Unicode.Scalar]] = []
        var current: [Unicode.Scalar] = []
        for scalar in s {
            if isSpace(scalar) {
                if !current.isEmpty { out.append(current) }
                current = []
            } else {
                current.append(scalar)
            }
        }
        if !current.isEmpty { out.append(current) }
        return out
    }

    struct Word: Equatable {
        /// The piece without the punctuation, symbols and spaces at its ends.
        let core: [Unicode.Scalar]
        /// Where `core` starts in the piece.
        let start: Int
        let lead: Bool
        let trail: Bool
    }

    /// A whitespace-separated piece as its word and whether punctuation comes
    /// before / after it; `nil` for a piece with no letters or digits.
    static func word(_ raw: [Unicode.Scalar]) -> Word? {
        var i = 0
        var j = raw.count
        while i < j, isEdge(raw[i]) { i += 1 }
        while j > i, isEdge(raw[j - 1]) { j -= 1 }
        let core = Array(raw[i..<j])
        guard core.contains(where: isAlphanumeric) else { return nil }
        return Word(core: core, start: i, lead: i > 0, trail: j < raw.count)
    }

    /// (before, after) for each word: the last letter of the word before and
    /// the first letter of the word after, lowercased, or `#` when punctuation
    /// or the end of the sentence is in between.
    static func contexts(_ words: [Word?]) -> [(before: String, after: String)?] {
        words.indices.map { i -> (before: String, after: String)? in
            guard let w = words[i] else { return nil }
            var before = "#"
            if i > 0, !w.lead, let previous = words[i - 1], !previous.trail, let letter = previous.core.last {
                before = String(letter).lowercased()
            }
            var after = "#"
            if i + 1 < words.count, !w.trail, let next = words[i + 1], !next.lead, let letter = next.core.first {
                after = String(letter).lowercased()
            }
            return (before, after)
        }
    }

    /// `re.findall(r"\d+|[^\d]+", core)`.
    static func digitRuns(_ s: [Unicode.Scalar]) -> [(isDigits: Bool, scalars: [Unicode.Scalar])] {
        var out: [(isDigits: Bool, scalars: [Unicode.Scalar])] = []
        for scalar in s {
            let digit = isDecimal(scalar)
            if let last = out.last, last.isDigits == digit {
                out[out.count - 1].scalars.append(scalar)
            } else {
                out.append((digit, [scalar]))
            }
        }
        return out
    }

    static func strip(_ s: [Unicode.Scalar], characters: String) -> [Unicode.Scalar] {
        let set = Set(characters.unicodeScalars)
        var i = 0
        var j = s.count
        while i < j, set.contains(s[i]) { i += 1 }
        while j > i, set.contains(s[j - 1]) { j -= 1 }
        return Array(s[i..<j])
    }

    static func value(_ digits: [Unicode.Scalar]) -> Int {
        digits.reduce(0) { $0 * 10 + Int($1.properties.numericValue ?? 0) }
    }

    static func string(_ s: [Unicode.Scalar]) -> String {
        String(String.UnicodeScalarView(s))
    }

    // MARK: - Character classes (Python `str` semantics)

    /// Punctuation, symbols, spaces and zero-width characters (upstream `_is_edge`).
    static func isEdge(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation, .initialPunctuation,
            .finalPunctuation, .otherPunctuation, .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol,
            .spaceSeparator, .lineSeparator, .paragraphSeparator:
            return true
        default:
            return zeroWidth.contains(c)
        }
    }

    /// `str.isalnum()`: letters (L*) and anything with a numeric value.
    static func isAlphanumeric(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return c.properties.numericType != nil
        }
    }

    /// Regex `\d`: decimal digits (Nd) of any script.
    static func isDecimal(_ c: Unicode.Scalar) -> Bool {
        c.properties.generalCategory == .decimalNumber
    }

    /// Regex `\w`.
    static func isWordCharacter(_ c: Unicode.Scalar) -> Bool {
        c == "_" || isAlphanumeric(c)
    }

    /// `str.isspace()`: White_Space plus the ASCII separators U+001C–U+001F.
    static func isSpace(_ c: Unicode.Scalar) -> Bool {
        c.properties.isWhitespace || (0x1C...0x1F).contains(c.value)
    }
}

/// `paradee_g2p.json`: the model's symbol tables, number pieces and word lists
/// (upstream `paradee/data/g2p.json.gz`, unzipped).
struct ParadeeG2PData: Sendable {

    struct Lexicon: Sendable {
        /// Word → phonemes.
        let words: [String: String]
        /// `word\tbefore\tafter` → phonemes, for words whose reading depends on
        /// the neighbouring letters.
        let inContext: [String: String]
    }

    /// Model input symbols: `<pad>`, `<s1>`, `<s2>`, `<hi>` … `<en-gb>`,
    /// `<unk>` and single letters.
    let inputs: [Unicode.Scalar: Int32]
    let specials: [String: Int32]
    /// CTC output symbols; 0 is the blank.
    let outputs: [String]
    /// Language → number piece → phonemes.
    let numbers: [String: [String: String]]
    let lexicon: [String: Lexicon]
    /// English word lists keyed by lowercased word: the English frontend hands
    /// over lowercased words, and some listed words are capitalized names.
    let englishLowercased: [String: [String: String]]

    static func contextKey(_ word: String, _ before: String, _ after: String) -> String {
        "\(word)\t\(before)\t\(after)"
    }

    static func load(from url: URL) throws -> ParadeeG2PData {
        struct Raw: Decodable {
            struct Vocab: Decodable {
                let `in`: [String]
                let out: [String]
            }
            struct Lexicon: Decodable {
                let w: [String: String]
                let c: [String: String]
            }
            let vocab: Vocab
            let numbers: [String: [String: String]]
            let lexicon: [String: Lexicon]
        }
        let raw: Raw
        do {
            raw = try JSONDecoder().decode(Raw.self, from: Data(contentsOf: url))
        } catch {
            throw ParadeeError.corruptedModel(url.lastPathComponent, underlying: "\(error)")
        }
        var inputs: [Unicode.Scalar: Int32] = [:]
        var specials: [String: Int32] = [:]
        for (index, symbol) in raw.vocab.in.enumerated() {
            let scalars = Array(symbol.unicodeScalars)
            if scalars.count == 1 {
                inputs[scalars[0]] = Int32(index)
            } else {
                specials[symbol] = Int32(index)
            }
        }
        for required in ["<pad>", "<s1>", "<s2>", "<unk>"] + ParadeeLanguage.allCases.map({ "<\($0.rawValue)>" })
        where specials[required] == nil {
            throw ParadeeError.corruptedModel(url.lastPathComponent, underlying: "symbol \(required) missing")
        }
        var englishLowercased: [String: [String: String]] = [:]
        for language in [ParadeeLanguage.americanEnglish, .britishEnglish] {
            guard let words = raw.lexicon[language.rawValue]?.w else { continue }
            var lowered: [String: String] = [:]
            for (word, phonemes) in words.sorted(by: { $0.key < $1.key }) where lowered[word.lowercased()] == nil {
                lowered[word.lowercased()] = words[word.lowercased()] ?? phonemes
            }
            englishLowercased[language.rawValue] = lowered
        }
        return ParadeeG2PData(
            inputs: inputs, specials: specials, outputs: raw.vocab.out, numbers: raw.numbers,
            lexicon: raw.lexicon.mapValues { Lexicon(words: $0.w, inContext: $0.c) },
            englishLowercased: englishLowercased)
    }

    /// `<lang> before <s1> letters <s2> after`, letters lowercased and cut at
    /// ``ParadeeG2P/maxWordLength`` (upstream `G2P.model`'s rows).
    func row(_ language: ParadeeLanguage, _ before: String, _ word: String, _ after: String) -> [Int32] {
        let unk = specials["<unk>"] ?? 0
        func symbol(_ s: String) -> Int32 {
            let scalars = Array(s.unicodeScalars)
            return scalars.count == 1 ? inputs[scalars[0]] ?? unk : unk
        }
        let letters = word.lowercased().unicodeScalars.prefix(ParadeeG2P.maxWordLength).map { inputs[$0] ?? unk }
        return [specials["<\(language.rawValue)>"] ?? unk, symbol(before), specials["<s1>"] ?? unk] + letters
            + [specials["<s2>"] ?? unk, symbol(after)]
    }

    /// CTC best path → phonemes: merge repeats, drop blanks (0).
    func decode(_ steps: [Int32]) -> String {
        var out = ""
        var previous: Int32 = 0
        for k in steps {
            if k != previous, k != 0, Int(k) < outputs.count { out += outputs[Int(k)] }
            previous = k
        }
        return out
    }
}
#endif
