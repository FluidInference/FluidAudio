import Foundation

extension VocabularyRescorer {
    /// `normalizeForSimilarity` after folding the curly apostrophe (which it would drop) into `'`.
    static func normalizedLexical(_ text: String) -> String {
        normalizeForSimilarity(text.replacingOccurrences(of: "’", with: "'"))
    }

    /// Slack for comparing TDT timestamps. A word's end (`start + duration`) and the next word's start
    /// (`frame * 0.08`) are computed differently and can differ by a few ulp, and CTC frames need not share
    /// TDT's 0.08 s grid, so exact comparisons in seconds or in CTC frames flip on rounding noise.
    static let timingTolerance = 0.001

    /// First CTC frame at or after `seconds`, ignoring rounding noise.
    static func frameCeil(_ seconds: Double, frameDuration: Double) -> Int {
        Int(((seconds - timingTolerance) / frameDuration).rounded(.up))
    }

    /// Last CTC frame boundary at or before `seconds`, ignoring rounding noise.
    static func frameFloor(_ seconds: Double, frameDuration: Double) -> Int {
        Int(((seconds + timingTolerance) / frameDuration).rounded(.down))
    }

    /// A correction changes the name, not an existing possessive inflection: "Quiltor's" → "Quilter's",
    /// "Jonnes'" → "Jones'". Left alone when the apostrophe split the term's own final "s"
    /// ("Envious Lab's" → "Envious Labs") or belongs to a neighbouring word in the span ("Quiltor it's").
    static func preservingPossessive(_ replacement: String, from original: String) -> String {
        let punctuation = CharacterSet.punctuationCharacters.subtracting(CharacterSet(charactersIn: "'’"))
        let lexicalOriginal = original.trimmingCharacters(in: punctuation)
        let folded = lexicalOriginal.replacingOccurrences(of: "’", with: "'").lowercased()
        let suffixLength: Int
        if folded.hasSuffix("'s") {
            suffixLength = 2
        } else if folded.hasSuffix("s'") {
            suffixLength = 1
        } else {
            return replacement
        }
        guard lexicalOriginal.count > 2 else { return replacement }
        let term = normalizedLexical(replacement)
        guard !term.hasSuffix("'s"), !term.hasSuffix("'") else { return replacement }

        // In a multi-word span the inflected word must belong to the name, not be a neighbour such as "it's":
        // dropping it must make the span a worse match. Letters, so a split rendering ("Quilt er's") counts.
        let stem = normalizedLexical(String(lexicalOriginal.dropLast(suffixLength)))
        let stemWords = stem.split(separator: " ")
        if stemWords.count > 1 {
            let termLetters = letters(term)
            let withoutLastWord = stemWords.dropLast().joined(separator: " ")
            guard
                stringSimilarity(letters(stem), termLetters)
                    > stringSimilarity(letters(withoutLastWord), termLetters)
            else { return replacement }
        }

        if suffixLength == 1 {
            // A plural possessive ("Jonnes'") only fits a term that ends in "s".
            return term.hasSuffix("s") ? replacement + "'" : replacement
        }
        // "Lab's" for "Labs": the stem is closer to the term without its final "s" than to the term.
        if term.hasSuffix("s"),
            StringUtils.levenshteinDistance(stem, String(term.dropLast()))
                < StringUtils.levenshteinDistance(stem, term)
        {
            return replacement
        }
        return replacement + String(lexicalOriginal.suffix(2))
    }

    /// Whether `original` already spells `canonical`, allowing a possessive inflection.
    static func isCanonicalMatch(_ original: String, canonical: String) -> Bool {
        isCanonicalMatch(lexical: normalizedLexical(original), canonicalKey: normalizedLexical(canonical))
    }

    /// `isCanonicalMatch` on precomputed `normalizedLexical` forms, for the term × word loops.
    static func isCanonicalMatch(lexical: String, canonicalKey: String) -> Bool {
        if lexical == canonicalKey { return true }
        if lexical.hasSuffix("'s"), !canonicalKey.hasSuffix("'s"), lexical.dropLast(2) == canonicalKey {
            return true
        }
        return lexical.hasSuffix("s'") && canonicalKey.hasSuffix("s") && lexical.dropLast() == canonicalKey
    }

    /// The `normalizedLexical` words of `indices`, joined like `normalizedLexical` of the joined words.
    static func lexicalPhrase(_ lexicalWords: [String], _ indices: [Int]) -> String {
        indices.map { lexicalWords[$0] }.filter { !$0.isEmpty }.joined(separator: " ")
    }

    /// A one-edit name spelling or a registered alias supports a genuine
    /// repetition. An unrelated neighboring word must not gain that same bonus.
    static func isTextSupportedRepetition(_ candidate: CTCMatchCandidate) -> Bool {
        if candidate.similarity == 1 { return true }  // Exact alias or compound spelling, including BK-tree matches.
        func lexicalCore(_ text: String) -> String {
            let normalized = normalizedLexical(text)
            if normalized.hasSuffix("'s") { return String(normalized.dropLast(2)) }
            if normalized.hasSuffix("s'") { return String(normalized.dropLast()) }
            return normalized
        }
        let original = lexicalCore(candidate.originalPhrase)
        let forms = [candidate.vocabTerm] + (candidate.matchedAlias.map { [$0] } ?? [])
        return forms.contains { StringUtils.levenshteinDistance(original, lexicalCore($0)) <= 1 }
    }

    struct RecognizedVocabularySpans {
        let ranges: [Range<Int>]
        let byCanonical: [String: [Range<Int>]]
    }

    /// Index recognized phrases once per request. Normalize each term/word once,
    /// then look up only the phrase lengths present in the vocabulary.
    static func recognizedVocabularySpans(
        in wordTimings: [WordTiming], canonicalTerms: [CustomVocabularyTerm]
    ) -> RecognizedVocabularySpans {
        var canonicalByLength: [Int: Set<String>] = [:]
        for term in canonicalTerms {
            let canonical = normalizedLexical(term.text)
            let count = canonical.split(separator: " ").count
            guard count > 0, count <= wordTimings.count else { continue }
            canonicalByLength[count, default: []].insert(canonical)
        }
        let words = wordTimings.map { normalizedLexical($0.word) }
        var ranges = Set<Range<Int>>()
        var byCanonical: [String: [Range<Int>]] = [:]
        for (count, canonicals) in canonicalByLength {
            for start in 0...(words.count - count) {
                let range = start..<(start + count)
                let phrase = words[range].filter { !$0.isEmpty }.joined(separator: " ")
                var forms = [phrase]
                if phrase.hasSuffix("'s"), phrase.count > 2 {
                    forms.append(String(phrase.dropLast(2)))
                } else if phrase.hasSuffix("s'"), phrase.count > 2 {
                    forms.append(String(phrase.dropLast()))
                }
                for form in forms where canonicals.contains(form) {
                    ranges.insert(range)
                    byCanonical[form, default: []].append(range)
                }
            }
        }
        return RecognizedVocabularySpans(
            ranges: ranges.sorted {
                $0.lowerBound == $1.lowerBound ? $0.upperBound < $1.upperBound : $0.lowerBound < $1.lowerBound
            }, byCanonical: byCanonical)
    }

    /// A replacement may not swallow a recognized vocabulary phrase inside its span. The recognized letters
    /// must survive in the replacement on the same side of the surrounding words: if the span had words
    /// before (after) the recognized phrase, the replacement needs letters before (after) it too. This
    /// permits "John Collar" → "John Collier" and "Open AI" → "OpenAI" when "John" or "AI" is also listed,
    /// and rejects "in Envious Labs" → "Envious Labs" or "mister Quilter" → "Quilter".
    static func preservesRecognizedSpans(
        _ candidate: CTCMatchCandidate,
        recognizedSpans: RecognizedVocabularySpans,
        wordTimings: [WordTiming]
    ) -> Bool {
        guard let first = candidate.spanIndices.first, let last = candidate.spanIndices.last else { return false }
        let sourceRange = first..<(last + 1)
        let replacement = letters(preservingPossessive(candidate.vocabTerm, from: candidate.originalPhrase))
        for span in recognizedSpans.ranges where span.overlaps(sourceRange) {
            guard sourceRange.lowerBound <= span.lowerBound, sourceRange.upperBound >= span.upperBound else {
                return false
            }
            let recognized = letters(wordTimings[span].map(\.word).joined())
            guard !recognized.isEmpty else { continue }
            guard
                occurs(
                    recognized, in: replacement,
                    withTextBefore: span.lowerBound > sourceRange.lowerBound,
                    withTextAfter: span.upperBound < sourceRange.upperBound)
            else { return false }
        }
        return true
    }

    /// Lowercased letters and digits of `text`.
    private static func letters(_ text: String) -> String {
        String(normalizedLexical(text).filter { $0.isLetter || $0.isNumber })
    }

    /// Whether `needle` occurs in `haystack` with text before it exactly when `withTextBefore`, and text after
    /// it exactly when `withTextAfter`.
    private static func occurs(
        _ needle: String, in haystack: String, withTextBefore: Bool, withTextAfter: Bool
    ) -> Bool {
        var searchStart = haystack.startIndex
        while searchStart < haystack.endIndex,
            let found = haystack.range(of: needle, range: searchStart..<haystack.endIndex)
        {
            let hasBefore = found.lowerBound > haystack.startIndex
            let hasAfter = found.upperBound < haystack.endIndex
            if hasBefore == withTextBefore && hasAfter == withTextAfter { return true }
            searchStart = haystack.index(after: found.lowerBound)
        }
        return false
    }

    /// The margin accommodates clock differences, but must not borrow a name
    /// already recognized in a different word span (e.g. "mister Quilter").
    /// Limit the search at those occurrences rather than disabling corrections
    /// for the term: another, misspelled occurrence can still be corrected.
    /// A recognized listed term that is part of the candidate's own name (a
    /// listed surname beside "Chandra Shankar" for "Chandra Shekhar Solanki")
    /// bounds the search the same way, so the full name cannot reuse it.
    static func protectedSearchWindow(
        for candidate: CTCMatchCandidate,
        wordTimings: [WordTiming],
        recognizedSpans: RecognizedVocabularySpans,
        frameDuration: Double,
        frameCount: Int,
        marginSeconds: Double
    ) -> (frames: Range<Int>, createsAdjacentDuplicate: Bool) {
        let marginFrames = Int(marginSeconds / frameDuration)
        var start = max(0, Int(candidate.spanStartTime / frameDuration) - marginFrames)
        var end = min(frameCount, Int(candidate.spanEndTime / frameDuration) + marginFrames)
        guard let first = candidate.spanIndices.first, let last = candidate.spanIndices.last else {
            return (start..<max(start, end), false)
        }
        let sourceRange = first..<(last + 1)
        let canonical = normalizedLexical(candidate.vocabTerm)
        let canonicalWords = canonical.split(separator: " ")
        let sourceWords = wordTimings[sourceRange].map { normalizedLexical($0.word) }.filter { !$0.isEmpty }
            .flatMap { $0.split(separator: " ") }

        // Anchors: recognized occurrences of the term, and recognized listed terms that would be duplicated
        // beside the span: a sub-phrase that starts the name and sits before the span (or ends it and sits
        // after), when the span does not already start (end) with it.
        var anchors: [Range<Int>] = []
        for (recognized, ranges) in recognizedSpans.byCanonical {
            if recognized == canonical {
                if ranges.contains(where: { $0.overlaps(sourceRange) }) { return (start..<start, false) }
                anchors.append(contentsOf: ranges)
                continue
            }
            let part = recognized.split(separator: " ")
            guard part.count < canonicalWords.count else { continue }
            let startsName =
                Array(canonicalWords.prefix(part.count)) == part
                && Array(sourceWords.prefix(part.count)) != part
            let endsName =
                Array(canonicalWords.suffix(part.count)) == part
                && Array(sourceWords.suffix(part.count)) != part
            anchors.append(
                contentsOf: ranges.filter {
                    !$0.overlaps(sourceRange) && ($0.upperBound <= first ? startsName : endsName)
                })
        }

        var createsAdjacentDuplicate = false
        for range in anchors {
            let anchorStart = wordTimings[range.lowerBound].startTime
            let anchorEnd = wordTimings[range.upperBound - 1].endTime
            // Coarse, overlapping clocks cannot establish which occurrence owns
            // a frame. Preserve independent text spans rather than emptying the
            // second occurrence's search window or removing its normal bias.
            let overlap = min(anchorEnd, candidate.spanEndTime) - max(anchorStart, candidate.spanStartTime)
            if overlap > timingTolerance { continue }
            let gap = max(anchorStart - candidate.spanEndTime, candidate.spanStartTime - anchorEnd)
            if range.upperBound == first || range.lowerBound == last + 1, gap <= marginSeconds + timingTolerance {
                createsAdjacentDuplicate = true
            }
            if range.upperBound <= first {
                start = max(start, frameCeil(anchorEnd, frameDuration: frameDuration))
            } else {
                end = min(end, frameFloor(anchorStart, frameDuration: frameDuration))
            }
        }
        return (start..<max(start, end), createsAdjacentDuplicate)
    }
}
