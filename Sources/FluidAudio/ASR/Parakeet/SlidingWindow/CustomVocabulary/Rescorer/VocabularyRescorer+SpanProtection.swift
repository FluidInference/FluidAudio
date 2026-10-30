import Foundation

extension VocabularyRescorer {
    /// `normalizeForSimilarity` after folding the curly apostrophe (which it would drop) into `'`.
    static func normalizedLexical(_ text: String) -> String {
        normalizeForSimilarity(text.replacingOccurrences(of: "’", with: "'"))
    }

    /// Nearest CTC frame of a TDT timestamp. Word boundaries are frame multiples computed two ways
    /// (`start + duration` vs the next word's `frame * frameDuration`), so adjacent words can differ
    /// by a few ulp; rounding makes them abut exactly instead of overlapping by 1e-16 s.
    static func frameIndex(_ seconds: Double, frameDuration: Double) -> Int {
        Int((seconds / frameDuration).rounded())
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

        // The inflected word must end the name, not be a neighbour such as "it's": the stem's trailing letters
        // must resemble the term. Letters, so a split rendering ("Quilt er's") still counts.
        let stem = normalizedLexical(String(lexicalOriginal.dropLast(suffixLength)))
        let termLetters = letters(term)
        let tail = String(letters(stem).suffix(termLetters.count))
        guard stringSimilarity(tail, termLetters) >= 0.5 else { return replacement }

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

        var anchors: [Range<Int>] = []
        for (recognized, ranges) in recognizedSpans.byCanonical {
            if recognized == canonical {
                if ranges.contains(where: { $0.overlaps(sourceRange) }) { return (start..<start, false) }
                anchors.append(contentsOf: ranges)
            } else if isSubPhrase(recognized.split(separator: " "), of: canonicalWords) {
                anchors.append(contentsOf: ranges.filter { !$0.overlaps(sourceRange) })
            }
        }

        let spanStartFrame = frameIndex(candidate.spanStartTime, frameDuration: frameDuration)
        let spanEndFrame = frameIndex(candidate.spanEndTime, frameDuration: frameDuration)
        var createsAdjacentDuplicate = false
        for range in anchors {
            let anchorStartFrame = frameIndex(wordTimings[range.lowerBound].startTime, frameDuration: frameDuration)
            let anchorEndFrame = frameIndex(wordTimings[range.upperBound - 1].endTime, frameDuration: frameDuration)
            // Coarse, overlapping clocks cannot establish which occurrence owns
            // a frame. Preserve independent text spans rather than emptying the
            // second occurrence's search window or removing its normal bias.
            if anchorStartFrame < spanEndFrame && anchorEndFrame > spanStartFrame { continue }
            let gapFrames = max(anchorStartFrame - spanEndFrame, spanStartFrame - anchorEndFrame)
            if range.upperBound == first || range.lowerBound == last + 1,
                Double(gapFrames) * frameDuration <= marginSeconds
            {
                createsAdjacentDuplicate = true
            }
            if range.upperBound <= first {
                start = max(start, anchorEndFrame)
            } else {
                end = min(end, anchorStartFrame)
            }
        }
        return (start..<max(start, end), createsAdjacentDuplicate)
    }

    /// Whether `part` is a contiguous, shorter run of `whole`'s words.
    private static func isSubPhrase(_ part: [Substring], of whole: [Substring]) -> Bool {
        guard !part.isEmpty, part.count < whole.count else { return false }
        return (0...(whole.count - part.count)).contains { Array(whole[$0..<($0 + part.count)]) == part }
    }
}
