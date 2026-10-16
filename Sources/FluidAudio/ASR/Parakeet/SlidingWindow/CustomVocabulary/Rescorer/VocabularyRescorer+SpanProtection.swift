import Foundation

extension VocabularyRescorer {
    /// A correction changes the name, not an existing singular possessive inflection.
    static func preservingPossessive(_ replacement: String, from original: String) -> String {
        let punctuation = CharacterSet.punctuationCharacters.subtracting(CharacterSet(charactersIn: "'’"))
        let lexicalOriginal = original.trimmingCharacters(in: punctuation)
        let suffix = String(lexicalOriginal.suffix(2))
        guard lexicalOriginal.count > 2, ["'s", "’s"].contains(suffix.lowercased()) else { return replacement }
        let normalizedReplacement = replacement.replacingOccurrences(of: "’", with: "'").lowercased()
        guard !normalizedReplacement.hasSuffix("'s") else { return replacement }
        return replacement + suffix
    }

    static func isCanonicalMatch(_ original: String, canonical: String) -> Bool {
        let inflected = preservingPossessive(canonical, from: original)
        return normalizeForSimilarity(original.replacingOccurrences(of: "’", with: "'"))
            == normalizeForSimilarity(inflected.replacingOccurrences(of: "’", with: "'"))
    }

    /// A one-edit name spelling or a registered alias supports a genuine
    /// repetition. An unrelated neighboring word must not gain that same bonus.
    static func isTextSupportedRepetition(_ candidate: CTCMatchCandidate) -> Bool {
        if candidate.similarity == 1 { return true }  // Exact alias or compound spelling, including BK-tree matches.
        func lexicalCore(_ text: String) -> String {
            let normalized = normalizeForSimilarity(text.replacingOccurrences(of: "’", with: "'"))
            return normalized.hasSuffix("'s") ? String(normalized.dropLast(2)) : normalized
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
            let canonical = Self.normalizeForSimilarity(term.text.replacingOccurrences(of: "’", with: "'"))
            let count = canonical.split(separator: " ").count
            guard count > 0, count <= wordTimings.count else { continue }
            canonicalByLength[count, default: []].insert(canonical)
        }
        let words = wordTimings.map {
            Self.normalizeForSimilarity($0.word.replacingOccurrences(of: "’", with: "'"))
        }
        var ranges = Set<Range<Int>>()
        var byCanonical: [String: [Range<Int>]] = [:]
        for (count, canonicals) in canonicalByLength {
            for start in 0...(words.count - count) {
                let range = start..<(start + count)
                let phrase = words[range].filter { !$0.isEmpty }.joined(separator: " ")
                var forms = [phrase]
                if phrase.hasSuffix("'s"), phrase.count > 2 {
                    forms.append(String(phrase.dropLast(2)))
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

    /// Keep recognized words at their existing offsets within a replacement.
    /// This permits "John Collar" → "John Collier" when "John" is also listed,
    /// while rejecting a wider replacement that would consume a recognized name.
    static func preservesRecognizedSpans(
        _ candidate: CTCMatchCandidate,
        recognizedSpans: RecognizedVocabularySpans,
        wordTimings: [WordTiming]
    ) -> Bool {
        guard let first = candidate.spanIndices.first, let last = candidate.spanIndices.last else { return false }
        let sourceRange = first..<(last + 1)
        let replacement = preservingPossessive(candidate.vocabTerm, from: candidate.originalPhrase)
        let words = replacement.split(separator: " ").map {
            normalizeForSimilarity(String($0).replacingOccurrences(of: "’", with: "'"))
        }
        for span in recognizedSpans.ranges where span.overlaps(sourceRange) {
            guard sourceRange.lowerBound <= span.lowerBound, sourceRange.upperBound >= span.upperBound else {
                return false
            }
            let offset = span.lowerBound - first
            guard offset + span.count <= words.count else { return false }
            let original = wordTimings[span].map {
                normalizeForSimilarity($0.word.replacingOccurrences(of: "’", with: "'"))
            }
            let retained = Array(words[offset..<(offset + span.count)])
            guard original == retained else { return false }
        }
        return true
    }

    /// The margin accommodates clock differences, but must not borrow a name
    /// already recognized in a different word span (e.g. "mister Quilter").
    /// Limit the search at those occurrences rather than disabling corrections
    /// for the term: another, misspelled occurrence can still be corrected.
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
        let canonical = normalizeForSimilarity(candidate.vocabTerm.replacingOccurrences(of: "’", with: "'"))
        guard let first = candidate.spanIndices.first, let last = candidate.spanIndices.last else {
            return (start..<max(start, end), false)
        }

        var createsAdjacentDuplicate = false
        for range in recognizedSpans.byCanonical[canonical, default: []] {
            let index = range.lowerBound
            if range.overlaps(first..<(last + 1)) { return (start..<start, false) }
            let anchorStart = wordTimings[index].startTime
            let anchorEnd = wordTimings[range.upperBound - 1].endTime
            // Coarse, overlapping clocks cannot establish which occurrence owns
            // a frame. Preserve independent text spans rather than emptying the
            // second occurrence's search window or removing its normal bias.
            if anchorStart < candidate.spanEndTime && anchorEnd > candidate.spanStartTime { continue }
            let gap = max(anchorStart - candidate.spanEndTime, candidate.spanStartTime - anchorEnd)
            if range.upperBound == first || range.lowerBound == last + 1, gap <= marginSeconds {
                createsAdjacentDuplicate = true
            }
            if range.upperBound <= first {
                start = max(start, Int(ceil(anchorEnd / frameDuration)))
            } else {
                end = min(end, Int(anchorStart / frameDuration))
            }
        }
        return (start..<max(start, end), createsAdjacentDuplicate)
    }
}
