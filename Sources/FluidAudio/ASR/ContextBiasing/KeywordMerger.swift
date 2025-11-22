import Foundation
import NaturalLanguage
import OSLog

/// Merges CTC keyword detections into an existing transcript by replacing similar words
/// with their canonical forms. This allows using CTC keyword spotting with any ASR system.
public struct KeywordMerger {

    private static let logger = Logger(subsystem: "com.fluidaudio", category: "KeywordMerger")

    /// Check if a word is likely a common word (not a proper noun) using NaturalLanguage framework
    /// - Parameters:
    ///   - word: The word to check
    ///   - transcript: Full transcript for context
    ///   - wordIndex: Index of the word in the transcript
    private static func isCommonWord(_ word: String, in transcript: String, at wordIndex: Int) -> Bool {
        // Use full transcript for context so NLTagger can properly identify word types
        let tagger = NLTagger(tagSchemes: [.lexicalClass, .nameType])
        tagger.string = transcript

        // Find the word's position in the transcript
        let words = transcript.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
        guard wordIndex < words.count else { return false }

        // Calculate character position of this word
        var charPosition = 0
        for i in 0..<wordIndex {
            charPosition += words[i].count + 1  // +1 for space
        }

        guard
            let startIndex = transcript.index(
                transcript.startIndex, offsetBy: charPosition, limitedBy: transcript.endIndex)
        else {
            return false
        }

        let lexicalTag = tagger.tag(at: startIndex, unit: .word, scheme: .lexicalClass).0
        let nameTag = tagger.tag(at: startIndex, unit: .word, scheme: .nameType).0

        // If it's a named entity (person, place, organization), it's NOT a common word
        if let tag = nameTag {
            // Check if it's actually a named entity (not just "OtherWord")
            if tag == .personalName || tag == .placeName || tag == .organizationName {
                return false  // It's a proper noun, allow replacement
            }
        }

        // Common word types that shouldn't be replaced: verbs, pronouns, determiners, prepositions, etc.
        switch lexicalTag {
        case .verb, .pronoun, .determiner, .preposition, .conjunction, .adverb, .particle, .interjection:
            return true
        default:
            return false
        }
    }

    /// Result of applying keyword corrections to a transcript
    public struct MergeResult {
        public let correctedText: String
        public let replacements: [Replacement]

        public struct Replacement {
            public let originalText: String
            public let canonicalText: String
            public let wordIndex: Int
            public let spanLength: Int
            public let similarity: Float
            public let combinedConfidence: Float
        }
    }

    /// Apply CTC keyword detections using token-level matching
    /// - Parameters:
    ///   - detections: Keywords detected by CTC keyword spotter
    ///   - transcript: Original transcript to correct
    ///   - ctcTokenSequence: Greedy-decoded CTC token sequence from audio
    ///   - vocabulary: Custom vocabulary context with correction thresholds
    ///   - debugMode: Enable debug logging
    /// - Returns: Corrected transcript and list of replacements made
    public static func applyCorrectionsWithTokens(
        detections: [CtcKeywordSpotter.KeywordDetection],
        toTranscript transcript: String,
        ctcTokenSequence: [(tokenId: Int, frameIndex: Int)],
        vocabulary: CustomVocabularyContext,
        debugMode: Bool = false
    ) -> MergeResult {

        let minSimilarity = vocabulary.minSimilarity
        let minCombinedConfidence = vocabulary.minCombinedConfidence

        guard !vocabulary.terms.isEmpty else {
            return MergeResult(correctedText: transcript, replacements: [])
        }

        guard !detections.isEmpty else {
            return MergeResult(correctedText: transcript, replacements: [])
        }

        guard !ctcTokenSequence.isEmpty else {
            if debugMode {
                logger.info("KeywordMerger: No CTC token sequence available, falling back to string matching")
            }
            return applyCorrections(
                detections: detections,
                toTranscript: transcript,
                vocabulary: vocabulary,
                debugMode: debugMode
            )
        }

        var words = transcript.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
        guard !words.isEmpty else {
            return MergeResult(correctedText: transcript, replacements: [])
        }

        if debugMode {
            logger.info(
                "KeywordMerger (Token Mode): Processing \(detections.count) detections, \(ctcTokenSequence.count) CTC tokens"
            )
        }

        struct InternalReplacement {
            let wordIndex: Int
            let canonical: String
            let similarity: Float
            let ctcScore: Float
            let originalWord: String

            var combinedConfidence: Float {
                let normalizedCtcScore = max(0, min(1, (ctcScore + 10) / 5))
                return 0.6 * similarity + 0.4 * normalizedCtcScore
            }
        }

        var replacements: [InternalReplacement] = []

        // For each detection, find matching token sequence in CTC output
        for detection in detections {
            let canonical = detection.term.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !canonical.isEmpty else { continue }

            // Skip if canonical form already exists verbatim
            if transcript.range(of: canonical, options: [.caseInsensitive]) != nil {
                if debugMode {
                    logger.info("KeywordMerger: '\(canonical)' already exists verbatim, skipping")
                }
                continue
            }

            // Get the keyword's token IDs
            guard let keywordTokenIds = detection.term.ctcTokenIds else {
                if debugMode {
                    logger.info("KeywordMerger: '\(canonical)' has no CTC token IDs, skipping")
                }
                continue
            }

            // Find matching token subsequence in CTC output near the detection frames
            let detectionFrameRange = detection.startFrame...detection.endFrame
            let searchWindow = 10  // frames before/after to search

            // Extract CTC tokens in the search window
            let relevantTokens = ctcTokenSequence.filter { token in
                let frame = token.frameIndex
                return frame >= (detection.startFrame - searchWindow)
                    && frame <= (detection.endFrame + searchWindow)
            }

            if relevantTokens.isEmpty {
                continue
            }

            // Find best matching subsequence using token-level edit distance
            var bestMatch: (startIdx: Int, similarity: Float)?
            let relevantTokenIds = relevantTokens.map { $0.tokenId }

            if debugMode {
                logger.info(
                    "  Searching for '\(canonical)' tokens=\(keywordTokenIds) in window with \(relevantTokenIds.count) tokens"
                )
            }

            for startIdx in 0..<relevantTokenIds.count {
                let endIdx = min(startIdx + keywordTokenIds.count, relevantTokenIds.count)
                let windowTokens = Array(relevantTokenIds[startIdx..<endIdx])

                let similarity = tokenSimilarity(windowTokens, keywordTokenIds)

                if debugMode && similarity > 0.3 {
                    logger.info(
                        "    Window[\(startIdx)]: tokens=\(windowTokens) sim=\(String(format: "%.2f", similarity))")
                }

                if similarity >= minSimilarity {
                    if let existing = bestMatch {
                        if similarity > existing.similarity {
                            bestMatch = (startIdx, similarity)
                        }
                    } else {
                        bestMatch = (startIdx, similarity)
                    }
                }
            }

            if let match = bestMatch {
                // Map token index back to word index (approximate)
                // For now, use frame timing to estimate word position
                let matchFrameIdx = relevantTokens[match.startIdx].frameIndex
                let wordIndex = estimateWordIndex(
                    frameIndex: matchFrameIdx,
                    totalWords: words.count,
                    detectionFrames: detectionFrameRange
                )

                if wordIndex < words.count {
                    let replacement = InternalReplacement(
                        wordIndex: wordIndex,
                        canonical: canonical,
                        similarity: match.similarity,
                        ctcScore: detection.score,
                        originalWord: words[wordIndex]
                    )

                    if replacement.combinedConfidence >= minCombinedConfidence {
                        replacements.append(replacement)

                        if debugMode {
                            logger.info(
                                "KeywordMerger: Token match '\(replacement.originalWord)' -> '\(canonical)' (token_sim: \(String(format: "%.2f", match.similarity)), conf: \(String(format: "%.2f", replacement.combinedConfidence)))"
                            )
                        }
                    }
                }
            }
        }

        guard !replacements.isEmpty else {
            if debugMode {
                logger.info("KeywordMerger: No suitable token-based replacements found")
            }
            return MergeResult(correctedText: transcript, replacements: [])
        }

        // Apply replacements (same logic as string-based version)
        replacements.sort { $0.similarity > $1.similarity }

        var usedIndices = Set<Int>()
        var finalReplacements: [InternalReplacement] = []

        for replacement in replacements {
            if !usedIndices.contains(replacement.wordIndex) {
                finalReplacements.append(replacement)
                usedIndices.insert(replacement.wordIndex)
            }
        }

        finalReplacements.sort { $0.wordIndex > $1.wordIndex }

        var publicReplacements: [MergeResult.Replacement] = []

        for replacement in finalReplacements {
            publicReplacements.append(
                MergeResult.Replacement(
                    originalText: replacement.originalWord,
                    canonicalText: replacement.canonical,
                    wordIndex: replacement.wordIndex,
                    spanLength: 1,
                    similarity: replacement.similarity,
                    combinedConfidence: replacement.combinedConfidence
                ))

            words[replacement.wordIndex] = replacement.canonical
        }

        let correctedText = words.joined(separator: " ")

        if debugMode {
            logger.info("KeywordMerger: Applied \(publicReplacements.count) token-based replacements")
        }

        return MergeResult(correctedText: correctedText, replacements: publicReplacements)
    }

    /// Estimate word index from frame index (approximate mapping)
    private static func estimateWordIndex(frameIndex: Int, totalWords: Int, detectionFrames: ClosedRange<Int>) -> Int {
        // Simple linear interpolation based on frame position
        let relativePosition = Float(frameIndex - detectionFrames.lowerBound) / Float(detectionFrames.count)
        return min(Int(relativePosition * Float(totalWords)), totalWords - 1)
    }

    /// Compute token-level similarity using normalized edit distance
    private static func tokenSimilarity(_ a: [Int], _ b: [Int]) -> Float {
        let distance = tokenLevenshteinDistance(a, b)
        let maxLen = max(a.count, b.count)
        guard maxLen > 0 else { return 1.0 }
        return 1.0 - Float(distance) / Float(maxLen)
    }

    /// Compute Levenshtein distance between token sequences
    private static func tokenLevenshteinDistance(_ a: [Int], _ b: [Int]) -> Int {
        let m = a.count
        let n = b.count

        guard m > 0 else { return n }
        guard n > 0 else { return m }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m {
            dp[i][0] = i
        }
        for j in 0...n {
            dp[0][j] = j
        }

        for i in 1...m {
            for j in 1...n {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            }
        }

        return dp[m][n]
    }

    /// Apply CTC keyword detections to correct a transcript
    /// - Parameters:
    ///   - detections: Keywords detected by CTC keyword spotter
    ///   - transcript: Original transcript to correct
    ///   - vocabulary: Custom vocabulary context with correction thresholds
    ///   - debugMode: Enable debug logging
    /// - Returns: Corrected transcript and list of replacements made
    public static func applyCorrections(
        detections: [CtcKeywordSpotter.KeywordDetection],
        toTranscript transcript: String,
        vocabulary: CustomVocabularyContext,
        debugMode: Bool = false
    ) -> MergeResult {

        let minSimilarity = vocabulary.minSimilarity
        let minCombinedConfidence = vocabulary.minCombinedConfidence

        guard !vocabulary.terms.isEmpty else {
            return MergeResult(correctedText: transcript, replacements: [])
        }

        guard !detections.isEmpty else {
            return MergeResult(correctedText: transcript, replacements: [])
        }

        // Split transcript into words
        var words = transcript.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
        guard !words.isEmpty else {
            return MergeResult(correctedText: transcript, replacements: [])
        }

        if debugMode {
            logger.info("KeywordMerger: Processing \(detections.count) detections for \(words.count) words")
        }

        // Internal replacement structure with confidence scoring
        struct InternalReplacement {
            let wordIndex: Int
            let canonical: String
            let similarity: Float
            let ctcScore: Float
            let originalWord: String

            var combinedConfidence: Float {
                // Normalize CTC score from [-10, -5] range to [0, 1]
                let normalizedCtcScore = max(0, min(1, (ctcScore + 10) / 5))
                // Weight similarity more heavily (60%) than CTC confidence (40%)
                return 0.6 * similarity + 0.4 * normalizedCtcScore
            }
        }

        var replacements: [InternalReplacement] = []

        // Find best matching words for each detection
        for detection in detections {
            let canonical = detection.term.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !canonical.isEmpty else { continue }

            // Build list of all forms to check (canonical + aliases)
            var allForms = [canonical]
            if let aliases = detection.term.aliases {
                allForms.append(contentsOf: aliases)
            }

            // Skip only if canonical form already exists (aliases should be replaced)
            if transcript.range(of: canonical, options: [.caseInsensitive]) != nil {
                if debugMode {
                    logger.info("KeywordMerger: '\(canonical)' already exists verbatim, skipping")
                }
                continue
            }

            // Find best matching word or phrase in transcript
            var bestMatch: (index: Int, similarity: Float, spanLength: Int)?

            // Try each form (canonical + aliases)
            for formToMatch in allForms {
                let formWords = formToMatch.split(whereSeparator: { $0.isWhitespace }).map { String($0) }

                // Try single-word matches
                for (i, word) in words.enumerated() {
                    let cleanWord = word.trimmingCharacters(in: .punctuationCharacters)

                    // Skip common dictionary words (verbs, pronouns, etc.) even if capitalized
                    guard !isCommonWord(cleanWord, in: transcript, at: i) else {
                        continue
                    }

                    // Only replace capitalized words (proper nouns), skip lowercase common words
                    guard cleanWord.first?.isUppercase == true else {
                        continue
                    }

                    let targetWord = formWords[0]
                    let similarity = combinedSimilarity(cleanWord, targetWord)

                    if similarity >= minSimilarity {
                        if let existing = bestMatch {
                            if similarity > existing.similarity {
                                bestMatch = (i, similarity, formWords.count)
                            }
                        } else {
                            bestMatch = (i, similarity, formWords.count)
                        }
                    }
                }

                // Try multi-word matches if form has multiple words
                if formWords.count > 1 {
                    for startIdx in 0..<words.count {
                        let endIdx = min(startIdx + formWords.count, words.count)
                        let span = words[startIdx..<endIdx]

                        // Skip phrases starting with common dictionary words
                        let firstWord = span.first?.trimmingCharacters(in: .punctuationCharacters) ?? ""
                        guard !isCommonWord(firstWord, in: transcript, at: startIdx) else {
                            continue
                        }

                        // Only replace if first word is capitalized (proper noun phrase)
                        guard firstWord.first?.isUppercase == true else {
                            continue
                        }

                        let spanText = span.map { $0.trimmingCharacters(in: .punctuationCharacters) }.joined(
                            separator: " ")

                        let similarity = combinedSimilarity(spanText, formToMatch)

                        if similarity >= minSimilarity {
                            if let existing = bestMatch {
                                if span.count > existing.spanLength
                                    || (span.count == existing.spanLength && similarity > existing.similarity)
                                {
                                    bestMatch = (startIdx, similarity, span.count)
                                }
                            } else {
                                bestMatch = (startIdx, similarity, span.count)
                            }
                        }
                    }
                }
            }

            if let match = bestMatch {
                let originalSpan = words[match.index..<min(match.index + match.spanLength, words.count)]
                let replacement = InternalReplacement(
                    wordIndex: match.index,
                    canonical: canonical,
                    similarity: match.similarity,
                    ctcScore: detection.score,
                    originalWord: originalSpan.joined(separator: " ")
                )

                // Filter by combined confidence
                if replacement.combinedConfidence >= minCombinedConfidence {
                    replacements.append(replacement)

                    if debugMode {
                        logger.info(
                            "KeywordMerger: Match '\(replacement.originalWord)' -> '\(canonical)' (sim: \(String(format: "%.2f", match.similarity)), conf: \(String(format: "%.2f", replacement.combinedConfidence)))"
                        )
                    }
                } else if debugMode {
                    logger.info(
                        "KeywordMerger: Rejected '\(replacement.originalWord)' -> '\(canonical)' (conf: \(String(format: "%.2f", replacement.combinedConfidence)) < \(String(format: "%.2f", minCombinedConfidence)))"
                    )
                }
            }
        }

        guard !replacements.isEmpty else {
            if debugMode {
                logger.info("KeywordMerger: No suitable replacements found")
            }
            return MergeResult(correctedText: transcript, replacements: [])
        }

        // Sort by span length (prefer multi-word), then similarity
        replacements.sort { lhs, rhs in
            let lhsSpanLength = lhs.canonical.split(whereSeparator: { $0.isWhitespace }).count
            let rhsSpanLength = rhs.canonical.split(whereSeparator: { $0.isWhitespace }).count

            if lhsSpanLength != rhsSpanLength {
                return lhsSpanLength > rhsSpanLength
            }

            return lhs.similarity > rhs.similarity
        }

        // Select non-overlapping replacements
        var usedIndices = Set<Int>()
        var finalReplacements: [InternalReplacement] = []

        for replacement in replacements {
            let spanLength = replacement.canonical.split(whereSeparator: { $0.isWhitespace }).count
            let range = replacement.wordIndex..<min(replacement.wordIndex + spanLength, words.count)

            let overlaps = range.contains { usedIndices.contains($0) }

            if !overlaps {
                finalReplacements.append(replacement)
                range.forEach { usedIndices.insert($0) }
            }
        }

        // Apply replacements from end to start to preserve indices
        finalReplacements.sort { $0.wordIndex > $1.wordIndex }

        var publicReplacements: [MergeResult.Replacement] = []

        for replacement in finalReplacements {
            let canonicalWords = replacement.canonical.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
            let spanLength = canonicalWords.count
            let endIdx = min(replacement.wordIndex + spanLength, words.count)

            // Preserve capitalization if original word started with uppercase
            let originalSpan = words[replacement.wordIndex..<endIdx]
            let finalWords = zip(
                canonicalWords,
                originalSpan + Array(repeating: "", count: max(0, canonicalWords.count - originalSpan.count))
            ).map { canonical, original in
                if !original.isEmpty && original.first?.isUppercase == true && canonical.first?.isLowercase == true {
                    return canonical.prefix(1).uppercased() + canonical.dropFirst()
                }
                return canonical
            }

            publicReplacements.append(
                MergeResult.Replacement(
                    originalText: replacement.originalWord,
                    canonicalText: replacement.canonical,
                    wordIndex: replacement.wordIndex,
                    spanLength: spanLength,
                    similarity: replacement.similarity,
                    combinedConfidence: replacement.combinedConfidence
                ))

            words.replaceSubrange(replacement.wordIndex..<endIdx, with: finalWords)
        }

        let correctedText = words.joined(separator: " ")

        if debugMode {
            logger.info("KeywordMerger: Applied \(publicReplacements.count) replacements")
        }

        return MergeResult(correctedText: correctedText, replacements: publicReplacements)
    }

    /// Compute phonetic similarity using Double Metaphone
    private static func phoneticSimilarity(_ a: String, _ b: String) -> Float {
        let (aPrimary, aAlternate) = DoubleMetaphone.encode(a)
        let (bPrimary, bAlternate) = DoubleMetaphone.encode(b)

        // Check for exact phonetic match
        if aPrimary == bPrimary || aPrimary == bAlternate || aAlternate == bPrimary
            || (aAlternate == bAlternate && !aAlternate.isEmpty)
        {
            return 1.0
        }

        // Compute edit distance on phonetic codes
        let dist1 = levenshteinDistance(aPrimary, bPrimary)
        let dist2 =
            !aAlternate.isEmpty && !bAlternate.isEmpty
            ? levenshteinDistance(aAlternate, bAlternate) : Int.max
        let dist = min(dist1, dist2)

        let maxLen = max(aPrimary.count, bPrimary.count)
        guard maxLen > 0 else { return 0.0 }

        return 1.0 - Float(dist) / Float(maxLen)
    }

    /// Compute combined character + phonetic similarity
    private static func combinedSimilarity(_ a: String, _ b: String) -> Float {
        let charSim = characterSimilarity(a, b)
        let phoneSim = phoneticSimilarity(a, b)

        // Weight phonetic similarity more heavily (60%) for proper nouns
        return 0.4 * charSim + 0.6 * phoneSim
    }

    /// Compute character-level Levenshtein similarity
    private static func characterSimilarity(_ a: String, _ b: String) -> Float {
        let aNorm = a.lowercased()
        let bNorm = b.lowercased()
        let distance = levenshteinDistance(aNorm, bNorm)
        let maxLen = max(aNorm.count, bNorm.count)
        guard maxLen > 0 else { return 1.0 }

        return 1.0 - Float(distance) / Float(maxLen)
    }

    /// Compute Levenshtein distance between two strings
    private static func levenshteinDistance(_ a: String, _ b: String) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        let m = aChars.count
        let n = bChars.count

        guard m > 0 else { return n }
        guard n > 0 else { return m }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m {
            dp[i][0] = i
        }
        for j in 0...n {
            dp[0][j] = j
        }

        for i in 1...m {
            for j in 1...n {
                let cost = aChars[i - 1] == bChars[j - 1] ? 0 : 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            }
        }

        return dp[m][n]
    }
}
