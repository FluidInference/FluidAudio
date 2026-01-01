import Foundation
import OSLog

#if canImport(Metaphone3)
import Metaphone3
#endif

/// Merges CTC keyword detections into an existing transcript by replacing similar words
/// with their canonical forms. This allows using CTC keyword spotting with any ASR system.
public struct KeywordMerger {

    private static let logger = Logger(subsystem: "com.fluidaudio", category: "KeywordMerger")

    // MARK: - Phonetic Encoder Selection

    /// Which phonetic encoder is being used (logged once at first use)
    private enum PhoneticEncoder {
        case metaphone3
        case doubleMetaphone

        var description: String {
            switch self {
            case .metaphone3: return "Metaphone3"
            case .doubleMetaphone: return "DoubleMetaphone"
            }
        }
    }

    /// Track whether we've logged the encoder selection
    private nonisolated(unsafe) static var hasLoggedEncoderSelection = false

    /// The active phonetic encoder (determined at compile time)
    private static var activeEncoder: PhoneticEncoder {
        #if canImport(Metaphone3)
        return .metaphone3
        #else
        return .doubleMetaphone
        #endif
    }

    /// Log which encoder is being used (once per session)
    private static func logEncoderSelectionOnce() {
        guard !hasLoggedEncoderSelection else { return }
        hasLoggedEncoderSelection = true

        let debugMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"
        if debugMode {
            logger.debug("[KeywordMerger] Using \(activeEncoder.description) for phonetic encoding")
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

        var encoder: Any? = nil
        #if canImport(Metaphone3)
        let mp = Metaphone3Encoder()
        mp.encodeVowels = false
        mp.encodeExact = false
        encoder = mp
        #endif

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

                // Use the last frame of the entire sequence as the total duration reference
                let totalFrames = ctcTokenSequence.last?.frameIndex ?? matchFrameIdx

                let wordIndex = estimateWordIndex(
                    frameIndex: matchFrameIdx,
                    totalWords: words.count,
                    totalFrames: totalFrames
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
    private static func estimateWordIndex(frameIndex: Int, totalWords: Int, totalFrames: Int) -> Int {
        // Simple linear interpolation based on frame position in the whole segment
        guard totalFrames > 0 else { return 0 }
        let relativePosition = Float(frameIndex) / Float(totalFrames)
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
        debugMode: Bool = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"
    ) -> MergeResult {

        let minSimilarity = vocabulary.minSimilarity
        let minCombinedConfidence = vocabulary.minCombinedConfidence

        var encoder: Any? = nil
        #if canImport(Metaphone3)
        let mp = Metaphone3Encoder()
        mp.encodeVowels = false
        mp.encodeExact = false
        encoder = mp
        #endif

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
            logger.info(
                "KeywordMerger (String-Based): Processing \(detections.count) detections for \(words.count) words")
        }

        // Pre-compute cleaned words (strip punctuation)
        let cleanedWords = words.map { $0.trimmingCharacters(in: .punctuationCharacters) }

        // Internal replacement structure with confidence scoring
        struct InternalReplacement {
            let wordIndex: Int
            let canonical: String
            let similarity: Float
            let ctcScore: Float
            let originalWord: String
            let spanLength: Int  // Number of original words being replaced

            var combinedConfidence: Float {
                // Normalize CTC score from [-12, -6] range to [0, 1]
                // Expanded range to accommodate weaker CTC detections
                let normalizedCtcScore = max(0, min(1, (ctcScore + 12) / 6))
                // Weight similarity heavily (70%) since character matching helps weak phonetic cases
                return 0.7 * similarity + 0.3 * normalizedCtcScore
            }
        }

        var replacements: [InternalReplacement] = []

        // Find best matching words for each detection
        for detection in detections {
            let canonical = detection.term.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !canonical.isEmpty else { continue }

            if debugMode && canonical.lowercased().contains("prevnar") {
                logger.debug("[Detection] Processing '\(canonical)', transcript: '\(transcript.prefix(100))...'")
            }

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

                // Try single-word matches (only for single-word vocabulary terms)
                // Multi-word phrases are handled in the multi-word matching section below
                if formWords.count == 1 {
                    for (i, _) in words.enumerated() {
                        let cleanWord = cleanedWords[i]

                        // Check if this word matches any alias (case-insensitive) - always allow alias corrections
                        let isAliasMatch = allForms.contains { $0.lowercased() == cleanWord.lowercased() }

                        // Only replace capitalized words (proper nouns), unless it's an alias match
                        // This naturally filters out most common words (verbs, pronouns, etc.) which are lowercase
                        guard cleanWord.first?.isUppercase == true || isAliasMatch else {
                            continue
                        }

                        let targetWord = formWords[0]
                        let similarity = combinedSimilarity(cleanWord, targetWord, encoder: encoder)

                        if similarity >= minSimilarity {
                            if let existing = bestMatch {
                                if similarity > existing.similarity {
                                    bestMatch = (i, similarity, 1)
                                }
                            } else {
                                bestMatch = (i, similarity, 1)
                            }
                        }
                    }
                }

                // Try joining adjacent transcript words to match single-word vocabulary term
                if formWords.count == 1 {
                    let targetWord = formWords[0]
                    for startIdx in 0..<words.count {
                        // Try joining 2 or 3 adjacent words
                        for spanLen in 2...3 {
                            let endIdx = startIdx + spanLen
                            guard endIdx <= words.count else { continue }

                            let span = words[startIdx..<endIdx]

                            // Only process if first word is capitalized (proper noun)
                            let firstWord = cleanedWords[startIdx]
                            guard firstWord.first?.isUppercase == true else { continue }

                            let joinedSpan = span.map { $0.trimmingCharacters(in: .punctuationCharacters) }.joined()

                            let similarity = combinedSimilarity(joinedSpan, targetWord, encoder: encoder)

                            if similarity >= minSimilarity {
                                // Prefer longer spans only if similarity is comparable to existing match
                                if let existing = bestMatch {
                                    // Only prefer longer span if similarity is within 0.15 of existing
                                    let similarityComparable = similarity >= existing.similarity - 0.15
                                    if (spanLen > existing.spanLength && similarityComparable)
                                        || (spanLen == existing.spanLength && similarity > existing.similarity)
                                    {
                                        bestMatch = (startIdx, similarity, spanLen)
                                    }
                                } else {
                                    bestMatch = (startIdx, similarity, spanLen)
                                }
                            }
                        }
                    }
                }

                // Try multi-word matches if form has multiple words
                if formWords.count > 1 {
                    let debugEnv = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"
                    if debugEnv && formToMatch.lowercased().contains("prevnar") {
                        logger.debug("[MultiWord] Trying to match '\(formToMatch)' (formWords: \(formWords))")
                    }

                    for startIdx in 0..<words.count {
                        let endIdx = startIdx + formWords.count
                        // Ensure we have enough words remaining for full phrase match
                        guard endIdx <= words.count else { continue }

                        let span = Array(words[startIdx..<endIdx])
                        let cleanedSpan = span.map { $0.trimmingCharacters(in: .punctuationCharacters) }

                        // Only replace if first word is capitalized (proper noun phrase)
                        guard cleanedSpan[0].first?.isUppercase == true else {
                            continue
                        }

                        if debugEnv && formToMatch.lowercased().contains("prevnar") {
                            logger.debug("[MultiWord] Checking span: \(cleanedSpan) vs \(formWords)")
                        }

                        // Compare word-by-word for better phonetic matching of phrases
                        let similarity = multiWordSimilarity(
                            cleanedSpan, formWords.map { String($0) }, encoder: encoder)

                        if similarity >= minSimilarity {
                            if let existing = bestMatch {
                                // Only prefer longer span if similarity is within 0.15 of existing
                                let similarityComparable = similarity >= existing.similarity - 0.15
                                if (span.count > existing.spanLength && similarityComparable)
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
                    originalWord: originalSpan.joined(separator: " "),
                    spanLength: match.spanLength
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

        // Group replacements by word index to handle conflicts
        var replacementsByIndex: [Int: [InternalReplacement]] = [:]
        for replacement in replacements {
            replacementsByIndex[replacement.wordIndex, default: []].append(replacement)
        }

        // For each word position, select the best replacement
        var selectedReplacements: [InternalReplacement] = []
        for (wordIdx, candidates) in replacementsByIndex {
            if debugMode && candidates.count > 1 {
                logger.debug("[Conflict] at word \(wordIdx) ('\(words[wordIdx])'): \(candidates.count) candidates")
                for c in candidates {
                    logger.debug(
                        "  - '\(c.canonical)' sim=\(String(format: "%.2f", c.similarity)) span=\(c.spanLength) conf=\(String(format: "%.2f", c.combinedConfidence))"
                    )
                }
            }

            // When multiple detections target the same word, prefer:
            // 1. Higher similarity (exact/near-exact matches)
            // 2. Longer span length (multi-word phrases)
            // 3. Higher combined confidence
            let best = candidates.max { lhs, rhs in
                // Prioritize similarity first to prefer exact matches
                if abs(lhs.similarity - rhs.similarity) > 0.1 {
                    return lhs.similarity < rhs.similarity
                }

                // Then prefer longer spans (number of original words being replaced)
                if lhs.spanLength != rhs.spanLength {
                    return lhs.spanLength < rhs.spanLength
                }

                // Finally use combined confidence as tiebreaker
                return lhs.combinedConfidence < rhs.combinedConfidence
            }

            if let best = best {
                if debugMode && candidates.count > 1 {
                    logger.debug("  → Selected '\(best.canonical)'")
                }
                selectedReplacements.append(best)
            }
        }

        // Select non-overlapping replacements (for multi-word phrases)
        var usedIndices = Set<Int>()
        var finalReplacements: [InternalReplacement] = []

        // Sort by similarity to process best matches first
        selectedReplacements.sort { $0.similarity > $1.similarity }

        for replacement in selectedReplacements {
            // Use the actual span length of original words being replaced
            let range = replacement.wordIndex..<min(replacement.wordIndex + replacement.spanLength, words.count)

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
            // Use the actual span of original words being replaced
            let endIdx = min(replacement.wordIndex + replacement.spanLength, words.count)

            // Preserve capitalization if original word started with uppercase
            let originalSpan = words[replacement.wordIndex..<endIdx]
            let firstOriginal = originalSpan.first ?? ""
            let finalWords = canonicalWords.enumerated().map { idx, canonical in
                // Capitalize first word if original started with uppercase
                if idx == 0 && !firstOriginal.isEmpty && firstOriginal.first?.isUppercase == true
                    && canonical.first?.isLowercase == true
                {
                    return canonical.prefix(1).uppercased() + canonical.dropFirst()
                }
                return canonical
            }

            publicReplacements.append(
                MergeResult.Replacement(
                    originalText: replacement.originalWord,
                    canonicalText: replacement.canonical,
                    wordIndex: replacement.wordIndex,
                    spanLength: replacement.spanLength,
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

    /// Compute phonetic similarity using the best available encoder
    /// Uses Metaphone3 if available (better accuracy), otherwise falls back to DoubleMetaphone
    private static func phoneticSimilarity(_ a: String, _ b: String, encoder: Any? = nil) -> Float {
        logEncoderSelectionOnce()

        #if canImport(Metaphone3)
        return metaphone3Similarity(a, b, encoder: encoder)
        #else
        return doubleMetaphoneSimilarity(a, b)
        #endif
    }

    #if canImport(Metaphone3)
    /// Compute phonetic similarity using Metaphone3 (more accurate for names)
    private static func metaphone3Similarity(_ a: String, _ b: String, encoder: Any? = nil) -> Float {
        // Use passed encoder or fallback
        let encoderInstance: Metaphone3Encoder
        if let passed = encoder as? Metaphone3Encoder {
            encoderInstance = passed
        } else {
            encoderInstance = Metaphone3Encoder()
            encoderInstance.encodeVowels = false
            encoderInstance.encodeExact = false
        }

        let resultA = encoderInstance.encode(a)
        let resultB = encoderInstance.encode(b)

        let aPrimary = resultA.metaph
        let aAlternate = resultA.alternateMetaph
        let bPrimary = resultB.metaph
        let bAlternate = resultB.alternateMetaph

        // Check for exact phonetic match (any combination)
        if aPrimary == bPrimary {
            return 1.0
        }
        if !aAlternate.isEmpty && aAlternate == bPrimary {
            return 1.0
        }
        if !bAlternate.isEmpty && aPrimary == bAlternate {
            return 1.0
        }
        if !aAlternate.isEmpty && !bAlternate.isEmpty && aAlternate == bAlternate {
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
    #endif

    /// Compute phonetic similarity using Double Metaphone (fallback)
    private static func doubleMetaphoneSimilarity(_ a: String, _ b: String) -> Float {
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

    /// Compute similarity for multi-word phrases by comparing word-by-word
    /// Returns average similarity across all word pairs
    private static func multiWordSimilarity(
        _ transcriptWords: [String],
        _ vocabWords: [String],
        encoder: Any? = nil
    ) -> Float {
        guard transcriptWords.count == vocabWords.count, !transcriptWords.isEmpty else {
            return 0.0
        }

        var totalSimilarity: Float = 0.0
        let debugMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"

        for (tWord, vWord) in zip(transcriptWords, vocabWords) {
            let wordSim = combinedSimilarity(tWord, vWord, encoder: encoder)
            totalSimilarity += wordSim
            if debugMode && (vocabWords.first?.lowercased().contains("prevnar") == true) {
                logger.debug("[MultiWordSim] '\(tWord)' vs '\(vWord)' = \(String(format: "%.3f", wordSim))")
            }
        }

        let avgSim = totalSimilarity / Float(transcriptWords.count)
        if debugMode && (vocabWords.first?.lowercased().contains("prevnar") == true) {
            logger.debug(
                "[MultiWordSim] Total: \(transcriptWords) vs \(vocabWords) = \(String(format: "%.3f", avgSim))")
        }

        return avgSim
    }

    /// Compute combined character + phonetic similarity with multi-factor gating
    private static func combinedSimilarity(_ a: String, _ b: String, encoder: Any? = nil) -> Float {
        let charSim = characterSimilarity(a, b)
        let lenRatio = Float(min(a.count, b.count)) / Float(max(a.count, b.count))

        // For purely numeric strings, use only character similarity
        // Phonetic encoding doesn't work well for numbers (e.g., "13" vs "20")
        let aIsNumeric = a.allSatisfy { $0.isNumber }
        let bIsNumeric = b.allSatisfy { $0.isNumber }
        if aIsNumeric || bIsNumeric {
            return charSim
        }

        let phoneSim = phoneticSimilarity(a, b, encoder: encoder)

        // Poor length ratio: heavy penalty
        if lenRatio < 0.6 {
            return charSim * lenRatio
        }

        // If phonetic match is very high, use it
        if phoneSim >= 0.9 {
            return max(phoneSim, charSim)
        }

        // Otherwise blend character and phonetic similarity
        return (charSim * 0.6) + (phoneSim * 0.4)
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
