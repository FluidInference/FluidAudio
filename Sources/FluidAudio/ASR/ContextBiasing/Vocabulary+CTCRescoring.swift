import Foundation

// MARK: - Constrained CTC Rescoring

extension VocabularyRescorer {

    // MARK: - Debug Logging

    /// Log debug message only when debug mode is enabled.
    /// Uses closure to avoid string evaluation when debug is off.
    @inline(__always)
    private func debugLog(_ message: @escaping @autoclosure () -> String) {
        guard debugMode else { return }
        logger.debug(message())
    }

    // MARK: - Stopwords (Shared)

    /// Common stopwords that should not be replaced by vocabulary terms.
    /// Defined once as a static constant to avoid recreation in hot loops.
    private static let stopwords: Set<String> = [
        // Articles and determiners
        "a", "an", "the", "some", "any", "no", "every", "each", "all",
        // Conjunctions
        "and", "or", "but", "so", "if", "then", "than", "as",
        // Prepositions
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down",
        "out", "about", "into", "over", "after", "before", "between", "under",
        // Be verbs
        "is", "are", "was", "were", "be", "been", "being", "am",
        // Common verbs
        "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
        "go", "goes", "went", "come", "comes", "came", "get", "got", "take", "took",
        "make", "made", "say", "said", "see", "saw", "know", "knew", "think", "thought",
        // Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
        "who", "what", "which", "where", "when", "how", "why",
        // Common short words
        "just", "also", "only", "even", "still", "now", "here", "there", "very",
        "well", "back", "way", "own", "new", "old", "good", "great", "first", "last",
    ]

    // MARK: - CTC Match Types

    /// Parameters for evaluating a CTC match candidate.
    private struct CTCMatchCandidate {
        let originalPhrase: String
        let vocabTerm: String
        let vocabTokens: [Int]
        let similarity: Float
        let spanLength: Int
        let spanIndices: [Int]
        let spanStartTime: Double
        let spanEndTime: Double
    }

    /// Result of CTC match evaluation.
    private struct CTCMatchResult {
        let shouldReplace: Bool
        let originalScore: Float
        let boostedVocabScore: Float
        let replacement: String
        let reason: String
    }

    /// Pending replacement candidate for two-pass selection.
    /// Stores all info needed to apply the replacement later.
    private struct PendingReplacement {
        let candidate: CTCMatchCandidate
        let result: CTCMatchResult
        let similarity: Float  // String similarity for sorting
    }

    // MARK: - Shared Finalization

    /// Finalize replacements: sort by span length, apply greedily, reconstruct transcript.
    ///
    /// This shared helper handles Pass 2 & 3 of the two-pass algorithm, used by both
    /// word-centric and term-centric approaches.
    private func finalizeReplacements(
        pendingReplacements: [PendingReplacement],
        modifiedWords: inout [(word: String, startTime: Double, endTime: Double)],
        replacedIndices: inout Set<Int>,
        replacements: inout [RescoringResult]
    ) -> RescoreOutput {
        // PASS 2: Sort by span length (ascending) then by similarity (descending)
        // This ensures shorter spans are preferred when multiple matches overlap
        let sortedReplacements = pendingReplacements.sorted { a, b in
            if a.candidate.spanLength != b.candidate.spanLength {
                return a.candidate.spanLength < b.candidate.spanLength  // Prefer shorter spans
            }
            // For same span length, prefer higher similarity
            return a.similarity > b.similarity
        }

        // PASS 3: Greedily apply non-overlapping replacements
        for pending in sortedReplacements {
            // Check if any index in this span is already replaced
            guard pending.candidate.spanIndices.allSatisfy({ !replacedIndices.contains($0) }) else {
                continue  // Skip - overlaps with already-accepted replacement
            }

            applyReplacement(
                result: pending.result,
                candidate: pending.candidate,
                modifiedWords: &modifiedWords,
                replacedIndices: &replacedIndices,
                replacements: &replacements
            )
        }

        // Reconstruct transcript from modified words (filter empty strings from multi-word replacements)
        let modifiedText = modifiedWords.map { $0.word }.filter { !$0.isEmpty }.joined(separator: " ")
        let wasModified = !replacements.isEmpty
        let replacementCount = replacements.count  // Capture before debugLog (inout can't be captured in @escaping)

        debugLog("Final: \(modifiedText)")
        debugLog("Replacements: \(replacementCount)")
        debugLog("===========================================")

        return RescoreOutput(
            text: modifiedText,
            replacements: replacements,
            wasModified: wasModified
        )
    }

    // MARK: - Public API

    /// Rescore using constrained CTC search around TDT word locations.
    ///
    /// Dispatches to either word-centric (USE_BK_TREE=1) or term-centric (default) algorithm.
    /// Term-centric is the default as it produces better results in benchmarks.
    ///
    /// - Parameters:
    ///   - transcript: Original transcript from TDT decoder
    ///   - tokenTimings: Token-level timings from TDT decoder
    ///   - logProbs: CTC log-probabilities from spotter
    ///   - frameDuration: Duration of each CTC frame in seconds
    ///   - cbw: Context-biasing weight (default 3.0 per NeMo paper)
    ///   - marginSeconds: Temporal margin around TDT word for CTC search (default 0.5s)
    ///   - minSimilarity: Minimum string similarity to consider a match (default 0.5)
    /// - Returns: Rescored transcript with constrained CTC replacements
    public func rescoreWithConstrainedCTC(
        transcript: String,
        tokenTimings: [TokenTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor
    ) -> RescoreOutput {
        if useBKTree {
            return rescoreWithConstrainedCTCWordCentric(
                transcript: transcript,
                tokenTimings: tokenTimings,
                logProbs: logProbs,
                frameDuration: frameDuration,
                cbw: cbw,
                marginSeconds: marginSeconds,
                minSimilarity: minSimilarity
            )
        } else {
            return rescoreWithConstrainedCTCTermCentric(
                transcript: transcript,
                tokenTimings: tokenTimings,
                logProbs: logProbs,
                frameDuration: frameDuration,
                cbw: cbw,
                marginSeconds: marginSeconds,
                minSimilarity: minSimilarity
            )
        }
    }

    /// Word-centric constrained CTC rescoring (USE_BK_TREE=1).
    ///
    /// Algorithm:
    /// 1. For each TDT word, query BK-tree to find candidate vocabulary terms (O(log V) per word)
    /// 2. For each candidate, run constrained CTC DP within the TDT word's timestamp window
    /// 3. Compare constrained CTC score with original word's CTC score to decide replacement
    ///
    /// Best used with BK-tree enabled for O(W × log V) performance.
    private func rescoreWithConstrainedCTCWordCentric(
        transcript: String,
        tokenTimings: [TokenTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor
    ) -> RescoreOutput {
        // Build word-level timings from token timings
        let wordTimings = buildWordTimings(from: tokenTimings)

        guard !wordTimings.isEmpty, !logProbs.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        debugLog("=== VocabularyRescorer (Constrained CTC - Word-Centric) ===")
        debugLog("Words: \(wordTimings.count), Frames: \(logProbs.count), Vocab: \(vocabulary.terms.count)")
        debugLog("Frame duration: \(String(format: "%.4f", frameDuration))s")
        debugLog("CBW: \(cbw), Margin: \(marginSeconds)s, MinSimilarity: \(minSimilarity)")
        debugLog("Mode: \(useBKTree ? "BK-tree O(W × log V)" : "Linear scan O(W × V)")")

        var replacements: [RescoringResult] = []
        var modifiedWords: [(word: String, startTime: Double, endTime: Double)] = wordTimings.map {
            (word: $0.word, startTime: $0.startTime, endTime: $0.endTime)
        }
        var replacedIndices = Set<Int>()
        var pendingReplacements: [PendingReplacement] = []  // Two-pass: collect first, apply later

        // Build normalized vocabulary set for guard checks
        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()

        // Pre-compute normalized words for all timings
        let normalizedWords = wordTimings.map { Self.normalizeForSimilarity($0.word) }

        // WORD-CENTRIC LOOP: For each TDT word, find candidate vocabulary terms
        for (wordIdx, timing) in wordTimings.enumerated() {
            guard !replacedIndices.contains(wordIdx) else { continue }

            let tdtWord = timing.word
            let normalizedWord = normalizedWords[wordIdx]
            guard !normalizedWord.isEmpty else { continue }

            // Skip stopwords for single-word matching (checked again for compounds later)
            if Self.stopwords.contains(normalizedWord) {
                debugLog("  [STOPWORD] Skipping '\(normalizedWord)' (single word)")
                // Don't skip entirely - still check compound matches starting from this word
            }

            // Build adjacent normalized words for compound detection
            var adjacentNormalized: [String] = []
            for offset in 1...3 {
                let idx = wordIdx + offset
                if idx < wordTimings.count && !replacedIndices.contains(idx) {
                    let norm = normalizedWords[idx]
                    if !norm.isEmpty {
                        adjacentNormalized.append(norm)
                    } else {
                        break  // Stop at first empty/invalid word
                    }
                } else {
                    break
                }
            }

            // Find candidate vocabulary terms using BK-tree or linear scan
            let candidates = findCandidateTermsForWord(
                normalizedWord: normalizedWord,
                adjacentNormalized: adjacentNormalized,
                minSimilarity: minSimilarity
            )

            if !candidates.isEmpty {
                let candidateInfo = candidates.prefix(5).map {
                    "\($0.term.text)(sim=\(String(format: "%.2f", $0.similarity)), span=\($0.spanLength))"
                }.joined(separator: ", ")
                debugLog("  '\(tdtWord)' -> \(candidates.count) candidates: \(candidateInfo)")
            }

            // Process each candidate
            for candidate in candidates {
                let term = candidate.term
                let vocabTerm = term.text
                let similarity = candidate.similarity
                let spanLength = candidate.spanLength

                // Skip short vocabulary terms (per NeMo CTC-WS paper)
                guard vocabTerm.count >= vocabulary.minTermLength else { continue }

                // Get vocabulary tokens
                guard let vocabTokens = term.ctcTokenIds ?? term.tokenIds, !vocabTokens.isEmpty else {
                    continue
                }

                // Build span indices
                let spanIndices = Array(wordIdx..<(wordIdx + spanLength))

                // Check if any word in the span is already replaced
                guard spanIndices.allSatisfy({ !replacedIndices.contains($0) }) else { continue }

                // Build the original phrase
                let originalPhrase =
                    spanLength == 1
                    ? tdtWord
                    : spanIndices.map { wordTimings[$0].word }.joined(separator: " ")
                let normalizedPhrase =
                    spanLength == 1
                    ? normalizedWord
                    : spanIndices.map { normalizedWords[$0] }.joined(separator: " ")

                // Skip if already exact match to canonical (no replacement needed)
                let normalizedCanonical = Self.normalizeForSimilarity(vocabTerm)
                if normalizedPhrase == normalizedCanonical {
                    continue
                }

                // Guard: Skip if original phrase matches a DIFFERENT vocabulary term
                let normalizedCurrentSet = Set(buildNormalizedForms(for: term).map { $0.normalized })
                if vocabularyNormalizedSet.contains(normalizedPhrase)
                    && !normalizedCurrentSet.contains(normalizedPhrase)
                {
                    debugLog("  Skipping '\(vocabTerm)': phrase '\(originalPhrase)' matches another vocab term")
                    continue
                }

                // Apply similarity threshold adjustments
                var minSimilarityForSpan = requiredSimilarity(
                    minSimilarity: minSimilarity,
                    spanLength: spanLength,
                    normalizedText: normalizedPhrase
                )

                // LENGTH RATIO CHECK for single words
                if spanLength == 1 {
                    minSimilarityForSpan = checkLengthRatioRules(
                        normalizedWord: normalizedWord,
                        vocabTerm: vocabTerm,
                        currentSimilarity: similarity,
                        minSimilarity: minSimilarityForSpan
                    )
                }

                // STOPWORD CHECKS
                let spanWords = spanLength >= 2 ? spanIndices.map { normalizedWords[$0] } : []
                let (shouldSkipStopword, adjustedSimilarity) = checkStopwordRules(
                    normalizedWord: normalizedWord,
                    spanLength: spanLength,
                    spanWords: spanWords,
                    vocabTerm: vocabTerm,
                    currentSimilarity: minSimilarityForSpan
                )
                if shouldSkipStopword { continue }
                minSimilarityForSpan = adjustedSimilarity

                // Check if similarity meets threshold after all adjustments
                guard similarity >= minSimilarityForSpan else { continue }

                // Get temporal window for the span
                let spanStartTime = wordTimings[wordIdx].startTime
                let spanEndTime = wordTimings[wordIdx + spanLength - 1].endTime

                // Evaluate CTC match using shared helper
                let candidate = CTCMatchCandidate(
                    originalPhrase: originalPhrase,
                    vocabTerm: vocabTerm,
                    vocabTokens: vocabTokens,
                    similarity: similarity,
                    spanLength: spanLength,
                    spanIndices: spanIndices,
                    spanStartTime: spanStartTime,
                    spanEndTime: spanEndTime
                )

                let result = evaluateCTCMatch(
                    candidate: candidate,
                    logProbs: logProbs,
                    frameDuration: frameDuration,
                    cbw: cbw,
                    marginSeconds: marginSeconds
                )

                if result.shouldReplace {
                    // Collect candidate instead of applying immediately
                    pendingReplacements.append(
                        PendingReplacement(
                            candidate: candidate,
                            result: result,
                            similarity: similarity
                        )
                    )
                }
            }
        }

        // PASS 2 & 3: Sort, apply, and reconstruct (shared logic)
        return finalizeReplacements(
            pendingReplacements: pendingReplacements,
            modifiedWords: &modifiedWords,
            replacedIndices: &replacedIndices,
            replacements: &replacements
        )
    }

    /// Term-centric constrained CTC rescoring (default, USE_BK_TREE=0).
    ///
    /// Algorithm:
    /// 1. For each vocabulary term, find TDT words phonetically similar (string similarity)
    /// 2. For each match, run constrained CTC DP within the TDT word's timestamp window
    /// 3. Compare constrained CTC score with original word's CTC score to decide replacement
    ///
    /// This approach processes vocabulary in file order and produces better benchmark results.
    private func rescoreWithConstrainedCTCTermCentric(
        transcript: String,
        tokenTimings: [TokenTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor
    ) -> RescoreOutput {
        // Build word-level timings from token timings
        let wordTimings = buildWordTimings(from: tokenTimings)

        guard !wordTimings.isEmpty, !logProbs.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        debugLog("=== VocabularyRescorer (Constrained CTC - Term-Centric) ===")
        debugLog("Words: \(wordTimings.count), Frames: \(logProbs.count)")
        debugLog("Frame duration: \(String(format: "%.4f", frameDuration))s")
        debugLog("CBW: \(cbw), Margin: \(marginSeconds)s, MinSimilarity: \(minSimilarity)")

        var replacements: [RescoringResult] = []
        var modifiedWords: [(word: String, startTime: Double, endTime: Double)] = wordTimings.map {
            (word: $0.word, startTime: $0.startTime, endTime: $0.endTime)
        }
        var replacedIndices = Set<Int>()
        var pendingReplacements: [PendingReplacement] = []  // Two-pass: collect first, apply later

        // Build normalized vocabulary set for guard checks
        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()

        // TERM-CENTRIC LOOP: For each vocabulary term, find similar TDT words and run constrained CTC
        for term in vocabulary.terms {
            let vocabTerm = term.text

            // Skip short vocabulary terms (per NeMo CTC-WS paper)
            guard vocabTerm.count >= vocabulary.minTermLength else {
                debugLog(
                    "  Skipping '\(vocabTerm)': too short (\(vocabTerm.count) < \(vocabulary.minTermLength) chars)")
                continue
            }

            let vocabTokens = term.ctcTokenIds ?? term.tokenIds

            guard let vocabTokens, !vocabTokens.isEmpty else {
                continue
            }

            // Build all normalized forms (canonical + aliases) for this term
            let normalizedForms = buildNormalizedForms(for: term)
            guard !normalizedForms.isEmpty else { continue }

            let normalizedCanonical = Self.normalizeForSimilarity(vocabTerm)
            let normalizedCurrentSet = Set(normalizedForms.map { $0.normalized })

            // Split forms by word count for appropriate matching
            let multiWordForms = normalizedForms.filter { $0.wordCount > 1 }
            let singleWordForms = normalizedForms.filter { $0.wordCount == 1 }

            if !multiWordForms.isEmpty {
                // Multi-word phrase matching: look for consecutive TDT words that match the phrase
                let maxWordCount = multiWordForms.map { $0.wordCount }.max() ?? 0
                let minWordCount = multiWordForms.map { $0.wordCount }.min() ?? 0
                let maxSpan = min(4, maxWordCount + 1)  // Allow some flexibility
                let minSpan = max(2, minWordCount)

                guard minSpan <= maxSpan else { continue }

                for spanLength in minSpan...maxSpan {
                    for startIdx in 0..<(wordTimings.count - spanLength + 1) {
                        // Check if any word in the span is already replaced
                        let spanIndices = Array(startIdx..<(startIdx + spanLength))
                        guard spanIndices.allSatisfy({ !replacedIndices.contains($0) }) else { continue }

                        // Build concatenated phrase from consecutive TDT words
                        let spanWords = spanIndices.map { wordTimings[$0].word }
                        let tdtPhrase = spanWords.joined(separator: " ")
                        let normalizedPhrase = Self.normalizeForSimilarity(tdtPhrase)
                        guard !normalizedPhrase.isEmpty else { continue }

                        // Check similarity against ALL forms (canonical + aliases)
                        var bestSimilarity: Float = 0
                        for form in multiWordForms {
                            let similarity = Self.stringSimilarity(normalizedPhrase, form.normalized)
                            bestSimilarity = max(bestSimilarity, similarity)
                        }

                        // Skip if already exact match to canonical (no replacement needed)
                        if normalizedPhrase == normalizedCanonical {
                            continue
                        }

                        // Guard: Skip if original phrase matches a DIFFERENT vocabulary term
                        if vocabularyNormalizedSet.contains(normalizedPhrase)
                            && !normalizedCurrentSet.contains(normalizedPhrase)
                        {
                            debugLog(
                                "  [MULTI] Skipping '\(vocabTerm)': phrase '\(tdtPhrase)' matches another vocab term")
                            continue
                        }

                        // Use adaptive similarity threshold
                        let minSimilarityForSpan = requiredSimilarity(
                            minSimilarity: minSimilarity,
                            spanLength: spanLength,
                            normalizedText: normalizedPhrase
                        )
                        guard bestSimilarity >= minSimilarityForSpan else { continue }

                        // Get temporal window for the entire span
                        guard let firstIdx = spanIndices.first, let lastIdx = spanIndices.last else { continue }
                        let spanStartTime = wordTimings[firstIdx].startTime
                        let spanEndTime = wordTimings[lastIdx].endTime

                        // Evaluate CTC match using shared helper
                        let candidate = CTCMatchCandidate(
                            originalPhrase: tdtPhrase,
                            vocabTerm: vocabTerm,
                            vocabTokens: vocabTokens,
                            similarity: bestSimilarity,
                            spanLength: spanLength,
                            spanIndices: spanIndices,
                            spanStartTime: spanStartTime,
                            spanEndTime: spanEndTime
                        )

                        let result = evaluateCTCMatch(
                            candidate: candidate,
                            logProbs: logProbs,
                            frameDuration: frameDuration,
                            cbw: cbw,
                            marginSeconds: marginSeconds
                        )

                        if result.shouldReplace {
                            // Collect candidate instead of applying immediately
                            pendingReplacements.append(
                                PendingReplacement(
                                    candidate: candidate,
                                    result: result,
                                    similarity: bestSimilarity
                                )
                            )
                        }
                    }
                }
            }

            if !singleWordForms.isEmpty {
                // Single-word matching (includes compound word detection)
                for (wordIdx, timing) in wordTimings.enumerated() {
                    guard !replacedIndices.contains(wordIdx) else { continue }

                    let tdtWord = timing.word
                    let normalizedWord = Self.normalizeForSimilarity(tdtWord)
                    guard !normalizedWord.isEmpty else { continue }

                    // Skip if already exact match to canonical (no replacement needed)
                    if normalizedWord == normalizedCanonical {
                        continue
                    }

                    // Guard: Skip if original word matches a DIFFERENT vocabulary term
                    if vocabularyNormalizedSet.contains(normalizedWord)
                        && !normalizedCurrentSet.contains(normalizedWord)
                    {
                        debugLog("  Skipping '\(vocabTerm)': word '\(tdtWord)' matches another vocab term")
                        continue
                    }

                    // Check similarity against ALL forms (single word)
                    var bestSimilarity: Float = 0
                    var matchedSpanLength = 1
                    var matchedConcatenation = normalizedWord
                    for form in singleWordForms {
                        let similarity = Self.stringSimilarity(normalizedWord, form.normalized)
                        bestSimilarity = max(bestSimilarity, similarity)
                    }

                    // COMPOUND WORD MATCHING: For single-word vocabulary terms, also try
                    // matching against concatenated adjacent TDT words.
                    // This handles cases like "Livmarli" being transcribed as "Liv Mali".
                    // Minimum vocab length of 4 for 2-word matching to avoid false positives on short words.
                    let minLengthFor2Word = 4
                    let minLengthFor3Word = 8

                    // Pre-compute normalized adjacent words (only if needed)
                    let normalized2: String? =
                        (wordIdx + 1 < wordTimings.count && !replacedIndices.contains(wordIdx + 1))
                        ? Self.normalizeForSimilarity(wordTimings[wordIdx + 1].word)
                        : nil
                    let normalized3: String? =
                        (wordIdx + 2 < wordTimings.count && !replacedIndices.contains(wordIdx + 2))
                        ? Self.normalizeForSimilarity(wordTimings[wordIdx + 2].word)
                        : nil

                    // 2-word compound matching
                    // Skip if the second word already matches the vocab term well on its own
                    if let norm2 = normalized2, !norm2.isEmpty, vocabTerm.count >= minLengthFor2Word {
                        let norm2MatchesVocab = singleWordForms.contains {
                            Self.stringSimilarity(norm2, $0.normalized) >= 0.9
                        }
                        if !norm2MatchesVocab {
                            let concatenated = normalizedWord + norm2  // No space
                            for form in singleWordForms {
                                let concatSimilarity = Self.stringSimilarity(concatenated, form.normalized)
                                if concatSimilarity > bestSimilarity {
                                    bestSimilarity = concatSimilarity
                                    matchedSpanLength = 2
                                    matchedConcatenation = concatenated
                                }
                            }
                        }
                    }

                    // 3-word compound matching (for longer vocabulary terms only)
                    // Skip if any of the later words already matches the vocab term well
                    if let norm2 = normalized2, let norm3 = normalized3,
                        !norm2.isEmpty, !norm3.isEmpty, vocabTerm.count >= minLengthFor3Word
                    {
                        let laterWordMatchesVocab = singleWordForms.contains {
                            Self.stringSimilarity(norm2, $0.normalized) >= 0.9
                                || Self.stringSimilarity(norm3, $0.normalized) >= 0.9
                        }
                        if !laterWordMatchesVocab {
                            let concatenated = normalizedWord + norm2 + norm3
                            for form in singleWordForms {
                                let concatSimilarity = Self.stringSimilarity(concatenated, form.normalized)
                                if concatSimilarity > bestSimilarity {
                                    bestSimilarity = concatSimilarity
                                    matchedSpanLength = 3
                                    matchedConcatenation = concatenated
                                }
                            }
                        }
                    }

                    // Use adaptive similarity threshold (use concatenated text for multi-word spans)
                    var minSimilarityForSpan = requiredSimilarity(
                        minSimilarity: minSimilarity,
                        spanLength: matchedSpanLength,
                        normalizedText: matchedConcatenation
                    )

                    // LENGTH RATIO CHECK for single words
                    if matchedSpanLength == 1 {
                        minSimilarityForSpan = checkLengthRatioRules(
                            normalizedWord: normalizedWord,
                            vocabTerm: vocabTerm,
                            currentSimilarity: bestSimilarity,
                            minSimilarity: minSimilarityForSpan
                        )
                    }

                    // STOPWORD CHECKS
                    let spanWords =
                        matchedSpanLength >= 2
                        ? (0..<matchedSpanLength).map { Self.normalizeForSimilarity(wordTimings[wordIdx + $0].word) }
                        : []
                    let (shouldSkipStopword, adjustedSimilarity) = checkStopwordRules(
                        normalizedWord: normalizedWord,
                        spanLength: matchedSpanLength,
                        spanWords: spanWords,
                        vocabTerm: vocabTerm,
                        currentSimilarity: minSimilarityForSpan
                    )
                    if shouldSkipStopword { continue }
                    minSimilarityForSpan = adjustedSimilarity

                    guard bestSimilarity >= minSimilarityForSpan else { continue }

                    // Build the original phrase (single word or concatenated span)
                    let spanIndices = Array(wordIdx..<(wordIdx + matchedSpanLength))
                    let originalPhrase =
                        matchedSpanLength == 1
                        ? tdtWord
                        : spanIndices.map { wordTimings[$0].word }.joined(separator: " ")

                    // Get temporal window for the span
                    let spanStartTime = wordTimings[wordIdx].startTime
                    let spanEndTime = wordTimings[wordIdx + matchedSpanLength - 1].endTime

                    // Evaluate CTC match using shared helper
                    let candidate = CTCMatchCandidate(
                        originalPhrase: originalPhrase,
                        vocabTerm: vocabTerm,
                        vocabTokens: vocabTokens,
                        similarity: bestSimilarity,
                        spanLength: matchedSpanLength,
                        spanIndices: spanIndices,
                        spanStartTime: spanStartTime,
                        spanEndTime: spanEndTime
                    )

                    let result = evaluateCTCMatch(
                        candidate: candidate,
                        logProbs: logProbs,
                        frameDuration: frameDuration,
                        cbw: cbw,
                        marginSeconds: marginSeconds
                    )

                    if result.shouldReplace {
                        // Collect candidate instead of applying immediately
                        pendingReplacements.append(
                            PendingReplacement(
                                candidate: candidate,
                                result: result,
                                similarity: bestSimilarity
                            )
                        )
                    }
                }
            }
        }

        // PASS 2 & 3: Sort, apply, and reconstruct (shared logic)
        return finalizeReplacements(
            pendingReplacements: pendingReplacements,
            modifiedWords: &modifiedWords,
            replacedIndices: &replacedIndices,
            replacements: &replacements
        )
    }

    // MARK: - CTC Evaluation Helpers

    /// Evaluate a CTC match candidate and determine if replacement should occur.
    ///
    /// This method encapsulates the core CTC scoring logic shared between word-centric
    /// and term-centric algorithms.
    ///
    /// - Parameters:
    ///   - candidate: The match candidate to evaluate
    ///   - logProbs: CTC log-probabilities
    ///   - frameDuration: Duration of each CTC frame in seconds
    ///   - cbw: Context-biasing weight
    ///   - marginSeconds: Temporal margin around word for CTC search
    /// - Returns: Evaluation result with replacement decision
    private func evaluateCTCMatch(
        candidate: CTCMatchCandidate,
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double
    ) -> CTCMatchResult {
        // Calculate frame window
        let marginFrames = Int(marginSeconds / frameDuration)
        let spanStartFrame = Int(candidate.spanStartTime / frameDuration)
        let spanEndFrame = Int(candidate.spanEndTime / frameDuration)

        let searchStart = max(0, spanStartFrame - marginFrames)
        let searchEnd = min(logProbs.count, spanEndFrame + marginFrames)

        // Score vocabulary term using constrained CTC
        let (vocabCtcScore, _, _) = spotter.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: candidate.vocabTokens,
            searchStartFrame: searchStart,
            searchEndFrame: searchEnd
        )

        // Score original phrase using constrained CTC
        guard let tokenizer = ctcTokenizer else {
            debugLog("  [WARN] No tokenizer - skipping CTC comparison for '\(candidate.originalPhrase)'")
            return CTCMatchResult(
                shouldReplace: false,
                originalScore: -Float.infinity,
                boostedVocabScore: vocabCtcScore,
                replacement: candidate.vocabTerm,
                reason: "No tokenizer available"
            )
        }

        let originalTokens = tokenizer.encode(candidate.originalPhrase)
        guard !originalTokens.isEmpty else {
            debugLog("  [WARN] Empty tokens for '\(candidate.originalPhrase)' - skipping")
            return CTCMatchResult(
                shouldReplace: false,
                originalScore: -Float.infinity,
                boostedVocabScore: vocabCtcScore,
                replacement: candidate.vocabTerm,
                reason: "Empty tokens for original phrase"
            )
        }

        let (originalCtcScore, _, _) = spotter.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: originalTokens,
            searchStartFrame: searchStart,
            searchEndFrame: searchEnd
        )

        // Apply adaptive context-biasing weight
        let adaptiveCbwValue = config.adaptiveCbw(baseCbw: cbw, tokenCount: candidate.vocabTokens.count)
        let boostedVocabScore = vocabCtcScore + adaptiveCbwValue

        // CTC-vs-CTC comparison
        let shouldReplace = boostedVocabScore > originalCtcScore

        // Debug output
        let label = candidate.spanLength > 1 ? "[MULTI] " : ""
        debugLog(
            "  \(label)'\(candidate.originalPhrase)' vs '\(candidate.vocabTerm)' "
                + "(sim=\(String(format: "%.2f", candidate.similarity)), span=\(candidate.spanLength))"
        )
        debugLog(
            "    TDT span: [\(String(format: "%.2f", candidate.spanStartTime))-"
                + "\(String(format: "%.2f", candidate.spanEndTime))s]"
        )
        debugLog("    CTC('\(candidate.originalPhrase)'): \(String(format: "%.2f", originalCtcScore))")
        let cbwInfo =
            config.useAdaptiveThresholds
            ? "adaptive=\(String(format: "%.2f", adaptiveCbwValue)) (base=\(cbw), tokens=\(candidate.vocabTokens.count))"
            : String(format: "%.2f", cbw)
        debugLog(
            "    CTC('\(candidate.vocabTerm)'): \(String(format: "%.2f", vocabCtcScore)) + cbw=\(cbwInfo) "
                + "= \(String(format: "%.2f", boostedVocabScore))"
        )
        debugLog("    -> \(shouldReplace ? "REPLACE" : "KEEP") (vocab \(shouldReplace ? ">" : "<=") original)")

        // Preserve capitalization from original
        let firstOriginalWord =
            candidate.originalPhrase.split(separator: " ").first.map(String.init)
            ?? candidate.originalPhrase
        let replacement = preserveCapitalization(original: firstOriginalWord, replacement: candidate.vocabTerm)

        let reasonPrefix = candidate.spanLength > 1 ? "CTC-vs-CTC (multi-word)" : "CTC-vs-CTC"
        let reason =
            "\(reasonPrefix): '\(candidate.vocabTerm)'=\(String(format: "%.2f", boostedVocabScore)) "
            + "> '\(candidate.originalPhrase)'=\(String(format: "%.2f", originalCtcScore))"

        return CTCMatchResult(
            shouldReplace: shouldReplace,
            originalScore: originalCtcScore,
            boostedVocabScore: boostedVocabScore,
            replacement: replacement,
            reason: reason
        )
    }

    /// Apply a replacement to the modified words array and update tracking sets.
    ///
    /// - Parameters:
    ///   - result: The CTC match result
    ///   - candidate: The original match candidate
    ///   - modifiedWords: Array of words being modified (mutated)
    ///   - replacedIndices: Set of already-replaced indices (mutated)
    ///   - replacements: Array of replacement results (mutated)
    private func applyReplacement(
        result: CTCMatchResult,
        candidate: CTCMatchCandidate,
        modifiedWords: inout [(word: String, startTime: Double, endTime: Double)],
        replacedIndices: inout Set<Int>,
        replacements: inout [RescoringResult]
    ) {
        guard let firstIdx = candidate.spanIndices.first else { return }

        // Replace first word with the replacement, mark rest as empty
        modifiedWords[firstIdx].word = result.replacement
        for idx in candidate.spanIndices.dropFirst() {
            modifiedWords[idx].word = ""  // Will be filtered out
        }

        // Mark all indices as replaced
        for idx in candidate.spanIndices {
            replacedIndices.insert(idx)
        }

        // Record the replacement
        replacements.append(
            RescoringResult(
                originalWord: candidate.originalPhrase,
                originalScore: result.originalScore,
                replacementWord: result.replacement,
                replacementScore: result.boostedVocabScore,
                shouldReplace: true,
                reason: result.reason
            )
        )
    }

    /// Check if a word or span should skip replacement due to stopword rules.
    ///
    /// - Parameters:
    ///   - normalizedWord: The normalized single word
    ///   - spanLength: Length of the span (1 for single word)
    ///   - spanWords: All normalized words in the span (for multi-word checks)
    ///   - vocabTerm: The vocabulary term being considered
    ///   - currentSimilarity: Current similarity threshold
    /// - Returns: Tuple of (shouldSkip, adjustedMinSimilarity)
    private func checkStopwordRules(
        normalizedWord: String,
        spanLength: Int,
        spanWords: [String],
        vocabTerm: String,
        currentSimilarity: Float
    ) -> (shouldSkip: Bool, adjustedMinSimilarity: Float) {
        var minSimilarity = currentSimilarity

        // Single-word stopword check - skip entirely
        if spanLength == 1 && Self.stopwords.contains(normalizedWord) {
            debugLog("    [STOPWORD] '\(normalizedWord)' is a stopword, skipping replacement with '\(vocabTerm)'")
            return (shouldSkip: true, adjustedMinSimilarity: minSimilarity)
        }

        // Multi-word span stopword check - raise threshold
        if spanLength >= 2 {
            let containsStopword = spanWords.contains { Self.stopwords.contains($0) }
            if containsStopword {
                minSimilarity = max(minSimilarity, ContextBiasingConstants.stopwordSpanSimilarity)
                debugLog(
                    "    [STOPWORD] span '\(spanWords.joined(separator: " "))' contains stopword, "
                        + "raising threshold to \(String(format: "%.2f", minSimilarity))"
                )
            }
        }

        return (shouldSkip: false, adjustedMinSimilarity: minSimilarity)
    }

    /// Check length ratio rules for single-word matches.
    ///
    /// - Parameters:
    ///   - normalizedWord: The normalized word
    ///   - vocabTerm: The vocabulary term
    ///   - currentSimilarity: Current similarity
    ///   - minSimilarity: Base minimum similarity
    /// - Returns: Adjusted minimum similarity threshold
    private func checkLengthRatioRules(
        normalizedWord: String,
        vocabTerm: String,
        currentSimilarity: Float,
        minSimilarity: Float
    ) -> Float {
        let lengthRatio = Float(normalizedWord.count) / Float(vocabTerm.count)
        if lengthRatio < ContextBiasingConstants.lengthRatioThreshold
            && normalizedWord.count <= ContextBiasingConstants.shortWordMaxLength
        {
            let adjusted = max(minSimilarity, ContextBiasingConstants.shortWordSimilarity)
            if currentSimilarity >= minSimilarity {
                debugLog(
                    "    [LENGTH] '\(normalizedWord)' too short (ratio=\(String(format: "%.2f", lengthRatio))), "
                        + "raising threshold to \(String(format: "%.2f", adjusted))"
                )
            }
            return adjusted
        }
        return minSimilarity
    }
}
