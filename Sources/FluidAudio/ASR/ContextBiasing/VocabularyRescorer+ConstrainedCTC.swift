import Foundation

// MARK: - Constrained CTC Rescoring

extension VocabularyRescorer {

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
        cbw: Float = 3.0,
        marginSeconds: Double = 0.5,
        minSimilarity: Float = 0.5
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
        cbw: Float = 3.0,
        marginSeconds: Double = 0.5,
        minSimilarity: Float = 0.5
    ) -> RescoreOutput {
        // Build word-level timings from token timings
        let wordTimings = buildWordTimings(from: tokenTimings)

        guard !wordTimings.isEmpty, !logProbs.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        if debugMode {
            print("=== VocabularyRescorer (Constrained CTC - Word-Centric) ===")
            print("Words: \(wordTimings.count), Frames: \(logProbs.count), Vocab: \(vocabulary.terms.count)")
            print("Frame duration: \(String(format: "%.4f", frameDuration))s")
            print("CBW: \(cbw), Margin: \(marginSeconds)s, MinSimilarity: \(minSimilarity)")
            print("Mode: \(useBKTree ? "BK-tree O(W × log V)" : "Linear scan O(W × V)")")
        }

        var replacements: [RescoringResult] = []
        var modifiedWords: [(word: String, startTime: Double, endTime: Double)] = wordTimings.map {
            (word: $0.word, startTime: $0.startTime, endTime: $0.endTime)
        }
        var replacedIndices = Set<Int>()

        // Build normalized vocabulary set for guard checks
        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()

        // Stopwords defined once outside the loop for efficiency
        let stopwords: Set<String> = [
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

        // Pre-compute normalized words for all timings
        let normalizedWords = wordTimings.map { Self.normalizeForSimilarity($0.word) }

        // WORD-CENTRIC LOOP: For each TDT word, find candidate vocabulary terms
        for (wordIdx, timing) in wordTimings.enumerated() {
            guard !replacedIndices.contains(wordIdx) else { continue }

            let tdtWord = timing.word
            let normalizedWord = normalizedWords[wordIdx]
            guard !normalizedWord.isEmpty else { continue }

            // Skip stopwords for single-word matching (checked again for compounds later)
            if stopwords.contains(normalizedWord) {
                if debugMode {
                    print("  [STOPWORD] Skipping '\(normalizedWord)' (single word)")
                }
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

            if debugMode && !candidates.isEmpty {
                let candidateInfo = candidates.prefix(5).map {
                    "\($0.term.text)(sim=\(String(format: "%.2f", $0.similarity)), span=\($0.spanLength))"
                }.joined(separator: ", ")
                print("  '\(tdtWord)' -> \(candidates.count) candidates: \(candidateInfo)")
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
                    if debugMode {
                        print("  Skipping '\(vocabTerm)': phrase '\(originalPhrase)' matches another vocab term")
                    }
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
                    let lengthRatio = Float(normalizedWord.count) / Float(vocabTerm.count)
                    if lengthRatio < 0.75 && normalizedWord.count <= 4 {
                        minSimilarityForSpan = max(minSimilarityForSpan, 0.80)
                        if debugMode && similarity >= minSimilarity {
                            print(
                                "    [LENGTH] '\(normalizedWord)' too short (ratio=\(String(format: "%.2f", lengthRatio))), "
                                    + "raising threshold to \(String(format: "%.2f", minSimilarityForSpan))"
                            )
                        }
                    }
                }

                // STOPWORD CHECKS
                if spanLength == 1 && stopwords.contains(normalizedWord) {
                    if debugMode {
                        print(
                            "    [STOPWORD] '\(normalizedWord)' is a stopword, skipping replacement with '\(vocabTerm)'"
                        )
                    }
                    continue
                }

                if spanLength >= 2 {
                    let spanWords = spanIndices.map { normalizedWords[$0] }
                    let containsStopword = spanWords.contains { stopwords.contains($0) }
                    if containsStopword {
                        minSimilarityForSpan = max(minSimilarityForSpan, 0.85)
                        if debugMode && similarity >= minSimilarity {
                            print(
                                "    [STOPWORD] span '\(spanWords.joined(separator: " "))' contains stopword, "
                                    + "raising threshold to \(String(format: "%.2f", minSimilarityForSpan))"
                            )
                        }
                    }
                }

                // Check if similarity meets threshold after all adjustments
                guard similarity >= minSimilarityForSpan else { continue }

                // Get temporal window for the span
                let spanStartTime = wordTimings[wordIdx].startTime
                let spanEndTime = wordTimings[wordIdx + spanLength - 1].endTime

                let marginFrames = Int(marginSeconds / frameDuration)
                let spanStartFrame = Int(spanStartTime / frameDuration)
                let spanEndFrame = Int(spanEndTime / frameDuration)

                let searchStart = max(0, spanStartFrame - marginFrames)
                let searchEnd = min(logProbs.count, spanEndFrame + marginFrames)

                // Score vocabulary term using constrained CTC
                let (vocabCtcScore, _, _) = spotter.ctcWordSpotConstrained(
                    logProbs: logProbs,
                    keywordTokens: vocabTokens,
                    searchStartFrame: searchStart,
                    searchEndFrame: searchEnd
                )

                // Score original phrase using constrained CTC (same window)
                var originalCtcScore: Float = -Float.infinity
                if let tokenizer = ctcTokenizer {
                    let originalTokens = tokenizer.encode(originalPhrase)
                    if !originalTokens.isEmpty {
                        let (score, _, _) = spotter.ctcWordSpotConstrained(
                            logProbs: logProbs,
                            keywordTokens: originalTokens,
                            searchStartFrame: searchStart,
                            searchEndFrame: searchEnd
                        )
                        originalCtcScore = score
                    }
                }

                // Apply adaptive context-biasing weight based on vocabulary token count
                let adaptiveCbwValue = config.adaptiveCbw(baseCbw: cbw, tokenCount: vocabTokens.count)
                let boostedVocabScore = vocabCtcScore + adaptiveCbwValue

                // CTC-vs-CTC comparison (same scale, per NeMo paper)
                let shouldReplace = boostedVocabScore > originalCtcScore

                if debugMode {
                    print(
                        "  '\(originalPhrase)' vs '\(vocabTerm)' (sim=\(String(format: "%.2f", similarity)), span=\(spanLength))"
                    )
                    print(
                        "    TDT span: [\(String(format: "%.2f", spanStartTime))-\(String(format: "%.2f", spanEndTime))s]"
                    )
                    print(
                        "    CTC('\(originalPhrase)'): \(String(format: "%.2f", originalCtcScore))"
                    )
                    let cbwInfo =
                        config.useAdaptiveThresholds
                        ? "adaptive=\(String(format: "%.2f", adaptiveCbwValue)) (base=\(cbw), tokens=\(vocabTokens.count))"
                        : String(format: "%.2f", cbw)
                    print(
                        "    CTC('\(vocabTerm)'): \(String(format: "%.2f", vocabCtcScore)) + cbw=\(cbwInfo) = \(String(format: "%.2f", boostedVocabScore))"
                    )
                    print(
                        "    -> \(shouldReplace ? "REPLACE" : "KEEP") (vocab \(boostedVocabScore > originalCtcScore ? ">" : "<=") original)"
                    )
                }

                if shouldReplace {
                    let replacement = preserveCapitalization(original: tdtWord, replacement: vocabTerm)
                    modifiedWords[wordIdx].word = replacement
                    // Mark additional words in span as empty (will be filtered out)
                    for idx in spanIndices.dropFirst() {
                        modifiedWords[idx].word = ""
                    }
                    // Mark all indices as replaced
                    for idx in spanIndices {
                        replacedIndices.insert(idx)
                    }

                    replacements.append(
                        RescoringResult(
                            originalWord: originalPhrase,
                            originalScore: originalCtcScore,
                            replacementWord: replacement,
                            replacementScore: boostedVocabScore,
                            shouldReplace: true,
                            reason:
                                "CTC-vs-CTC: '\(vocabTerm)'=\(String(format: "%.2f", boostedVocabScore)) > '\(originalPhrase)'=\(String(format: "%.2f", originalCtcScore))"
                        ))

                    // Break out of candidate loop - this word is now replaced
                    break
                }
            }
        }

        // Reconstruct transcript from modified words (filter empty strings from multi-word replacements)
        let modifiedText = modifiedWords.map { $0.word }.filter { !$0.isEmpty }.joined(separator: " ")
        let wasModified = !replacements.isEmpty

        if debugMode {
            print("Final: \(modifiedText)")
            print("Replacements: \(replacements.count)")
            print("===========================================")
        }

        return RescoreOutput(
            text: modifiedText,
            replacements: replacements,
            wasModified: wasModified
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
        cbw: Float = 3.0,
        marginSeconds: Double = 0.5,
        minSimilarity: Float = 0.5
    ) -> RescoreOutput {
        // Build word-level timings from token timings
        let wordTimings = buildWordTimings(from: tokenTimings)

        guard !wordTimings.isEmpty, !logProbs.isEmpty else {
            return RescoreOutput(text: transcript, replacements: [], wasModified: false)
        }

        if debugMode {
            print("=== VocabularyRescorer (Constrained CTC - Term-Centric) ===")
            print("Words: \(wordTimings.count), Frames: \(logProbs.count)")
            print("Frame duration: \(String(format: "%.4f", frameDuration))s")
            print("CBW: \(cbw), Margin: \(marginSeconds)s, MinSimilarity: \(minSimilarity)")
        }

        var replacements: [RescoringResult] = []
        var modifiedWords: [(word: String, startTime: Double, endTime: Double)] = wordTimings.map {
            (word: $0.word, startTime: $0.startTime, endTime: $0.endTime)
        }
        var replacedIndices = Set<Int>()

        // Build normalized vocabulary set for guard checks
        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()

        // TERM-CENTRIC LOOP: For each vocabulary term, find similar TDT words and run constrained CTC
        for term in vocabulary.terms {
            let vocabTerm = term.text

            // Skip short vocabulary terms (per NeMo CTC-WS paper)
            guard vocabTerm.count >= vocabulary.minTermLength else {
                if debugMode {
                    print(
                        "  Skipping '\(vocabTerm)': too short (\(vocabTerm.count) < \(vocabulary.minTermLength) chars)")
                }
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
                            if debugMode {
                                print(
                                    "  [MULTI] Skipping '\(vocabTerm)': phrase '\(tdtPhrase)' matches another vocab term"
                                )
                            }
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
                        let spanStartTime = wordTimings[spanIndices.first!].startTime
                        let spanEndTime = wordTimings[spanIndices.last!].endTime

                        let marginFrames = Int(marginSeconds / frameDuration)
                        let spanStartFrame = Int(spanStartTime / frameDuration)
                        let spanEndFrame = Int(spanEndTime / frameDuration)

                        let searchStart = max(0, spanStartFrame - marginFrames)
                        let searchEnd = min(logProbs.count, spanEndFrame + marginFrames)

                        // Score vocabulary phrase using constrained CTC
                        let (vocabCtcScore, _, _) = spotter.ctcWordSpotConstrained(
                            logProbs: logProbs,
                            keywordTokens: vocabTokens,
                            searchStartFrame: searchStart,
                            searchEndFrame: searchEnd
                        )

                        // Score original TDT phrase using constrained CTC
                        var originalCtcScore: Float = -Float.infinity
                        if let tokenizer = ctcTokenizer {
                            let originalTokens = tokenizer.encode(tdtPhrase)
                            if !originalTokens.isEmpty {
                                let (score, _, _) = spotter.ctcWordSpotConstrained(
                                    logProbs: logProbs,
                                    keywordTokens: originalTokens,
                                    searchStartFrame: searchStart,
                                    searchEndFrame: searchEnd
                                )
                                originalCtcScore = score
                            }
                        }

                        // Compute adaptive CBW based on vocabulary token count
                        let adaptiveCbwValue = config.adaptiveCbw(baseCbw: cbw, tokenCount: vocabTokens.count)
                        let boostedVocabScore = vocabCtcScore + adaptiveCbwValue
                        let shouldReplace = boostedVocabScore > originalCtcScore

                        if debugMode {
                            print(
                                "  [MULTI] '\(tdtPhrase)' vs '\(vocabTerm)' (sim=\(String(format: "%.2f", bestSimilarity)))"
                            )
                            print(
                                "    TDT span: [\(String(format: "%.2f", spanStartTime))-\(String(format: "%.2f", spanEndTime))s] words=\(spanLength)"
                            )
                            print(
                                "    CTC('\(tdtPhrase)'): \(String(format: "%.2f", originalCtcScore))"
                            )
                            let cbwInfo =
                                config.useAdaptiveThresholds
                                ? "adaptive=\(String(format: "%.2f", adaptiveCbwValue)) (base=\(cbw), tokens=\(vocabTokens.count))"
                                : String(format: "%.2f", cbw)
                            print(
                                "    CTC('\(vocabTerm)'): \(String(format: "%.2f", vocabCtcScore)) + cbw=\(cbwInfo) = \(String(format: "%.2f", boostedVocabScore))"
                            )
                            print(
                                "    -> \(shouldReplace ? "REPLACE" : "KEEP") (vocab \(boostedVocabScore > originalCtcScore ? ">" : "<=") original)"
                            )
                        }

                        if shouldReplace {
                            // Replace first word with full phrase, mark rest as empty
                            let replacement = preserveCapitalization(
                                original: spanWords.first ?? tdtPhrase,
                                replacement: vocabTerm
                            )
                            modifiedWords[spanIndices.first!].word = replacement
                            for idx in spanIndices.dropFirst() {
                                modifiedWords[idx].word = ""  // Will be filtered out
                            }
                            // Mark all indices as replaced
                            for idx in spanIndices {
                                replacedIndices.insert(idx)
                            }

                            replacements.append(
                                RescoringResult(
                                    originalWord: tdtPhrase,
                                    originalScore: originalCtcScore,
                                    replacementWord: replacement,
                                    replacementScore: boostedVocabScore,
                                    shouldReplace: true,
                                    reason:
                                        "CTC-vs-CTC (multi-word): '\(vocabTerm)'=\(String(format: "%.2f", boostedVocabScore)) > '\(tdtPhrase)'=\(String(format: "%.2f", originalCtcScore))"
                                ))
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
                        if debugMode {
                            print("  Skipping '\(vocabTerm)': word '\(tdtWord)' matches another vocab term")
                        }
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
                    if let norm2 = normalized2, !norm2.isEmpty, vocabTerm.count >= minLengthFor2Word {
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

                    // 3-word compound matching (for longer vocabulary terms only)
                    if let norm2 = normalized2, let norm3 = normalized3,
                        !norm2.isEmpty, !norm3.isEmpty, vocabTerm.count >= minLengthFor3Word
                    {
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

                    // Use adaptive similarity threshold (use concatenated text for multi-word spans)
                    var minSimilarityForSpan = requiredSimilarity(
                        minSimilarity: minSimilarity,
                        spanLength: matchedSpanLength,
                        normalizedText: matchedConcatenation
                    )

                    // LENGTH RATIO CHECK: Prevent short common words from matching longer vocab terms
                    // e.g., "and" (3 chars) should not match "Andre" (5 chars) even with ~60% similarity
                    if matchedSpanLength == 1 {
                        let lengthRatio = Float(normalizedWord.count) / Float(vocabTerm.count)
                        if lengthRatio < 0.75 && normalizedWord.count <= 4 {
                            // For short words with low length ratio, require much higher similarity
                            minSimilarityForSpan = max(minSimilarityForSpan, 0.80)
                            if debugMode && bestSimilarity >= minSimilarity {
                                print(
                                    "    [LENGTH] '\(normalizedWord)' too short (ratio=\(String(format: "%.2f", lengthRatio))), "
                                        + "raising threshold to \(String(format: "%.2f", minSimilarityForSpan))"
                                )
                            }
                        }
                    }

                    // STOPWORD CHECK: Prevent common words from being replaced
                    let stopwords: Set<String> = [
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

                    // For single-word matches, skip entirely if the TDT word is a stopword
                    // This prevents "and" → "Jane", "comes" → "James", etc.
                    if matchedSpanLength == 1 && stopwords.contains(normalizedWord) {
                        if debugMode {
                            print(
                                "    [STOPWORD] '\(normalizedWord)' is a stopword, skipping replacement with '\(vocabTerm)'"
                            )
                        }
                        continue
                    }

                    // For multi-word spans, check if any word is a stopword
                    // This prevents "and we" → "Andre", "at this" → "Matthew", etc.
                    if matchedSpanLength >= 2 {
                        let spanWords = (0..<matchedSpanLength).map {
                            Self.normalizeForSimilarity(wordTimings[wordIdx + $0].word)
                        }
                        let containsStopword = spanWords.contains { stopwords.contains($0) }
                        if containsStopword {
                            // Require very high similarity when span contains stopwords
                            minSimilarityForSpan = max(minSimilarityForSpan, 0.85)
                            if debugMode && bestSimilarity >= minSimilarity {
                                print(
                                    "    [STOPWORD] span '\(spanWords.joined(separator: " "))' contains stopword, "
                                        + "raising threshold to \(String(format: "%.2f", minSimilarityForSpan))"
                                )
                            }
                        }
                    }

                    guard bestSimilarity >= minSimilarityForSpan else { continue }

                    // Build the original phrase (single word or concatenated span)
                    let spanIndices = Array(wordIdx..<(wordIdx + matchedSpanLength))
                    let originalPhrase =
                        matchedSpanLength == 1
                        ? tdtWord
                        : spanIndices.map { wordTimings[$0].word }.joined(separator: " ")

                    // Get temporal window for the span (use direct indexing to avoid force unwraps)
                    let spanStartTime = wordTimings[wordIdx].startTime
                    let spanEndTime = wordTimings[wordIdx + matchedSpanLength - 1].endTime

                    let marginFrames = Int(marginSeconds / frameDuration)
                    let spanStartFrame = Int(spanStartTime / frameDuration)
                    let spanEndFrame = Int(spanEndTime / frameDuration)

                    let searchStart = max(0, spanStartFrame - marginFrames)
                    let searchEnd = min(logProbs.count, spanEndFrame + marginFrames)

                    // Score vocabulary term using constrained CTC
                    let (vocabCtcScore, _, _) = spotter.ctcWordSpotConstrained(
                        logProbs: logProbs,
                        keywordTokens: vocabTokens,
                        searchStartFrame: searchStart,
                        searchEndFrame: searchEnd
                    )

                    // Score original phrase using constrained CTC (same window)
                    var originalCtcScore: Float = -Float.infinity
                    if let tokenizer = ctcTokenizer {
                        let originalTokens = tokenizer.encode(originalPhrase)
                        if !originalTokens.isEmpty {
                            let (score, _, _) = spotter.ctcWordSpotConstrained(
                                logProbs: logProbs,
                                keywordTokens: originalTokens,
                                searchStartFrame: searchStart,
                                searchEndFrame: searchEnd
                            )
                            originalCtcScore = score
                        }
                    }

                    // Apply adaptive context-biasing weight based on vocabulary token count
                    let adaptiveCbwValue = config.adaptiveCbw(baseCbw: cbw, tokenCount: vocabTokens.count)
                    let boostedVocabScore = vocabCtcScore + adaptiveCbwValue

                    // CTC-vs-CTC comparison (same scale, per NeMo paper)
                    let shouldReplace = boostedVocabScore > originalCtcScore

                    if debugMode {
                        print(
                            "  '\(originalPhrase)' vs '\(vocabTerm)' (sim=\(String(format: "%.2f", bestSimilarity)), span=\(matchedSpanLength))"
                        )
                        print(
                            "    TDT span: [\(String(format: "%.2f", spanStartTime))-\(String(format: "%.2f", spanEndTime))s]"
                        )
                        print(
                            "    CTC('\(originalPhrase)'): \(String(format: "%.2f", originalCtcScore))"
                        )
                        let cbwInfo =
                            config.useAdaptiveThresholds
                            ? "adaptive=\(String(format: "%.2f", adaptiveCbwValue)) (base=\(cbw), tokens=\(vocabTokens.count))"
                            : String(format: "%.2f", cbw)
                        print(
                            "    CTC('\(vocabTerm)'): \(String(format: "%.2f", vocabCtcScore)) + cbw=\(cbwInfo) = \(String(format: "%.2f", boostedVocabScore))"
                        )
                        print(
                            "    -> \(shouldReplace ? "REPLACE" : "KEEP") (vocab \(boostedVocabScore > originalCtcScore ? ">" : "<=") original)"
                        )
                    }

                    if shouldReplace {
                        let replacement = preserveCapitalization(original: tdtWord, replacement: vocabTerm)
                        modifiedWords[wordIdx].word = replacement
                        // Mark additional words in span as empty (will be filtered out)
                        for idx in spanIndices.dropFirst() {
                            modifiedWords[idx].word = ""
                        }
                        // Mark all indices as replaced
                        for idx in spanIndices {
                            replacedIndices.insert(idx)
                        }

                        replacements.append(
                            RescoringResult(
                                originalWord: originalPhrase,
                                originalScore: originalCtcScore,
                                replacementWord: replacement,
                                replacementScore: boostedVocabScore,
                                shouldReplace: true,
                                reason:
                                    "CTC-vs-CTC: '\(vocabTerm)'=\(String(format: "%.2f", boostedVocabScore)) > '\(originalPhrase)'=\(String(format: "%.2f", originalCtcScore))"
                            ))
                    }
                }
            }
        }

        // Reconstruct transcript from modified words (filter empty strings from multi-word replacements)
        let modifiedText = modifiedWords.map { $0.word }.filter { !$0.isEmpty }.joined(separator: " ")
        let wasModified = !replacements.isEmpty

        if debugMode {
            print("Final: \(modifiedText)")
            print("Replacements: \(replacements.count)")
            print("===========================================")
        }

        return RescoreOutput(
            text: modifiedText,
            replacements: replacements,
            wasModified: wasModified
        )
    }
}
