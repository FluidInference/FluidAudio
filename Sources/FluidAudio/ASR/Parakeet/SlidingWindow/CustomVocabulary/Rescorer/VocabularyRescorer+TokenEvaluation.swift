import Foundation

// MARK: - CTC Token Evaluation

extension VocabularyRescorer {

    /// Log debug message only when debug mode is enabled.
    /// Uses closure to avoid string evaluation when debug is off.
    @inline(__always)
    private func debugLog(_ message: @escaping @autoclosure () -> String) {
        guard debugMode else { return }
        logger.debug(message())
    }

    // MARK: - CTC Match Evaluation

    /// Evaluate a CTC match candidate and determine if replacement should occur.
    ///
    /// This method encapsulates the core CTC scoring logic shared between word-centric
    /// and term-centric algorithms.
    ///
    /// - Parameters:
    ///   - candidate: The match candidate to evaluate
    ///   - wordTimings: Original words, used to protect recognized vocabulary occurrences
    ///   - logProbs: CTC log-probabilities
    ///   - frameDuration: Duration of each CTC frame in seconds
    ///   - cbw: Context-biasing weight
    ///   - marginSeconds: Temporal margin around word for CTC search
    /// - Returns: Evaluation result with replacement decision
    func evaluateCTCMatch(
        candidate: CTCMatchCandidate,
        wordTimings: [WordTiming],
        recognizedSpans: RecognizedVocabularySpans,
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double
    ) -> CTCMatchResult {
        let searchWindow = Self.protectedSearchWindow(
            for: candidate, wordTimings: wordTimings, recognizedSpans: recognizedSpans, frameDuration: frameDuration,
            frameCount: logProbs.count, marginSeconds: marginSeconds)
        let searchStart = searchWindow.frames.lowerBound
        let searchEnd = searchWindow.frames.upperBound
        let inflectedTerm = Self.preservingPossessive(candidate.vocabTerm, from: candidate.originalPhrase)
        let boundaryTokens =
            inflectedTerm == candidate.vocabTerm
            ? candidate.vocabTokens : (ctcTokenizer?.encode(inflectedTerm) ?? candidate.vocabTokens)

        // Score vocabulary term against the constrained window. We try the
        // standard `▁`-prefixed tokenization AND the mid-utterance variant
        // (no leading boundary), keeping whichever scores higher. Compound
        // matches like "Liv" + "marli" → "Livmarli" do not begin at a CTC
        // word boundary, so the leading `▁` token has no acoustic
        // counterpart and penalizes the DP unfairly.
        var vocabTokensUsed = boundaryTokens
        var vocabCtcScore: Float = -.infinity
        var acousticFrameRange: Range<Int>?
        do {
            let (score, start, end) = spotter.ctcWordSpotConstrained(
                logProbs: logProbs,
                keywordTokens: boundaryTokens,
                searchStartFrame: searchStart,
                searchEndFrame: searchEnd
            )
            vocabCtcScore = score
            if start < end { acousticFrameRange = start..<end }
        }
        if let tokenizer = ctcTokenizer {
            let altTokens = tokenizer.encodeWithoutBoundary(inflectedTerm)
            if !altTokens.isEmpty && altTokens != boundaryTokens {
                let (altScore, start, end) = spotter.ctcWordSpotConstrained(
                    logProbs: logProbs,
                    keywordTokens: altTokens,
                    searchStartFrame: searchStart,
                    searchEndFrame: searchEnd
                )
                if altScore > vocabCtcScore {
                    vocabCtcScore = altScore
                    vocabTokensUsed = altTokens
                    acousticFrameRange = start < end ? start..<end : nil
                }
            }
        }

        // Score original phrase using constrained CTC
        guard let tokenizer = ctcTokenizer else {
            debugLog("  [WARN] No tokenizer - skipping CTC comparison for '\(candidate.originalPhrase)'")
            return CTCMatchResult(
                shouldReplace: false,
                comparisonWasPerformed: false,
                originalScore: -Float.infinity,
                boostedVocabScore: vocabCtcScore,
                rawVocabularyCTCScore: vocabCtcScore.isFinite ? vocabCtcScore : nil,
                rawOriginalCTCScore: nil,
                effectiveBoost: nil,
                replacement: candidate.vocabTerm,
                reason: "No tokenizer available"
            )
        }

        let originalTokens = tokenizer.encode(candidate.originalPhrase)
        guard !originalTokens.isEmpty else {
            debugLog("  [WARN] Empty tokens for '\(candidate.originalPhrase)' - skipping")
            return CTCMatchResult(
                shouldReplace: false,
                comparisonWasPerformed: false,
                originalScore: -Float.infinity,
                boostedVocabScore: vocabCtcScore,
                rawVocabularyCTCScore: vocabCtcScore.isFinite ? vocabCtcScore : nil,
                rawOriginalCTCScore: nil,
                effectiveBoost: nil,
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

        // Apply adaptive context-biasing weight, scaled to the variant
        // tokenization that actually scored best.
        // A weak textual match must not gain an extra occurrence beside an
        // already recognized name on bias alone. Close spellings and aliases
        // retain the normal bonus so genuine repetitions can still be corrected.
        let configuredBoost = config.adaptiveCbw(baseCbw: cbw, tokenCount: vocabTokensUsed.count)
        let adaptiveCbwValue =
            searchWindow.createsAdjacentDuplicate && !Self.isTextSupportedRepetition(candidate)
            ? min(0, configuredBoost) : configuredBoost
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
        let variantLabel = vocabTokensUsed == boundaryTokens ? "boundary" : "no-boundary"
        let cbwInfo =
            "effective=\(String(format: "%.2f", adaptiveCbwValue)) (base=\(cbw), tokens=\(vocabTokensUsed.count))"
        debugLog(
            "    CTC('\(inflectedTerm)' [\(variantLabel)]): \(String(format: "%.2f", vocabCtcScore)) + cbw=\(cbwInfo) "
                + "= \(String(format: "%.2f", boostedVocabScore))"
        )
        debugLog("    -> \(shouldReplace ? "REPLACE" : "KEEP") (vocab \(shouldReplace ? ">" : "<=") original)")

        // Preserve capitalization from original
        let firstOriginalWord =
            candidate.originalPhrase.split(separator: " ").first.map(String.init)
            ?? candidate.originalPhrase
        let replacement = preserveCapitalization(original: firstOriginalWord, replacement: inflectedTerm)

        let reasonPrefix = candidate.spanLength > 1 ? "CTC-vs-CTC (multi-word)" : "CTC-vs-CTC"
        let reason = Self.ctcComparisonReason(
            prefix: reasonPrefix,
            vocabularyTerm: inflectedTerm,
            boostedVocabularyScore: boostedVocabScore,
            originalPhrase: candidate.originalPhrase,
            originalScore: originalCtcScore,
            comparisonPassed: shouldReplace
        )

        return CTCMatchResult(
            shouldReplace: shouldReplace,
            comparisonWasPerformed: true,
            originalScore: originalCtcScore,
            boostedVocabScore: boostedVocabScore,
            rawVocabularyCTCScore: vocabCtcScore.isFinite ? vocabCtcScore : nil,
            rawOriginalCTCScore: originalCtcScore.isFinite ? originalCtcScore : nil,
            effectiveBoost: adaptiveCbwValue.isFinite ? adaptiveCbwValue : nil,
            replacement: replacement,
            reason: reason,
            acousticFrameRange: acousticFrameRange,
            sourceFrameRange: Int(
                candidate.spanStartTime / frameDuration)..<max(
                    Int(candidate.spanStartTime / frameDuration), Int(ceil(candidate.spanEndTime / frameDuration)))
        )
    }

    /// Build a truthful display reason for an available numeric CTC comparison.
    static func ctcComparisonReason(
        prefix: String,
        vocabularyTerm: String,
        boostedVocabularyScore: Float,
        originalPhrase: String,
        originalScore: Float,
        comparisonPassed: Bool
    ) -> String {
        let comparisonOperator = comparisonPassed ? ">" : "<="
        return
            "\(prefix): '\(vocabularyTerm)'=\(String(format: "%.2f", boostedVocabularyScore)) "
            + "\(comparisonOperator) '\(originalPhrase)'=\(String(format: "%.2f", originalScore))"
    }

    /// Convert the legacy CTC evaluation into stable, non-sentinel diagnostic evidence.
    static func makeCandidateEvidence(
        candidateID: Int,
        candidate: CTCMatchCandidate,
        result: CTCMatchResult,
        wordRange: Range<Int>,
        baseTextUTF8Range: Range<Int>?
    ) -> CandidateEvidence {
        let rawVocabularyScore = finite(result.rawVocabularyCTCScore)
        let rawOriginalScore = finite(result.rawOriginalCTCScore)
        let effectiveBoost = finite(result.effectiveBoost)
        let finiteScoresAvailable =
            rawVocabularyScore != nil && rawOriginalScore != nil && effectiveBoost != nil
        let comparisonPassed = result.comparisonWasPerformed && result.shouldReplace

        let legacyOutcome: LegacyApplicationOutcome
        if !result.comparisonWasPerformed {
            legacyOutcome = .unavailableEvidence
        } else if comparisonPassed {
            // The final overlap pass rewrites this to `.supersededByOverlap` when another candidate wins.
            legacyOutcome = .applied
        } else {
            legacyOutcome = .rejectedByComparison
        }
        let reason =
            finiteScoresAvailable
            ? result.reason
            : nonNumericComparisonReason(
                legacyReason: result.reason,
                comparisonWasPerformed: result.comparisonWasPerformed,
                comparisonPassed: comparisonPassed
            )

        return CandidateEvidence(
            candidateID: candidateID,
            origin: candidate.origin,
            basePhrase: candidate.originalPhrase,
            canonicalTerm: candidate.vocabTerm,
            matchedAlias: candidate.matchedAlias,
            similarity: candidate.similarity,
            rawVocabularyCTCScore: rawVocabularyScore,
            rawOriginalCTCScore: rawOriginalScore,
            effectiveBoost: effectiveBoost,
            wordRange: wordRange,
            tokenRange: candidate.tokenRange,
            baseTextUTF8Range: baseTextUTF8Range,
            startTime: candidate.spanStartTime.isFinite ? candidate.spanStartTime : nil,
            endTime: candidate.spanEndTime.isFinite ? candidate.spanEndTime : nil,
            comparisonPassed: comparisonPassed,
            legacyOutcome: legacyOutcome,
            reason: reason
        )
    }

    private static func nonNumericComparisonReason(
        legacyReason: String,
        comparisonWasPerformed: Bool,
        comparisonPassed: Bool
    ) -> String {
        guard legacyReason.contains(" > ") || legacyReason.contains(" <= ") else {
            return legacyReason
        }
        if comparisonWasPerformed {
            return
                "CTC comparison \(comparisonPassed ? "passed" : "rejected"): "
                + "one or more finite diagnostic scores were unavailable"
        }
        return "CTC comparison unavailable: required finite score evidence was missing"
    }

    private static func finite(_ value: Float?) -> Float? {
        guard let value, value.isFinite else { return nil }
        return value
    }

    // MARK: - Replacement Application

    /// Apply a replacement to the modified words array and update tracking sets.
    ///
    /// - Parameters:
    ///   - result: The CTC match result
    ///   - candidate: The original match candidate
    ///   - modifiedWords: Array of words being modified (mutated)
    ///   - replacedIndices: Set of already-replaced indices (mutated)
    ///   - replacements: Array of replacement results (mutated)
    func applyReplacement(
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

    // MARK: - Validation Rules

    /// Check if a word or span should skip replacement due to stopword rules.
    ///
    /// - Parameters:
    ///   - normalizedWord: The normalized single word
    ///   - spanLength: Length of the span (1 for single word)
    ///   - spanWords: All normalized words in the span (for multi-word checks)
    ///   - vocabTerm: The vocabulary term being considered
    ///   - currentSimilarity: Current similarity threshold
    /// - Returns: Tuple of (shouldSkip, adjustedMinSimilarity)
    func checkStopwordRules(
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
            let containsStopword = spanWords.contains { Self.multiWordStopwords.contains($0) }
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
    func checkLengthRatioRules(
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
