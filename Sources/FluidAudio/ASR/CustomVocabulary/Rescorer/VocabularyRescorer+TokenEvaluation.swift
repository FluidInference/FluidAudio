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
    ///   - logProbs: CTC log-probabilities
    ///   - frameDuration: Duration of each CTC frame in seconds
    ///   - cbw: Context-biasing weight
    ///   - marginSeconds: Temporal margin around word for CTC search
    /// - Returns: Evaluation result with replacement decision
    func evaluateCTCMatch(
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
