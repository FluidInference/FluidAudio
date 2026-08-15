import Foundation

// MARK: - CTC Token Rescoring

extension VocabularyRescorer {

    /// Log debug message only when debug mode is enabled.
    /// Uses closure to avoid string evaluation when debug is off.
    @inline(__always)
    private func debugLog(_ message: @escaping @autoclosure () -> String) {
        guard debugMode else { return }
        logger.debug(message())
    }

    // MARK: - Stopwords

    /// Common stopwords used by the **single-word** rescue path to skip
    /// substituting vocabulary terms over short common words (prevents
    /// false positives like `just` → `Wyost`). Wider than the multi-word
    /// set because lone-word substitutions are the most error-prone.
    static let stopwords: Set<String> = [
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

    /// Subset of `stopwords` used by the **multi-word** rescue path to
    /// raise the similarity floor on phrases like `at this` → `Matthew`.
    /// Restricted to true function words so content words like
    /// `new red` → `Newrez` (sim 0.83) are not silently upgraded to a
    /// 0.85 threshold and rejected.
    static let multiWordStopwords: Set<String> = [
        // Articles and determiners
        "a", "an", "the", "some", "any", "no", "every", "each", "all",
        // Conjunctions
        "and", "or", "but", "so", "if", "then", "than", "as",
        // Prepositions
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down",
        "out", "about", "into", "over", "after", "before", "between", "under",
        // Be verbs
        "is", "are", "was", "were", "be", "been", "being", "am",
        // Auxiliaries
        "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
        // Pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
        "who", "what", "which", "where", "when", "how", "why",
    ]

    // MARK: - CTC Match Types

    /// Parameters for evaluating a CTC match candidate.
    struct CTCMatchCandidate {
        let origin: CandidateOrigin
        let originalPhrase: String
        let vocabTerm: String
        let matchedAlias: String?
        let vocabTokens: [Int]
        let similarity: Float
        let spanLength: Int
        let spanIndices: [Int]
        let tokenRange: Range<Int>?
        let spanStartTime: Double
        let spanEndTime: Double
    }

    /// Result of CTC match evaluation.
    struct CTCMatchResult {
        let shouldReplace: Bool
        let comparisonWasPerformed: Bool
        let originalScore: Float
        let boostedVocabScore: Float
        let rawVocabularyCTCScore: Float?
        let rawOriginalCTCScore: Float?
        let effectiveBoost: Float?
        let replacement: String
        let reason: String
    }

    /// Pending replacement candidate for two-pass selection.
    /// Stores all info needed to apply the replacement later.
    struct PendingReplacement {
        let candidateID: Int?
        let candidate: CTCMatchCandidate
        let result: CTCMatchResult
        let similarity: Float  // String similarity for sorting
    }

    /// Evidence-only state. Legacy rescoring keeps this optional `nil` and allocates none of its arrays.
    struct CandidateEvidenceCollector {
        var baseText: String = ""
        var baseWords: [String] = []
        var alignedWordRanges: [Range<Int>?] = []
        var candidates: [CandidateEvidence] = []
    }

    /// Result of the legacy greedy overlap pass before text mutation is applied.
    struct PendingReplacementArbitration {
        let applied: [PendingReplacement]
        let supersededCandidateIDs: Set<Int>
    }

    /// Return a half-open token range when the word span has contiguous token provenance.
    static func tokenRange(
        for spanIndices: [Int],
        in wordTimings: [WordTiming]
    ) -> Range<Int>? {
        guard let firstIndex = spanIndices.first,
            spanIndices.allSatisfy({ wordTimings.indices.contains($0) })
        else {
            return nil
        }

        guard let firstRange = wordTimings[firstIndex].tokenRange else { return nil }
        var tokenEnd = firstRange.upperBound
        for (leftIndex, rightIndex) in zip(spanIndices, spanIndices.dropFirst()) {
            guard rightIndex == leftIndex + 1 else { return nil }
            guard let rightRange = wordTimings[rightIndex].tokenRange,
                rightRange.lowerBound == tokenEnd
            else {
                return nil
            }
            tokenEnd = rightRange.upperBound
        }

        return firstRange.lowerBound..<tokenEnd
    }

    /// Append one evidence row only when the caller requested evidence collection.
    @discardableResult
    static func recordCandidateEvidence(
        candidate: CTCMatchCandidate,
        result: CTCMatchResult,
        candidateEvidence: inout CandidateEvidenceCollector?
    ) -> Int? {
        guard let candidateID = candidateEvidence?.candidates.count else { return nil }

        let wordRange: Range<Int>
        if let firstIndex = candidate.spanIndices.first,
            let lastIndex = candidate.spanIndices.last,
            candidate.spanIndices == Array(firstIndex...lastIndex)
        {
            wordRange = firstIndex..<(lastIndex + 1)
        } else {
            wordRange = 0..<0
        }

        let textRange: Range<Int>?
        if let collector = candidateEvidence {
            textRange = candidateBaseTextUTF8Range(
                wordRange: wordRange,
                alignedWordRanges: collector.alignedWordRanges,
                baseText: collector.baseText,
                basePhrase: candidate.originalPhrase
            )
        } else {
            textRange = nil
        }

        candidateEvidence?.candidates.append(
            makeCandidateEvidence(
                candidateID: candidateID,
                candidate: candidate,
                result: result,
                wordRange: wordRange,
                baseTextUTF8Range: textRange
            )
        )
        return candidateID
    }

    // MARK: - Shared Finalization

    /// Finalize replacements: sort by span length, apply greedily, reconstruct transcript.
    ///
    /// This helper handles Pass 2 & 3 of the two-pass algorithm.
    private func finalizeReplacements(
        pendingReplacements: [PendingReplacement],
        modifiedWords: inout [(word: String, startTime: Double, endTime: Double)],
        replacedIndices: inout Set<Int>,
        replacements: inout [RescoringResult],
        candidateEvidence: inout CandidateEvidenceCollector?
    ) -> RescoreOutput {
        // PASS 2: Sort by similarity (descending), with span length used
        // only as a tiebreak.
        //
        // Rationale: when a longer multi-word span has substantially higher
        // similarity than a shorter overlapping span, the longer match
        // should win. This handles the FDA-extended failure pattern where
        // TDT splits an unfamiliar drug like `Romvimza` into `Rom vimza`,
        // and the rescorer finds:
        //   - 2-word: `rom vimza` → `Romvimza` (sim 0.89, real drug)
        //   - 1-word: `vimza` → `Cimzia` (sim 0.67, distractor)
        // The previous "shortest span wins" rule picked the 1-word
        // distractor purely because it was shorter, ignoring that the
        // 2-word match has 22pp better similarity.
        //
        // Span length still tiebreaks at near-equal similarity to keep the
        // existing preference for compact replacements when both candidates
        // are equally plausible.
        // Quantize similarity into 0.05-wide buckets so candidates whose raw
        // similarities differ by less than the bucket width are treated as
        // equivalent for ranking. Span length breaks ties within a bucket;
        // raw similarity breaks ties at equal span. Quantization yields a
        // strict weak ordering (transitive by construction) — the previous
        // implementation compared `abs(simDiff) > 0.05` directly, which is
        // non-transitive: e.g. for similarities 0.70/0.66/0.62 across span
        // lengths 3/2/1, A vs B and B vs C dispatch to the span tiebreaker
        // while A vs C dispatches to similarity, producing a cycle.
        let arbitration = Self.arbitratePendingReplacements(
            pendingReplacements,
            occupiedIndices: replacedIndices
        )

        // PASS 3: Greedily apply non-overlapping replacements
        var appliedCandidateIDs = Set<Int>()
        for pending in arbitration.applied {
            applyReplacement(
                result: pending.result,
                candidate: pending.candidate,
                modifiedWords: &modifiedWords,
                replacedIndices: &replacedIndices,
                replacements: &replacements
            )
            if let candidateID = pending.candidateID {
                appliedCandidateIDs.insert(candidateID)
            }
        }

        if var collector = candidateEvidence {
            Self.reconcileLegacyOutcomes(
                candidates: &collector.candidates,
                appliedCandidateIDs: appliedCandidateIDs
            )
            candidateEvidence = collector
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

    /// Run the legacy similarity ordering and greedy overlap selection without mutating text.
    static func arbitratePendingReplacements(
        _ pendingReplacements: [PendingReplacement],
        occupiedIndices: Set<Int> = []
    ) -> PendingReplacementArbitration {
        let quantized: (Float) -> Int = { Int(($0 / 0.05).rounded()) }
        let sortedReplacements = pendingReplacements.sorted { a, b in
            let aBucket = quantized(a.similarity)
            let bBucket = quantized(b.similarity)
            if aBucket != bBucket {
                return aBucket > bBucket
            }
            if a.candidate.spanLength != b.candidate.spanLength {
                return a.candidate.spanLength < b.candidate.spanLength
            }
            return a.similarity > b.similarity
        }

        var occupiedIndices = occupiedIndices
        var applied: [PendingReplacement] = []
        var supersededCandidateIDs = Set<Int>()
        for pending in sortedReplacements {
            guard pending.candidate.spanIndices.allSatisfy({ !occupiedIndices.contains($0) }) else {
                if let candidateID = pending.candidateID {
                    supersededCandidateIDs.insert(candidateID)
                }
                continue
            }
            applied.append(pending)
            occupiedIndices.formUnion(pending.candidate.spanIndices)
        }

        return PendingReplacementArbitration(
            applied: applied,
            supersededCandidateIDs: supersededCandidateIDs
        )
    }

    /// Resolve final outcomes after the legacy greedy overlap pass has selected its winners.
    static func reconcileLegacyOutcomes(
        candidates: inout [CandidateEvidence],
        appliedCandidateIDs: Set<Int>
    ) {
        for index in candidates.indices where candidates[index].comparisonPassed {
            let outcome: LegacyApplicationOutcome =
                appliedCandidateIDs.contains(candidates[index].candidateID)
                ? .applied
                : .supersededByOverlap
            candidates[index] = candidates[index].replacingLegacyOutcome(outcome)
        }
    }

    // MARK: - Public API

    /// Rescore using constrained CTC token scoring around TDT word locations.
    ///
    /// Dispatches to either word-centric (BK-tree enabled) or term-centric (default) algorithm.
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
    public func ctcTokenRescore(
        transcript: String,
        tokenTimings: [TokenTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor
    ) -> RescoreOutput {
        var candidateEvidence: CandidateEvidenceCollector?
        return evaluateTokenCandidates(
            transcript: transcript,
            tokenTimings: tokenTimings,
            logProbs: logProbs,
            frameDuration: frameDuration,
            cbw: cbw,
            marginSeconds: marginSeconds,
            minSimilarity: minSimilarity,
            candidateEvidence: &candidateEvidence
        )
    }

    /// Evaluate vocabulary candidates without returning the rewritten transcript.
    ///
    /// This runs the same candidate discovery and CTC comparison as ``ctcTokenRescore`` while
    /// preserving the supplied transcript as ``CandidateEvidenceOutput/baseText``. The output
    /// includes comparison failures, unavailable comparisons, applied candidates, and candidates
    /// superseded by the legacy overlap pass.
    ///
    /// - Parameters:
    ///   - transcript: Untouched transcript from the TDT decoder.
    ///   - tokenTimings: Token-level timings from the TDT decoder.
    ///   - logProbs: CTC log-probabilities from the spotter.
    ///   - frameDuration: Duration of each CTC frame in seconds.
    ///   - cbw: Context-biasing weight.
    ///   - marginSeconds: Temporal margin around TDT words for CTC search.
    ///   - minSimilarity: Minimum string similarity to consider a match.
    /// - Returns: Untouched base text, its exact internal word sequence, and finalized evidence for
    ///   every CTC-evaluated candidate.
    public func ctcTokenEvaluateCandidates(
        transcript: String,
        tokenTimings: [TokenTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor
    ) -> CandidateEvidenceOutput {
        var candidateEvidence: CandidateEvidenceCollector? = CandidateEvidenceCollector()
        _ = evaluateTokenCandidates(
            transcript: transcript,
            tokenTimings: tokenTimings,
            logProbs: logProbs,
            frameDuration: frameDuration,
            cbw: cbw,
            marginSeconds: marginSeconds,
            minSimilarity: minSimilarity,
            candidateEvidence: &candidateEvidence
        )
        return CandidateEvidenceOutput(
            baseText: transcript,
            baseWords: candidateEvidence?.baseWords ?? [],
            candidates: candidateEvidence?.candidates ?? []
        )
    }

    private func evaluateTokenCandidates(
        transcript: String,
        tokenTimings: [TokenTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double,
        minSimilarity: Float,
        candidateEvidence: inout CandidateEvidenceCollector?
    ) -> RescoreOutput {
        // Build word-level timings once at the entrypoint and pass into both
        // dispatch paths. Computing this once instead of twice avoids
        // duplicate work for the BK-tree branch and keeps the private
        // functions parameterized by `[WordTiming]` rather than the raw
        // `[TokenTiming]` (cleaner contract; useful if a caller ever wants
        // to supply pre-computed timings from another source).
        let wordTimings = Self.buildWordTimings(from: tokenTimings)
        if candidateEvidence != nil {
            let baseWords = wordTimings.map(\.word)
            candidateEvidence?.baseText = transcript
            candidateEvidence?.baseWords = baseWords
            candidateEvidence?.alignedWordRanges = Self.alignBaseWordsToUTF8Ranges(
                baseText: transcript,
                baseWords: baseWords
            )
        }

        if useBKTree {
            return rescoreWithConstrainedCTCWordCentric(
                transcript: transcript,
                wordTimings: wordTimings,
                logProbs: logProbs,
                frameDuration: frameDuration,
                cbw: cbw,
                marginSeconds: marginSeconds,
                minSimilarity: minSimilarity,
                candidateEvidence: &candidateEvidence
            )
        } else {
            return rescoreWithConstrainedCTCTermCentric(
                transcript: transcript,
                wordTimings: wordTimings,
                logProbs: logProbs,
                frameDuration: frameDuration,
                cbw: cbw,
                marginSeconds: marginSeconds,
                minSimilarity: minSimilarity,
                candidateEvidence: &candidateEvidence
            )
        }
    }

    // MARK: - Word-Centric Algorithm (Experimental)

    /// Word-centric constrained CTC rescoring (BK-tree enabled).
    ///
    /// Algorithm:
    /// 1. For each TDT word, query BK-tree to find candidate vocabulary terms (O(log V) per word)
    /// 2. For each candidate, run constrained CTC DP within the TDT word's timestamp window
    /// 3. Compare constrained CTC score with original word's CTC score to decide replacement
    ///
    /// Best used with BK-tree enabled for O(W x log V) performance.
    private func rescoreWithConstrainedCTCWordCentric(
        transcript: String,
        wordTimings: [WordTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor,
        candidateEvidence: inout CandidateEvidenceCollector?
    ) -> RescoreOutput {
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
        var pendingReplacements: [PendingReplacement] = []

        // Build normalized vocabulary set for guard checks
        let vocabularyNormalizedSet = buildVocabularyNormalizedSet()

        // Lowest per-term similarity across the vocabulary. The BK-tree search
        // bound is derived from this floor so that terms with a lower per-term
        // `minSimilarity` are not pruned before per-candidate filtering applies
        // each term's own threshold.
        let searchFloor = vocabulary.terms.reduce(minSimilarity) { min($0, $1.minSimilarity ?? minSimilarity) }

        // Pre-compute normalized words for all timings
        let normalizedWords = wordTimings.map { Self.normalizeForSimilarity($0.word) }

        // WORD-CENTRIC LOOP: For each TDT word, find candidate vocabulary terms
        for (wordIdx, timing) in wordTimings.enumerated() {
            guard !replacedIndices.contains(wordIdx) else { continue }

            let tdtWord = timing.word
            let normalizedWord = normalizedWords[wordIdx]
            guard !normalizedWord.isEmpty else { continue }

            // Build adjacent normalized words for compound detection
            var adjacentNormalized: [String] = []
            for offset in 1...3 {
                let idx = wordIdx + offset
                if idx < wordTimings.count && !replacedIndices.contains(idx) {
                    let norm = normalizedWords[idx]
                    if !norm.isEmpty {
                        adjacentNormalized.append(norm)
                    } else {
                        break
                    }
                } else {
                    break
                }
            }

            // Find candidate vocabulary terms using BK-tree or linear scan.
            // `searchFloor` widens the BK-tree edit-distance bound to cover the
            // most permissive per-term override; each candidate is then filtered
            // against its own threshold inside the helper.
            let candidates = findCandidateTermsForWord(
                normalizedWord: normalizedWord,
                adjacentNormalized: adjacentNormalized,
                minSimilarity: minSimilarity,
                searchFloor: searchFloor
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

                // Apply similarity threshold adjustments. Per-term override
                // falls back to the vocabulary-level threshold; guards below
                // still clamp it upward for short/stopword spans.
                let termMinSimilarity = term.minSimilarity ?? minSimilarity
                var minSimilarityForSpan = requiredSimilarity(
                    minSimilarity: termMinSimilarity,
                    spanLength: spanLength
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
                let matchCandidate = CTCMatchCandidate(
                    origin: .wordCentric,
                    originalPhrase: originalPhrase,
                    vocabTerm: vocabTerm,
                    matchedAlias: nil,
                    vocabTokens: vocabTokens,
                    similarity: similarity,
                    spanLength: spanLength,
                    spanIndices: spanIndices,
                    tokenRange: Self.tokenRange(for: spanIndices, in: wordTimings),
                    spanStartTime: spanStartTime,
                    spanEndTime: spanEndTime
                )

                let result = evaluateCTCMatch(
                    candidate: matchCandidate,
                    logProbs: logProbs,
                    frameDuration: frameDuration,
                    cbw: cbw,
                    marginSeconds: marginSeconds
                )
                let candidateID = Self.recordCandidateEvidence(
                    candidate: matchCandidate,
                    result: result,
                    candidateEvidence: &candidateEvidence
                )

                if result.shouldReplace {
                    pendingReplacements.append(
                        PendingReplacement(
                            candidateID: candidateID,
                            candidate: matchCandidate,
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
            replacements: &replacements,
            candidateEvidence: &candidateEvidence
        )
    }

    // MARK: - Term-Centric Algorithm (Default)

    /// Term-centric constrained CTC rescoring.
    ///
    /// Algorithm:
    /// 1. For each vocabulary term, find TDT words phonetically similar (string similarity)
    /// 2. For each match, run constrained CTC DP within the TDT word's timestamp window
    /// 3. Compare constrained CTC score with original word's CTC score to decide replacement
    ///
    /// This approach processes vocabulary in file order and produces better benchmark results.
    private func rescoreWithConstrainedCTCTermCentric(
        transcript: String,
        wordTimings: [WordTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float = ContextBiasingConstants.defaultCbw,
        marginSeconds: Double = ContextBiasingConstants.defaultMarginSeconds,
        minSimilarity: Float = ContextBiasingConstants.minSimilarityFloor,
        candidateEvidence: inout CandidateEvidenceCollector?
    ) -> RescoreOutput {
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

            // Per-term similarity override (falls back to the vocabulary-level
            // threshold). Safety guards in requiredSimilarity/checkStopwordRules/
            // checkLengthRatioRules still clamp this upward for short/stopword spans.
            let termMinSimilarity = term.minSimilarity ?? minSimilarity

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
                    guard spanLength <= wordTimings.count else { break }
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
                        var matchedAlias: String?
                        for form in multiWordForms {
                            let similarity = Self.stringSimilarity(normalizedPhrase, form.normalized)
                            if similarity > bestSimilarity {
                                bestSimilarity = similarity
                                matchedAlias = form.matchedAlias
                            }
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
                            minSimilarity: termMinSimilarity,
                            spanLength: spanLength
                        )
                        if bestSimilarity < minSimilarityForSpan { continue }

                        // Get temporal window for the entire span
                        guard let firstIdx = spanIndices.first, let lastIdx = spanIndices.last else { continue }
                        let spanStartTime = wordTimings[firstIdx].startTime
                        let spanEndTime = wordTimings[lastIdx].endTime

                        // Evaluate CTC match using shared helper
                        let matchCandidate = CTCMatchCandidate(
                            origin: .termCentricMultiWord,
                            originalPhrase: tdtPhrase,
                            vocabTerm: vocabTerm,
                            matchedAlias: matchedAlias,
                            vocabTokens: vocabTokens,
                            similarity: bestSimilarity,
                            spanLength: spanLength,
                            spanIndices: spanIndices,
                            tokenRange: Self.tokenRange(for: spanIndices, in: wordTimings),
                            spanStartTime: spanStartTime,
                            spanEndTime: spanEndTime
                        )

                        let result = evaluateCTCMatch(
                            candidate: matchCandidate,
                            logProbs: logProbs,
                            frameDuration: frameDuration,
                            cbw: cbw,
                            marginSeconds: marginSeconds
                        )
                        let candidateID = Self.recordCandidateEvidence(
                            candidate: matchCandidate,
                            result: result,
                            candidateEvidence: &candidateEvidence
                        )

                        if result.shouldReplace {
                            // Collect candidate instead of applying immediately
                            pendingReplacements.append(
                                PendingReplacement(
                                    candidateID: candidateID,
                                    candidate: matchCandidate,
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
                    var matchedAlias: String?
                    for form in singleWordForms {
                        let similarity = Self.stringSimilarity(normalizedWord, form.normalized)
                        if similarity > bestSimilarity {
                            bestSimilarity = similarity
                            matchedAlias = form.matchedAlias
                        }
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
                                    matchedAlias = form.matchedAlias
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
                                    matchedAlias = form.matchedAlias
                                }
                            }
                        }
                    }

                    // Use adaptive similarity threshold
                    var minSimilarityForSpan = requiredSimilarity(
                        minSimilarity: termMinSimilarity,
                        spanLength: matchedSpanLength
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

                    if bestSimilarity < minSimilarityForSpan { continue }

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
                    let matchCandidate = CTCMatchCandidate(
                        origin: .termCentricSingleWord,
                        originalPhrase: originalPhrase,
                        vocabTerm: vocabTerm,
                        matchedAlias: matchedAlias,
                        vocabTokens: vocabTokens,
                        similarity: bestSimilarity,
                        spanLength: matchedSpanLength,
                        spanIndices: spanIndices,
                        tokenRange: Self.tokenRange(for: spanIndices, in: wordTimings),
                        spanStartTime: spanStartTime,
                        spanEndTime: spanEndTime
                    )

                    let result = evaluateCTCMatch(
                        candidate: matchCandidate,
                        logProbs: logProbs,
                        frameDuration: frameDuration,
                        cbw: cbw,
                        marginSeconds: marginSeconds
                    )
                    let candidateID = Self.recordCandidateEvidence(
                        candidate: matchCandidate,
                        result: result,
                        candidateEvidence: &candidateEvidence
                    )

                    if result.shouldReplace {
                        // Collect candidate instead of applying immediately
                        pendingReplacements.append(
                            PendingReplacement(
                                candidateID: candidateID,
                                candidate: matchCandidate,
                                result: result,
                                similarity: bestSimilarity
                            )
                        )
                    }
                }
            }
        }

        // SPOTTER-ANCHORED RESCUE PASS:
        //
        // The string-similarity gate above only fires when TDT's
        // hypothesis is near (in edit-distance) some vocabulary term.
        // For rare brand names that TDT mangles beyond ~0.55 similarity
        // (e.g. `Quhuo` → `chuhua`, `HepsiJet` → `the jet`,
        // `DiaSorin` → `the solar`), the CTC keyword spotter still
        // detects the keyword acoustically. This pass collects those
        // detections and proposes them as replacements.
        //
        // **Size-gated**: only runs when vocabulary is small enough that
        // the spotter's keyword-vs-keyword competition is reliable. On
        // extra-large vocabularies (e.g. 670 drug names with ~600
        // distractors), the spotter has too many phonetically-similar
        // terms producing false-positive rescues like
        // `and` → `Evenity`. The TDT-anchored path above already covers
        // those cases via Levenshtein matching against the larger
        // distractor pool.
        // Opt-out (#724): `spotterRescueEnabled = false` skips the acoustic
        // rescue entirely (pre-#634 / 0.14.5 behavior) for short-vocab KWS
        // where it over-fires more than it recovers.
        if config.spotterRescueEnabled,
            vocabulary.terms.count <= ContextBiasingConstants.largeVocabThreshold
        {
            collectSpotterAnchoredCandidates(
                wordTimings: wordTimings,
                logProbs: logProbs,
                frameDuration: frameDuration,
                cbw: cbw,
                marginSeconds: marginSeconds,
                vocabularyNormalizedSet: vocabularyNormalizedSet,
                pendingReplacements: &pendingReplacements,
                candidateEvidence: &candidateEvidence
            )
        }

        // PASS 2 & 3: Sort, apply, and reconstruct (shared logic)
        return finalizeReplacements(
            pendingReplacements: pendingReplacements,
            modifiedWords: &modifiedWords,
            replacedIndices: &replacedIndices,
            replacements: &replacements,
            candidateEvidence: &candidateEvidence
        )
    }

    // MARK: - Spotter-Anchored Rescue

    /// Minimum CTC keyword-spotter score to consider a detection for the
    /// rescue path. Looser than the production spotter floor (-15) but
    /// strict enough that the CTC-vs-CTC comparison can still reject
    /// borderline acoustic matches.
    private static let spotterRescueMinScore: Float = -10.0

    /// Collect candidate replacements grounded in CTC-keyword-spotter
    /// detections rather than in TDT string similarity.
    ///
    /// For each detection, locate the TDT word indices whose timestamp
    /// window overlaps the detection. Build a `CTCMatchCandidate`
    /// (similarity is computed for ranking only — the gate is the
    /// CTC-vs-CTC comparison plus cbw boost performed by
    /// `evaluateCTCMatch`).
    private func collectSpotterAnchoredCandidates(
        wordTimings: [WordTiming],
        logProbs: [[Float]],
        frameDuration: Double,
        cbw: Float,
        marginSeconds: Double,
        vocabularyNormalizedSet: Set<String>,
        pendingReplacements: inout [PendingReplacement],
        candidateEvidence: inout CandidateEvidenceCollector?
    ) {
        let result = spotter.spotKeywordsFromLogProbs(
            logProbs: logProbs,
            frameDuration: frameDuration,
            customVocabulary: vocabulary,
            minScore: Self.spotterRescueMinScore
        )
        guard !result.detections.isEmpty else { return }

        // Sort detections by score descending — the highest-confidence
        // acoustic detection for each (term, time-window) pair gets to
        // race for the span first.
        let detections = result.detections.sorted { $0.score > $1.score }
        var seenSpans = Set<String>()  // dedupe by termLower:firstIdx:spanLength

        for detection in detections {
            let term = detection.term
            let vocabTerm = term.text
            guard vocabTerm.count >= vocabulary.minTermLength else { continue }

            // Map detection time window → TDT word indices.
            let span = wordIndices(
                in: wordTimings,
                overlapping: detection.startTime,
                end: detection.endTime
            )
            guard !span.isEmpty else { continue }

            // Bound to a sensible width so we don't replace whole
            // sentences with a single keyword on a misaligned spotter
            // hit. 4 mirrors the multi-word ceiling in the term-centric
            // loop above.
            guard span.count <= 4 else { continue }

            let dedupeKey = "\(term.textLowercased):\(span.first!):\(span.count)"
            guard seenSpans.insert(dedupeKey).inserted else { continue }

            let spanWords = span.map { wordTimings[$0].word }
            let originalPhrase = spanWords.joined(separator: " ")
            let normalizedPhrase = Self.normalizeForSimilarity(originalPhrase)

            // If the TDT phrase already matches some other vocabulary
            // term verbatim, leave it alone — that's the same guard the
            // primary path applies.
            let normalizedForms = buildNormalizedForms(for: term)
            let normalizedCurrentSet = Set(normalizedForms.map { $0.normalized })
            if vocabularyNormalizedSet.contains(normalizedPhrase),
                !normalizedCurrentSet.contains(normalizedPhrase)
            {
                continue
            }

            // If the TDT phrase already equals the canonical (or an
            // alias), there's nothing to replace.
            let normalizedCanonical = Self.normalizeForSimilarity(vocabTerm)
            if normalizedPhrase == normalizedCanonical { continue }
            if normalizedCurrentSet.contains(normalizedPhrase) { continue }

            guard let vocabTokens = term.ctcTokenIds ?? term.tokenIds, !vocabTokens.isEmpty else {
                continue
            }

            // STOPWORD/CONTENT GUARD:
            // Reject rescues whose entire span is short common words.
            // Without this, drug-list FPs like `and` → `Evenity`,
            // `the` → `ELEVIDYS`, `by` → `Stelara` slip through because
            // the spotter score, evaluated in its own (acoustic) window,
            // beats the original stopword's CTC score in that same
            // window — yet the spotter window is several frames wider
            // than the lone TDT word it lands on.
            let normalizedSpanWords =
                span.map { Self.normalizeForSimilarity(wordTimings[$0].word) }
            if span.count == 1 && Self.stopwords.contains(normalizedSpanWords[0]) {
                continue
            }
            if span.count >= 2 {
                let allStopwords = normalizedSpanWords.allSatisfy { Self.stopwords.contains($0) }
                if allStopwords { continue }
            }

            // Compute similarity (best over canonical + aliases).
            var bestSimilarity: Float = 0
            var matchedAlias: String?
            for form in normalizedForms {
                let similarity = Self.stringSimilarity(normalizedPhrase, form.normalized)
                if similarity > bestSimilarity {
                    bestSimilarity = similarity
                    matchedAlias = form.matchedAlias
                }
            }

            // SIMILARITY FLOOR for the acoustic rescue path. The spotter
            // rescue is acoustic-evidence driven and otherwise ignores
            // similarity, which lets it force-replace low-similarity spans.
            // The effective floor is the stricter of:
            //   - the term's EXPLICIT per-term override (#647), and
            //   - the opt-in span-aware config floor (#702; higher for
            //     multi-word spans, which are the most error-prone).
            // Both sources default to disabled, so by default this is a no-op
            // and similarity stays "for ranking only".
            let configSpotterFloor =
                span.count >= 2
                ? config.spotterRescueMultiWordMinSimilarity
                : config.spotterRescueMinSimilarity
            let spotterSimFloor = max(term.minSimilarity ?? 0, configSpotterFloor)
            if bestSimilarity < spotterSimFloor {
                debugLog(
                    "  [SPOTTER-RESCUE] Skipping '\(vocabTerm)' over '\(originalPhrase)': "
                        + "similarity \(String(format: "%.2f", bestSimilarity)) < floor "
                        + "\(String(format: "%.2f", spotterSimFloor)) (span=\(span.count))")
                continue
            }

            let firstIdx = span.first!
            let lastIdx = span.last!
            let candidate = CTCMatchCandidate(
                origin: .spotterRescue,
                originalPhrase: originalPhrase,
                vocabTerm: vocabTerm,
                matchedAlias: matchedAlias,
                vocabTokens: vocabTokens,
                similarity: bestSimilarity,
                spanLength: span.count,
                spanIndices: span,
                tokenRange: Self.tokenRange(for: span, in: wordTimings),
                spanStartTime: wordTimings[firstIdx].startTime,
                spanEndTime: wordTimings[lastIdx].endTime
            )

            let evalResult = evaluateCTCMatch(
                candidate: candidate,
                logProbs: logProbs,
                frameDuration: frameDuration,
                cbw: cbw,
                marginSeconds: marginSeconds
            )
            let candidateID = Self.recordCandidateEvidence(
                candidate: candidate,
                result: evalResult,
                candidateEvidence: &candidateEvidence
            )
            guard evalResult.shouldReplace else { continue }

            debugLog(
                "  [SPOTTER-RESCUE] '\(originalPhrase)' → '\(vocabTerm)' "
                    + "spotter=\(String(format: "%.2f", detection.score)), "
                    + "sim=\(String(format: "%.2f", bestSimilarity))"
            )

            pendingReplacements.append(
                PendingReplacement(
                    candidateID: candidateID,
                    candidate: candidate,
                    result: evalResult,
                    similarity: bestSimilarity
                )
            )
        }
    }

    /// Find the indices of TDT words whose [startTime, endTime] window
    /// overlaps the supplied detection range. Returns at most a small
    /// contiguous run; non-contiguous overlaps are reduced to the run
    /// containing the time-center of the detection.
    private func wordIndices(
        in wordTimings: [WordTiming],
        overlapping start: Double,
        end: Double
    ) -> [Int] {
        guard !wordTimings.isEmpty, start < end else { return [] }

        var overlapping: [Int] = []
        for (idx, w) in wordTimings.enumerated() {
            if w.endTime < start { continue }
            if w.startTime > end { break }
            overlapping.append(idx)
        }
        if overlapping.isEmpty {
            // Fall back to nearest word to the detection center.
            let center = (start + end) / 2.0
            var bestIdx = 0
            var bestDelta = Double.infinity
            for (idx, w) in wordTimings.enumerated() {
                let mid = (w.startTime + w.endTime) / 2.0
                let delta = abs(mid - center)
                if delta < bestDelta {
                    bestDelta = delta
                    bestIdx = idx
                }
            }
            return [bestIdx]
        }
        // Ensure contiguity (consecutive indices).
        var contiguous: [Int] = [overlapping[0]]
        for i in 1..<overlapping.count {
            if overlapping[i] == contiguous.last! + 1 {
                contiguous.append(overlapping[i])
            } else {
                break
            }
        }
        return contiguous
    }
}
