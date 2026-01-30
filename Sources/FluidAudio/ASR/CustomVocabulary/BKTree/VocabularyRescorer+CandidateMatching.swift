import Foundation

extension VocabularyRescorer {

    /// Candidate vocabulary term match with span information
    struct CandidateMatch {
        let term: CustomVocabularyTerm
        let similarity: Float
        let spanLength: Int  // Number of TDT words matched (1 for single, 2+ for compound)
    }

    /// Find candidate vocabulary terms for a TDT word, including compound word detection.
    ///
    /// This method queries the BK-tree (or performs linear scan) for:
    /// 1. Single word matches
    /// 2. Two-word compound matches (word + next word concatenated)
    /// 3. Three-word compound matches (for longer vocabulary terms)
    ///
    /// - Parameters:
    ///   - normalizedWord: The normalized TDT word
    ///   - adjacentNormalized: Array of normalized adjacent words (for compound detection)
    ///   - minSimilarity: Minimum similarity threshold
    /// - Returns: Array of candidate matches sorted by similarity (descending)
    func findCandidateTermsForWord(
        normalizedWord: String,
        adjacentNormalized: [String],
        minSimilarity: Float
    ) -> [CandidateMatch] {
        guard !normalizedWord.isEmpty else { return [] }

        var candidates: [CandidateMatch] = []

        if useBKTree, let tree = bkTree {
            // BK-tree path: O(log V) per query

            // 1. Single word query
            let maxLen1 = max(normalizedWord.count, 3)
            let maxDist1 = min(bkTreeMaxDistance, Int((1.0 - minSimilarity) * Float(maxLen1)))
            let results1 = tree.search(query: normalizedWord, maxDistance: maxDist1)

            for result in results1 {
                let similarity = Self.stringSimilarity(normalizedWord, result.normalizedText)
                if similarity >= minSimilarity {
                    candidates.append(
                        CandidateMatch(
                            term: result.term,
                            similarity: similarity,
                            spanLength: 1
                        ))
                }
            }

            // 2. Two-word compound query (e.g., "new" + "res" -> "newres" matches "Newrez")
            if !adjacentNormalized.isEmpty, let word2 = adjacentNormalized.first, !word2.isEmpty {
                let compound2 = normalizedWord + word2
                let maxLen2 = max(compound2.count, 3)
                let maxDist2 = min(bkTreeMaxDistance, Int((1.0 - minSimilarity) * Float(maxLen2)))
                let results2 = tree.search(query: compound2, maxDistance: maxDist2)

                for result in results2 {
                    // Use length-penalized similarity to prevent prefix/suffix mismatches
                    let similarity = Self.lengthPenalizedSimilarity(compound2, result.normalizedText)
                    if similarity >= minSimilarity {
                        candidates.append(
                            CandidateMatch(
                                term: result.term,
                                similarity: similarity,
                                spanLength: 2
                            ))
                    }
                }
            }

            // 3. Three-word compound query (for longer terms like "livmarli" from "liv" + "mar" + "li")
            if adjacentNormalized.count >= 2,
                let word2 = adjacentNormalized.first, !word2.isEmpty,
                let word3 = adjacentNormalized.dropFirst().first, !word3.isEmpty
            {
                let compound3 = normalizedWord + word2 + word3
                // Only search for 3-word compounds if the compound is long enough
                if compound3.count >= 6 {
                    let maxLen3 = compound3.count
                    let maxDist3 = min(bkTreeMaxDistance, Int((1.0 - minSimilarity) * Float(maxLen3)))
                    let results3 = tree.search(query: compound3, maxDistance: maxDist3)

                    for result in results3 {
                        // Use length-penalized similarity to prevent prefix/suffix mismatches
                        let similarity = Self.lengthPenalizedSimilarity(compound3, result.normalizedText)
                        if similarity >= minSimilarity {
                            candidates.append(
                                CandidateMatch(
                                    term: result.term,
                                    similarity: similarity,
                                    spanLength: 3
                                ))
                        }
                    }
                }
            }

            // 4. Multi-word phrase query (e.g., "bank of america" as space-separated phrase)
            // This handles multi-word vocabulary terms
            // Guard: only attempt if we have adjacent words (need at least 1 for a 2-word phrase)
            if !adjacentNormalized.isEmpty {
                for spanLen in 2...min(4, adjacentNormalized.count + 1) {
                    let phraseWords = [normalizedWord] + Array(adjacentNormalized.prefix(spanLen - 1))
                    let phrase = phraseWords.joined(separator: " ")
                    let maxLenPhrase = max(phrase.count, 3)
                    let maxDistPhrase = min(
                        bkTreeMaxDistance + 1, Int((1.0 - minSimilarity) * Float(maxLenPhrase)))
                    let resultsPhrase = tree.search(query: phrase, maxDistance: maxDistPhrase)

                    for result in resultsPhrase {
                        let similarity = Self.stringSimilarity(phrase, result.normalizedText)
                        if similarity >= minSimilarity {
                            candidates.append(
                                CandidateMatch(
                                    term: result.term,
                                    similarity: similarity,
                                    spanLength: spanLen
                                ))
                        }
                    }
                }
            }

        } else {
            // Linear scan fallback: O(V) per word
            for term in vocabulary.terms {
                let termNormalized = Self.normalizeForSimilarity(term.text)
                guard !termNormalized.isEmpty else { continue }

                let termWordCount = termNormalized.split(separator: " ").count

                if termWordCount == 1 {
                    // Single word term - check single word and compounds
                    let similarity1 = Self.stringSimilarity(normalizedWord, termNormalized)
                    if similarity1 >= minSimilarity {
                        candidates.append(
                            CandidateMatch(
                                term: term,
                                similarity: similarity1,
                                spanLength: 1
                            ))
                    }

                    // Check 2-word compound
                    if !adjacentNormalized.isEmpty, let word2 = adjacentNormalized.first, !word2.isEmpty {
                        let compound2 = normalizedWord + word2
                        // Use length-penalized similarity to prevent prefix/suffix mismatches
                        let similarity2 = Self.lengthPenalizedSimilarity(compound2, termNormalized)
                        if similarity2 >= minSimilarity {
                            candidates.append(
                                CandidateMatch(
                                    term: term,
                                    similarity: similarity2,
                                    spanLength: 2
                                ))
                        }
                    }

                    // Check 3-word compound
                    if adjacentNormalized.count >= 2 {
                        let word2 = adjacentNormalized[0]
                        let word3 = adjacentNormalized[1]
                        if !word2.isEmpty && !word3.isEmpty {
                            let compound3 = normalizedWord + word2 + word3
                            if compound3.count >= 6 {
                                // Use length-penalized similarity to prevent prefix/suffix mismatches
                                let similarity3 = Self.lengthPenalizedSimilarity(compound3, termNormalized)
                                if similarity3 >= minSimilarity {
                                    candidates.append(
                                        CandidateMatch(
                                            term: term,
                                            similarity: similarity3,
                                            spanLength: 3
                                        ))
                                }
                            }
                        }
                    }
                } else {
                    // Multi-word term - check phrases
                    // Guard: only attempt if we have adjacent words
                    if !adjacentNormalized.isEmpty {
                        for spanLen in 2...min(4, adjacentNormalized.count + 1) {
                            let phraseWords = [normalizedWord] + Array(adjacentNormalized.prefix(spanLen - 1))
                            let phrase = phraseWords.joined(separator: " ")
                            let similarity = Self.stringSimilarity(phrase, termNormalized)
                            if similarity >= minSimilarity {
                                candidates.append(
                                    CandidateMatch(
                                        term: term,
                                        similarity: similarity,
                                        spanLength: spanLen
                                    ))
                            }
                        }
                    }
                }
            }
        }

        // Sort by similarity (descending), then by span length (prefer longer matches)
        return candidates.sorted {
            if $0.similarity != $1.similarity {
                return $0.similarity > $1.similarity
            }
            return $0.spanLength > $1.spanLength
        }
    }
}
