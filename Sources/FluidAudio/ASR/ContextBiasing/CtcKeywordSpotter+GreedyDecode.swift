import Foundation

// MARK: - Greedy CTC Decoding (NeMo CTC-WS Algorithm)

extension CtcKeywordSpotter {

    /// A word from greedy CTC decoding with frame-level alignment.
    /// Used for comparison with vocabulary detections per the NeMo CTC-WS paper.
    public struct GreedyCtcWord: Sendable {
        /// The decoded word text
        public let text: String
        /// Start frame (inclusive)
        public let startFrame: Int
        /// End frame (exclusive)
        public let endFrame: Int
        /// Accumulated log-probability score (sum of log-probs for non-blank tokens)
        public let score: Float
        /// Token IDs that make up this word
        public let tokenIds: [Int]

        /// Normalized score (per-token average)
        public var normalizedScore: Float {
            tokenIds.isEmpty ? score : score / Float(tokenIds.count)
        }

        /// Start time in seconds
        public func startTime(frameDuration: Double) -> Double {
            Double(startFrame) * frameDuration
        }

        /// End time in seconds
        public func endTime(frameDuration: Double) -> Double {
            Double(endFrame) * frameDuration
        }
    }

    /// Result of greedy CTC decoding with word-level alignment.
    public struct GreedyCtcResult: Sendable {
        /// Decoded words with frame boundaries and scores
        public let words: [GreedyCtcWord]
        /// Full transcript text
        public let text: String
        /// Duration of each CTC frame in seconds
        public let frameDuration: Double
        /// Total number of frames
        public let totalFrames: Int
    }

    /// Perform greedy CTC decoding with word-level alignment.
    ///
    /// This implements the "greedy CTC decoding" step from the NeMo CTC-WS paper.
    /// For each frame, takes the best non-blank token and determines if blank is
    /// "dominant" (significantly higher probability). Only non-blank-dominant frames
    /// contribute to the decoded output.
    ///
    /// Word boundaries are detected using SentencePiece's "▁" prefix convention:
    /// tokens starting with "▁" begin a new word.
    ///
    /// **Note on CTC model compatibility:**
    /// Some CTC models (including parakeet-ctc-110m) are "blank-dominant" - they predict
    /// blank with very high probability for most frames. For these models, greedy decoding
    /// may produce poor results. The `blankDominanceThreshold` parameter can be tuned:
    /// - Lower threshold (e.g., 6.0): More aggressive, may include noise
    /// - Higher threshold (e.g., 15.0): More conservative, may miss tokens
    /// - Default (10.0): Balanced for most models
    ///
    /// For blank-dominant models, consider using `rescoreWithConstrainedCTC` instead,
    /// which uses dynamic programming to find optimal paths.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities [T, vocab_size]
    ///   - tokenizer: tokenizer for id-to-piece conversion
    ///   - frameDuration: Duration of each frame in seconds
    ///   - blankDominanceThreshold: Threshold for considering blank as dominant (default: 10.0)
    /// - Returns: GreedyCtcResult with words, their boundaries, and scores
    public func greedyCtcDecode(
        logProbs: [[Float]],
        tokenizer: CtcTokenizer,
        frameDuration: Double,
        blankDominanceThreshold: Float = 10.0,
        blankBiasCorrection: Float = 0.0
    ) -> GreedyCtcResult {
        let T = logProbs.count
        guard T > 0 else {
            return GreedyCtcResult(words: [], text: "", frameDuration: frameDuration, totalFrames: 0)
        }

        // Step 1: Get best non-blank token for each frame
        // CTC models often predict blank with very high probability for most frames.
        // Instead of taking argmax (which would always be blank), we take the
        // best non-blank token and check if blank is "dominant" (much higher prob).
        // The blankDominanceThreshold controls how much higher blank must be.

        var frameTokens: [(tokenId: Int, logProb: Float, blankDominant: Bool)] = []
        var nonBlankCount = 0
        for t in 0..<T {
            let frame = logProbs[t]
            let blankLogProb = frame[blankId]

            // Find best non-blank token
            var bestNonBlankId = 0
            var bestNonBlankLogProb: Float = -.infinity
            for v in 0..<blankId {
                if frame[v] > bestNonBlankLogProb {
                    bestNonBlankLogProb = frame[v]
                    bestNonBlankId = v
                }
            }

            // Check if blank is dominant (apply bias correction to blank logprob)
            let correctedBlankLogProb = blankLogProb - blankBiasCorrection
            let blankDominant = correctedBlankLogProb - bestNonBlankLogProb > blankDominanceThreshold

            frameTokens.append((tokenId: bestNonBlankId, logProb: bestNonBlankLogProb, blankDominant: blankDominant))
            if !blankDominant {
                nonBlankCount += 1
            }
        }

        if debugMode {
            logger.debug(
                "Greedy CTC: \(T) frames, \(nonBlankCount) non-blank-dominant (threshold=\(blankDominanceThreshold))"
            )
        }

        // Step 2: Collapse consecutive duplicates and blanks, tracking boundaries
        // A "collapsed token" represents a sequence of identical tokens or blanks
        struct CollapsedToken {
            let tokenId: Int
            let startFrame: Int
            let endFrame: Int  // exclusive
            let score: Float  // sum of log-probs across frames
        }

        var collapsedTokens: [CollapsedToken] = []
        var currentTokenId: Int = -1  // -1 means "blank/silent"
        var currentStartFrame: Int = 0
        var currentScore: Float = 0

        for (t, (tokenId, logProb, blankDominant)) in frameTokens.enumerated() {
            // Treat blank-dominant frames as blank (tokenId = -1)
            let effectiveTokenId = blankDominant ? -1 : tokenId

            if effectiveTokenId == currentTokenId {
                // Same token (or still blank), accumulate score
                if effectiveTokenId != -1 {
                    currentScore += logProb
                }
            } else {
                // New token, save previous if it was a real token (not blank)
                if currentTokenId != -1 {
                    collapsedTokens.append(
                        CollapsedToken(
                            tokenId: currentTokenId,
                            startFrame: currentStartFrame,
                            endFrame: t,
                            score: currentScore
                        ))
                }
                currentTokenId = effectiveTokenId
                currentStartFrame = t
                currentScore = effectiveTokenId != -1 ? logProb : 0
            }
        }
        // Don't forget the last token
        if currentTokenId != -1 {
            collapsedTokens.append(
                CollapsedToken(
                    tokenId: currentTokenId,
                    startFrame: currentStartFrame,
                    endFrame: T,
                    score: currentScore
                ))
        }

        // Step 3: Group tokens into words based on "▁" prefix
        // SentencePiece tokens starting with "▁" indicate word start
        var words: [GreedyCtcWord] = []
        var currentWordTokens: [Int] = []
        var currentWordScore: Float = 0
        var currentWordStartFrame: Int = 0
        var currentWordEndFrame: Int = 0

        for collapsed in collapsedTokens {
            let piece = tokenizer.idToPiece(collapsed.tokenId) ?? ""

            // Check if this token starts a new word (has "▁" prefix)
            let startsNewWord = piece.hasPrefix("▁") || piece.hasPrefix(" ")

            if startsNewWord && !currentWordTokens.isEmpty {
                // Finish current word
                let wordText = buildWordText(tokenIds: currentWordTokens, tokenizer: tokenizer)
                if !wordText.isEmpty {
                    words.append(
                        GreedyCtcWord(
                            text: wordText,
                            startFrame: currentWordStartFrame,
                            endFrame: currentWordEndFrame,
                            score: currentWordScore,
                            tokenIds: currentWordTokens
                        ))
                }
                // Start new word
                currentWordTokens = [collapsed.tokenId]
                currentWordScore = collapsed.score
                currentWordStartFrame = collapsed.startFrame
                currentWordEndFrame = collapsed.endFrame
            } else {
                // Continue current word or start first word
                if currentWordTokens.isEmpty {
                    currentWordStartFrame = collapsed.startFrame
                }
                currentWordTokens.append(collapsed.tokenId)
                currentWordScore += collapsed.score
                currentWordEndFrame = collapsed.endFrame
            }
        }

        // Finish last word
        if !currentWordTokens.isEmpty {
            let wordText = buildWordText(tokenIds: currentWordTokens, tokenizer: tokenizer)
            if !wordText.isEmpty {
                words.append(
                    GreedyCtcWord(
                        text: wordText,
                        startFrame: currentWordStartFrame,
                        endFrame: currentWordEndFrame,
                        score: currentWordScore,
                        tokenIds: currentWordTokens
                    ))
            }
        }

        // Build full transcript text
        let fullText = words.map { $0.text }.joined(separator: " ")

        if debugMode {
            logger.debug("=== Greedy CTC Decode ===")
            logger.debug("Frames: \(T), Words: \(words.count)")
            for word in words {
                let startTime = word.startTime(frameDuration: frameDuration)
                let endTime = word.endTime(frameDuration: frameDuration)
                logger.debug(
                    "  '\(word.text)' [\(String(format: "%.2f", startTime))-\(String(format: "%.2f", endTime))s] "
                        + "score=\(String(format: "%.2f", word.normalizedScore))"
                )
            }
            logger.debug("=========================")
        }

        return GreedyCtcResult(
            words: words,
            text: fullText,
            frameDuration: frameDuration,
            totalFrames: T
        )
    }

    /// Build word text from token IDs, handling SentencePiece formatting.
    private func buildWordText(tokenIds: [Int], tokenizer: CtcTokenizer) -> String {
        var text = ""
        for tokenId in tokenIds {
            if let piece = tokenizer.idToPiece(tokenId) {
                // Strip "▁" prefix (SentencePiece word boundary marker)
                if piece.hasPrefix("▁") {
                    text += String(piece.dropFirst())
                } else if piece.hasPrefix(" ") {
                    text += String(piece.dropFirst())
                } else {
                    text += piece
                }
            }
        }
        return text.trimmingCharacters(in: .whitespaces)
    }

    /// Convenience method: Run greedy CTC decode on audio samples.
    /// Computes log-probs and then performs greedy decoding.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono audio samples
    ///   - tokenizer: tokenizer for id-to-piece conversion
    /// - Returns: GreedyCtcResult with words, their boundaries, and scores
    public func greedyCtcDecode(
        audioSamples: [Float],
        tokenizer: CtcTokenizer
    ) async throws -> GreedyCtcResult {
        let ctcResult = try await computeLogProbs(for: audioSamples)
        return greedyCtcDecode(
            logProbs: ctcResult.logProbs,
            tokenizer: tokenizer,
            frameDuration: ctcResult.frameDuration
        )
    }

}
