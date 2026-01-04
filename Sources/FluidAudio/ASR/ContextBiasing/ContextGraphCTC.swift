import Foundation
import OSLog

/// A state in the CTC context graph (Trie node with CTC topology).
///
/// Each state represents a position in recognizing vocabulary words.
/// Transitions map CTC token IDs to successor states.
public struct ContextState: Sendable {
    /// Unique index of this state in the graph
    public let index: Int

    /// Whether this state represents the end of a vocabulary word
    public var isEndState: Bool

    /// The vocabulary word if this is an end state
    public var word: String?

    /// Maps token_id -> next_state_index
    public var transitions: [Int: Int]

    /// Create a new context state
    public init(index: Int, isEndState: Bool = false, word: String? = nil) {
        self.index = index
        self.isEndState = isEndState
        self.word = word
        self.transitions = [:]
    }
}

/// Token for beam search in the Token Passing Algorithm.
///
/// Tracks the current position in the context graph and accumulated score
/// during frame-by-frame decoding.
public struct WSToken: Sendable {
    /// Current state index in the context graph
    public let stateIndex: Int

    /// Accumulated log-probability score
    public var score: Float

    /// Frame index where this hypothesis started
    public let startFrame: Int

    /// Frame index of the last non-blank transition (for tighter end bounds)
    public let lastNonBlankFrame: Int

    /// Target word this token is tracking toward (for per-word beam pruning)
    /// Set when token reaches an end state's prefix path
    public let targetWord: String?

    public init(
        stateIndex: Int,
        score: Float,
        startFrame: Int,
        lastNonBlankFrame: Int? = nil,
        targetWord: String? = nil
    ) {
        self.stateIndex = stateIndex
        self.score = score
        self.startFrame = startFrame
        self.lastNonBlankFrame = lastNonBlankFrame ?? startFrame
        self.targetWord = targetWord
    }
}

/// A detected word hypothesis with precise timestamps.
///
/// Produced by the context graph word spotter when a vocabulary word
/// is recognized in the audio.
public struct WSHypothesis: Sendable {
    /// The detected vocabulary word
    public let word: String

    /// Accumulated CTC score (including cbw bonuses)
    public let score: Float

    /// Frame index where the word starts
    public let startFrame: Int

    /// Frame index where the word ends
    public let endFrame: Int

    /// Frame duration in seconds (for converting to timestamps)
    public let frameDuration: Double

    /// Start time in seconds
    public var startTime: TimeInterval {
        TimeInterval(startFrame) * frameDuration
    }

    /// End time in seconds
    public var endTime: TimeInterval {
        TimeInterval(endFrame) * frameDuration
    }

    public init(
        word: String,
        score: Float,
        startFrame: Int,
        endFrame: Int,
        frameDuration: Double
    ) {
        self.word = word
        self.score = score
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.frameDuration = frameDuration
    }
}

/// Context graph for CTC-based word spotting (NeMo CTC-WS algorithm).
///
/// This is a Trie (prefix tree) composed with CTC transition topology.
/// Vocabulary words share prefix paths, enabling efficient multi-word
/// spotting in a single beam search pass.
///
/// Reference: "Fast Context-Biasing for CTC and Transducer ASR models
/// with CTC-based Word Spotter" (arXiv:2406.07096)
public struct ContextGraphCTC: Sendable {
    private let logger = Logger(subsystem: "com.fluidaudio", category: "ContextGraphCTC")

    /// All states in the graph
    public private(set) var states: [ContextState]

    /// The blank token ID for CTC
    public let blankId: Int

    /// Root state index (always 0)
    public let rootIndex: Int = 0

    /// Number of vocabulary words added
    public private(set) var wordCount: Int = 0

    /// Debug mode from environment variable
    private let debugMode: Bool

    /// Create an empty context graph
    ///
    /// - Parameter blankId: The CTC blank token ID (typically vocab_size)
    public init(blankId: Int) {
        self.blankId = blankId
        self.debugMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"

        // Initialize with root state
        self.states = [ContextState(index: 0)]
    }

    /// Add a single word to the context graph.
    ///
    /// Creates a path through the Trie for the word's token sequence.
    /// If prefixes are shared with existing words, reuses those states.
    ///
    /// - Parameters:
    ///   - word: The vocabulary word text
    ///   - tokenIds: CTC token IDs for the word
    public mutating func addWord(_ word: String, tokenIds: [Int]) {
        guard !tokenIds.isEmpty else {
            if debugMode {
                logger.debug("Skipping '\(word)': empty token IDs")
            }
            return
        }

        var currentIndex = rootIndex

        for tokenId in tokenIds {
            // Check if transition already exists
            if let nextIndex = states[currentIndex].transitions[tokenId] {
                currentIndex = nextIndex
            } else {
                // Create new state
                let newIndex = states.count
                let newState = ContextState(index: newIndex)
                states.append(newState)

                // Add transition from current state
                states[currentIndex].transitions[tokenId] = newIndex
                currentIndex = newIndex
            }
        }

        // Mark final state as end state with the word
        states[currentIndex].isEndState = true
        states[currentIndex].word = word
        wordCount += 1

        if debugMode {
            let tokenCount = tokenIds.count
            let finalState = currentIndex
            logger.debug("Added '\(word)' with \(tokenCount) tokens, path ends at state \(finalState)")
        }
    }

    /// Build context graph from a CustomVocabularyContext.
    ///
    /// Tokenizes each vocabulary term using the CTC tokenizer and adds
    /// to the graph. Terms with pre-computed ctcTokenIds use those directly.
    ///
    /// - Parameters:
    ///   - vocabulary: The vocabulary context containing terms
    ///   - tokenizer: Optional SentencePiece tokenizer for terms without ctcTokenIds
    public mutating func addVocabulary(
        _ vocabulary: CustomVocabularyContext,
        tokenizer: SentencePieceCtcTokenizer? = nil
    ) {
        for term in vocabulary.terms {
            // Use pre-computed CTC token IDs if available
            if let ctcTokenIds = term.ctcTokenIds, !ctcTokenIds.isEmpty {
                addWord(term.text, tokenIds: ctcTokenIds)
                continue
            }

            // Fall back to tokenizing the text
            if let tokenizer = tokenizer {
                let tokenIds = tokenizer.encode(term.text)
                if !tokenIds.isEmpty {
                    addWord(term.text, tokenIds: tokenIds)
                }
            } else if debugMode {
                logger.debug("Skipping '\(term.text)': no ctcTokenIds and no tokenizer")
            }
        }

        if debugMode {
            let words = wordCount
            let stateCount = states.count
            logger.info("Built context graph: \(words) words, \(stateCount) states")
        }
    }

    /// Get the state at a given index
    public func state(at index: Int) -> ContextState? {
        guard index >= 0 && index < states.count else { return nil }
        return states[index]
    }

    /// Check if a transition exists from a state
    public func transition(from stateIndex: Int, with tokenId: Int) -> Int? {
        guard stateIndex >= 0 && stateIndex < states.count else { return nil }
        return states[stateIndex].transitions[tokenId]
    }

    /// Get all possible transitions from a state
    public func transitions(from stateIndex: Int) -> [Int: Int] {
        guard stateIndex >= 0 && stateIndex < states.count else { return [:] }
        return states[stateIndex].transitions
    }

    /// Check if a state is an end state (vocabulary word boundary)
    public func isEndState(_ stateIndex: Int) -> Bool {
        guard stateIndex >= 0 && stateIndex < states.count else { return false }
        return states[stateIndex].isEndState
    }

    /// Get the word at an end state
    public func word(at stateIndex: Int) -> String? {
        guard stateIndex >= 0 && stateIndex < states.count else { return nil }
        return states[stateIndex].word
    }

    /// Print graph statistics for debugging
    public func printStats() {
        var totalTransitions = 0
        var maxTransitions = 0
        var endStateCount = 0

        for state in states {
            totalTransitions += state.transitions.count
            maxTransitions = max(maxTransitions, state.transitions.count)
            if state.isEndState {
                endStateCount += 1
            }
        }

        print("=== Context Graph Stats ===")
        print("  Words: \(wordCount)")
        print("  States: \(states.count)")
        print("  End states: \(endStateCount)")
        print("  Total transitions: \(totalTransitions)")
        print("  Max transitions per state: \(maxTransitions)")
        print("  Blank ID: \(blankId)")
        print("===========================")
    }

    /// Compute the target word for each state (the word this state leads to).
    /// For a prefix tree, each state leads to exactly one word.
    /// Returns a mapping from state index to target word.
    public func computeTargetWords() -> [Int: String] {
        var stateToWord: [Int: String] = [:]

        // DFS from each end state back to root to mark all states on the path
        func findTargetWord(stateIndex: Int, visited: inout Set<Int>) -> String? {
            if visited.contains(stateIndex) { return stateToWord[stateIndex] }
            visited.insert(stateIndex)

            let state = states[stateIndex]
            if state.isEndState, let word = state.word {
                stateToWord[stateIndex] = word
                return word
            }

            // Follow transitions to find the end state
            for (_, nextIndex) in state.transitions {
                if let word = findTargetWord(stateIndex: nextIndex, visited: &visited) {
                    stateToWord[stateIndex] = word
                    return word
                }
            }

            return nil
        }

        // Process all states
        var visited = Set<Int>()
        for i in 0..<states.count {
            _ = findTargetWord(stateIndex: i, visited: &visited)
        }

        return stateToWord
    }
}

// MARK: - Beam Search Word Spotter

extension ContextGraphCTC {
    /// Configuration for the beam search word spotter
    public struct SpotterConfig: Sendable {
        /// Context-biasing weight added to non-blank transitions (default: 3.0 per NeMo paper)
        public let cbw: Float

        /// Beam pruning threshold - tokens with score < best - threshold are pruned
        public let beamThreshold: Float

        /// Blank probability threshold for frame skipping optimization
        public let blankThreshold: Float

        /// Minimum score threshold for detected words
        public let minScore: Float

        public static let `default` = SpotterConfig(
            cbw: 3.0,
            beamThreshold: 30.0,  // Relaxed to allow more vocabulary tokens to survive
            blankThreshold: 0.8,
            minScore: -50.0  // Very relaxed to capture more detections
        )

        public init(
            cbw: Float = 3.0,
            beamThreshold: Float = 7.0,
            blankThreshold: Float = 0.8,
            minScore: Float = -15.0
        ) {
            self.cbw = cbw
            self.beamThreshold = beamThreshold
            self.blankThreshold = blankThreshold
            self.minScore = minScore
        }
    }

    /// Spot vocabulary words using beam search over the context graph.
    ///
    /// This implements the NeMo CTC-WS algorithm: frame-by-frame token passing
    /// with precise start/end frame tracking for each detected word.
    ///
    /// Uses per-word beam pruning: each vocabulary word has its own beam threshold,
    /// so tokens for different words don't compete with each other. This prevents
    /// high-scoring paths for one word from pruning tokens for other words.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities [T, vocab_size]
    ///   - frameDuration: Duration of each frame in seconds
    ///   - config: Spotter configuration
    /// - Returns: Array of detected word hypotheses with precise timestamps
    public func spotWords(
        logProbs: [[Float]],
        frameDuration: Double,
        config: SpotterConfig = .default
    ) -> [WSHypothesis] {
        let T = logProbs.count
        guard T > 0, !states.isEmpty else { return [] }

        // Precompute which target word each state leads to
        let stateToTargetWord = computeTargetWords()

        var activeTokens: [WSToken] = []
        var spottedHyps: [WSHypothesis] = []

        // Track best score per (startFrame, word) pair to keep only the best detection
        // from each starting position. We still allow multiple detections from different
        // start frames, and resolveOverlaps will pick the best overall match.
        var bestScorePerPair: [String: Float] = [:]

        let debugMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_DEBUG_CTC_BOOSTING"] == "1"

        for t in 0..<T {
            let frame = logProbs[t]
            guard frame.count > blankId else { continue }

            // Add new token at root (enables detection at any position)
            activeTokens.append(WSToken(stateIndex: rootIndex, score: 0, startFrame: t, targetWord: nil))

            // Per-word beam tracking: best score for each vocabulary word
            var bestScorePerWord: [String: Float] = [:]
            var stateToToken: [Int: WSToken] = [:]  // For state pruning

            for token in activeTokens {
                let state = states[token.stateIndex]

                // Try all transitions from current state
                for (tokenId, nextStateIndex) in state.transitions {
                    guard tokenId < frame.count else { continue }

                    let transitionScore = frame[tokenId]

                    // Apply cbw bonus for non-blank transitions
                    let bonus: Float = (tokenId == blankId) ? 0 : config.cbw
                    let newScore = token.score + transitionScore + bonus

                    // Determine target word for this token path
                    let targetWord = token.targetWord ?? stateToTargetWord[nextStateIndex]

                    // Per-word beam pruning: only compare against same word's best score
                    if let target = targetWord {
                        let wordBest = bestScorePerWord[target] ?? -.infinity
                        if newScore < wordBest - config.beamThreshold {
                            continue
                        }
                        bestScorePerWord[target] = max(wordBest, newScore)
                    }
                    // Tokens without target word (at root) are not pruned

                    // Track last non-blank frame for tighter end bounds
                    let isBlankTransition = (tokenId == blankId)
                    let newLastNonBlankFrame = isBlankTransition ? token.lastNonBlankFrame : t

                    let newToken = WSToken(
                        stateIndex: nextStateIndex,
                        score: newScore,
                        startFrame: token.startFrame,
                        lastNonBlankFrame: newLastNonBlankFrame,
                        targetWord: targetWord
                    )

                    // Check for word completion (end state reached)
                    let nextState = states[nextStateIndex]
                    if nextState.isEndState, let word = nextState.word {
                        // Use lastNonBlankFrame for tighter end bounds (current frame t for final token)
                        let endFrame = t

                        // Normalize score by number of tokens (approximate)
                        let frameSpan = endFrame - token.startFrame + 1
                        let normalizedScore = newScore / Float(max(1, frameSpan / 10))

                        if normalizedScore >= config.minScore {
                            // Track best score per (startFrame, word) pair
                            let pairKey = "\(token.startFrame):\(word)"
                            let existingScore = bestScorePerPair[pairKey] ?? -.infinity

                            // Only add detection if this is better than existing for same pair
                            if newScore > existingScore {
                                bestScorePerPair[pairKey] = newScore
                                spottedHyps.append(
                                    WSHypothesis(
                                        word: word,
                                        score: newScore,
                                        startFrame: token.startFrame,
                                        endFrame: endFrame,
                                        frameDuration: frameDuration
                                    ))

                                if debugMode {
                                    print(
                                        "  [SPOT] '\(word)' at frames [\(token.startFrame)-\(endFrame)] score=\(String(format: "%.2f", newScore))"
                                    )
                                }
                            }
                        }
                        // Don't add end-state tokens to stateToToken - they've completed
                        continue
                    }

                    // State pruning: keep only best token per state (for non-end states)
                    if let existing = stateToToken[nextStateIndex] {
                        if newScore > existing.score {
                            stateToToken[nextStateIndex] = newToken
                        }
                    } else {
                        stateToToken[nextStateIndex] = newToken
                    }
                }

                // Also allow staying in current state (blank transition)
                // This enables the CTC blank-skipping behavior
                // IMPORTANT: Only allow blank staying for:
                //   - Non-root tokens (root tokens shouldn't accumulate via blanks)
                //   - Non-end-state tokens (end state tokens have already detected, should terminate)
                if token.stateIndex != rootIndex && !state.isEndState && state.transitions[blankId] == nil {
                    let blankScore = token.score + frame[blankId]
                    let targetWord = token.targetWord ?? stateToTargetWord[token.stateIndex]

                    // Per-word beam pruning for blank transitions
                    var shouldKeep = true
                    if let target = targetWord {
                        let wordBest = bestScorePerWord[target] ?? -.infinity
                        if blankScore < wordBest - config.beamThreshold {
                            shouldKeep = false
                        } else {
                            bestScorePerWord[target] = max(wordBest, blankScore)
                        }
                    }

                    if shouldKeep {
                        // Preserve lastNonBlankFrame for blank staying (blank doesn't update it)
                        let stayToken = WSToken(
                            stateIndex: token.stateIndex,
                            score: blankScore,
                            startFrame: token.startFrame,
                            lastNonBlankFrame: token.lastNonBlankFrame,
                            targetWord: targetWord
                        )
                        if let existing = stateToToken[token.stateIndex] {
                            if blankScore > existing.score {
                                stateToToken[token.stateIndex] = stayToken
                            }
                        } else {
                            stateToToken[token.stateIndex] = stayToken
                        }
                    }
                }
            }

            // Collect surviving tokens with per-word beam pruning
            activeTokens = stateToToken.values.filter { token in
                guard let target = token.targetWord else { return true }  // Keep root tokens
                let wordBest = bestScorePerWord[target] ?? -.infinity
                return token.score >= wordBest - config.beamThreshold
            }
        }

        // Post-processing: resolve overlapping detections
        return resolveOverlaps(spottedHyps)
    }

    /// Resolve overlapping detections by keeping the best one per word.
    ///
    /// For each vocabulary word, we keep the highest-scoring detection across
    /// all start/end frame combinations. This ensures we find the globally
    /// best match rather than just the first match.
    private func resolveOverlaps(_ hypotheses: [WSHypothesis]) -> [WSHypothesis] {
        guard !hypotheses.isEmpty else { return hypotheses }

        // Group by word and keep the best detection for each word
        var bestPerWord: [String: WSHypothesis] = [:]

        for hyp in hypotheses {
            if let existing = bestPerWord[hyp.word] {
                // Keep the higher-scoring detection
                if hyp.score > existing.score {
                    bestPerWord[hyp.word] = hyp
                }
            } else {
                bestPerWord[hyp.word] = hyp
            }
        }

        return Array(bestPerWord.values)
    }
}

// MARK: - Factory Methods

extension ContextGraphCTC {
    /// Create a context graph from a vocabulary context.
    ///
    /// - Parameters:
    ///   - vocabulary: The vocabulary context containing terms
    ///   - blankId: The CTC blank token ID
    ///   - tokenizer: Optional tokenizer for terms without ctcTokenIds
    /// - Returns: A context graph populated with the vocabulary terms
    public static func build(
        from vocabulary: CustomVocabularyContext,
        blankId: Int,
        tokenizer: SentencePieceCtcTokenizer? = nil
    ) -> ContextGraphCTC {
        var graph = ContextGraphCTC(blankId: blankId)
        graph.addVocabulary(vocabulary, tokenizer: tokenizer)
        return graph
    }
}
