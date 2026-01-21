import Foundation

/// Centralized constants for the ContextBiasing module.
///
/// This file consolidates magic numbers and thresholds used across vocabulary boosting,
/// CTC keyword spotting, and rescoring components.
///
/// ## Similarity Threshold Hierarchy
/// The similarity thresholds form a hierarchy from lenient to strict:
/// ```
/// 0.50 (floor) < 0.52 (default) < 0.55 (single-word) < 0.65 (alias) < 0.75 (length-ratio) < 0.80 (multi-word/short) < 0.85 (stopword)
/// ```
public enum ContextBiasingConstants {

    // MARK: - Token IDs

    /// Sentinel value representing blank/silent token in CTC decoding.
    ///
    /// In CTC (Connectionist Temporal Classification), frames can output either a vocabulary
    /// token or a "blank" indicating no speech. This sentinel value (-1) marks frames where
    /// the blank token dominates, used during greedy decode collapse.
    ///
    /// - Value: `-1` (invalid token ID, distinct from all valid vocabulary indices)
    /// - Used in: `CtcKeywordSpotter+GreedyDecode.swift` for frame collapsing
    public static let blankSilentTokenId: Int = -1

    /// Wildcard token ID for pattern matching in CTC dynamic programming paths.
    ///
    /// Represents "*" in keyword patterns that matches any token at zero cost.
    /// Used in the DP alignment algorithm ported from NeMo's `ctc_word_spotter.py`.
    ///
    /// - Value: `-1` (matches any token during path scoring)
    /// - Used in: `CtcKeywordSpotter.swift` for flexible keyword matching
    public static let wildcardTokenId: Int = -1

    /// Default blank token ID for CTC models.
    ///
    /// In CTC models, the blank token is typically the last token in the vocabulary.
    /// For parakeet-ctc-110m, the vocabulary has 1024 tokens (indices 0-1023),
    /// so the blank token is at index 1024.
    ///
    /// - Value: `1024` (vocab_size for parakeet-ctc-110m)
    /// - Used in: `CtcKeywordSpotter.init()` as default parameter
    public static let defaultBlankId: Int = 1024

    // MARK: - CTC Score Thresholds

    /// Default minimum CTC score for keyword spotting detections.
    ///
    /// Keywords with CTC scores below this threshold are filtered out as low-confidence.
    /// CTC scores are log-probabilities (negative values), so -15.0 represents very low
    /// probability. This lenient default allows rescoring to make final decisions.
    ///
    /// - Value: `-15.0` (log-probability, ~3e-7 probability)
    /// - Range: Typically -20.0 (very lenient) to -5.0 (strict)
    /// - Used in: `CtcKeywordSpotter.spotKeywords()` for initial filtering
    public static let defaultMinSpotterScore: Float = -15.0

    /// Default minimum CTC score for vocabulary context matching.
    ///
    /// Slightly stricter than spotter score since this is used after initial detection.
    /// Balances catching valid vocabulary terms vs. reducing false positives.
    ///
    /// - Value: `-12.0` (log-probability, ~6e-6 probability)
    /// - Used in: `CustomVocabularyContext.init()` as default
    public static let defaultMinVocabCtcScore: Float = -12.0

    /// Threshold for blank dominance in greedy CTC decode.
    ///
    /// During greedy decoding, if (blank_logprob - best_nonblank_logprob) exceeds this
    /// threshold, the frame is treated as silence. Higher values are more conservative
    /// (may miss quiet speech), lower values are more aggressive (may include noise).
    ///
    /// - Value: `10.0` (log-probability difference)
    /// - Tuning: 6.0 (aggressive) → 10.0 (balanced) → 15.0 (conservative)
    /// - Used in: `CtcKeywordSpotter+GreedyDecode.greedyCtcDecode()`
    public static let defaultBlankDominanceThreshold: Float = 10.0

    /// CTC temperature for softmax probability distribution.
    ///
    /// Controls the "sharpness" of the probability distribution:
    /// - Higher values (>1.0): Softer, more uniform probabilities
    /// - Lower values (<1.0): Sharper, more peaked at top prediction
    /// - Value 1.0: Standard softmax
    ///
    /// - Value: `1.0` (standard softmax)
    /// - Used in: `CtcKeywordSpotter.swift` for log-prob computation
    public static let ctcTemperature: Float = 1.0

    /// Blank bias correction applied to CTC log-probabilities.
    ///
    /// Positive values penalize the blank token, making non-blank tokens
    /// more likely. Useful for models that over-predict blank.
    ///
    /// - Value: `0.0` (no bias)
    /// - Used in: `CtcKeywordSpotter.swift` for log-prob computation
    public static let blankBias: Float = 0.0

    // MARK: - Similarity Thresholds

    /// Absolute minimum similarity floor for any vocabulary matching.
    ///
    /// No replacement is considered if string similarity falls below this floor.
    /// Uses Levenshtein-based similarity: 1 - (editDistance / maxLength).
    ///
    /// - Value: `0.50` (50% character overlap required)
    /// - Example: "nvidia" vs "nvida" = 0.83 ✓, "nvidia" vs "intel" = 0.17 ✗
    /// - Used in: Debug logging, BK-tree candidate filtering
    public static let minSimilarityFloor: Float = 0.50

    /// Default minimum similarity for vocabulary term matching.
    ///
    /// Slightly above floor to reduce false positives while remaining permissive.
    /// Can be overridden per-vocabulary via `CustomVocabularyContext.minSimilarity`.
    ///
    /// - Value: `0.52` (52% similarity required)
    /// - Used in: `CustomVocabularyContext.init()` as default parameter
    public static let defaultMinSimilarity: Float = 0.52

    /// Default minimum combined confidence threshold.
    ///
    /// Used when combining CTC acoustic score with string similarity score.
    /// The combined score must exceed this threshold for replacement.
    ///
    /// - Value: `0.54` (slightly above default similarity)
    /// - Used in: `CustomVocabularyContext.init()` as default parameter
    public static let defaultMinCombinedConfidence: Float = 0.54

    /// Similarity threshold for single-word span replacements.
    ///
    /// When replacing a single transcript word (span=1) with a vocabulary term,
    /// this moderate threshold allows corrections like "NECI" → "Nequi" while
    /// still being stricter than the floor.
    ///
    /// - Value: `0.55` (55% similarity for single words)
    /// - Example: "neci" vs "nequi" = 0.60 ✓
    /// - Used in: `VocabularyRescorer.swift` span-based decision logic
    public static let singleWordSpanSimilarity: Float = 0.55

    /// Similarity threshold for user-defined alias matches.
    ///
    /// When a vocabulary term has explicit aliases defined by the user,
    /// we trust those mappings with moderate similarity. This allows
    /// intentional phonetic aliases like "newres" → "Newrez".
    ///
    /// - Value: `0.65` (65% similarity for trusted aliases)
    /// - Used in: `VocabularyRescorer.swift` high-confidence alias path
    public static let highConfidenceAliasSimilarity: Float = 0.65

    /// Length ratio threshold below which stricter similarity is required.
    ///
    /// When original word is significantly shorter than vocabulary term
    /// (ratio < 0.75), we require higher similarity to prevent false positives
    /// like "and" (3 chars) matching "Andre" (5 chars) at 60% similarity.
    ///
    /// - Value: `0.75` (original must be at least 75% of vocab term length)
    /// - Formula: `originalWord.count / vocabTerm.count`
    /// - Example: "and"/"Andre" = 0.60 < 0.75 → requires stricter threshold
    /// - Used in: `VocabularyRescorer+ConstrainedCTC.swift` length ratio check
    public static let lengthRatioThreshold: Float = 0.75

    /// Similarity threshold for multi-word span replacements.
    ///
    /// When replacing multiple transcript words with a single vocabulary term
    /// (e.g., "new res" → "Newrez"), high similarity is required since we're
    /// modifying more text. Prevents "want to" → "Santoro", "and I" → "Audi".
    ///
    /// - Value: `0.80` (80% similarity for multi-word spans)
    /// - Raised from 0.75 after "and I" → "Audi" false positive (sim=0.75)
    /// - Used in: `VocabularyRescorer.swift` compound word matching
    public static let multiWordSpanSimilarity: Float = 0.80

    /// Similarity threshold for short words with low length ratio.
    ///
    /// Short common words (≤4 chars) with low length ratio need very high
    /// similarity to replace, preventing "you" → "Yu", "or" → "VR".
    ///
    /// - Value: `0.80` (80% similarity for short words)
    /// - Applies when: `word.count <= 4` AND `lengthRatio < 0.75`
    /// - Used in: `VocabularyRescorer+ConstrainedCTC.swift` short word guard
    public static let shortWordSimilarity: Float = 0.80

    /// Similarity threshold for spans containing stopwords.
    ///
    /// When a multi-word span contains common stopwords (articles, prepositions,
    /// pronouns), require very high similarity. Prevents replacing common
    /// phrases like "and we" → "Andre", "at this" → "Matthew".
    ///
    /// - Value: `0.85` (85% similarity when stopwords present)
    /// - Stopwords: "a", "the", "and", "or", "is", "to", "for", "in", etc.
    /// - Used in: `VocabularyRescorer+ConstrainedCTC.swift` stopword check
    public static let stopwordSpanSimilarity: Float = 0.85

    // MARK: - Context Biasing Weights

    /// Default context-biasing weight (CBW) per NeMo paper.
    ///
    /// Added to vocabulary term CTC scores to boost their likelihood.
    /// Higher values make vocabulary terms more likely to be selected.
    /// From NVIDIA NeMo's context biasing implementation.
    ///
    /// - Value: `3.0` (log-probability boost)
    /// - Effect: Multiplies vocabulary term probability by ~20x (e^3.0)
    /// - Used in: `VocabularyRescorer.rescoreWithTimings()`, constrained CTC methods
    public static let defaultCbw: Float = 3.0

    /// Default alpha value for weighted score combination.
    ///
    /// Used when combining acoustic and language model scores:
    /// `combinedScore = alpha * acousticScore + (1-alpha) * lmScore`
    ///
    /// - Value: `0.5` (equal weighting)
    /// - Range: 0.0 (LM only) to 1.0 (acoustic only)
    /// - Used in: `CustomVocabularyContext.init()` as default parameter
    public static let defaultAlpha: Float = 0.5

    /// Default margin in seconds for CTC frame alignment.
    ///
    /// When aligning vocabulary terms to transcript words, this margin
    /// allows for timing imprecision in word boundaries.
    ///
    /// - Value: `0.5` seconds (500ms tolerance)
    /// - Used in: `VocabularyRescorer+ConstrainedCTC.rescoreWithConstrainedCTC()`
    public static let defaultMarginSeconds: Double = 0.5

    /// Default vocabulary boost weight for rescorer config.
    ///
    /// Multiplier applied to vocabulary term scores in the rescorer.
    /// Lower than CBW since this is applied after initial detection.
    ///
    /// - Value: `1.5` (50% score boost)
    /// - Used in: `VocabularyRescorer.Config.default`
    public static let defaultVocabBoostWeight: Float = 1.5

    /// Score advantage threshold for multi-word span replacement.
    ///
    /// When replacing multi-word spans based on high similarity alone
    /// (without strong CTC evidence), require at least this score advantage.
    ///
    /// - Value: `0.5` (vocabulary score must be 0.5 higher than original)
    /// - Used in: `VocabularyRescorer.swift` compound word decision
    public static let scoreAdvantageThreshold: Float = 0.5

    // MARK: - Word Length Thresholds

    /// Maximum character count for "short word" classification.
    ///
    /// Words at or below this length are considered "short" and receive
    /// stricter similarity requirements to prevent false positives on
    /// common short words like "the", "and", "you", "or".
    ///
    /// - Value: `4` characters
    /// - Examples: "you" (3) is short, "nvidia" (6) is not
    /// - Used in: `VocabularyRescorer+ConstrainedCTC.swift` length checks
    public static let shortWordMaxLength: Int = 4

    // MARK: - BK-Tree Configuration

    /// Whether to use BK-tree for fuzzy string matching.
    ///
    /// BK-trees organize vocabulary by edit distance for efficient O(log V)
    /// approximate matching. Beneficial for large vocabularies (>100 terms).
    /// When false, uses O(V) linear scan which is faster for small vocabularies.
    ///
    /// - Value: `false` (linear scan)
    /// - Used in: `VocabularyRescorer.swift` initialization
    public static let useBkTree: Bool = false

    /// Maximum edit distance for BK-tree fuzzy search.
    ///
    /// Limits how "fuzzy" the BK-tree search can be. Higher values
    /// find more distant matches but increase search time.
    ///
    /// - Value: `3` (max 3 character edits)
    /// - Used in: `VocabularyRescorer.swift` when BK-tree is enabled
    public static let bkTreeMaxDistance: Int = 3
}
