import Foundation

/// Vocabulary-boosting pipeline shared by the ASR engines: CTC keyword
/// spotting over the audio plus constrained rescoring of the transcript.
///
/// The CTC pass runs a separate CTC model (e.g. parakeet-ctc-110m) on the raw
/// audio, so the session is independent of which primary model produced the
/// transcript — any engine that can supply its transcript, token timings, and
/// the audio behind them can use it (SlidingWindow TDT, Unified batch,
/// Unified streaming).
public struct VocabularyBoostingSession: Sendable {
    private let logger = AppLogger(category: "VocabBoosting")

    public let vocabulary: CustomVocabularyContext
    public let thresholds: Thresholds
    let spotter: CtcKeywordSpotter
    let rescorer: VocabularyRescorer
    let vocabSizeConfig: ContextBiasingConstants.VocabSizeConfig

    /// Rescoring thresholds. `nil` fields keep the vocabulary-size defaults of
    /// `ContextBiasingConstants.rescorerConfig(forVocabSize:)` (cbw 4.5; similarity 0.50 up to 10 terms,
    /// 0.55 up to 100, 0.60 above).
    ///
    /// The defaults favour recall. For dictation, where a wrong replacement costs more than a miss,
    /// `cbw: 0, minSimilarity: 0.7` replaces far fewer correctly heard words (#967).
    public struct Thresholds: Sendable, Equatable {
        /// Weight added to a term's per-token CTC score before it is compared with the transcribed words'
        /// own score over the same window. 0 makes the term win on acoustics alone; a negative value makes
        /// it win by that margin (scaled up for terms over 3 tokens when adaptive thresholds are on).
        public var cbw: Float?
        /// Minimum string similarity between a transcript span and a term. The vocabulary's own
        /// `minSimilarity` still applies when it is stricter.
        public var minSimilarity: Float?
        /// Audio searched on each side of a span, in seconds, for both the term and the transcribed words.
        public var marginSeconds: Double
        /// Minimum per-token spotter score (average log-probability) for a term to be listed in
        /// `RescoreOutput.detectedTerms`. Does not affect which replacements are made.
        public var detectionMinScore: Float

        public init(
            cbw: Float? = nil,
            minSimilarity: Float? = nil,
            marginSeconds: Double = 0.5,
            detectionMinScore: Float = Thresholds.defaultDetectionMinScore
        ) {
            self.cbw = cbw
            self.minSimilarity = minSimilarity
            self.marginSeconds = marginSeconds
            self.detectionMinScore = detectionMinScore
        }

        /// Measured on real speech (13 dictation clips from #967 and 40 LibriSpeech test-clean utterances):
        /// at -6.5 every spoken term stayed listed, including all 13 that Parakeet misheard (the weakest
        /// at -6.13), while unspoken terms fell from 299/300 listed at the spotter's own -15 to 53/300, and a
        /// junk term (best -8.53) was never listed. Sound-alikes at real locations (`Sora` inside "Saurabh",
        /// `Codex` on "Xcode") score like spoken terms, so no floor removes them.
        public static let defaultDetectionMinScore: Float = -6.5
        public static let `default` = Thresholds()
    }

    /// Recommended rescorer config for engines whose output text is
    /// inverse-text-normalized (parakeet-unified writes "11.4 billion" and
    /// "35.3%"). ITN packs a multi-second spoken number into one or two
    /// written words, so short word spans cover long acoustic stretches and
    /// become eligible for the spotter-anchored rescue pass at garbage
    /// spotter scores ('yen, down by 35.3%' → term at sim 0.20). The #702
    /// similarity floors close exactly that hole; spelled-out engines
    /// (SlidingWindow TDT) keep `.default`, where the rescue recovers brand
    /// names the model mangles past the similarity gate.
    public static let itnDefaultConfig = VocabularyRescorer.Config(
        spotterRescueMinSimilarity: 0.30,
        spotterRescueMultiWordMinSimilarity: 0.50
    )

    /// Create a session from a tokenized vocabulary and pre-loaded CTC models.
    ///
    /// - Parameters:
    ///   - vocabulary: Custom vocabulary context with terms to detect. Terms
    ///     without `ctcTokenIds` are tokenized here with the CTC tokenizer
    ///     shipped alongside `ctcModels` (#851); pre-tokenized terms (e.g. via
    ///     `CustomVocabularyContext.loadWithCtcTokens(from:ctcVariant:)`) are
    ///     used as-is.
    ///   - ctcModels: Pre-loaded CTC models for keyword spotting
    ///   - config: Optional rescorer configuration (default: `.default`)
    ///   - thresholds: Rescoring thresholds (default: vocabulary-size defaults)
    /// - Throws: Error if the CTC tokenizer cannot be loaded for untokenized
    ///   terms, or if rescorer initialization fails
    public init(
        vocabulary: CustomVocabularyContext,
        ctcModels: CtcModels,
        config: VocabularyRescorer.Config? = nil,
        thresholds: Thresholds = .default
    ) async throws {
        self.thresholds = thresholds
        let ctcModelDir = CtcModels.defaultCacheDirectory(for: ctcModels.variant)

        // Terms built in code arrive without CTC token IDs, and every consumer
        // (spotter, rescorer) skips such terms without a word — the documented
        // `CustomVocabularyContext(terms: [CustomVocabularyTerm(text:)])` path
        // was a silent no-op on every engine (#851). Tokenize them here.
        var vocabulary = vocabulary
        let needsTokens = vocabulary.terms.contains { ($0.ctcTokenIds ?? []).isEmpty }
        if needsTokens {
            let tokenizer = try await CtcTokenizer.load(from: ctcModelDir)
            let result = vocabulary.tokenizingMissingCtcTokens(using: tokenizer.encode)
            vocabulary = result.context
            logger.info("Tokenized \(result.tokenized) vocabulary term(s) with the CTC tokenizer")
            for text in result.dropped {
                logger.warning("Vocabulary term '\(text)' produced no CTC tokens; dropped")
            }
        }
        if vocabulary.terms.isEmpty {
            logger.warning("Vocabulary boosting configured with no usable terms; rescoring will be a no-op")
        }
        self.vocabulary = vocabulary

        let blankId = ctcModels.vocabulary.count
        let spotter = CtcKeywordSpotter(models: ctcModels, blankId: blankId)
        self.spotter = spotter

        self.vocabSizeConfig = ContextBiasingConstants.rescorerConfig(
            forVocabSize: vocabulary.terms.count)

        self.rescorer = try await VocabularyRescorer.create(
            spotter: spotter,
            vocabulary: vocabulary,
            config: config ?? .default,
            ctcModelDirectory: ctcModelDir
        )
    }

    /// Whether transcript rescoring needs CTC inference. This does not predict
    /// standalone keyword detections; callers requiring those must still rescore.
    public func hasCTCRescoringCandidates(text: String, tokenTimings: [TokenTiming]) -> Bool {
        rescorer.hasCTCRescoringCandidates(
            transcript: text,
            tokenTimings: tokenTimings,
            minSimilarity: minSimilarity
        )
    }

    /// Similarity floor for rescoring.
    var minSimilarity: Float {
        Self.minSimilarity(
            override: thresholds.minSimilarity, sizeDefault: vocabSizeConfig.minSimilarity,
            vocabularyFloor: vocabulary.minSimilarity)
    }

    /// The `ctcTokenRescore` arguments for `thresholds`: overrides where set, otherwise the vocabulary-size
    /// defaults. Pure, for testability.
    static func rescoreParameters(
        _ thresholds: Thresholds, sizeConfig: ContextBiasingConstants.VocabSizeConfig, vocabularyFloor: Float
    ) -> (cbw: Float, minSimilarity: Float, marginSeconds: Double) {
        (
            cbw: thresholds.cbw ?? sizeConfig.cbw,
            minSimilarity: minSimilarity(
                override: thresholds.minSimilarity, sizeDefault: sizeConfig.minSimilarity,
                vocabularyFloor: vocabularyFloor),
            marginSeconds: thresholds.marginSeconds
        )
    }

    /// The override (or the vocabulary-size default), raised to the vocabulary's own threshold when that
    /// is stricter. Pure, for testability.
    static func minSimilarity(override: Float?, sizeDefault: Float, vocabularyFloor: Float) -> Float {
        max(override ?? sizeDefault, vocabularyFloor)
    }

    /// Rescore a transcript against CTC acoustic evidence from its audio.
    ///
    /// `tokenTimings` must be on the same clock as `audioSamples`: time zero
    /// is the first sample. Callers rescoring a window or segment out of a
    /// longer stream pass segment-local timings with the segment's audio.
    ///
    /// - Returns: The rescore output when a replacement was applied **or** the
    ///   spotter detected at least one vocabulary term (`detectedTerms`); the
    ///   text is the caller's own when nothing was replaced. Nil when there is
    ///   nothing to report, and on CTC inference failure, which is logged and
    ///   absorbed — boosting must never break transcription.
    public func rescore(
        text: String,
        tokenTimings: [TokenTiming],
        audioSamples: [Float]
    ) async -> VocabularyRescorer.RescoreOutput? {
        guard !tokenTimings.isEmpty, !audioSamples.isEmpty else { return nil }

        do {
            let spotResult = try await spotter.spotKeywordsWithLogProbs(
                audioSamples: audioSamples,
                customVocabulary: vocabulary,
                minScore: nil
            )

            let logProbs = spotResult.logProbs
            guard !logProbs.isEmpty else {
                logger.debug("Vocabulary rescoring skipped: no log probs from CTC")
                return nil
            }

            let parameters = Self.rescoreParameters(
                thresholds, sizeConfig: vocabSizeConfig, vocabularyFloor: vocabulary.minSimilarity)
            let rescoreOutput = rescorer.ctcTokenRescore(
                transcript: text,
                tokenTimings: tokenTimings,
                logProbs: logProbs,
                frameDuration: spotResult.frameDuration,
                cbw: parameters.cbw,
                marginSeconds: parameters.marginSeconds,
                minSimilarity: parameters.minSimilarity
            )

            // The spotter runs with its lenient default floor (-15 per token), which lists nearly every
            // term on any speech (#967); report only detections above `detectionMinScore`.
            let detectedTerms = Self.detectedTermTexts(
                spotResult.detections, minScore: thresholds.detectionMinScore)
            guard rescoreOutput.wasModified || !detectedTerms.isEmpty else { return nil }

            if rescoreOutput.wasModified {
                logger.info(
                    "Vocabulary rescoring applied \(rescoreOutput.replacements.count) replacement(s)"
                )
                for replacement in rescoreOutput.replacements where replacement.shouldReplace {
                    logger.debug(
                        "  '\(replacement.originalWord)' → '\(replacement.replacementWord ?? "")'"
                    )
                }
            }
            // The rescorer rebuilds its text from word timings (single spaces,
            // timing-backed words only). Only hand that back when it actually
            // changed something; otherwise the caller keeps its own text.
            return VocabularyRescorer.RescoreOutput(
                text: rescoreOutput.wasModified ? rescoreOutput.text : text,
                replacements: rescoreOutput.replacements,
                wasModified: rescoreOutput.wasModified,
                detectedTerms: detectedTerms
            )
        } catch {
            logger.warning("Vocabulary rescoring failed: \(error.localizedDescription)")
            return nil
        }
    }

    /// Canonical texts of the spotter's detections scoring at least `minScore`, in time order, without
    /// repeats. Pure, for testability.
    static func detectedTermTexts(
        _ detections: [CtcKeywordSpotter.KeywordDetection], minScore: Float = -.infinity
    ) -> [String] {
        var seen = Set<String>()
        var texts: [String] = []
        for detection in detections.sorted(by: { $0.startTime < $1.startTime }) where detection.score >= minScore {
            let text = detection.term.text
            if seen.insert(text.lowercased()).inserted {
                texts.append(text)
            }
        }
        return texts
    }
}
