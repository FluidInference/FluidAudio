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
    let spotter: CtcKeywordSpotter
    let rescorer: VocabularyRescorer
    let vocabSizeConfig: ContextBiasingConstants.VocabSizeConfig

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
    ///     must carry `ctcTokenIds` (e.g. via
    ///     `CustomVocabularyContext.loadWithCtcTokens(from:ctcVariant:)`).
    ///   - ctcModels: Pre-loaded CTC models for keyword spotting
    ///   - config: Optional rescorer configuration (default: `.default`)
    /// - Throws: Error if rescorer initialization fails
    public init(
        vocabulary: CustomVocabularyContext,
        ctcModels: CtcModels,
        config: VocabularyRescorer.Config? = nil
    ) async throws {
        self.vocabulary = vocabulary

        let blankId = ctcModels.vocabulary.count
        let spotter = CtcKeywordSpotter(models: ctcModels, blankId: blankId)
        self.spotter = spotter

        self.vocabSizeConfig = ContextBiasingConstants.rescorerConfig(
            forVocabSize: vocabulary.terms.count)

        let ctcModelDir = CtcModels.defaultCacheDirectory(for: ctcModels.variant)
        self.rescorer = try await VocabularyRescorer.create(
            spotter: spotter,
            vocabulary: vocabulary,
            config: config ?? .default,
            ctcModelDirectory: ctcModelDir
        )
    }

    /// Rescore a transcript against CTC acoustic evidence from its audio.
    ///
    /// `tokenTimings` must be on the same clock as `audioSamples`: time zero
    /// is the first sample. Callers rescoring a window or segment out of a
    /// longer stream pass segment-local timings with the segment's audio.
    ///
    /// - Returns: The rescore output if any replacement was applied,
    ///   nil otherwise (including on CTC inference failure, which is logged
    ///   and absorbed — boosting must never break transcription).
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

            // Vocabulary-size-aware thresholds, respecting the caller-specified
            // threshold when stricter.
            let minSimilarity = max(vocabSizeConfig.minSimilarity, vocabulary.minSimilarity)

            let rescoreOutput = rescorer.ctcTokenRescore(
                transcript: text,
                tokenTimings: tokenTimings,
                logProbs: logProbs,
                frameDuration: spotResult.frameDuration,
                cbw: vocabSizeConfig.cbw,
                marginSeconds: 0.5,
                minSimilarity: minSimilarity
            )

            guard rescoreOutput.wasModified else { return nil }

            logger.info(
                "Vocabulary rescoring applied \(rescoreOutput.replacements.count) replacement(s)"
            )
            for replacement in rescoreOutput.replacements where replacement.shouldReplace {
                logger.debug(
                    "  '\(replacement.originalWord)' → '\(replacement.replacementWord ?? "")'"
                )
            }
            return rescoreOutput
        } catch {
            logger.warning("Vocabulary rescoring failed: \(error.localizedDescription)")
            return nil
        }
    }
}
