import Foundation

/// Applies custom-vocabulary keyword boosting to a Canary (AED) transcript using
/// the existing CTC keyword spotter — the same detector the parakeet "ctc custom
/// vocab" path uses.
///
/// Canary decodes autoregressively and emits no per-frame timestamps, so the
/// timestamp-constrained CTC rescorer (`VocabularyRescorer.ctcTokenRescore`)
/// cannot be applied directly. Instead this reuses the engine-independent
/// `CtcKeywordSpotter` to detect dictionary terms from the audio, then injects
/// each detected term into Canary's transcript by fuzzy string match: a span that
/// is close-but-not-exact to a detected term (i.e. Canary mis-spelled the domain
/// word) is replaced with the canonical term.
public struct CanaryKeywordBooster: Sendable {

    public struct Result: Sendable {
        public let text: String
        /// Distinct terms the CTC spotter detected in the audio.
        public let detected: [String]
        /// Terms actually substituted into the transcript.
        public let applied: [String]
    }

    private let spotter: CtcKeywordSpotter
    private let tokenizer: CtcTokenizer
    /// CTC detection score floor (log-prob; higher = stronger). Matches the
    /// permissive detection threshold the earnings benchmark uses.
    private let minScore: Float
    /// Replace a transcript span only when its similarity to the term is at least
    /// this (close enough to be the same word mis-transcribed).
    private let minSimilarity: Float
    /// …and below this (above it the span is already essentially the term).
    private let maxSimilarity: Float
    /// When a detected term has no fuzzy-matchable span (canary missed it entirely),
    /// insert it at the position implied by the CTC detection time.
    private let insertOnMiss: Bool
    /// Only insert (vs replace) when the detection score clears this stronger floor —
    /// protects precision against weak detections being force-inserted.
    private let insertScoreFloor: Float

    private static let logger = AppLogger(category: "CanaryKeywordBooster")

    public init(
        spotter: CtcKeywordSpotter,
        tokenizer: CtcTokenizer,
        minScore: Float = -15.0,
        minSimilarity: Float = 0.60,
        maxSimilarity: Float = 0.97,
        insertOnMiss: Bool = true,
        insertScoreFloor: Float = -6.0
    ) {
        self.spotter = spotter
        self.tokenizer = tokenizer
        self.minScore = minScore
        self.minSimilarity = minSimilarity
        self.maxSimilarity = maxSimilarity
        self.insertOnMiss = insertOnMiss
        self.insertScoreFloor = insertScoreFloor
    }

    /// Load the CTC spotter + tokenizer (parakeet-tdt_ctc-110m) and build a booster.
    public static func load(
        minScore: Float = -15.0,
        minSimilarity: Float = 0.60,
        insertOnMiss: Bool = true,
        insertScoreFloor: Float = -6.0
    ) async throws -> CanaryKeywordBooster {
        let models = try await CtcModels.downloadAndLoad()
        let tokenizer = try await CtcTokenizer.load()
        return CanaryKeywordBooster(
            spotter: CtcKeywordSpotter(models: models), tokenizer: tokenizer, minScore: minScore,
            minSimilarity: minSimilarity, insertOnMiss: insertOnMiss, insertScoreFloor: insertScoreFloor)
    }

    /// Ensure every term carries CTC token IDs (the spotter scores by them).
    private func tokenized(_ vocabulary: CustomVocabularyContext) -> CustomVocabularyContext {
        let terms = vocabulary.terms.map { term -> CustomVocabularyTerm in
            if let ids = term.ctcTokenIds, !ids.isEmpty { return term }
            let ids = tokenizer.encode(term.text)
            return CustomVocabularyTerm(
                text: term.text, weight: term.weight, aliases: term.aliases, tokenIds: term.tokenIds,
                ctcTokenIds: ids)
        }
        return CustomVocabularyContext(
            terms: terms, alpha: vocabulary.alpha, minCtcScore: vocabulary.minCtcScore,
            minSimilarity: vocabulary.minSimilarity, minCombinedConfidence: vocabulary.minCombinedConfidence,
            minTermLength: vocabulary.minTermLength)
    }

    /// Inject CTC-spotted custom-vocabulary terms into `transcript`.
    public func boost(
        transcript: String, audioSamples: [Float], vocabulary: CustomVocabularyContext
    ) async throws -> Result {
        let vocab = tokenized(vocabulary)
        let spot = try await spotter.spotKeywordsWithLogProbs(
            audioSamples: audioSamples, customVocabulary: vocab, minScore: minScore)

        // Best CTC detection (score + start time) per detected term.
        var detByTerm: [String: (term: CustomVocabularyTerm, score: Float, startTime: TimeInterval)] = [:]
        for d in spot.detections where d.score >= minScore {
            let key = d.term.textLowercased
            if let cur = detByTerm[key], cur.score >= d.score { continue }
            detByTerm[key] = (d.term, d.score, d.startTime)
        }
        let detected = detByTerm.values.map { $0.term.text }.sorted()
        guard !detByTerm.isEmpty else {
            return Result(text: transcript, detected: detected, applied: [])
        }
        let duration = max(0.001, Double(audioSamples.count) / 16000.0)

        // Strongest detections first; longer phrases before shorter to avoid
        // a single word stealing a multi-word match.
        let ordered = detByTerm.values.sorted {
            $0.term.text.split(separator: " ").count != $1.term.text.split(separator: " ").count
                ? $0.term.text.split(separator: " ").count > $1.term.text.split(separator: " ").count
                : $0.score > $1.score
        }

        var words = transcript.split(separator: " ").map(String.init)
        var applied: [String] = []

        for entry in ordered {
            let term = entry.term
            let termLower = term.textLowercased
            // Already present (case-insensitive substring) → nothing to fix.
            if words.joined(separator: " ").lowercased().contains(termLower) { continue }

            let termWords = term.text.split(separator: " ").map(String.init)
            let span = max(1, termWords.count)

            // 1) Fuzzy replace: a close-but-wrong span is canary mis-spelling the term.
            var bestIdx = -1
            var bestSim: Float = 0
            if words.count >= span {
                for i in 0...(words.count - span) {
                    let window = normalize(words[i..<(i + span)].joined(separator: " "))
                    let sim = VocabularyRescorer.stringSimilarity(window, termLower)
                    if sim > bestSim {
                        bestSim = sim
                        bestIdx = i
                    }
                }
            }

            if bestIdx >= 0, bestSim >= minSimilarity, bestSim < maxSimilarity {
                words.replaceSubrange(bestIdx..<(bestIdx + span), with: termWords)
                applied.append(term.text)
                continue
            }

            // 2) Timestamp-guided insertion: canary missed the word entirely (no fuzzy
            // span). The CTC detection still localizes it in time, so insert it at the
            // proportional word position. Gated by a stronger score floor to protect
            // precision.
            if insertOnMiss, entry.score >= insertScoreFloor, !words.isEmpty {
                let frac = min(1.0, max(0.0, entry.startTime / duration))
                let pos = min(words.count, Int((frac * Double(words.count)).rounded()))
                words.insert(contentsOf: termWords, at: pos)
                applied.append(term.text)
            }
        }

        return Result(text: words.joined(separator: " "), detected: detected, applied: applied)
    }

    private func normalize(_ s: String) -> String {
        s.lowercased().filter { !$0.isPunctuation }
    }
}
