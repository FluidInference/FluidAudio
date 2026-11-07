import Foundation

/// One candidate for a SenseVoice query slot (language, emotion or audio event)
/// and how likely the model thinks it is.
public struct SenseVoiceTagScore: Sendable, Equatable {
    /// The tag without its `<|` `|>` markers, e.g. `HAPPY`, `Speech`, `zh`.
    public let tag: String
    /// Softmax over the slot's candidate tags only, so the scores of one slot sum to 1.
    public let probability: Float

    public init(tag: String, probability: Float) {
        self.tag = tag
        self.probability = probability
    }
}

/// Scores for the leading query slots, read straight from the CTC logits rather
/// than from the decoded text. Each list is sorted most likely first.
public struct SenseVoiceTagScores: Sendable, Equatable {
    public let language: [SenseVoiceTagScore]
    public let emotion: [SenseVoiceTagScore]
    public let event: [SenseVoiceTagScore]

    public static let empty = SenseVoiceTagScores(language: [], emotion: [], event: [])

    public init(language: [SenseVoiceTagScore], emotion: [SenseVoiceTagScore], event: [SenseVoiceTagScore]) {
        self.language = language
        self.emotion = emotion
        self.event = event
    }
}

/// Turns the first logit rows into per-slot tag probabilities.
///
/// SenseVoice emits its query tokens in the order language, emotion, event,
/// text-norm (`<|zh|><|NEUTRAL|><|Speech|><|woitn|>`), one per leading logit row.
/// Restricting each row's softmax to the tags that belong in that slot gives a
/// usable confidence even when the greedy decode would pick a token from a
/// different slot or CTC collapsing merges two equal tags.
enum SenseVoiceTagScoring {

    static let emotionTags = [
        "HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "OTHER", "EMO_UNKNOWN",
    ]

    static let eventTags = [
        "Speech", "BGM", "Laughter", "Applause", "Cry", "Sneeze", "Breath", "Cough", "Sing",
        "Speech_Noise", "Event_UNK",
    ]

    /// Logit row for each slot.
    static let languageRow = 0
    static let emotionRow = 1
    static let eventRow = 2

    /// Token ids for each slot, looked up by name so the vocabulary order doesn't matter.
    struct Candidates: Sendable {
        let language: [(id: Int, tag: String)]
        let emotion: [(id: Int, tag: String)]
        let event: [(id: Int, tag: String)]

        init(vocabulary: [Int: String]) {
            var byTag: [String: Int] = [:]
            for (id, piece) in vocabulary {
                guard let tag = SenseVoiceTagScoring.tagName(piece) else { continue }
                byTag[tag] = id
            }
            func lookup(_ tags: [String]) -> [(id: Int, tag: String)] {
                tags.compactMap { tag in byTag[tag].map { (id: $0, tag: tag) } }
            }
            emotion = lookup(SenseVoiceTagScoring.emotionTags)
            event = lookup(SenseVoiceTagScoring.eventTags)
            // Languages are the lowercase tags (`zh`, `en`, `zh/en`, …) plus `nospeech`.
            language =
                byTag
                .filter { tag, _ in tag == tag.lowercased() && !tag.contains("itn") }
                .map { (id: $0.value, tag: $0.key) }
                .sorted { $0.id < $1.id }
        }
    }

    /// `<|HAPPY|>` → `HAPPY`; nil for ordinary pieces.
    static func tagName(_ piece: String) -> String? {
        guard piece.hasPrefix("<|"), piece.hasSuffix("|>"), piece.count > 4 else { return nil }
        return String(piece.dropFirst(2).dropLast(2))
    }

    /// `rows` holds the first logit rows (at least three) as full-vocabulary vectors.
    static func scores(rows: [[Float]], candidates: Candidates) -> SenseVoiceTagScores {
        guard rows.count > eventRow else { return .empty }
        return SenseVoiceTagScores(
            language: score(rows[languageRow], candidates.language),
            emotion: score(rows[emotionRow], candidates.emotion),
            event: score(rows[eventRow], candidates.event))
    }

    static func score(_ row: [Float], _ candidates: [(id: Int, tag: String)]) -> [SenseVoiceTagScore] {
        let usable = candidates.filter { $0.id < row.count && row[$0.id].isFinite }
        guard let maxLogit = usable.map({ row[$0.id] }).max() else { return [] }
        let weights = usable.map { exp(row[$0.id] - maxLogit) }
        let total = weights.reduce(0, +)
        guard total > 0 else { return [] }
        return zip(usable, weights)
            .map { SenseVoiceTagScore(tag: $0.tag, probability: $1 / total) }
            .sorted { $0.probability > $1.probability }
    }
}
