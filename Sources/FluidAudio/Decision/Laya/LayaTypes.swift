import Foundation

/// Errors from laya prompt construction, model loading, or prediction validation.
public enum LayaError: Error, LocalizedError, Sendable, Equatable {
    /// A question needs nonempty instructions.
    case emptyInstructions
    /// Choice and score questions need between two and the model's maximum number of options.
    case invalidOptionCount(Int)
    /// The option at this zero-based index is empty.
    case emptyOption(Int)
    /// The rendered options do not fit the option budget of the checkpoint, even after shrinking.
    case optionsExceedBudget(Int)
    /// No loaded length bucket can hold the head of this prompt.
    case promptTooLong(tokens: Int, maximumLength: Int)
    /// A required file (model bundle, tokenizer) is missing or malformed.
    case invalidAsset(String)
    /// The model does not have the supported laya tensor interface.
    case invalidModel(String)
    /// The model returned invalid logits or probabilities.
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .emptyInstructions:
            return "laya requires nonempty question instructions."
        case .invalidOptionCount(let count):
            return "laya requires 2–\(LayaManager.maximumOptions) options; received \(count)."
        case .emptyOption(let index):
            return "laya option \(index) is empty."
        case .optionsExceedBudget(let budget):
            return "laya options exceed the checkpoint's option budget of \(budget) tokens."
        case .promptTooLong(let tokens, let maximumLength):
            return "laya prompt head needs \(tokens) tokens; the largest loaded bucket is \(maximumLength)."
        case .invalidAsset(let reason):
            return "Invalid laya asset: \(reason)"
        case .invalidModel(let reason):
            return "Invalid laya model: \(reason)"
        case .invalidOutput(let reason):
            return "Invalid laya output: \(reason)"
        }
    }
}

/// The three typed question primitives laya answers in one encoder pass.
public enum LayaQuestionType: String, Sendable, Codable, CaseIterable {
    /// Pick one labelled option; probabilities are returned per option.
    case choice
    /// Ordinal levels; the answer is the expected level index plus the distribution.
    case score
    /// A calibrated yes/no probability that the statement holds.
    case noul

    /// Index of the question-type embedding row in the checkpoint.
    var index: Int {
        switch self {
        case .choice: return 0
        case .score: return 1
        case .noul: return 2
        }
    }
}

/// One typed question about a state, mirroring laya's `{"type", "instructions", "criteria"}` schema.
public struct LayaQuestion: Sendable, Equatable {
    /// A choice option: the label that is returned, plus an optional description shown to the model.
    public struct Choice: Sendable, Equatable {
        public let label: String
        public let description: String?

        public init(_ label: String, description: String? = nil) {
            self.label = label
            self.description = description
        }
    }

    public enum Kind: Sendable, Equatable {
        case choice([Choice])
        case score(levels: [String])
        case noul(falseDescription: String?, trueDescription: String?)
    }

    public let instructions: String
    public let kind: Kind

    public init(instructions: String, kind: Kind) {
        self.instructions = instructions
        self.kind = kind
    }

    /// A choice question with plain labels.
    public static func choice(_ instructions: String, options: [String]) -> LayaQuestion {
        LayaQuestion(instructions: instructions, kind: .choice(options.map { Choice($0) }))
    }

    /// A choice question with labels and descriptions.
    public static func choice(_ instructions: String, options: [Choice]) -> LayaQuestion {
        LayaQuestion(instructions: instructions, kind: .choice(options))
    }

    /// An ordinal question; `levels[0]` is the lowest level.
    public static func score(_ instructions: String, levels: [String]) -> LayaQuestion {
        LayaQuestion(instructions: instructions, kind: .score(levels: levels))
    }

    /// A yes/no question. The answer probability is P(true).
    public static func noul(
        _ instructions: String, falseDescription: String? = nil, trueDescription: String? = nil
    ) -> LayaQuestion {
        LayaQuestion(
            instructions: instructions,
            kind: .noul(falseDescription: falseDescription, trueDescription: trueDescription))
    }

    public var type: LayaQuestionType {
        switch kind {
        case .choice: return .choice
        case .score: return .score
        case .noul: return .noul
        }
    }

    /// Labels in option order: choice labels, `"0"…` for score levels, `"false"`/`"true"` for noul.
    public var labels: [String] {
        switch kind {
        case .choice(let options): return options.map(\.label)
        case .score(let levels): return levels.indices.map(String.init)
        case .noul: return ["false", "true"]
        }
    }

    /// Option texts exactly as laya renders them (`render_options`).
    var renderedOptions: [String] {
        switch kind {
        case .choice(let options):
            return options.map { option in
                guard let description = option.description, !description.isEmpty else { return option.label }
                return "\(option.label): \(description)"
            }
        case .score(let levels):
            return levels.enumerated().map { "level \($0.offset): \($0.element)" }
        case .noul(let falseDescription, let trueDescription):
            let no = (falseDescription?.isEmpty == false) ? falseDescription! : "no, the statement does not hold"
            let yes = (trueDescription?.isEmpty == false) ? trueDescription! : "yes, the statement holds"
            return ["false: \(no)", "true: \(yes)"]
        }
    }

    func validate() throws {
        guard !instructions.isEmpty else { throw LayaError.emptyInstructions }
        switch kind {
        case .choice(let options):
            guard (2...LayaManager.maximumOptions).contains(options.count) else {
                throw LayaError.invalidOptionCount(options.count)
            }
            for (index, option) in options.enumerated() where option.label.isEmpty {
                throw LayaError.emptyOption(index)
            }
        case .score(let levels):
            guard (2...LayaManager.maximumOptions).contains(levels.count) else {
                throw LayaError.invalidOptionCount(levels.count)
            }
            for (index, level) in levels.enumerated() where level.isEmpty {
                throw LayaError.emptyOption(index)
            }
        case .noul:
            break
        }
    }
}

/// One answered question. Probabilities are scores, not guarantees of correctness.
public struct LayaAnswer: Sendable {
    public let question: LayaQuestion
    /// Temperature-calibrated probabilities per option, in option order, computed on the host in Double.
    public let probabilities: [Float]
    /// The model's own softmax output (temperature 1) for the supplied options, for conversion comparisons.
    public let rawProbabilities: [Float]
    /// One raw score per supplied option.
    public let logits: [Float]
    /// `1 - H(p) / log(k)`: 1 when one option takes all mass, 0 when uniform.
    public let confidence: Float
    /// Probability that the action head keeps the decision (versus escalating).
    public let actionProbability: Float
    /// Real tokens in the encoded prompt.
    public let tokenCount: Int
    /// Fixed sequence length of the model bucket that ran this question.
    public let bucketLength: Int
    /// Whether the state text was cut to fit the bucket.
    public let stateWasTruncated: Bool

    /// Zero-based index of the highest-probability option.
    public var selectedIndex: Int {
        probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
    }

    /// Label of the selected option (`question.labels[selectedIndex]`).
    public var selectedLabel: String {
        question.labels[selectedIndex]
    }

    /// Expected level index for score questions; nil otherwise.
    public var expectedScore: Float? {
        guard question.type == .score else { return nil }
        return probabilities.enumerated().reduce(0) { $0 + Float($1.offset) * $1.element }
    }

    /// P(true) for noul questions; nil otherwise.
    public var noul: Float? {
        guard question.type == .noul, probabilities.count == 2 else { return nil }
        return probabilities[1]
    }
}
