import Foundation

/// Port of laya's `build_sequence`: `[CLS] <type> question: instructions [SEP] ([MASK] option)* [SEP] state [SEP]`.
///
/// Options are capped at 48 tokens each and share a `headMaxLength` budget with the instructions;
/// whatever remains of `maximumLength` goes to the state, which is truncated on the right.
struct LayaSequenceBuilder: Sendable {
    struct Sequence: Sendable, Equatable {
        let ids: [Int]
        /// Position of the `[MASK]` marker for each option, in option order.
        let markers: [Int]
        let stateWasTruncated: Bool
    }

    static let optionTokenCap = 48
    static let minimumOptionBudget = 16
    static let minimumInstructionTokens = 8
    static let minimumTokensPerOption = 4

    let encode: @Sendable (String) -> [Int]
    let maskTokenId: Int
    let clsTokenId: Int
    let sepTokenId: Int
    let maskToken: String
    let headMaxLength: Int

    init(tokenizer: LayaTokenizer, headMaxLength: Int) {
        self.encode = { tokenizer.encode($0) }
        self.maskTokenId = tokenizer.maskTokenId
        self.clsTokenId = tokenizer.clsTokenId
        self.sepTokenId = tokenizer.sepTokenId
        self.maskToken = tokenizer.maskToken
        self.headMaxLength = headMaxLength
    }

    init(
        encode: @escaping @Sendable (String) -> [Int], maskTokenId: Int, clsTokenId: Int, sepTokenId: Int,
        maskToken: String, headMaxLength: Int
    ) {
        self.encode = encode
        self.maskTokenId = maskTokenId
        self.clsTokenId = clsTokenId
        self.sepTokenId = sepTokenId
        self.maskToken = maskToken
        self.headMaxLength = headMaxLength
    }

    /// The prompt head (everything before the state) plus the encoded state, so callers can pick a bucket.
    struct Parts: Sendable {
        let head: [Int]
        let markers: [Int]
        let state: [Int]
        let optionCount: Int

        /// Sequence length before any state truncation: head + state + trailing separator.
        var untruncatedLength: Int { head.count + state.count + 1 }
    }

    func parts(state: String, question: LayaQuestion) throws -> Parts {
        try question.validate()
        let options = question.renderedOptions
        let instructions = question.instructions.replacingOccurrences(of: maskToken, with: " ")
        var headIds = encode("\(question.type.rawValue) question: \(instructions)")
        var optionIds = options.map { option in
            [maskTokenId]
                + Array(encode(" " + option.replacingOccurrences(of: maskToken, with: " ")).prefix(Self.optionTokenCap))
        }
        var optionBudget = headMaxLength - optionIds.reduce(0) { $0 + $1.count }
        if optionBudget < Self.minimumOptionBudget {
            let perOption = max(
                Self.minimumTokensPerOption, (headMaxLength - Self.minimumOptionBudget) / max(1, optionIds.count))
            optionIds = optionIds.map { Array($0.prefix(perOption)) }
            optionBudget = headMaxLength - optionIds.reduce(0) { $0 + $1.count }
        }
        headIds = Array(headIds.prefix(max(Self.minimumInstructionTokens, optionBudget)))
        var ids = [clsTokenId] + headIds + [sepTokenId]
        var markers: [Int] = []
        for option in optionIds {
            markers.append(ids.count)
            ids.append(contentsOf: option)
        }
        ids.append(sepTokenId)
        let stateIds = encode(state.replacingOccurrences(of: maskToken, with: " "))
        return Parts(head: ids, markers: markers, state: stateIds, optionCount: options.count)
    }

    /// Fit the parts into `maximumLength` tokens the way laya does for its `max_len`.
    func sequence(from parts: Parts, maximumLength: Int) throws -> Sequence {
        let room = max(0, maximumLength - parts.head.count - 1)
        let state = Array(parts.state.prefix(room))
        var ids = parts.head + state + [sepTokenId]
        ids = Array(ids.prefix(maximumLength))
        // Upstream keeps a marker only while it is inside the window and refuses the question otherwise.
        let markers = parts.markers.filter { $0 < maximumLength }
        guard markers.count == parts.optionCount else {
            throw LayaError.promptTooLong(tokens: parts.head.count + 1, maximumLength: maximumLength)
        }
        return Sequence(ids: ids, markers: markers, stateWasTruncated: state.count < parts.state.count)
    }

    func sequence(state: String, question: LayaQuestion, maximumLength: Int) throws -> Sequence {
        try sequence(from: parts(state: state, question: question), maximumLength: maximumLength)
    }
}
