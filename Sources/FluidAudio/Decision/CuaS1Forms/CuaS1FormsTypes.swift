import Foundation

/// Errors from form-option encoding, model loading, or prediction validation.
public enum CuaS1FormsError: Error, LocalizedError, Sendable, Equatable {
    /// The model requires a nonempty description of the task and UI element.
    case emptyContext
    /// A request must contain between two and the model's maximum number of options.
    case invalidOptionCount(Int)
    /// The option at this zero-based index is empty.
    case emptyOption(Int)
    /// The model does not have the supported CUA-S1-FORMS tensor interface.
    case invalidModel(String)
    /// The model returned invalid logits or probabilities.
    case invalidOutput(String)

    /// A description of the invalid input, model, or prediction.
    public var errorDescription: String? {
        switch self {
        case .emptyContext:
            return "CUA-S1-FORMS requires a nonempty context."
        case .invalidOptionCount(let count):
            return "CUA-S1-FORMS requires 2–32 options; received \(count)."
        case .emptyOption(let index):
            return "CUA-S1-FORMS option \(index) is empty."
        case .invalidModel(let reason):
            return "Invalid CUA-S1-FORMS model: \(reason)"
        case .invalidOutput(let reason):
            return "Invalid CUA-S1-FORMS output: \(reason)"
        }
    }
}

/// Scores for the supplied options, in their original order.
///
/// Probabilities are model outputs, not guarantees of correctness. This result
/// describes one form decision; it does not execute or authorize a GUI action.
public struct CuaS1FormsResult: Sendable {
    /// Zero-based index of the highest-probability supplied option.
    public let selectedIndex: Int
    /// The original, untruncated option string at `selectedIndex`.
    public let selectedOption: String
    /// One probability per supplied option; padding is omitted.
    public let probabilities: [Float]
    /// One raw score per supplied option; padding is omitted.
    public let logits: [Float]
    /// Whether context encoding exceeded the model's 224-byte input limit.
    public let contextWasTruncated: Bool
    /// Indices of options whose encoding exceeded the model's 96-byte input limit.
    public let truncatedOptionIndices: [Int]
}
