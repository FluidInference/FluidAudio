import Foundation

/// Decode the emitted logits with a stable softmax, retaining the model's raw probabilities.
struct CuaS1FormsOutput {
    let selectedIndex: Int
    let logits: [Float]
    let probabilities: [Float]
    let rawProbabilities: [Float]

    init(logits: [Float], rawProbabilities: [Float], optionCount: Int) throws {
        let capacity = CuaS1FormsInput.maximumOptions
        guard (2...capacity).contains(optionCount), logits.count == capacity, rawProbabilities.count == capacity else {
            throw CuaS1FormsError.invalidOutput("Unexpected output size")
        }
        guard logits.allSatisfy(\.isFinite), rawProbabilities.allSatisfy(\.isFinite) else {
            throw CuaS1FormsError.invalidOutput("Model outputs contain a nonfinite value")
        }
        guard rawProbabilities.allSatisfy({ $0 >= 0 && $0 <= 1 }) else {
            throw CuaS1FormsError.invalidOutput("Probabilities must be between zero and one")
        }
        guard rawProbabilities.dropFirst(optionCount).allSatisfy({ $0 == 0 }) else {
            throw CuaS1FormsError.invalidOutput("Padded options received probability mass")
        }
        let liveLogits = Array(logits.prefix(optionCount))
        guard let maximum = liveLogits.max() else {
            throw CuaS1FormsError.invalidOutput("No live logits")
        }
        // Subtract before exponentiation to prevent overflow. Double arithmetic
        // avoids the FP16 model softmax's observed probability-sum rounding error.
        let weights = liveLogits.map { exp(Double($0) - Double(maximum)) }
        let total = weights.reduce(0, +)
        let probabilities = weights.map { Float($0 / total) }
        var selected = 0
        for index in probabilities.indices.dropFirst() where probabilities[index] > probabilities[selected] {
            selected = index
        }
        self.selectedIndex = selected
        self.logits = liveLogits
        self.probabilities = probabilities
        self.rawProbabilities = Array(rawProbabilities.prefix(optionCount))
    }
}
