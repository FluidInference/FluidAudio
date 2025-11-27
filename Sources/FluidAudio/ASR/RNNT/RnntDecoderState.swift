import Accelerate
import CoreML
import Foundation

/// Manages LSTM hidden and cell states for the RNNT decoder (Parakeet EOU)
/// This model uses a 1-layer LSTM with 640 hidden units
struct RnntDecoderState {
    /// LSTM hidden state [1, 1, 640]
    var hiddenState: MLMultiArray

    /// LSTM cell state [1, 1, 640]
    var cellState: MLMultiArray

    /// Last decoded token from previous chunk for context continuity
    var lastToken: Int?

    /// Cached decoder output for optimization
    var predictorOutput: MLMultiArray?

    /// Whether end-of-utterance was detected
    var eouDetected: Bool = false

    /// Hidden size for this decoder (640 for Parakeet EOU)
    private let hiddenSize: Int

    init(hiddenSize: Int = 640) throws {
        self.hiddenSize = hiddenSize

        // EOU model uses 1-layer LSTM with shape [1, 1, hiddenSize]
        hiddenState = try ANEOptimizer.createANEAlignedArray(
            shape: [1, 1, NSNumber(value: hiddenSize)],
            dataType: .float32
        )
        cellState = try ANEOptimizer.createANEAlignedArray(
            shape: [1, 1, NSNumber(value: hiddenSize)],
            dataType: .float32
        )

        // Initialize to zeros
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
    }

    static func make(hiddenSize: Int = 640) -> RnntDecoderState {
        do {
            return try RnntDecoderState(hiddenSize: hiddenSize)
        } catch {
            fatalError("Failed to allocate RNNT decoder state: \(error)")
        }
    }

    mutating func update(from decoderOutput: MLFeatureProvider) {
        if let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue {
            hiddenState = hOut
        }
        if let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue {
            cellState = cOut
        }
    }

    init(from other: RnntDecoderState) throws {
        self.hiddenSize = other.hiddenSize
        hiddenState = try MLMultiArray(shape: other.hiddenState.shape, dataType: .float32)
        cellState = try MLMultiArray(shape: other.cellState.shape, dataType: .float32)
        lastToken = other.lastToken
        eouDetected = other.eouDetected

        hiddenState.copyData(from: other.hiddenState)
        cellState.copyData(from: other.cellState)

        if let pred = other.predictorOutput {
            predictorOutput = try MLMultiArray(shape: pred.shape, dataType: .float32)
            predictorOutput?.copyData(from: pred)
        }
    }

    /// Reset all state variables to initial values
    mutating func reset() {
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
        lastToken = nil
        predictorOutput = nil
        eouDetected = false
    }

    /// Mark end-of-utterance detected and reset for next utterance
    mutating func markEndOfUtterance() {
        eouDetected = true
        // Reset LSTM state for fresh utterance
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
        lastToken = nil
        predictorOutput = nil
    }
}
