import CoreML
import Foundation

/// Manages LSTM hidden and cell states for the Parakeet decoder
struct TdtDecoderState {
    var hiddenState: MLMultiArray
    var cellState: MLMultiArray
    /// Stores the last decoded token from the previous audio chunk.
    /// Used for maintaining linguistic context across chunk boundaries in streaming ASR.
    /// When processing a new chunk, the decoder starts with this token instead of SOS,
    /// ensuring proper context continuity for real-time transcription.
    var lastToken: Int?

    // Cached CoreML "decoder" output used for the very first Joint at utterance/chunk start.
    // This mirrors NeMo's behavior where SOS == blankId is used only to prime the predictor.
    var predictorOutput: MLMultiArray?

    /// Time jump tracking for streaming TDT decoding.
    /// Represents how far ahead the decoder has progressed relative to encoder frames.
    /// Formula: timeJump = timeIndices - encoderSequenceLength
    /// - nil: First chunk or no streaming context
    /// - negative: Decoder hasn't processed all encoder frames yet
    /// - positive: Decoder has advanced beyond current encoder frames
    /// - zero: Decoder exactly at the end of encoder frames
    var timeJump: Int?

    init() throws {
        // Use ANE-aligned arrays for optimal performance
        let decoderHiddenSize = ASRConstants.decoderHiddenSize
        hiddenState = try ANEOptimizer.createANEAlignedArray(
            shape: [2, 1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )
        cellState = try ANEOptimizer.createANEAlignedArray(
            shape: [2, 1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )

        // Initialize to zeros using Accelerate
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
    }

    static func make() -> TdtDecoderState {
        do {
            return try TdtDecoderState()
        } catch {
            fatalError("Failed to allocate decoder state: \(error)")
        }
    }

    mutating func update(from decoderOutput: MLFeatureProvider) {
        hiddenState = decoderOutput.featureValue(for: "h_out")?.multiArrayValue ?? hiddenState
        cellState = decoderOutput.featureValue(for: "c_out")?.multiArrayValue ?? cellState
    }

    init(from other: TdtDecoderState) throws {
        hiddenState = try MLMultiArray(shape: other.hiddenState.shape, dataType: .float32)
        cellState = try MLMultiArray(shape: other.cellState.shape, dataType: .float32)
        lastToken = other.lastToken
        timeJump = other.timeJump

        hiddenState.copyData(from: other.hiddenState)
        cellState.copyData(from: other.cellState)
    }

    /// Reset all state variables to initial values
    mutating func reset() {
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
        lastToken = nil
        predictorOutput = nil
        timeJump = nil
    }

    /// Finalize the decoder state for the last chunk
    /// Clears cached outputs and streaming-specific state while preserving linguistic context
    mutating func finalizeLastChunk() {
        // Clear predictor output cache to ensure clean state
        predictorOutput = nil

        // Clear time jump since there are no more chunks
        timeJump = nil

        // Keep lastToken for linguistic context - this may be useful for final post-processing
        // Keep LSTM states as they represent the final linguistic context
    }
}

extension MLMultiArray {
    func resetData(to value: NSNumber) {
        for i in 0..<count {
            self[i] = value
        }
    }

    func copyData(from source: MLMultiArray) {
        for i in 0..<count {
            self[i] = source[i]
        }
    }
}
