@preconcurrency import CoreML
import Foundation

/// Manages LSTM hidden and cell states for the Parakeet decoder
public struct TdtDecoderState: Sendable {
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

    /// Initialize decoder state with specified number of LSTM layers.
    /// - Parameter decoderLayers: Number of decoder LSTM layers (default: 2)
    ///   - v2 and v3 models: 2 layers (default)
    ///   - tdtCtc110m model: 1 layer
    ///   Default of 2 matches the most common Parakeet TDT architecture (v2/v3)
    public init(decoderLayers: Int = 2) throws {
        // Use ANE-aligned arrays for optimal performance
        let decoderHiddenSize = ASRConstants.decoderHiddenSize
        hiddenState = try ANEMemoryUtils.createAlignedArray(
            shape: [NSNumber(value: decoderLayers), 1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )
        cellState = try ANEMemoryUtils.createAlignedArray(
            shape: [NSNumber(value: decoderLayers), 1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )

        // Initialize to zeros using Accelerate
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
    }

    /// Create decoder state with specified number of LSTM layers (cannot throw).
    /// - Parameter decoderLayers: Number of decoder LSTM layers (default: 2)
    ///   Default of 2 matches v2/v3 models. Use 1 for tdtCtc110m.
    public static func make(decoderLayers: Int = 2) -> TdtDecoderState {
        do {
            return try TdtDecoderState(decoderLayers: decoderLayers)
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
    /// Fills every element with `value` through the backing storage, so padded strides are covered.
    /// Zero is a single `memset`; other values fill through a typed pointer where the type allows.
    func resetData(to value: NSNumber) {
        let filled = withUnsafeMutableBytes { bytes, _ -> Bool in
            guard let base = bytes.baseAddress else {
                return true
            }
            if value == 0 {
                memset(base, 0, bytes.count)
                return true
            }
            switch dataType {
            case .float32:
                bytes.bindMemory(to: Float.self).update(repeating: value.floatValue)
            case .float64:
                bytes.bindMemory(to: Double.self).update(repeating: value.doubleValue)
            case .int32:
                bytes.bindMemory(to: Int32.self).update(repeating: value.int32Value)
            default:
                return false
            }
            return true
        }
        if filled {
            return
        }
        for i in 0..<count {
            self[i] = value
        }
    }

    /// Copies every element from `source`. Identical layouts copy the backing storage in one
    /// `memcpy`; anything else goes element by element.
    func copyData(from source: MLMultiArray) {
        let copied = withUnsafeMutableBytes { destination, _ -> Bool in
            source.withUnsafeBytes { origin -> Bool in
                guard dataType == source.dataType, shape == source.shape, strides == source.strides,
                    destination.count == origin.count, let to = destination.baseAddress,
                    let from = origin.baseAddress
                else {
                    return false
                }
                memcpy(to, from, destination.count)
                return true
            }
        }
        if copied {
            return
        }
        for i in 0..<count {
            self[i] = source[i]
        }
    }
}
