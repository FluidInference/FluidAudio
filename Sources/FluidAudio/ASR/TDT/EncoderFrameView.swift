import Accelerate
import CoreML
import Foundation

/// Lightweight view over encoder frames that preserves original strides for zero-copy access.
/// Provides contiguous frame vectors on demand without materializing intermediate arrays.
struct EncoderFrameView {
    let hiddenSize: Int
    let count: Int

    private let array: MLMultiArray
    private let timeAxis: Int
    private let hiddenAxis: Int
    private let timeStride: Int
    private let hiddenStride: Int
    private let timeBaseOffset: Int
    private let basePointer: UnsafeMutablePointer<Float>

    init(encoderOutput: MLMultiArray, validLength: Int) throws {
        let shape = encoderOutput.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid encoder output shape: \(shape)")
        }
        guard shape[0] == 1 else {
            throw ASRError.processingFailed("Unsupported batch dimension: \(shape[0])")
        }

        let hiddenSize = ASRConstants.encoderHiddenSize
        let axis1MatchesHidden = shape[1] == hiddenSize
        let axis2MatchesHidden = shape[2] == hiddenSize
        guard axis1MatchesHidden || axis2MatchesHidden else {
            throw ASRError.processingFailed("Encoder hidden size mismatch: \(shape)")
        }

        self.hiddenAxis = axis1MatchesHidden ? 1 : 2
        self.timeAxis = axis1MatchesHidden ? 2 : 1
        self.hiddenSize = hiddenSize

        let strides = encoderOutput.strides.map { $0.intValue }
        self.hiddenStride = strides[self.hiddenAxis]
        self.timeStride = strides[self.timeAxis]

        let availableFrames = shape[self.timeAxis]
        self.count = min(validLength, availableFrames)
        guard count > 0 else {
            throw ASRError.processingFailed("Encoder output has no frames")
        }
        self.array = encoderOutput

        guard encoderOutput.dataType == .float32 else {
            throw ASRError.processingFailed("Unsupported encoder output type: \(encoderOutput.dataType)")
        }

        self.basePointer = encoderOutput.dataPointer.bindMemory(
            to: Float.self, capacity: encoderOutput.count)

        if timeStride >= 0 {
            self.timeBaseOffset = 0
        } else {
            self.timeBaseOffset = (availableFrames - 1) * timeStride
        }
    }

    func copyFrame(
        at index: Int,
        into destination: UnsafeMutablePointer<Float>,
        destinationStride: Int
    ) throws {
        guard index >= 0 && index < count else {
            throw ASRError.processingFailed("Encoder frame index out of range: \(index)")
        }

        let frameOffset = timeBaseOffset + index * timeStride
        let frameStart = basePointer.advanced(by: frameOffset)

        guard hiddenStride != 0 else {
            throw ASRError.processingFailed("Invalid hidden stride: 0")
        }
        let sourcePointer = UnsafePointer<Float>(frameStart)
        let count = try makeBlasIndex(hiddenSize, label: "Hidden size")
        let incX = try makeBlasIndex(hiddenStride, label: "Hidden stride")
        let destStrideCblas = try makeBlasIndex(destinationStride, label: "Destination stride")

        if hiddenStride == 1 && destinationStride == 1 {
            destination.update(from: sourcePointer, count: hiddenSize)
        } else {
            cblas_scopy(count, sourcePointer, incX, destination, destStrideCblas)
        }
    }
}
