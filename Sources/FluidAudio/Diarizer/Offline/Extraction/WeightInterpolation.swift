import Foundation

/// Utility helpers to resample soft VAD weights with the same half-pixel offset
/// mapping used by `scipy.ndimage.zoom`. The offline diarization pipeline relies
/// on these helpers to match the reference implementation produced by the
/// Pyannote/Core ML exporters when interpolating segmentation masks.
enum WeightInterpolation {

    /// Pre-computed interpolation metadata that allows repeated resampling with
    /// minimal per-call overhead.
    struct InterpolationCoefficients {
        private let leftIndices: [Int]
        private let rightIndices: [Int]
        private let leftWeights: [Float]
        private let rightWeights: [Float]
        private let maxInputIndex: Int
        private let outputLength: Int

        init(inputLength: Int, outputLength: Int) {
            precondition(inputLength > 0 && outputLength > 0, "Lengths must be positive")

            var left: [Int] = []
            var right: [Int] = []
            var lWeights: [Float] = []
            var rWeights: [Float] = []
            var maxIndex = 0

            let scale = Float(outputLength) / Float(inputLength)
            left.reserveCapacity(outputLength)
            right.reserveCapacity(outputLength)
            lWeights.reserveCapacity(outputLength)
            rWeights.reserveCapacity(outputLength)

            for index in 0..<outputLength {
                // Half-pixel offset mapping to match scipy.ndimage.zoom(order=1)
                let position = (Float(index) + 0.5) / scale - 0.5
                let clamped = min(max(position, 0), Float(inputLength - 1))

                let leftIndex = Int(floor(clamped))
                let rightIndex = min(leftIndex + 1, inputLength - 1)
                let weightRight = clamped - Float(leftIndex)
                let weightLeft = 1 - weightRight

                maxIndex = max(maxIndex, rightIndex)

                left.append(leftIndex)
                right.append(rightIndex)
                lWeights.append(weightLeft)
                rWeights.append(weightRight)
            }

            self.leftIndices = left
            self.rightIndices = right
            self.leftWeights = lWeights
            self.rightWeights = rWeights
            self.maxInputIndex = maxIndex
            self.outputLength = outputLength
        }

        func interpolate(_ input: [Float]) -> [Float] {
            guard !input.isEmpty else { return [] }
            precondition(input.count > maxInputIndex, "Input shorter than interpolation map")

            var output = [Float](repeating: 0, count: outputLength)
            for index in 0..<outputLength {
                let leftValue = input[leftIndices[index]]
                let rightValue = input[rightIndices[index]]
                output[index] = leftValue * leftWeights[index] + rightValue * rightWeights[index]
            }
            return output
        }

        func interpolate(_ input: [Float], into output: inout [Float]) {
            guard !input.isEmpty else {
                if output.count == outputLength {
                    for index in 0..<outputLength {
                        output[index] = 0
                    }
                } else {
                    output = [Float](repeating: 0, count: outputLength)
                }
                return
            }

            precondition(input.count > maxInputIndex, "Input shorter than interpolation map")

            if output.count != outputLength {
                output = [Float](repeating: 0, count: outputLength)
            }

            for index in 0..<outputLength {
                let leftValue = input[leftIndices[index]]
                let rightValue = input[rightIndices[index]]
                output[index] = leftValue * leftWeights[index] + rightValue * rightWeights[index]
            }
        }
    }

    /// Resample a 1-dimensional array to the requested length.
    static func resample(_ input: [Float], to outputLength: Int) -> [Float] {
        guard !input.isEmpty, outputLength > 0 else {
            return []
        }

        if input.count == outputLength {
            return input
        }

        let coefficients = InterpolationCoefficients(
            inputLength: input.count,
            outputLength: outputLength
        )

        return coefficients.interpolate(input)
    }

    /// Resample each row independently using the same interpolation map.
    static func resample2D(_ input: [[Float]], to outputLength: Int) -> [[Float]] {
        guard let firstRow = input.first, !firstRow.isEmpty, outputLength > 0 else {
            return []
        }

        let coefficients = InterpolationCoefficients(
            inputLength: firstRow.count,
            outputLength: outputLength
        )

        return input.map { row in
            if row.count == firstRow.count {
                return coefficients.interpolate(row)
            } else {
                // Fallback to per-row computation if the layout differs
                return resample(row, to: outputLength)
            }
        }
    }

    /// Convenience helper mirroring `scipy.ndimage.zoom` for linear interpolation.
    static func zoom(_ input: [Float], factor: Float) -> [Float] {
        guard !input.isEmpty, factor > 0 else {
            return []
        }

        let outputLength = max(1, Int(round(Float(input.count) * factor)))
        return resample(input, to: outputLength)
    }
}
