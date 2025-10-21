import Accelerate
import Foundation

/// Thin wrapper around common vDSP routines used by the offline diarization
/// pipeline. Centralising this logic keeps the clustering implementation more
/// readable while guaranteeing we stay away from unsafe pointer juggling in
/// the hot path.
enum VDSPOperations {

    private static let epsilon: Float = 1e-12

    static func l2Normalize(_ input: [Float]) -> [Float] {
        guard !input.isEmpty else { return input }

        var dot: Float = 0
        vDSP_dotpr(input, 1, input, 1, &dot, vDSP_Length(input.count))
        let norm = max(sqrt(dot), epsilon)
        var scale = 1 / norm

        var output = [Float](repeating: 0, count: input.count)
        vDSP_vsmul(input, 1, &scale, &output, 1, vDSP_Length(input.count))
        return output
    }

    static func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count, "Vectors must have the same dimension")
        var dot: Float = 0
        vDSP_dotpr(lhs, 1, rhs, 1, &dot, vDSP_Length(lhs.count))
        return dot
    }

    static func matrixVectorMultiply(matrix: [[Float]], vector: [Float]) -> [Float] {
        guard let columns = matrix.first?.count else { return [] }
        precondition(columns == vector.count, "Dimension mismatch")

        return matrix.map { row in
            precondition(row.count == vector.count, "Jagged matrix not supported")
            return dotProduct(row, vector)
        }
    }

    static func matrixMultiply(a: [[Float]], b: [[Float]]) -> [[Float]] {
        guard
            let aColumns = a.first?.count,
            !a.isEmpty,
            !b.isEmpty
        else {
            return []
        }

        precondition(
            aColumns == b.count,
            "Inner dimensions must match for matrix multiplication"
        )

        let rowsA = a.count
        let columnsB = b.first!.count

        // Pre-compute columns of B to avoid repeated gathering
        var columns: [[Float]] = Array(
            repeating: Array(repeating: 0, count: b.count),
            count: columnsB
        )

        for rowIndex in 0..<b.count {
            let row = b[rowIndex]
            precondition(row.count == columnsB, "Jagged matrix not supported")
            for columnIndex in 0..<columnsB {
                columns[columnIndex][rowIndex] = row[columnIndex]
            }
        }

        var result = Array(
            repeating: Array(repeating: 0 as Float, count: columnsB),
            count: rowsA
        )

        for rowIndex in 0..<rowsA {
            for columnIndex in 0..<columnsB {
                result[rowIndex][columnIndex] = dotProduct(a[rowIndex], columns[columnIndex])
            }
        }

        return result
    }

    static func logSumExp(_ input: [Float]) -> Float {
        guard let maxElement = input.max() else { return -Float.infinity }
        var sum: Float = 0
        let shift = -maxElement
        var mutableShift = shift

        var shifted = [Float](repeating: 0, count: input.count)
        vDSP_vsadd(input, 1, &mutableShift, &shifted, 1, vDSP_Length(input.count))
        var count = Int32(input.count)
        vvexpf(&shifted, shifted, &count)
        vDSP_sve(shifted, 1, &sum, vDSP_Length(input.count))

        return log(sum) + maxElement
    }

    static func softmax(_ input: [Float]) -> [Float] {
        guard let maxElement = input.max() else { return [] }

        let shift = -maxElement
        var mutableShift = shift
        var shifted = [Float](repeating: 0, count: input.count)
        var sum: Float = 0

        vDSP_vsadd(input, 1, &mutableShift, &shifted, 1, vDSP_Length(input.count))
        var count = Int32(input.count)
        vvexpf(&shifted, shifted, &count)
        vDSP_sve(shifted, 1, &sum, vDSP_Length(input.count))

        guard sum > 0 else {
            return Array(repeating: 1.0 / Float(input.count), count: input.count)
        }

        var scale = 1 / sum
        vDSP_vsmul(shifted, 1, &scale, &shifted, 1, vDSP_Length(input.count))
        return shifted
    }

    static func sum(_ input: [Float]) -> Float {
        guard !input.isEmpty else { return 0 }
        var total: Float = 0
        vDSP_sve(input, 1, &total, vDSP_Length(input.count))
        return total
    }

    static func sum(_ input: [Double]) -> Double {
        guard !input.isEmpty else { return 0 }
        var total: Double = 0
        vDSP_sveD(input, 1, &total, vDSP_Length(input.count))
        return total
    }

    static func pairwiseEuclideanDistances(a: [[Float]], b: [[Float]]) -> [[Float]] {
        guard let dimension = a.first?.count, dimension == b.first?.count else {
            return []
        }

        let rowsA = a.count
        let rowsB = b.count

        var result = Array(
            repeating: Array(repeating: 0 as Float, count: rowsB),
            count: rowsA
        )

        for i in 0..<rowsA {
            for j in 0..<rowsB {
                var sum: Float = 0
                for k in 0..<dimension {
                    let diff = a[i][k] - b[j][k]
                    sum += diff * diff
                }
                result[i][j] = sqrt(sum)
            }
        }

        return result
    }
}
