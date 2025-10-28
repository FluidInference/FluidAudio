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
        guard let columns = matrix.first?.count, !matrix.isEmpty else { return [] }
        precondition(columns == vector.count, "Dimension mismatch")
        if columns == 0 {
            return [Float](repeating: 0, count: matrix.count)
        }

        let flatMatrix = matrix.flatMap { row in
            precondition(row.count == columns, "Jagged matrix not supported")
            return row
        }

        var result = [Float](repeating: 0, count: matrix.count)
        flatMatrix.withUnsafeBufferPointer { matrixPointer in
            vector.withUnsafeBufferPointer { vectorPointer in
                result.withUnsafeMutableBufferPointer { resultPointer in
                    cblas_sgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        Int32(matrix.count),
                        Int32(columns),
                        1.0,
                        matrixPointer.baseAddress!,
                        Int32(columns),
                        vectorPointer.baseAddress!,
                        1,
                        0.0,
                        resultPointer.baseAddress!,
                        1
                    )
                }
            }
        }

        return result
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

        if aColumns == 0 || b.first?.isEmpty == true {
            return Array(
                repeating: Array(repeating: 0 as Float, count: b.first?.count ?? 0),
                count: a.count
            )
        }

        let rowsA = a.count
        let columnsB = b.first!.count

        let flatA = a.flatMap { row in
            precondition(row.count == aColumns, "Jagged matrix not supported")
            return row
        }

        let flatB = b.flatMap { row in
            precondition(row.count == columnsB, "Jagged matrix not supported")
            return row
        }

        var flatResult = [Float](repeating: 0, count: rowsA * columnsB)
        flatA.withUnsafeBufferPointer { aPointer in
            flatB.withUnsafeBufferPointer { bPointer in
                flatResult.withUnsafeMutableBufferPointer { resultPointer in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        Int32(rowsA),
                        Int32(columnsB),
                        Int32(aColumns),
                        1.0,
                        aPointer.baseAddress!,
                        Int32(aColumns),
                        bPointer.baseAddress!,
                        Int32(columnsB),
                        0.0,
                        resultPointer.baseAddress!,
                        Int32(columnsB)
                    )
                }
            }
        }

        var result = Array(
            repeating: Array(repeating: 0 as Float, count: columnsB),
            count: rowsA
        )

        for rowIndex in 0..<rowsA {
            let base = rowIndex * columnsB
            for columnIndex in 0..<columnsB {
                result[rowIndex][columnIndex] = flatResult[base + columnIndex]
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

        if rowsA == 0 || rowsB == 0 || dimension == 0 {
            return Array(
                repeating: Array(repeating: 0 as Float, count: rowsB),
                count: rowsA
            )
        }

        let flatA = a.flatMap { row in
            precondition(row.count == dimension, "Jagged matrix not supported")
            return row
        }

        let flatB = b.flatMap { row in
            precondition(row.count == dimension, "Jagged matrix not supported")
            return row
        }

        var normsA = [Float](repeating: 0, count: rowsA)
        var normsB = [Float](repeating: 0, count: rowsB)

        flatA.withUnsafeBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            for row in 0..<rowsA {
                vDSP_svesq(
                    base.advanced(by: row * dimension),
                    1,
                    &normsA[row],
                    vDSP_Length(dimension)
                )
            }
        }

        flatB.withUnsafeBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            for row in 0..<rowsB {
                vDSP_svesq(
                    base.advanced(by: row * dimension),
                    1,
                    &normsB[row],
                    vDSP_Length(dimension)
                )
            }
        }

        var dotProducts = [Float](repeating: 0, count: rowsA * rowsB)
        flatA.withUnsafeBufferPointer { aPointer in
            flatB.withUnsafeBufferPointer { bPointer in
                dotProducts.withUnsafeMutableBufferPointer { resultPointer in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        Int32(rowsA),
                        Int32(rowsB),
                        Int32(dimension),
                        1.0,
                        aPointer.baseAddress!,
                        Int32(dimension),
                        bPointer.baseAddress!,
                        Int32(dimension),
                        0.0,
                        resultPointer.baseAddress!,
                        Int32(rowsB)
                    )
                }
            }
        }

        var negativeTwo: Float = -2
        let dotElementCount = dotProducts.count
        dotProducts.withUnsafeMutableBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress else { return }
            vDSP_vsmul(
                baseAddress,
                1,
                &negativeTwo,
                baseAddress,
                1,
                vDSP_Length(dotElementCount)
            )
        }

        normsB.withUnsafeBufferPointer { normsBPointer in
            guard let normsBBase = normsBPointer.baseAddress else { return }
            dotProducts.withUnsafeMutableBufferPointer { pointer in
                guard let baseAddress = pointer.baseAddress else { return }
                for rowIndex in 0..<rowsA {
                    let rowPointer = baseAddress.advanced(by: rowIndex * rowsB)
                    var normA = normsA[rowIndex]
                    vDSP_vsadd(
                        rowPointer,
                        1,
                        &normA,
                        rowPointer,
                        1,
                        vDSP_Length(rowsB)
                    )
                    vDSP_vadd(
                        rowPointer,
                        1,
                        normsBBase,
                        1,
                        rowPointer,
                        1,
                        vDSP_Length(rowsB)
                    )
                }
            }
        }

        var zero: Float = 0
        dotProducts.withUnsafeMutableBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress else { return }
            vDSP_vthres(
                baseAddress,
                1,
                &zero,
                baseAddress,
                1,
                vDSP_Length(dotElementCount)
            )
        }

        dotProducts.withUnsafeMutableBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress else { return }
            var elementCount = Int32(pointer.count)
            vvsqrtf(baseAddress, baseAddress, &elementCount)
        }

        var result = Array(
            repeating: Array(repeating: 0 as Float, count: rowsB),
            count: rowsA
        )

        for rowIndex in 0..<rowsA {
            let base = rowIndex * rowsB
            for columnIndex in 0..<rowsB {
                result[rowIndex][columnIndex] = dotProducts[base + columnIndex]
            }
        }

        return result
    }
}
