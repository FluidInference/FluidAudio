import Foundation
import Accelerate
private import LinearAssignmentWrapper

internal func solveRectangularLinearAssignment<C>(
    numRows: Int,
    numCols: Int,
    costMatrix: C,
    maximize: Bool = false
) -> (rows: [Int], cols: [Int])?
where C: AccelerateBuffer, C.Element: LinearAssignmentCost {
    guard costMatrix.count > 0 else { return ([], []) }
    guard costMatrix.count == numRows * numCols else { return nil }
    
    var rows = [Int](repeating: 0, count: numRows)
    var cols = [Int](repeating: 0, count: numCols)
    
    let statusCode = costMatrix.withUnsafeBufferPointer { costPtr in
        C.Element.solveRLAP(numRows: numRows, numCols: numCols,
                            costMatrix: costPtr.baseAddress,
                            maximize: maximize,
                            rowsOut: &rows, colsOut: &cols)
    }
    
    guard statusCode == 0 else { return nil }
    return (rows, cols)
}


internal protocol LinearAssignmentCost {
    static func solveRLAP(
        numRows: Int, numCols: Int,
        costMatrix: UnsafePointer<Self>?,
        maximize: Bool,
        rowsOut: inout [Int], colsOut: inout [Int]
    ) -> Int
}

extension Int: LinearAssignmentCost {
    static func solveRLAP(
        numRows: Int, numCols: Int,
        costMatrix: UnsafePointer<Self>?,
        maximize: Bool,
        rowsOut: inout [Int], colsOut: inout [Int]
    ) -> Int {
        solveRectangularLinearAssignment(
            numRows, numCols,
            costMatrix, maximize,
            &rowsOut, &colsOut
        )
    }
}

extension Float: LinearAssignmentCost {
    static func solveRLAP(
        numRows: Int, numCols: Int,
        costMatrix: UnsafePointer<Self>?,
        maximize: Bool,
        rowsOut: inout [Int], colsOut: inout [Int]
    ) -> Int {
        solveRectangularLinearAssignment(
            numRows, numCols,
            costMatrix, maximize,
            &rowsOut, &colsOut
        )
    }
}

extension Double: LinearAssignmentCost {
    static func solveRLAP(
        numRows: Int, numCols: Int,
        costMatrix: UnsafePointer<Self>?,
        maximize: Bool,
        rowsOut: inout [Int], colsOut: inout [Int]
    ) -> Int {
        solveRectangularLinearAssignment(
            numRows, numCols,
            costMatrix, maximize,
            &rowsOut, &colsOut
        )
    }
}
