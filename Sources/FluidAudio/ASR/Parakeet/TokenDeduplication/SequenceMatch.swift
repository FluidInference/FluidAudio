import Foundation

/// Represents a matching sequence between two token arrays
struct SequenceMatch {
    let leftStartIndex: Int
    let rightStartIndex: Int
    let length: Int

    var leftRange: Range<Int> {
        leftStartIndex..<(leftStartIndex + length)
    }

    var rightRange: Range<Int> {
        rightStartIndex..<(rightStartIndex + length)
    }
}
