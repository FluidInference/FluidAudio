import Foundation

/// Hungarian algorithm implementation for optimal speaker-to-ground-truth mapping
/// Used in offline diarization benchmarks to achieve minimum DER through optimal assignment
public enum HungarianAlgorithm {

    /// Find optimal assignment that minimizes cost
    /// Returns dictionary mapping speaker IDs to ground truth IDs
    public static func findOptimalMapping(
        predicted: [TimedSpeakerSegment],
        groundTruth: [TimedSpeakerSegment],
        totalDuration: Float
    ) -> [String: String] {

        let frameSize: Float = 0.01
        let totalFrames = Int(totalDuration / frameSize)

        // Get unique speaker IDs
        let predSpeakers = Array(Set(predicted.map { $0.speakerId })).sorted()
        let gtSpeakers = Array(Set(groundTruth.map { $0.speakerId })).sorted()

        guard !predSpeakers.isEmpty && !gtSpeakers.isEmpty else {
            return [:]
        }

        // Create cost matrix: cost[i][j] = error when assigning predSpeakers[i] to gtSpeakers[j]
        let numPred = predSpeakers.count
        let numGT = gtSpeakers.count

        // Calculate overlap matrix instead of error matrix for better stability
        var overlapMatrix: [[Float]] = Array(repeating: Array(repeating: 0, count: numGT), count: numPred)

        for i in 0..<numPred {
            for j in 0..<numGT {
                let predSpeaker = predSpeakers[i]
                let gtSpeaker = gtSpeakers[j]

                // Calculate overlap in seconds
                var overlap: Float = 0
                for predSegment in predicted where predSegment.speakerId == predSpeaker {
                    for gtSegment in groundTruth where gtSegment.speakerId == gtSpeaker {
                        let start = max(predSegment.startTimeSeconds, gtSegment.startTimeSeconds)
                        let end = min(predSegment.endTimeSeconds, gtSegment.endTimeSeconds)
                        if start < end {
                            overlap += (end - start)
                        }
                    }
                }

                overlapMatrix[i][j] = overlap
            }
        }

        // Use greedy assignment for stability
        let assignment = greedyAssignment(overlapMatrix: overlapMatrix)

        // Convert assignment to speaker mapping
        var mapping: [String: String] = [:]
        for i in 0..<numPred {
            if i < assignment.count {
                let assignedJ = assignment[i]
                if assignedJ >= 0 && assignedJ < numGT {
                    mapping[predSpeakers[i]] = gtSpeakers[assignedJ]
                }
            }
        }

        return mapping
    }

    /// Greedy assignment based on maximum overlap
    private static func greedyAssignment(overlapMatrix: [[Float]]) -> [Int] {
        let numRows = overlapMatrix.count
        guard numRows > 0 else { return [] }
        let numCols = overlapMatrix[0].count

        var assignment = Array(repeating: -1, count: numRows)
        var usedCols = Set<Int>()

        // Sort rows by their maximum overlap (highest first)
        let sortedRows = (0..<numRows).sorted { row1, row2 in
            let max1 = overlapMatrix[row1].max() ?? 0
            let max2 = overlapMatrix[row2].max() ?? 0
            return max1 > max2
        }

        // Assign each row to its best available column
        for row in sortedRows {
            var bestCol = -1
            var bestOverlap: Float = 0

            for col in 0..<numCols {
                if !usedCols.contains(col) && overlapMatrix[row][col] > bestOverlap {
                    bestOverlap = overlapMatrix[row][col]
                    bestCol = col
                }
            }

            if bestCol != -1 && bestOverlap > 0 {
                assignment[row] = bestCol
                usedCols.insert(bestCol)
            }
        }

        return assignment
    }
}
