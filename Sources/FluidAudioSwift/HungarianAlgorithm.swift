import Foundation

/// Hungarian Algorithm implementation for optimal assignment problems
/// Used for finding minimum cost assignment between predicted and ground truth speakers
public struct HungarianAlgorithm {
    
    /// Solve the assignment problem using Hungarian Algorithm
    /// - Parameter costMatrix: Matrix where costMatrix[i][j] is cost of assigning row i to column j
    /// - Returns: Array of (row, column) pairs representing optimal assignment
    public static func solve(costMatrix: [[Float]]) -> [(row: Int, col: Int)] {
        guard !costMatrix.isEmpty, !costMatrix[0].isEmpty else {
            return []
        }
        
        let result = minimumCostAssignment(costs: costMatrix)
        var assignments: [(row: Int, col: Int)] = []
        
        for (row, col) in result.assignments.enumerated() {
            if col != -1 {  // -1 indicates unassigned
                assignments.append((row: row, col: col))
            }
        }
        
        return assignments
    }
    
    /// Find minimum cost assignment using Hungarian Algorithm
    /// - Parameter costs: Cost matrix (rows = workers, cols = tasks)
    /// - Returns: Tuple with assignments array and total cost
    public static func minimumCostAssignment(costs: [[Float]]) -> (assignments: [Int], totalCost: Float) {
        guard !costs.isEmpty, !costs[0].isEmpty else {
            return ([], 0.0)
        }
        
        let rows = costs.count
        let cols = costs[0].count
        let size = max(rows, cols)
        
        // Create square matrix padded with zeros
        var matrix = Array(repeating: Array(repeating: Float(0), count: size), count: size)
        for i in 0..<rows {
            for j in 0..<cols {
                matrix[i][j] = costs[i][j]
            }
        }
        
        // Step 1: Subtract row minimums
        for i in 0..<size {
            let rowMin = matrix[i].min() ?? 0
            for j in 0..<size {
                matrix[i][j] -= rowMin
            }
        }
        
        // Step 2: Subtract column minimums
        for j in 0..<size {
            var colMin = Float.greatestFiniteMagnitude
            for i in 0..<size {
                colMin = min(colMin, matrix[i][j])
            }
            for i in 0..<size {
                matrix[i][j] -= colMin
            }
        }
        
        // Track assignments and coverage
        var assignments = Array(repeating: -1, count: size)
        var revAssignments = Array(repeating: -1, count: size)
        
        // Try to find initial assignments using zeros
        for _ in 0..<size {
            var rowCovered = Array(repeating: false, count: size)
            var colCovered = Array(repeating: false, count: size)
            
            // Mark current assignments as covered
            for i in 0..<size {
                if assignments[i] != -1 {
                    rowCovered[i] = true
                    colCovered[assignments[i]] = true
                }
            }
            
            // Try to find augmenting paths
            var improved = false
            for i in 0..<size {
                if assignments[i] == -1 && !rowCovered[i] {
                    if findAugmentingPath(matrix: matrix, row: i, 
                                        assignments: &assignments, 
                                        revAssignments: &revAssignments,
                                        rowCovered: &rowCovered, 
                                        colCovered: &colCovered) {
                        improved = true
                        break
                    }
                }
            }
            
            if !improved {
                break
            }
        }
        
        // If we don't have a complete assignment, use the minimum weight approach
        while assignments.filter({ $0 != -1 }).count < min(rows, cols) {
            var rowCovered = Array(repeating: false, count: size)
            var colCovered = Array(repeating: false, count: size)
            
            // Mark assigned rows and columns as covered
            for i in 0..<size {
                if assignments[i] != -1 {
                    rowCovered[i] = true
                    colCovered[assignments[i]] = true
                }
            }
            
            // Cover all rows with assignments
            for i in 0..<size {
                if assignments[i] != -1 {
                    rowCovered[i] = true
                }
            }
            
            // Cover columns using minimum vertex cover
            coverColumnsOptimally(matrix: matrix, 
                                rowCovered: &rowCovered, 
                                colCovered: &colCovered)
            
            // Find minimum uncovered value
            var minUncovered = Float.greatestFiniteMagnitude
            for i in 0..<size {
                for j in 0..<size {
                    if !rowCovered[i] && !colCovered[j] {
                        minUncovered = min(minUncovered, matrix[i][j])
                    }
                }
            }
            
            if minUncovered == Float.greatestFiniteMagnitude {
                break
            }
            
            // Adjust matrix: subtract from uncovered, add to double-covered
            for i in 0..<size {
                for j in 0..<size {
                    if !rowCovered[i] && !colCovered[j] {
                        matrix[i][j] -= minUncovered
                    } else if rowCovered[i] && colCovered[j] {
                        matrix[i][j] += minUncovered
                    }
                }
            }
            
            // Try to improve assignment
            var improved = false
            for i in 0..<size {
                if assignments[i] == -1 {
                    rowCovered = Array(repeating: false, count: size)
                    colCovered = Array(repeating: false, count: size)
                    if findAugmentingPath(matrix: matrix, row: i,
                                        assignments: &assignments,
                                        revAssignments: &revAssignments,
                                        rowCovered: &rowCovered,
                                        colCovered: &colCovered) {
                        improved = true
                        break
                    }
                }
            }
            
            if !improved {
                break
            }
        }
        
        // Calculate total cost and filter assignments for original matrix size
        var totalCost: Float = 0
        var finalAssignments = Array(repeating: -1, count: rows)
        
        for i in 0..<rows {
            if assignments[i] != -1 && assignments[i] < cols {
                finalAssignments[i] = assignments[i]
                totalCost += costs[i][assignments[i]]
            }
        }
        
        return (finalAssignments, totalCost)
    }
    
    /// Find augmenting path for assignment improvement
    private static func findAugmentingPath(matrix: [[Float]], row: Int,
                                         assignments: inout [Int],
                                         revAssignments: inout [Int],
                                         rowCovered: inout [Bool],
                                         colCovered: inout [Bool]) -> Bool {
        
        // Look for zeros in current row
        for col in 0..<matrix[row].count {
            if matrix[row][col] == 0 && !colCovered[col] {
                colCovered[col] = true
                
                // If column is unassigned or we can find alternating path
                if revAssignments[col] == -1 || 
                   findAugmentingPath(matrix: matrix, row: revAssignments[col],
                                    assignments: &assignments,
                                    revAssignments: &revAssignments,
                                    rowCovered: &rowCovered,
                                    colCovered: &colCovered) {
                    
                    // Update assignments
                    if revAssignments[col] != -1 {
                        assignments[revAssignments[col]] = -1
                    }
                    assignments[row] = col
                    revAssignments[col] = row
                    return true
                }
            }
        }
        
        return false
    }
    
    /// Cover columns optimally for minimum vertex cover
    private static func coverColumnsOptimally(matrix: [[Float]],
                                            rowCovered: inout [Bool],
                                            colCovered: inout [Bool]) {
        let size = matrix.count
        
        // Simple greedy approach: cover columns that have zeros in uncovered rows
        for i in 0..<size {
            if !rowCovered[i] {
                for j in 0..<size {
                    if matrix[i][j] == 0 && !colCovered[j] {
                        colCovered[j] = true
                    }
                }
            }
        }
    }
}

// MARK: - Speaker Assignment Utilities

extension HungarianAlgorithm {
    
    /// Convert speaker overlap matrix to cost matrix for Hungarian algorithm
    /// - Parameter overlapMatrix: Matrix where higher values = better overlap
    /// - Returns: Cost matrix where lower values = better assignment
    public static func overlapToCostMatrix(_ overlapMatrix: [[Int]]) -> [[Float]] {
        guard !overlapMatrix.isEmpty, !overlapMatrix[0].isEmpty else {
            return []
        }
        
        // Find maximum overlap to convert to cost (cost = max - overlap)
        let maxOverlap = overlapMatrix.flatMap { $0 }.max() ?? 0
        
        return overlapMatrix.map { row in
            row.map { overlap in
                Float(maxOverlap - overlap)
            }
        }
    }
    
    /// Create assignment mapping from Hungarian algorithm result
    /// - Parameters:
    ///   - assignments: Result from Hungarian algorithm
    ///   - predSpeakers: Array of predicted speaker IDs
    ///   - gtSpeakers: Array of ground truth speaker IDs
    /// - Returns: Dictionary mapping predicted speaker ID to ground truth speaker ID
    public static func createSpeakerMapping(assignments: [Int],
                                          predSpeakers: [String],
                                          gtSpeakers: [String]) -> [String: String] {
        var mapping: [String: String] = [:]
        
        for (predIndex, gtIndex) in assignments.enumerated() {
            if gtIndex != -1 && predIndex < predSpeakers.count && gtIndex < gtSpeakers.count {
                mapping[predSpeakers[predIndex]] = gtSpeakers[gtIndex]
            }
        }
        
        return mapping
    }
}