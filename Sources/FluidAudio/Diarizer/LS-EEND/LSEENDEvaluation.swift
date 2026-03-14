import Foundation

public struct LSEENDRTTMEntry: Sendable, Codable {
    public let recordingID: String
    public let start: Double
    public let duration: Double
    public let speaker: String
}

public struct LSEENDEvaluationSettings: Sendable, Codable {
    public let threshold: Float
    public let medianWidth: Int
    public let collarSeconds: Double
    public let frameRate: Double

    public init(threshold: Float, medianWidth: Int, collarSeconds: Double, frameRate: Double) {
        self.threshold = threshold
        self.medianWidth = medianWidth
        self.collarSeconds = collarSeconds
        self.frameRate = frameRate
    }
}

public struct LSEENDEvaluationResult: Sendable {
    public let der: Double
    public let speakerScored: Double
    public let speakerMiss: Double
    public let speakerFalseAlarm: Double
    public let speakerError: Double
    public let threshold: Float
    public let medianWidth: Int
    public let collarSeconds: Double
    public let mappedBinary: LSEENDMatrix
    public let mappedProbabilities: LSEENDMatrix
    public let validMask: [Bool]
    public let assignment: [Int: Int]
    public let unmatchedPredictionIndices: [Int]
}

public enum LSEENDEvaluation {
    public static func parseRTTM(url: URL) throws -> (entries: [LSEENDRTTMEntry], speakers: [String]) {
        let text = try String(contentsOf: url, encoding: .utf8)
        var entries: [LSEENDRTTMEntry] = []
        var speakers: [String] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: " ")
            guard parts.count >= 8 else { continue }
            let speaker = String(parts[7])
            if !speakers.contains(speaker) {
                speakers.append(speaker)
            }
            entries.append(
                LSEENDRTTMEntry(
                    recordingID: String(parts[1]),
                    start: Double(parts[3]) ?? 0,
                    duration: Double(parts[4]) ?? 0,
                    speaker: speaker
                )
            )
        }
        return (entries, speakers)
    }

    public static func rttmToFrameMatrix(
        entries: [LSEENDRTTMEntry],
        speakers: [String],
        numFrames: Int,
        frameRate: Double
    ) -> LSEENDMatrix {
        var matrix = LSEENDMatrix.zeros(rows: numFrames, columns: speakers.count)
        let speakerToIndex = Dictionary(uniqueKeysWithValues: speakers.enumerated().map { ($1, $0) })
        for entry in entries {
            guard let speakerIndex = speakerToIndex[entry.speaker] else { continue }
            let start = pythonRoundedInt(entry.start * frameRate)
            let stop = pythonRoundedInt((entry.start + entry.duration) * frameRate)
            guard stop > start else { continue }
            for rowIndex in max(0, start)..<min(numFrames, stop) {
                matrix[rowIndex, speakerIndex] = 1
            }
        }
        return matrix
    }

    public static func writeRTTM(
        recordingID: String,
        binaryPrediction: LSEENDMatrix,
        outputURL: URL,
        frameRate: Double,
        speakerLabels: [String]? = nil
    ) throws {
        let labels = speakerLabels ?? (0..<binaryPrediction.columns).map { "spk\($0)" }
        var lines: [String] = []
        for speakerIndex in 0..<min(labels.count, binaryPrediction.columns) {
            var previous: Float = 0
            var startIndex: Int?
            for rowIndex in 0..<binaryPrediction.rows {
                let value = binaryPrediction[rowIndex, speakerIndex]
                if previous == 0, value > 0 {
                    startIndex = rowIndex
                } else if previous > 0, value == 0, let activeStart = startIndex {
                    let startSeconds = Double(activeStart) / frameRate
                    let durationSeconds = Double(rowIndex - activeStart) / frameRate
                    lines.append(
                        String(
                            format: "SPEAKER %@ 1 %.3f %.3f <NA> <NA> %@ <NA> <NA>",
                            recordingID,
                            startSeconds,
                            durationSeconds,
                            labels[speakerIndex]
                        )
                    )
                    startIndex = nil
                }
                previous = value
            }
            if previous > 0, let activeStart = startIndex {
                let startSeconds = Double(activeStart) / frameRate
                let durationSeconds = Double(binaryPrediction.rows - activeStart) / frameRate
                lines.append(
                    String(
                        format: "SPEAKER %@ 1 %.3f %.3f <NA> <NA> %@ <NA> <NA>",
                        recordingID,
                        startSeconds,
                        durationSeconds,
                        labels[speakerIndex]
                    )
                )
            }
        }
        try lines.joined(separator: "\n").appending("\n").write(to: outputURL, atomically: true, encoding: .utf8)
    }

    public static func collarMask(reference: LSEENDMatrix, collarFrames: Int) -> [Bool] {
        guard collarFrames > 0 else {
            return [Bool](repeating: true, count: reference.rows)
        }
        var mask = [Bool](repeating: true, count: reference.rows)
        for columnIndex in 0..<reference.columns {
            var previous: Float = 0
            for rowIndex in 0..<reference.rows {
                let current = reference[rowIndex, columnIndex]
                if current != previous {
                    let start = max(0, rowIndex - collarFrames)
                    let stop = min(reference.rows, rowIndex + collarFrames)
                    for maskedIndex in start..<stop {
                        mask[maskedIndex] = false
                    }
                }
                previous = current
            }
            if previous > 0 {
                let start = max(0, reference.rows - collarFrames)
                for maskedIndex in start..<reference.rows {
                    mask[maskedIndex] = false
                }
            }
        }
        return mask
    }

    public static func threshold(probabilities: LSEENDMatrix, value: Float) -> LSEENDMatrix {
        var binary = probabilities
        for index in binary.values.indices {
            binary.values[index] = binary.values[index] > value ? 1 : 0
        }
        return binary
    }

    public static func medianFilter(binary: LSEENDMatrix, width: Int) -> LSEENDMatrix {
        guard width > 1, binary.rows > 0, binary.columns > 0 else {
            return binary
        }
        let kernel = width % 2 == 0 ? width + 1 : width
        let radius = kernel / 2
        var output = binary
        for columnIndex in 0..<binary.columns {
            for rowIndex in 0..<binary.rows {
                let start = max(0, rowIndex - radius)
                let stop = min(binary.rows - 1, rowIndex + radius)
                var ones = 0
                let count = stop - start + 1
                for sampleIndex in start...stop {
                    if binary[sampleIndex, columnIndex] > 0 {
                        ones += 1
                    }
                }
                output[rowIndex, columnIndex] = ones * 2 >= count ? 1 : 0
            }
        }
        return output
    }

    public static func computeDER(
        probabilities: LSEENDMatrix,
        referenceBinary: LSEENDMatrix,
        settings: LSEENDEvaluationSettings
    ) -> LSEENDEvaluationResult {
        let predictionBinary = medianFilter(
            binary: threshold(probabilities: probabilities, value: settings.threshold),
            width: settings.medianWidth
        )
        let validMask = collarMask(
            reference: referenceBinary,
            collarFrames: pythonRoundedInt(settings.collarSeconds * settings.frameRate)
        )
        let mapping = mapPredictions(
            predictionBinary: predictionBinary,
            referenceBinary: referenceBinary,
            validMask: validMask
        )
        var mappedProbabilities = LSEENDMatrix.zeros(rows: probabilities.rows, columns: referenceBinary.columns)
        for (referenceIndex, predictionIndex) in mapping.assignment {
            for rowIndex in 0..<probabilities.rows {
                mappedProbabilities[rowIndex, referenceIndex] = probabilities[rowIndex, predictionIndex]
            }
        }
        let extraBinary = mapping.unmatchedPredictionIndices.isEmpty
            ? LSEENDMatrix.empty(columns: 0)
            : selectColumns(from: predictionBinary, indices: mapping.unmatchedPredictionIndices)

        var scoredReference = LSEENDMatrix.zeros(
            rows: referenceBinary.rows,
            columns: referenceBinary.columns + extraBinary.columns
        )
        for rowIndex in 0..<referenceBinary.rows {
            for columnIndex in 0..<referenceBinary.columns {
                scoredReference[rowIndex, columnIndex] = referenceBinary[rowIndex, columnIndex]
            }
        }
        var scoredPrediction = LSEENDMatrix.zeros(
            rows: mapping.mappedBinary.rows,
            columns: mapping.mappedBinary.columns + extraBinary.columns
        )
        for rowIndex in 0..<mapping.mappedBinary.rows {
            for columnIndex in 0..<mapping.mappedBinary.columns {
                scoredPrediction[rowIndex, columnIndex] = mapping.mappedBinary[rowIndex, columnIndex]
            }
            for columnIndex in 0..<extraBinary.columns {
                scoredPrediction[rowIndex, mapping.mappedBinary.columns + columnIndex] = extraBinary[rowIndex, columnIndex]
            }
        }

        var miss: Double = 0
        var falseAlarm: Double = 0
        var speakerError: Double = 0
        var speakerScored: Double = 0
        for rowIndex in 0..<scoredReference.rows where validMask[rowIndex] {
            var referenceActive = 0
            var predictionActive = 0
            var mappedOverlap = 0
            for columnIndex in 0..<scoredReference.columns {
                let refValue = scoredReference[rowIndex, columnIndex] > 0
                let predValue = scoredPrediction[rowIndex, columnIndex] > 0
                if refValue { referenceActive += 1 }
                if predValue { predictionActive += 1 }
                if refValue && predValue { mappedOverlap += 1 }
            }
            miss += Double(max(referenceActive - predictionActive, 0))
            falseAlarm += Double(max(predictionActive - referenceActive, 0))
            speakerError += Double(min(referenceActive, predictionActive) - mappedOverlap)
            speakerScored += Double(referenceActive)
        }
        let der = speakerScored > 0 ? (miss + falseAlarm + speakerError) / speakerScored : 0
        return LSEENDEvaluationResult(
            der: der,
            speakerScored: speakerScored,
            speakerMiss: miss,
            speakerFalseAlarm: falseAlarm,
            speakerError: speakerError,
            threshold: settings.threshold,
            medianWidth: settings.medianWidth,
            collarSeconds: settings.collarSeconds,
            mappedBinary: mapping.mappedBinary,
            mappedProbabilities: mappedProbabilities,
            validMask: validMask,
            assignment: mapping.assignment,
            unmatchedPredictionIndices: mapping.unmatchedPredictionIndices
        )
    }

    private static func mapPredictions(
        predictionBinary: LSEENDMatrix,
        referenceBinary: LSEENDMatrix,
        validMask: [Bool]
    ) -> (mappedBinary: LSEENDMatrix, assignment: [Int: Int], unmatchedPredictionIndices: [Int]) {
        let numPred = predictionBinary.columns
        let numRef = referenceBinary.columns
        var mapped = LSEENDMatrix.zeros(rows: predictionBinary.rows, columns: numRef)
        guard numPred > 0, numRef > 0 else {
            return (mapped, [:], Array(0..<numPred))
        }

        var cost = [Float](repeating: 0, count: numPred * numRef)
        for predIndex in 0..<numPred {
            for refIndex in 0..<numRef {
                cost[predIndex * numRef + refIndex] = pairCost(
                    predictionBinary: predictionBinary,
                    predictionIndex: predIndex,
                    referenceBinary: referenceBinary,
                    referenceIndex: refIndex,
                    validMask: validMask
                )
            }
        }

        let assignment = solveRectangularAssignment(cost: cost, rows: numPred, columns: numRef)
        var mappedAssignment: [Int: Int] = [:]
        var matchedPredictions = Set<Int>()
        for (predIndex, refIndex) in assignment {
            matchedPredictions.insert(predIndex)
            mappedAssignment[refIndex] = predIndex
            for rowIndex in 0..<predictionBinary.rows {
                mapped[rowIndex, refIndex] = predictionBinary[rowIndex, predIndex]
            }
        }
        let unmatched = (0..<numPred).filter { !matchedPredictions.contains($0) }
        return (mapped, mappedAssignment, unmatched)
    }

    private static func pairCost(
        predictionBinary: LSEENDMatrix,
        predictionIndex: Int,
        referenceBinary: LSEENDMatrix,
        referenceIndex: Int,
        validMask: [Bool]
    ) -> Float {
        var refCount = 0
        var predCount = 0
        var overlap = 0
        for rowIndex in 0..<predictionBinary.rows where validMask[rowIndex] {
            let pred = predictionBinary[rowIndex, predictionIndex] > 0
            let ref = referenceBinary[rowIndex, referenceIndex] > 0
            if ref { refCount += 1 }
            if pred { predCount += 1 }
            if ref && pred { overlap += 1 }
        }
        let miss = max(refCount - predCount, 0)
        let falseAlarm = max(predCount - refCount, 0)
        let speakerError = min(refCount, predCount) - overlap
        return Float(miss + falseAlarm + speakerError)
    }

    private static func solveRectangularAssignment(cost: [Float], rows: Int, columns: Int) -> [(Int, Int)] {
        if rows <= columns {
            let solution = solveAssignmentRowsToColumns(cost: cost, rows: rows, columns: columns)
            return solution.enumerated().map { ($0.offset, $0.element) }
        }
        let transposed = transpose(cost: cost, rows: rows, columns: columns)
        let solution = solveAssignmentRowsToColumns(cost: transposed, rows: columns, columns: rows)
        return solution.enumerated().map { ($0.element, $0.offset) }
    }

    private static func solveAssignmentRowsToColumns(cost: [Float], rows: Int, columns: Int) -> [Int] {
        let stateCount = 1 << columns
        var dp = [Float](repeating: .greatestFiniteMagnitude, count: stateCount)
        var parent = [Int](repeating: -1, count: stateCount)
        var parentColumn = [Int](repeating: -1, count: stateCount)
        dp[0] = 0

        for mask in 0..<stateCount {
            let assignedRows = mask.nonzeroBitCount
            guard assignedRows < rows else { continue }
            let baseCost = dp[mask]
            guard baseCost.isFinite else { continue }
            for column in 0..<columns where (mask & (1 << column)) == 0 {
                let nextMask = mask | (1 << column)
                let nextCost = baseCost + cost[assignedRows * columns + column]
                if nextCost < dp[nextMask] {
                    dp[nextMask] = nextCost
                    parent[nextMask] = mask
                    parentColumn[nextMask] = column
                }
            }
        }

        var bestMask = 0
        var bestCost = Float.greatestFiniteMagnitude
        for mask in 0..<stateCount where mask.nonzeroBitCount == rows {
            if dp[mask] < bestCost {
                bestCost = dp[mask]
                bestMask = mask
            }
        }

        var assignment = [Int](repeating: -1, count: rows)
        var currentMask = bestMask
        for rowIndex in stride(from: rows - 1, through: 0, by: -1) {
            assignment[rowIndex] = parentColumn[currentMask]
            currentMask = parent[currentMask]
        }
        return assignment
    }

    private static func transpose(cost: [Float], rows: Int, columns: Int) -> [Float] {
        var output = [Float](repeating: 0, count: cost.count)
        for rowIndex in 0..<rows {
            for columnIndex in 0..<columns {
                output[columnIndex * rows + rowIndex] = cost[rowIndex * columns + columnIndex]
            }
        }
        return output
    }

    private static func selectColumns(from matrix: LSEENDMatrix, indices: [Int]) -> LSEENDMatrix {
        guard !indices.isEmpty else { return .empty(columns: 0) }
        var output = [Float](repeating: 0, count: matrix.rows * indices.count)
        for rowIndex in 0..<matrix.rows {
            let destinationBase = rowIndex * indices.count
            for (outputColumn, sourceColumn) in indices.enumerated() {
                output[destinationBase + outputColumn] = matrix[rowIndex, sourceColumn]
            }
        }
        return LSEENDMatrix(validatingRows: matrix.rows, columns: indices.count, values: output)
    }

    private static func pythonRoundedInt(_ value: Double) -> Int {
        Int(value.rounded(.toNearestOrEven))
    }
}
