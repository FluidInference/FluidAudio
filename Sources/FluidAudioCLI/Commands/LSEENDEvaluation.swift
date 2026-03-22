import FluidAudio
import Foundation

/// A single speaker turn entry from an RTTM (Rich Transcription Time Marked) file.
public struct LSEENDRTTMEntry: Sendable, Codable {
    /// The recording or file identifier.
    public let recordingID: String
    /// Start time of the speaker turn in seconds.
    public let start: Double
    /// Duration of the speaker turn in seconds.
    public let duration: Double
    /// Speaker label (e.g. `"spk0"`, `"speaker_A"`).
    public let speaker: String

    public init(recordingID: String, start: Double, duration: Double, speaker: String) {
        self.recordingID = recordingID
        self.start = start
        self.duration = duration
        self.speaker = speaker
    }
}

/// Configuration for DER (Diarization Error Rate) evaluation.
public struct LSEENDEvaluationSettings: Sendable, Codable {
    /// Probability threshold for binarizing speaker predictions (e.g. 0.5).
    public let threshold: Float
    /// Median filter kernel width applied after thresholding (0 or 1 to disable).
    public let medianWidth: Int
    /// Collar duration in seconds around reference speaker transitions to exclude from scoring.
    public let collarSeconds: Double
    /// Frame rate in Hz used to convert between time and frame indices.
    public let frameRate: Double

    /// Creates evaluation settings.
    public init(threshold: Float, medianWidth: Int, collarSeconds: Double, frameRate: Double) {
        self.threshold = threshold
        self.medianWidth = medianWidth
        self.collarSeconds = collarSeconds
        self.frameRate = frameRate
    }
}

/// Detailed results of a DER evaluation, including error breakdown and speaker mapping.
public struct LSEENDEvaluationResult: Sendable {
    /// Overall Diarization Error Rate: `(miss + falseAlarm + speakerError) / speakerScored`.
    public let der: Double
    /// Total number of reference speaker-active frames scored (after collar exclusion).
    public let speakerScored: Double
    /// Missed speech: reference-active frames with no corresponding prediction.
    public let speakerMiss: Double
    /// False alarm: predicted-active frames with no corresponding reference.
    public let speakerFalseAlarm: Double
    /// Speaker confusion: frames where both reference and prediction are active but mapped to different speakers.
    public let speakerError: Double
    /// The probability threshold used for binarization.
    public let threshold: Float
    /// The median filter width applied after thresholding.
    public let medianWidth: Int
    /// The collar duration in seconds used during scoring.
    public let collarSeconds: Double
    /// Binary predictions remapped to reference speaker order via optimal assignment.
    public let mappedBinary: LSEENDMatrix
    /// Continuous probabilities remapped to reference speaker order.
    public let mappedProbabilities: LSEENDMatrix
    /// Per-frame mask: `true` for frames included in scoring, `false` for collar-excluded frames.
    public let validMask: [Bool]
    /// Optimal speaker assignment mapping: `[referenceIndex: predictionIndex]`.
    public let assignment: [Int: Int]
    /// Prediction column indices that were not matched to any reference speaker.
    public let unmatchedPredictionIndices: [Int]
}

/// Utilities for evaluating LS-EEND diarization output against reference annotations.
///
/// Provides RTTM parsing/writing, post-processing (threshold + median filter),
/// and DER computation with collar masking and optimal speaker assignment.
public enum LSEENDEvaluation {
    /// Parses an RTTM file into speaker turn entries.
    ///
    /// - Parameter url: Path to the RTTM file.
    /// - Returns: A tuple of parsed entries and an ordered list of unique speaker labels.
    public static func parseRTTM(url: URL) throws -> (entries: [LSEENDRTTMEntry], speakers: [String]) {
        let text = try String(contentsOf: url, encoding: .utf8)
        var entries: [LSEENDRTTMEntry] = []
        var speakers: [String] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: " ")
            guard parts.count >= 8, parts[0] == "SPEAKER" else { continue }
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

    /// Converts RTTM entries into a binary frame-level matrix.
    ///
    /// - Parameters:
    ///   - entries: Speaker turn entries from ``parseRTTM(url:)``.
    ///   - speakers: Ordered speaker labels defining column order.
    ///   - numFrames: Total number of output frames.
    ///   - frameRate: Frame rate in Hz for time-to-frame conversion.
    /// - Returns: A binary matrix of shape `[numFrames, speakers.count]` where 1 indicates active speech.
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

    /// Writes a binary prediction matrix to an RTTM file.
    ///
    /// - Parameters:
    ///   - recordingID: The recording identifier to use in each RTTM line.
    ///   - binaryPrediction: Binary matrix of shape `[frames, speakers]`.
    ///   - outputURL: Path where the RTTM file will be written.
    ///   - frameRate: Frame rate in Hz for frame-to-time conversion.
    ///   - speakerLabels: Optional speaker label names; defaults to `"spk0"`, `"spk1"`, etc.
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

    /// Computes a validity mask that excludes frames near speaker transitions.
    ///
    /// Frames within `collarFrames` of any speaker onset or offset in the reference
    /// are marked `false` (excluded from DER scoring).
    ///
    /// - Parameters:
    ///   - reference: Binary reference matrix of shape `[frames, speakers]`.
    ///   - collarFrames: Number of frames on each side of a transition to exclude.
    /// - Returns: Boolean mask of length `reference.rows`.
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

    /// Binarizes a probability matrix: values above `value` become 1, others become 0.
    ///
    /// - Parameters:
    ///   - probabilities: Continuous probability matrix.
    ///   - value: Threshold (exclusive). Values strictly greater than this are set to 1.
    /// - Returns: Binary matrix with the same shape.
    public static func threshold(probabilities: LSEENDMatrix, value: Float) -> LSEENDMatrix {
        var binary = probabilities
        for index in binary.values.indices {
            binary.values[index] = binary.values[index] > value ? 1 : 0
        }
        return binary
    }

    /// Applies a 1D median filter along the time axis of each speaker column.
    ///
    /// Smooths binary predictions to remove brief spurious activations or gaps.
    /// Even widths are rounded up to the next odd number.
    ///
    /// - Parameters:
    ///   - binary: Binary matrix to filter.
    ///   - width: Kernel width in frames (1 or 0 to skip filtering).
    /// - Returns: Filtered binary matrix with the same shape.
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

    /// Computes the Diarization Error Rate (DER) between predictions and a reference.
    ///
    /// Applies thresholding, median filtering, collar masking, and optimal speaker
    /// assignment (Hungarian-style) before computing miss, false alarm, and speaker error rates.
    ///
    /// - Parameters:
    ///   - probabilities: Continuous prediction matrix of shape `[frames, predSpeakers]`.
    ///   - referenceBinary: Binary reference matrix of shape `[frames, refSpeakers]`.
    ///   - settings: Evaluation parameters (threshold, median width, collar, frame rate).
    /// - Returns: Detailed evaluation result including DER, error breakdown, and speaker mapping.
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
        let extraBinary =
            mapping.unmatchedPredictionIndices.isEmpty
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
                scoredPrediction[rowIndex, mapping.mappedBinary.columns + columnIndex] =
                    extraBinary[rowIndex, columnIndex]
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
        precondition(columns <= 20, "Assignment solver is O(2^columns); columns=\(columns) is too large")
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
