#if os(macOS)
import FluidAudio
import Foundation

/// Aggregate diarization quality metrics.
struct DiarizationMetrics: Codable {
    let der: Float
    let missRate: Float
    let falseAlarmRate: Float
    let speakerErrorRate: Float
    let jer: Float
    let speakerMapping: [String: String]
}

/// Utility for computing diarization metrics that can be shared between CLI commands.
enum DiarizationMetricsCalculator {

    /// Compute offline diarization metrics using frame-level evaluation and optimal speaker mapping.
    /// - Parameters:
    ///   - predicted: Predicted speaker segments.
    ///   - groundTruth: Ground-truth speaker segments.
    ///   - frameSize: Frame size (seconds) for evaluation grid.
    ///   - logger: Optional logger for emitting debug information.
    /// - Returns: Aggregate metrics including DER, JER, and the speaker mapping.
    static func offlineMetrics(
        predicted: [TimedSpeakerSegment],
        groundTruth: [TimedSpeakerSegment],
        frameSize: Float = 0.01,
        logger: AppLogger? = nil
    ) -> DiarizationMetrics {

        guard !groundTruth.isEmpty else {
            return DiarizationMetrics(
                der: 0,
                missRate: 0,
                falseAlarmRate: 0,
                speakerErrorRate: 0,
                jer: 0,
                speakerMapping: [:]
            )
        }

        let maxTruthEnd = groundTruth.map { $0.endTimeSeconds }.max() ?? 0
        let totalDuration = maxTruthEnd
        let totalFrames = max(1, Int(ceil(totalDuration / frameSize)))

        // Build frame-wise timelines.
        let groundTruthTimeline = buildTimeline(
            segments: groundTruth,
            frameSize: frameSize,
            totalFrames: totalFrames
        )
        let predictedTimeline = buildTimeline(
            segments: predicted,
            frameSize: frameSize,
            totalFrames: totalFrames
        )

        // Enumerate unique speakers.
        let groundTruthSpeakers = Array(Set(groundTruth.map { $0.speakerId })).sorted()
        let predictedSpeakers = Array(Set(predicted.map { $0.speakerId })).sorted()

        var speakerMapping: [String: String] = [:]

        if !predictedSpeakers.isEmpty && !groundTruthSpeakers.isEmpty {
            let mapping = computeOptimalMapping(
                groundTruthTimeline: groundTruthTimeline,
                predictedTimeline: predictedTimeline,
                groundTruthSpeakers: groundTruthSpeakers,
                predictedSpeakers: predictedSpeakers
            )

            for (predIndex, truthIndex) in mapping {
                if predIndex < predictedSpeakers.count && truthIndex < groundTruthSpeakers.count {
                    speakerMapping[predictedSpeakers[predIndex]] = groundTruthSpeakers[truthIndex]
                }
            }
        }

        // Frame-level metrics.
        var referenceSpeakerFrames = 0
        var missedSpeakerFrames = 0
        var falseAlarmSpeakerFrames = 0
        var confusionSpeakerFrames = 0
        var activeFramesForJer = 0
        var accumulatedJaccard: Float = 0

        for frame in 0..<totalFrames {
            let truthSpeakers = groundTruthTimeline[frame]
            let predictedSpeakersAtFrame = predictedTimeline[frame]

            if truthSpeakers.isEmpty && predictedSpeakersAtFrame.isEmpty {
                continue
            }

            activeFramesForJer += 1

            let mappedPredictions = Set(
                predictedSpeakersAtFrame.map { speakerMapping[$0] ?? $0 }
            )

            let matchCounts = computeMatchCounts(
                referenceSpeakers: Array(truthSpeakers),
                predictedSpeakers: Array(mappedPredictions)
            )

            referenceSpeakerFrames += matchCounts.referenceTotal
            missedSpeakerFrames += matchCounts.missed
            falseAlarmSpeakerFrames += matchCounts.falseAlarms
            confusionSpeakerFrames += matchCounts.confusions

            let unionSet = truthSpeakers.union(mappedPredictions)
            if !unionSet.isEmpty {
                let intersectionCount = truthSpeakers.intersection(mappedPredictions).count
                accumulatedJaccard += Float(intersectionCount) / Float(unionSet.count)
            }
        }

        let referenceSpeakerDenominator = max(1, referenceSpeakerFrames)
        let missRate =
            Float(missedSpeakerFrames) / Float(referenceSpeakerDenominator) * 100
        let falseAlarmRate =
            Float(falseAlarmSpeakerFrames) / Float(referenceSpeakerDenominator) * 100
        let speakerErrorRate =
            Float(confusionSpeakerFrames) / Float(referenceSpeakerDenominator) * 100
        let der = missRate + falseAlarmRate + speakerErrorRate
        let jer =
            activeFramesForJer > 0
            ? (1 - accumulatedJaccard / Float(activeFramesForJer)) * 100 : 0

        if let logger = logger {
            logger.debug("🎯 Offline mapping: \(speakerMapping)")
            logger.info(
                "📊 OFFLINE METRICS: DER=\(String(format: "%.1f", der))% (Miss=\(String(format: "%.1f", missRate))%, FA=\(String(format: "%.1f", falseAlarmRate))%, SE=\(String(format: "%.1f", speakerErrorRate))%, JER=\(String(format: "%.1f", jer))%)"
            )
        }

        return DiarizationMetrics(
            der: der,
            missRate: missRate,
            falseAlarmRate: falseAlarmRate,
            speakerErrorRate: speakerErrorRate,
            jer: jer,
            speakerMapping: speakerMapping
        )
    }

    // MARK: - Timeline helpers

    private static func buildTimeline(
        segments: [TimedSpeakerSegment],
        frameSize: Float,
        totalFrames: Int
    ) -> [Set<String>] {
        var timeline = Array(repeating: Set<String>(), count: totalFrames)

        for segment in segments {
            let startFrame = max(0, Int(floor(segment.startTimeSeconds / frameSize)))
            let endFrame = max(startFrame, Int(ceil(segment.endTimeSeconds / frameSize)))

            guard endFrame > startFrame else { continue }

            let cappedEnd = min(totalFrames, endFrame)

            if cappedEnd <= startFrame {
                continue
            }

            for frame in startFrame..<cappedEnd {
                timeline[frame].insert(segment.speakerId)
            }
        }

        return timeline
    }

    // MARK: - Optimal assignment

    private static func computeOptimalMapping(
        groundTruthTimeline: [Set<String>],
        predictedTimeline: [Set<String>],
        groundTruthSpeakers: [String],
        predictedSpeakers: [String]
    ) -> [Int: Int] {
        let gtCount = groundTruthSpeakers.count
        let predCount = predictedSpeakers.count

        guard gtCount > 0, predCount > 0 else { return [:] }

        var confusionMatrix = Array(
            repeating: Array(repeating: 0, count: predCount),
            count: gtCount
        )

        var groundTruthIndex: [String: Int] = [:]
        for (index, speaker) in groundTruthSpeakers.enumerated() {
            groundTruthIndex[speaker] = index
        }

        var predictedIndex: [String: Int] = [:]
        for (index, speaker) in predictedSpeakers.enumerated() {
            predictedIndex[speaker] = index
        }

        let frameCount = groundTruthTimeline.count
        for frame in 0..<frameCount {
            let truthSpeakers = groundTruthTimeline[frame]
            let predictedSpeakersAtFrame = predictedTimeline[frame]

            if truthSpeakers.isEmpty || predictedSpeakersAtFrame.isEmpty {
                continue
            }

            for truthSpeaker in truthSpeakers {
                guard let gtIdx = groundTruthIndex[truthSpeaker] else { continue }
                for predictedSpeaker in predictedSpeakersAtFrame {
                    guard let predIdx = predictedIndex[predictedSpeaker] else { continue }
                    confusionMatrix[gtIdx][predIdx] += 1
                }
            }
        }

        return AssignmentSolver.bestAssignment(confusionMatrix: confusionMatrix)
    }

    // MARK: - Assignment solver (DP over subsets)

    private enum AssignmentSolver {

        struct Key: Hashable {
            let predIndex: Int
            let mask: Int
        }

        struct Result {
            let score: Int
            let mapping: [Int: Int]
        }

        static func bestAssignment(confusionMatrix: [[Int]]) -> [Int: Int] {
            let gtCount = confusionMatrix.count
            guard gtCount > 0 else { return [:] }
            let predCount = confusionMatrix.first?.count ?? 0
            guard predCount > 0 else { return [:] }

            // Fall back to greedy matching if the number of ground-truth speakers exceeds bit-mask capacity.
            if gtCount >= Int.bitWidth {
                return greedyAssignment(confusionMatrix: confusionMatrix)
            }

            var memo: [Key: Result] = [:]

            func dfs(predIndex: Int, mask: Int) -> Result {
                if predIndex == predCount {
                    return Result(score: 0, mapping: [:])
                }

                let key = Key(predIndex: predIndex, mask: mask)
                if let cached = memo[key] {
                    return cached
                }

                // Option 1: skip assigning this predicted speaker.
                var bestResult = dfs(predIndex: predIndex + 1, mask: mask)

                // Option 2: assign to an unused ground-truth speaker.
                for gtIndex in 0..<gtCount where (mask & (1 << gtIndex)) == 0 {
                    let nextResult = dfs(predIndex: predIndex + 1, mask: mask | (1 << gtIndex))
                    let candidateScore = nextResult.score + confusionMatrix[gtIndex][predIndex]

                    if candidateScore > bestResult.score {
                        var updatedMapping = nextResult.mapping
                        updatedMapping[predIndex] = gtIndex
                        bestResult = Result(score: candidateScore, mapping: updatedMapping)
                    }
                }

                memo[key] = bestResult
                return bestResult
            }

            return dfs(predIndex: 0, mask: 0).mapping
        }

        private static func greedyAssignment(confusionMatrix: [[Int]]) -> [Int: Int] {
            let gtCount = confusionMatrix.count
            let predCount = confusionMatrix.first?.count ?? 0

            var assignments: [Int: Int] = [:]
            var usedGroundTruth = Set<Int>()

            for predIndex in 0..<predCount {
                var bestGt = -1
                var bestScore = Int.min

                for gtIndex in 0..<gtCount where !usedGroundTruth.contains(gtIndex) {
                    let score = confusionMatrix[gtIndex][predIndex]
                    if score > bestScore {
                        bestScore = score
                        bestGt = gtIndex
                    }
                }

                if bestGt >= 0 {
                    assignments[predIndex] = bestGt
                    usedGroundTruth.insert(bestGt)
                }
            }

            return assignments
        }
    }

    private struct MatchCounts {
        let referenceTotal: Int
        let missed: Int
        let falseAlarms: Int
        let confusions: Int
    }

    private static func computeMatchCounts(
        referenceSpeakers: [String],
        predictedSpeakers: [String]
    ) -> MatchCounts {
        let reference = referenceSpeakers.sorted()
        let hypothesis = predictedSpeakers.sorted()

        let referenceCount = reference.count
        let hypothesisCount = hypothesis.count
        let dimension = max(referenceCount, hypothesisCount)

        guard dimension > 0 else {
            return MatchCounts(referenceTotal: referenceCount, missed: 0, falseAlarms: 0, confusions: 0)
        }

        var bestAssignment: [Int] = Array(repeating: 0, count: dimension)
        var used = Array(repeating: false, count: dimension)
        var currentAssignment = Array(repeating: 0, count: dimension)
        var bestCost = Int.max

        func costFor(row: Int, column: Int) -> Int {
            if row < referenceCount, column < hypothesisCount {
                return reference[row] == hypothesis[column] ? 0 : 1
            }
            return 1
        }

        func search(row: Int, accumulatedCost: Int) {
            if accumulatedCost > bestCost {
                return
            }

            if row == dimension {
                bestCost = accumulatedCost
                bestAssignment = currentAssignment
                return
            }

            for column in 0..<dimension where !used[column] {
                used[column] = true
                currentAssignment[row] = column
                let nextCost = accumulatedCost + costFor(row: row, column: column)
                search(row: row + 1, accumulatedCost: nextCost)
                used[column] = false
            }
        }

        search(row: 0, accumulatedCost: 0)

        var missed = 0
        var falseAlarms = 0
        var confusions = 0

        for row in 0..<dimension {
            let column = bestAssignment[row]

            if row >= referenceCount {
                if column < hypothesisCount {
                    falseAlarms += 1
                }
                continue
            }

            if column >= hypothesisCount {
                missed += 1
                continue
            }

            if reference[row] == hypothesis[column] {
                continue
            }

            confusions += 1
        }

        if referenceCount == 0 && hypothesisCount > 0 {
            falseAlarms = hypothesisCount
        }

        return MatchCounts(
            referenceTotal: referenceCount,
            missed: missed,
            falseAlarms: falseAlarms,
            confusions: confusions
        )
    }
}
#endif
