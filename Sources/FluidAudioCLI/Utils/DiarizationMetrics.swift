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
    let evaluationCollarSeconds: Float
    let evaluationIgnoresOverlap: Bool

    private enum CodingKeys: String, CodingKey {
        case der
        case missRate
        case falseAlarmRate
        case speakerErrorRate
        case jer
        case speakerMapping
        case evaluationCollarSeconds
        case evaluationIgnoresOverlap
    }

    init(
        der: Float,
        missRate: Float,
        falseAlarmRate: Float,
        speakerErrorRate: Float,
        jer: Float,
        speakerMapping: [String: String],
        evaluationCollarSeconds: Float,
        evaluationIgnoresOverlap: Bool
    ) {
        self.der = der
        self.missRate = missRate
        self.falseAlarmRate = falseAlarmRate
        self.speakerErrorRate = speakerErrorRate
        self.jer = jer
        self.speakerMapping = speakerMapping
        self.evaluationCollarSeconds = evaluationCollarSeconds
        self.evaluationIgnoresOverlap = evaluationIgnoresOverlap
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        der = try container.decode(Float.self, forKey: .der)
        missRate = try container.decode(Float.self, forKey: .missRate)
        falseAlarmRate = try container.decode(Float.self, forKey: .falseAlarmRate)
        speakerErrorRate = try container.decode(Float.self, forKey: .speakerErrorRate)
        jer = try container.decode(Float.self, forKey: .jer)
        speakerMapping = try container.decode([String: String].self, forKey: .speakerMapping)
        evaluationCollarSeconds =
            try container.decodeIfPresent(
                Float.self,
                forKey: .evaluationCollarSeconds
            ) ?? 0.25
        evaluationIgnoresOverlap =
            try container.decodeIfPresent(
                Bool.self,
                forKey: .evaluationIgnoresOverlap
            ) ?? true
    }
}

/// Utility for computing diarization metrics that can be shared between CLI commands.
enum DiarizationMetricsCalculator {

    private static let scoringCollarSeconds: Double = 0.25
    private static let ignoreOverlap = true

    private struct ScoringSegment {
        let start: Double
        let end: Double
        let speaker: String
    }

    private enum EventType: Int {
        case start = 0
        case end = 1
    }

    private static func zeroMetrics() -> DiarizationMetrics {
        DiarizationMetrics(
            der: 0,
            missRate: 0,
            falseAlarmRate: 0,
            speakerErrorRate: 0,
            jer: 0,
            speakerMapping: [:],
            evaluationCollarSeconds: Float(scoringCollarSeconds),
            evaluationIgnoresOverlap: ignoreOverlap
        )
    }

    /// Compute offline diarization metrics using segment-level overlap analysis.
    /// - Parameters:
    ///   - predicted: Predicted speaker segments.
    ///   - groundTruth: Ground-truth speaker segments.
    ///   - frameSize: Retained for compatibility; unused in segment-based evaluation.
    ///   - logger: Optional logger for emitting debug information.
    /// - Returns: Aggregate metrics including DER, JER, and the speaker mapping.
    static func offlineMetrics(
        predicted: [TimedSpeakerSegment],
        groundTruth: [TimedSpeakerSegment],
        frameSize: Float = 0.01,
        audioDurationSeconds: Double? = nil,
        logger: AppLogger? = nil
    ) -> DiarizationMetrics {

        _ = frameSize  // Retained for API compatibility; no longer needed.

        guard !groundTruth.isEmpty else {
            return zeroMetrics()
        }

        let predictedSegments =
            predicted
            .map {
                ScoringSegment(
                    start: Double($0.startTimeSeconds),
                    end: Double($0.endTimeSeconds),
                    speaker: $0.speakerId
                )
            }
            .sorted { $0.start < $1.start }

        let groundTruthSegments =
            groundTruth
            .map {
                ScoringSegment(
                    start: Double($0.startTimeSeconds),
                    end: Double($0.endTimeSeconds),
                    speaker: $0.speakerId
                )
            }
            .sorted { $0.start < $1.start }

        let (processedGroundTruth, excludedIntervals) = applyOfficialReferenceProcessing(groundTruthSegments)

        let maxPredictedEnd = predictedSegments.map { $0.end }.max() ?? 0
        let maxGroundTruthEnd = groundTruthSegments.map { $0.end }.max() ?? 0
        let evaluationDuration = max(audioDurationSeconds ?? 0, maxPredictedEnd, maxGroundTruthEnd)

        guard evaluationDuration > 0 else {
            return zeroMetrics()
        }

        let evaluationIntervals = subtractIntervals(
            from: (0.0, evaluationDuration),
            removing: excludedIntervals
        )

        guard !evaluationIntervals.isEmpty else {
            return zeroMetrics()
        }

        let predictedEvaluated = clipSegments(predictedSegments, to: evaluationIntervals)
        let processedGroundTruthEvaluated = clipSegments(processedGroundTruth, to: evaluationIntervals)

        guard !processedGroundTruthEvaluated.isEmpty else {
            return zeroMetrics()
        }

        let groundTruthIntervals = mergeIntervals(processedGroundTruthEvaluated.map { ($0.start, $0.end) })
        let predictedIntervals = mergeIntervals(predictedEvaluated.map { ($0.start, $0.end) })

        let referenceSpeech = groundTruthIntervals.reduce(0.0) { $0 + ($1.1 - $1.0) }
        guard referenceSpeech > 0 else {
            return zeroMetrics()
        }

        let predictedSpeech = predictedIntervals.reduce(0.0) { $0 + ($1.1 - $1.0) }
        let overlapSpeech = intervalsOverlap(groundTruthIntervals, predictedIntervals)

        let miss = max(0.0, referenceSpeech - overlapSpeech)
        let falseAlarm = max(0.0, predictedSpeech - overlapSpeech)

        let groundTruthBySpeaker = segmentsBySpeaker(processedGroundTruthEvaluated)
        let predictedBySpeaker = segmentsBySpeaker(predictedEvaluated)

        let speakerMapping = computeSpeakerMapping(
            predicted: predictedBySpeaker,
            groundTruth: groundTruthBySpeaker
        )

        var correctlyAssigned = 0.0
        for (predId, truthId) in speakerMapping {
            if let predSegments = predictedBySpeaker[predId],
                let truthSegments = groundTruthBySpeaker[truthId]
            {
                correctlyAssigned += overlapDuration(predSegments, truthSegments)
            }
        }

        let confusion = max(0.0, overlapSpeech - correctlyAssigned)

        let missRate = Float((miss / referenceSpeech) * 100)
        let falseAlarmRate = Float((falseAlarm / referenceSpeech) * 100)
        let speakerErrorRate = Float((confusion / referenceSpeech) * 100)
        let der = missRate + falseAlarmRate + speakerErrorRate

        var jaccardScores: [Double] = []
        let inverseMapping = Dictionary(uniqueKeysWithValues: speakerMapping.map { ($0.value, $0.key) })

        for (truthId, truthSegments) in groundTruthBySpeaker {
            let matchedPred = inverseMapping[truthId]
            let predictedSegmentsForSpeaker = matchedPred.flatMap { predictedBySpeaker[$0] } ?? []
            let intersection = overlapDuration(predictedSegmentsForSpeaker, truthSegments)
            let union = unionDuration(predictedSegmentsForSpeaker, truthSegments)
            if union > 0 {
                jaccardScores.append(intersection / union)
            }
        }

        for (predId, predSegments) in predictedBySpeaker where speakerMapping[predId] == nil {
            if unionDuration(predSegments) > 0 {
                jaccardScores.append(0.0)
            }
        }

        let jer: Float
        if jaccardScores.isEmpty {
            jer = 0
        } else {
            let averageJaccard = jaccardScores.reduce(0.0, +) / Double(jaccardScores.count)
            jer = Float((1.0 - averageJaccard) * 100)
        }

        if let logger = logger {
            logger.debug("🎯 Offline mapping: \(speakerMapping)")
            let formattedDer = String(format: "%.1f", der)
            let formattedMiss = String(format: "%.1f", missRate)
            let formattedFalseAlarm = String(format: "%.1f", falseAlarmRate)
            let formattedSpeakerError = String(format: "%.1f", speakerErrorRate)
            let formattedJer = String(format: "%.1f", jer)
            let formattedCollar = String(format: "%.2f", scoringCollarSeconds)
            let summary =
                "📊 OFFLINE METRICS: DER=\(formattedDer)% "
                + "(Miss=\(formattedMiss)%, FA=\(formattedFalseAlarm)%, "
                + "SE=\(formattedSpeakerError)%, JER=\(formattedJer)%) "
                + "(collar=\(formattedCollar)s, ignoreOverlap=\(ignoreOverlap))"
            logger.info(summary)
        }

        return DiarizationMetrics(
            der: der,
            missRate: missRate,
            falseAlarmRate: falseAlarmRate,
            speakerErrorRate: speakerErrorRate,
            jer: jer,
            speakerMapping: speakerMapping,
            evaluationCollarSeconds: Float(scoringCollarSeconds),
            evaluationIgnoresOverlap: ignoreOverlap
        )
    }

    // MARK: - Segment-level helpers

    private static func applyOfficialReferenceProcessing(
        _ segments: [ScoringSegment]
    ) -> ([ScoringSegment], [(Double, Double)]) {
        guard !segments.isEmpty else { return ([], []) }

        var trimmed: [ScoringSegment] = []
        var excluded: [(Double, Double)] = []

        for segment in segments {
            let trimmedStart = segment.start + scoringCollarSeconds
            let trimmedEnd = segment.end - scoringCollarSeconds

            if trimmedEnd <= trimmedStart {
                excluded.append((segment.start, segment.end))
                continue
            }

            if trimmedStart > segment.start {
                excluded.append((segment.start, trimmedStart))
            }
            if trimmedEnd < segment.end {
                excluded.append((trimmedEnd, segment.end))
            }

            trimmed.append(
                ScoringSegment(start: trimmedStart, end: trimmedEnd, speaker: segment.speaker)
            )
        }

        if trimmed.isEmpty {
            return ([], mergeIntervals(excluded))
        }

        var processed = trimmed
        if ignoreOverlap {
            let isolated = isolateSingleSpeakerSegments(processed)
            processed = isolated.segments
            excluded.append(contentsOf: isolated.excluded)
        } else {
            processed = mergeAdjacentSegments(processed)
        }

        processed.removeAll { $0.end <= $0.start }

        return (processed, mergeIntervals(excluded))
    }

    private static func mergeAdjacentSegments(
        _ segments: [ScoringSegment]
    ) -> [ScoringSegment] {
        var merged: [ScoringSegment] = []

        for segment in segments {
            guard segment.end > segment.start else { continue }
            if let last = merged.last, last.speaker == segment.speaker, segment.start <= last.end {
                let updated = ScoringSegment(
                    start: last.start,
                    end: max(last.end, segment.end),
                    speaker: last.speaker
                )
                merged[merged.count - 1] = updated
            } else if let last = merged.last,
                last.speaker == segment.speaker,
                abs(segment.start - last.end) < 1e-9
            {
                let updated = ScoringSegment(
                    start: last.start,
                    end: max(last.end, segment.end),
                    speaker: last.speaker
                )
                merged[merged.count - 1] = updated
            } else {
                merged.append(segment)
            }
        }

        return merged
    }

    private static func isolateSingleSpeakerSegments(
        _ segments: [ScoringSegment]
    ) -> (segments: [ScoringSegment], excluded: [(Double, Double)]) {
        struct Event {
            let time: Double
            let type: EventType
            let speaker: String
        }

        var events: [Event] = []
        for segment in segments where segment.end > segment.start {
            events.append(Event(time: segment.start, type: .start, speaker: segment.speaker))
            events.append(Event(time: segment.end, type: .end, speaker: segment.speaker))
        }

        guard !events.isEmpty else { return ([], []) }

        events.sort { lhs, rhs in
            if lhs.time == rhs.time {
                return lhs.type.rawValue < rhs.type.rawValue
            }
            return lhs.time < rhs.time
        }

        var activeCounts: [String: Int] = [:]
        var singleSpeaker: [ScoringSegment] = []
        var excluded: [(Double, Double)] = []
        var index = 0
        var previousTime: Double?

        while index < events.count {
            let currentTime = events[index].time
            if let prev = previousTime, currentTime > prev {
                let activeSpeakers = activeCounts.filter { $0.value > 0 }.map(\.key)
                if activeSpeakers.count == 1, let speaker = activeSpeakers.first {
                    singleSpeaker.append(
                        ScoringSegment(start: prev, end: currentTime, speaker: speaker)
                    )
                } else if activeSpeakers.count > 1 {
                    excluded.append((prev, currentTime))
                }
            }

            while index < events.count && events[index].time == currentTime {
                let event = events[index]
                switch event.type {
                case .start:
                    activeCounts[event.speaker, default: 0] += 1
                case .end:
                    let count = activeCounts[event.speaker] ?? 0
                    if count <= 1 {
                        activeCounts.removeValue(forKey: event.speaker)
                    } else {
                        activeCounts[event.speaker] = count - 1
                    }
                }
                index += 1
            }
            previousTime = currentTime
        }

        return (mergeAdjacentSegments(singleSpeaker), mergeIntervals(excluded))
    }

    private static func clipSegments(
        _ segments: [ScoringSegment],
        to intervals: [(Double, Double)]
    ) -> [ScoringSegment] {
        guard !segments.isEmpty, !intervals.isEmpty else { return [] }

        let sortedSegments = segments.sorted { $0.start < $1.start }
        let sortedIntervals = intervals.sorted { $0.0 < $1.0 }

        var clipped: [ScoringSegment] = []
        var intervalIndex = 0

        for segment in sortedSegments where segment.end > segment.start {
            while intervalIndex < sortedIntervals.count && sortedIntervals[intervalIndex].1 <= segment.start {
                intervalIndex += 1
            }

            var probeIndex = intervalIndex
            while probeIndex < sortedIntervals.count {
                let interval = sortedIntervals[probeIndex]
                if interval.0 >= segment.end {
                    break
                }

                let overlapStart = max(segment.start, interval.0)
                let overlapEnd = min(segment.end, interval.1)
                if overlapEnd > overlapStart {
                    clipped.append(
                        ScoringSegment(start: overlapStart, end: overlapEnd, speaker: segment.speaker)
                    )
                }

                if interval.1 >= segment.end {
                    break
                }
                probeIndex += 1
            }
        }

        return clipped
    }

    private static func subtractIntervals(
        from span: (Double, Double),
        removing intervals: [(Double, Double)]
    ) -> [(Double, Double)] {
        let (start, end) = span
        guard end > start else { return [] }

        let merged = mergeIntervals(
            intervals.map { (max(start, $0.0), min(end, $0.1)) }
                .filter { $0.1 > start && $0.0 < end }
        )

        var remaining: [(Double, Double)] = []
        var cursor = start

        for interval in merged {
            if interval.0 > cursor {
                remaining.append((cursor, min(interval.0, end)))
            }
            cursor = max(cursor, interval.1)
            if cursor >= end {
                break
            }
        }

        if cursor < end {
            remaining.append((cursor, end))
        }

        return remaining
    }

    private static func mergeIntervals(
        _ intervals: [(Double, Double)]
    ) -> [(Double, Double)] {
        guard !intervals.isEmpty else { return [] }

        let sorted = intervals.sorted { lhs, rhs in
            if lhs.0 == rhs.0 {
                return lhs.1 < rhs.1
            }
            return lhs.0 < rhs.0
        }

        var merged: [(Double, Double)] = []
        var current = sorted[0]

        for interval in sorted.dropFirst() {
            if interval.0 <= current.1 {
                current.1 = max(current.1, interval.1)
            } else {
                merged.append(current)
                current = interval
            }
        }

        merged.append(current)
        return merged
    }

    private static func intervalsOverlap(
        _ lhs: [(Double, Double)],
        _ rhs: [(Double, Double)]
    ) -> Double {
        var total = 0.0
        var i = 0
        var j = 0

        while i < lhs.count && j < rhs.count {
            let a = lhs[i]
            let b = rhs[j]
            let start = max(a.0, b.0)
            let end = min(a.1, b.1)

            if end > start {
                total += end - start
            }

            if a.1 <= b.1 {
                i += 1
            } else {
                j += 1
            }
        }

        return total
    }

    private static func segmentsBySpeaker(
        _ segments: [ScoringSegment]
    ) -> [String: [ScoringSegment]] {
        var grouped: [String: [ScoringSegment]] = [:]
        for segment in segments {
            grouped[segment.speaker, default: []].append(segment)
        }
        for key in grouped.keys {
            grouped[key]?.sort { $0.start < $1.start }
        }
        return grouped
    }

    private static func overlapDuration(
        _ lhs: [ScoringSegment],
        _ rhs: [ScoringSegment]
    ) -> Double {
        var total = 0.0
        var i = 0
        var j = 0

        while i < lhs.count && j < rhs.count {
            let a = lhs[i]
            let b = rhs[j]
            let start = max(a.start, b.start)
            let end = min(a.end, b.end)

            if end > start {
                total += end - start
            }

            if a.end <= b.end {
                i += 1
            } else {
                j += 1
            }
        }

        return total
    }

    private static func unionDuration(
        _ segments: [ScoringSegment]
    ) -> Double {
        let intervals = segments.map { ($0.start, $0.end) }
        return mergeIntervals(intervals).reduce(0.0) { $0 + ($1.1 - $1.0) }
    }

    private static func unionDuration(
        _ lhs: [ScoringSegment],
        _ rhs: [ScoringSegment]
    ) -> Double {
        return unionDuration(lhs + rhs)
    }

    private static func computeSpeakerMapping(
        predicted: [String: [ScoringSegment]],
        groundTruth: [String: [ScoringSegment]]
    ) -> [String: String] {
        guard !predicted.isEmpty, !groundTruth.isEmpty else { return [:] }

        let predictedIds = Array(predicted.keys)
        let groundTruthIds = Array(groundTruth.keys)

        var confusionMatrix = Array(
            repeating: Array(repeating: 0, count: predictedIds.count),
            count: groundTruthIds.count
        )
        let scale = 1_000.0

        for (gtIndex, gtId) in groundTruthIds.enumerated() {
            for (predIndex, predId) in predictedIds.enumerated() {
                let overlap = overlapDuration(
                    predicted[predId] ?? [],
                    groundTruth[gtId] ?? []
                )
                confusionMatrix[gtIndex][predIndex] = Int((overlap * scale).rounded())
            }
        }

        let assignment = AssignmentSolver.bestAssignment(confusionMatrix: confusionMatrix)
        var mapping: [String: String] = [:]

        for (predIndex, gtIndex) in assignment {
            guard predIndex < predictedIds.count, gtIndex < groundTruthIds.count else { continue }
            if confusionMatrix[gtIndex][predIndex] > 0 {
                mapping[predictedIds[predIndex]] = groundTruthIds[gtIndex]
            }
        }

        return mapping
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

                var bestResult = dfs(predIndex: predIndex + 1, mask: mask)

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
}
#endif
