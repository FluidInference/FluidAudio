#if os(macOS)
import FluidAudio
import Foundation

/// Metrics calculation for diarization evaluation
struct MetricsCalculator {
    
    static func calculateDiarizationMetrics(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment], totalDuration: Float
    ) -> DiarizationMetrics {
        let frameSize: Float = 0.01
        let totalFrames = Int(totalDuration / frameSize)

        // Step 1: Find optimal speaker assignment using frame-based overlap
        let speakerMapping = findOptimalSpeakerMapping(
            predicted: predicted, groundTruth: groundTruth, totalDuration: totalDuration)

        print("🔍 SPEAKER MAPPING: \(speakerMapping)")

        var missedFrames = 0
        var falseAlarmFrames = 0
        var speakerErrorFrames = 0

        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            let gtSpeaker = findSpeakerAtTime(frameTime, in: groundTruth)
            let predSpeaker = findSpeakerAtTime(frameTime, in: predicted)

            switch (gtSpeaker, predSpeaker) {
            case (nil, nil):
                continue
            case (nil, _):
                falseAlarmFrames += 1
            case (_, nil):
                missedFrames += 1
            case let (gt?, pred?):
                // Map predicted speaker ID to ground truth speaker ID
                let mappedPredSpeaker = speakerMapping[pred] ?? pred
                if gt != mappedPredSpeaker {
                    speakerErrorFrames += 1
                    // Debug first few mismatches
                    if speakerErrorFrames <= 5 {
                        print(
                            "🔍 DER DEBUG: Speaker mismatch at \(String(format: "%.2f", frameTime))s - GT: '\(gt)' vs Pred: '\(pred)' (mapped: '\(mappedPredSpeaker)')"
                        )
                    }
                }
            }
        }

        let der =
            Float(missedFrames + falseAlarmFrames + speakerErrorFrames) / Float(totalFrames) * 100
        let jer = calculateJaccardErrorRate(predicted: predicted, groundTruth: groundTruth)

        // Debug error breakdown
        print(
            "🔍 DER BREAKDOWN: Missed: \(missedFrames), FalseAlarm: \(falseAlarmFrames), SpeakerError: \(speakerErrorFrames), Total: \(totalFrames)"
        )
        print(
            "🔍 DER RATES: Miss: \(String(format: "%.1f", Float(missedFrames) / Float(totalFrames) * 100))%, FA: \(String(format: "%.1f", Float(falseAlarmFrames) / Float(totalFrames) * 100))%, SE: \(String(format: "%.1f", Float(speakerErrorFrames) / Float(totalFrames) * 100))%"
        )

        // Count mapped speakers (those that successfully mapped to ground truth)
        let mappedSpeakerCount = speakerMapping.count

        return DiarizationMetrics(
            der: der,
            jer: jer,
            missRate: Float(missedFrames) / Float(totalFrames) * 100,
            falseAlarmRate: Float(falseAlarmFrames) / Float(totalFrames) * 100,
            speakerErrorRate: Float(speakerErrorFrames) / Float(totalFrames) * 100,
            mappedSpeakerCount: mappedSpeakerCount
        )
    }

    static func calculateJaccardErrorRate(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment]
    ) -> Float {
        // If no segments in either prediction or ground truth, return 100% error
        if predicted.isEmpty && groundTruth.isEmpty {
            return 0.0  // Perfect match - both empty
        } else if predicted.isEmpty || groundTruth.isEmpty {
            return 100.0  // Complete mismatch - one empty, one not
        }

        // Use the same frame size as DER calculation for consistency
        let frameSize: Float = 0.01
        let totalDuration = max(
            predicted.map { $0.endTimeSeconds }.max() ?? 0,
            groundTruth.map { $0.endTimeSeconds }.max() ?? 0
        )
        let totalFrames = Int(totalDuration / frameSize)

        // Get optimal speaker mapping using existing Hungarian algorithm
        let speakerMapping = findOptimalSpeakerMapping(
            predicted: predicted,
            groundTruth: groundTruth,
            totalDuration: totalDuration
        )

        var intersectionFrames = 0
        var unionFrames = 0

        // Calculate frame-by-frame Jaccard
        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            let gtSpeaker = findSpeakerAtTime(frameTime, in: groundTruth)
            let predSpeaker = findSpeakerAtTime(frameTime, in: predicted)

            // Map predicted speaker to ground truth speaker using optimal mapping
            let mappedPredSpeaker = predSpeaker.flatMap { speakerMapping[$0] }

            switch (gtSpeaker, mappedPredSpeaker) {
            case (nil, nil):
                // Both silent - no contribution to intersection or union
                continue
            case (nil, _):
                // Ground truth silent, prediction has speaker
                unionFrames += 1
            case (_, nil):
                // Ground truth has speaker, prediction silent
                unionFrames += 1
            case let (gt?, pred?):
                // Both have speakers
                unionFrames += 1
                if gt == pred {
                    // Same speaker - contributes to intersection
                    intersectionFrames += 1
                }
            // Different speakers - only contributes to union
            }
        }

        // Calculate Jaccard Index
        let jaccardIndex = unionFrames > 0 ? Float(intersectionFrames) / Float(unionFrames) : 0.0

        // Convert to error rate: JER = 1 - Jaccard Index
        let jer = (1.0 - jaccardIndex) * 100.0

        // Debug logging for first few calculations
        if predicted.count > 0 && groundTruth.count > 0 {
            print(
                "🔍 JER DEBUG: Intersection: \(intersectionFrames), Union: \(unionFrames), Jaccard Index: \(String(format: "%.3f", jaccardIndex)), JER: \(String(format: "%.1f", jer))%"
            )
        }

        return jer
    }

    static func findSpeakerAtTime(_ time: Float, in segments: [TimedSpeakerSegment]) -> String? {
        for segment in segments {
            if time >= segment.startTimeSeconds && time < segment.endTimeSeconds {
                return segment.speakerId
            }
        }
        return nil
    }

    /// Find optimal speaker mapping using frame-by-frame overlap analysis
    static func findOptimalSpeakerMapping(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment], totalDuration: Float
    ) -> [String: String] {
        let frameSize: Float = 0.01
        let totalFrames = Int(totalDuration / frameSize)

        // Get all unique speaker IDs
        let predSpeakers = Set(predicted.map { $0.speakerId })
        let gtSpeakers = Set(groundTruth.map { $0.speakerId })

        // Build overlap matrix: [predSpeaker][gtSpeaker] = overlap_frames
        var overlapMatrix: [String: [String: Int]] = [:]

        for predSpeaker in predSpeakers {
            overlapMatrix[predSpeaker] = [:]
            for gtSpeaker in gtSpeakers {
                overlapMatrix[predSpeaker]![gtSpeaker] = 0
            }
        }

        // Calculate frame-by-frame overlaps
        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            let gtSpeaker = findSpeakerAtTime(frameTime, in: groundTruth)
            let predSpeaker = findSpeakerAtTime(frameTime, in: predicted)

            if let gt = gtSpeaker, let pred = predSpeaker {
                overlapMatrix[pred]![gt]! += 1
            }
        }

        // Find optimal assignment using Hungarian Algorithm for globally optimal solution
        let predSpeakerArray = Array(predSpeakers).sorted()  // Consistent ordering
        let gtSpeakerArray = Array(gtSpeakers).sorted()  // Consistent ordering

        // Build numerical overlap matrix for Hungarian algorithm
        var numericalOverlapMatrix: [[Int]] = []
        for predSpeaker in predSpeakerArray {
            var row: [Int] = []
            for gtSpeaker in gtSpeakerArray {
                row.append(overlapMatrix[predSpeaker]![gtSpeaker]!)
            }
            numericalOverlapMatrix.append(row)
        }

        // Convert overlap matrix to cost matrix (higher overlap = lower cost)
        let costMatrix = HungarianAlgorithm.overlapToCostMatrix(numericalOverlapMatrix)

        // Solve optimal assignment
        let assignments = HungarianAlgorithm.minimumCostAssignment(costs: costMatrix)

        // Create speaker mapping from Hungarian result
        var mapping: [String: String] = [:]
        var totalAssignmentCost: Float = 0
        var totalOverlap = 0

        for (predIndex, gtIndex) in assignments.assignments.enumerated() {
            if gtIndex != -1 && predIndex < predSpeakerArray.count && gtIndex < gtSpeakerArray.count
            {
                let predSpeaker = predSpeakerArray[predIndex]
                let gtSpeaker = gtSpeakerArray[gtIndex]
                let overlap = overlapMatrix[predSpeaker]![gtSpeaker]!

                if overlap > 0 {  // Only assign if there's actual overlap
                    mapping[predSpeaker] = gtSpeaker
                    totalOverlap += overlap
                    print(
                        "🔍 HUNGARIAN MAPPING: '\(predSpeaker)' → '\(gtSpeaker)' (overlap: \(overlap) frames)"
                    )
                }
            }
        }

        totalAssignmentCost = assignments.totalCost
        print(
            "🔍 HUNGARIAN RESULT: Total assignment cost: \(String(format: "%.1f", totalAssignmentCost)), Total overlap: \(totalOverlap) frames"
        )

        // Handle unassigned predicted speakers
        for predSpeaker in predSpeakerArray {
            if mapping[predSpeaker] == nil {
                print("🔍 HUNGARIAN MAPPING: '\(predSpeaker)' → NO_MATCH (no beneficial assignment)")
            }
        }

        return mapping
    }
}

#endif