#if os(macOS)
    import Foundation
    import FluidAudio

    // MARK: - Metrics Calculation

    public struct MetricsCalculator {

        public static func calculateDiarizationMetrics(
            predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment],
            totalDuration: Float
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
                Float(missedFrames + falseAlarmFrames + speakerErrorFrames) / Float(totalFrames)
                * 100
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

        public static func calculateJaccardErrorRate(
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
            let jaccardIndex =
                unionFrames > 0 ? Float(intersectionFrames) / Float(unionFrames) : 0.0

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

        static func findSpeakerAtTime(_ time: Float, in segments: [TimedSpeakerSegment]) -> String?
        {
            for segment in segments {
                if time >= segment.startTimeSeconds && time < segment.endTimeSeconds {
                    return segment.speakerId
                }
            }
            return nil
        }

        /// Find optimal speaker mapping using frame-by-frame overlap analysis
        static func findOptimalSpeakerMapping(
            predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment],
            totalDuration: Float
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
                if gtIndex != -1 && predIndex < predSpeakerArray.count
                    && gtIndex < gtSpeakerArray.count
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
                    print(
                        "🔍 HUNGARIAN MAPPING: '\(predSpeaker)' → NO_MATCH (no beneficial assignment)"
                    )
                }
            }

            return mapping
        }

        // MARK: - VAD Metrics

        public static func calculateVadMetrics(predictions: [Int], groundTruth: [Int]) -> (
            accuracy: Float, precision: Float, recall: Float, f1Score: Float
        ) {
            guard predictions.count == groundTruth.count && !predictions.isEmpty else {
                return (0, 0, 0, 0)
            }

            var truePositives = 0
            var falsePositives = 0
            var trueNegatives = 0
            var falseNegatives = 0

            for (pred, truth) in zip(predictions, groundTruth) {
                switch (pred, truth) {
                case (1, 1): truePositives += 1
                case (1, 0): falsePositives += 1
                case (0, 0): trueNegatives += 1
                case (0, 1): falseNegatives += 1
                default: break
                }
            }

            let accuracy = Float(truePositives + trueNegatives) / Float(predictions.count) * 100
            let precision =
                truePositives + falsePositives > 0
                ? Float(truePositives) / Float(truePositives + falsePositives) * 100 : 0
            let recall =
                truePositives + falseNegatives > 0
                ? Float(truePositives) / Float(truePositives + falseNegatives) * 100 : 0
            let f1Score =
                precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0

            return (accuracy, precision, recall, f1Score)
        }
    }

    // MARK: - Output Formatting

    public struct OutputFormatter {

        public static func formatTime(_ seconds: Float) -> String {
            let minutes = Int(seconds) / 60
            let remainingSeconds = Int(seconds) % 60
            return String(format: "%02d:%02d", minutes, remainingSeconds)
        }

        public static func printResults(_ result: ProcessingResult) async {
            print("\n📊 Diarization Results:")
            print("   Audio File: \(result.audioFile)")
            print("   Duration: \(String(format: "%.1f", result.durationSeconds))s")
            print("   Processing Time: \(String(format: "%.1f", result.processingTimeSeconds))s")
            print("   Real-time Factor: \(String(format: "%.2f", result.realTimeFactor))x")
            print("   Detected Speakers: \(result.speakerCount)")
            print("\n🎤 Speaker Segments:")

            for (index, segment) in result.segments.enumerated() {
                let startTime = formatTime(segment.startTimeSeconds)
                let endTime = formatTime(segment.endTimeSeconds)
                let duration = segment.endTimeSeconds - segment.startTimeSeconds

                print(
                    "   \(index + 1). \(segment.speakerId): \(startTime) - \(endTime) (\(String(format: "%.1f", duration))s)"
                )
            }
        }

        public static func saveResults(_ result: ProcessingResult, to file: String) async throws {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601

            let data = try encoder.encode(result)
            try data.write(to: URL(fileURLWithPath: file))
        }

        public static func saveBenchmarkResults(_ summary: BenchmarkSummary, to file: String)
            async throws
        {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601

            let data = try encoder.encode(summary)
            try data.write(to: URL(fileURLWithPath: file))
        }

        public static func saveVadBenchmarkResults(_ result: VadBenchmarkResult, to file: String)
            throws
        {
            let resultsDict: [String: Any] = [
                "test_name": result.testName,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1Score,
                "processing_time_seconds": result.processingTime,
                "total_files": result.totalFiles,
                "correct_predictions": result.correctPredictions,
                "timestamp": ISO8601DateFormatter().string(from: Date()),
                "environment": "CLI",
            ]

            let jsonData = try JSONSerialization.data(
                withJSONObject: resultsDict, options: .prettyPrinted)
            try jsonData.write(to: URL(fileURLWithPath: file))
        }

        /// Calculate average timings across all benchmark results
        public static func calculateAverageTimings(_ results: [BenchmarkResult]) -> PipelineTimings
        {
            let count = Double(results.count)
            guard count > 0 else { return PipelineTimings() }

            let avgModelDownload =
                results.reduce(0.0) { $0 + $1.timings.modelDownloadSeconds } / count
            let avgModelCompilation =
                results.reduce(0.0) { $0 + $1.timings.modelCompilationSeconds } / count
            let avgAudioLoading =
                results.reduce(0.0) { $0 + $1.timings.audioLoadingSeconds } / count
            let avgSegmentation =
                results.reduce(0.0) { $0 + $1.timings.segmentationSeconds } / count
            let avgEmbedding =
                results.reduce(0.0) { $0 + $1.timings.embeddingExtractionSeconds } / count
            let avgClustering =
                results.reduce(0.0) { $0 + $1.timings.speakerClusteringSeconds } / count
            let avgPostProcessing =
                results.reduce(0.0) { $0 + $1.timings.postProcessingSeconds } / count

            return PipelineTimings(
                modelDownloadSeconds: avgModelDownload,
                modelCompilationSeconds: avgModelCompilation,
                audioLoadingSeconds: avgAudioLoading,
                segmentationSeconds: avgSegmentation,
                embeddingExtractionSeconds: avgEmbedding,
                speakerClusteringSeconds: avgClustering,
                postProcessingSeconds: avgPostProcessing
            )
        }

        public static func calculateStandardDeviation(_ values: [Float]) -> Float {
            guard values.count > 1 else { return 0.0 }
            let mean = values.reduce(0, +) / Float(values.count)
            let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Float(values.count - 1)
            return sqrt(variance)
        }
    }
#endif
