#if os(macOS)
import FluidAudio
import Foundation

/// Results formatting and output handling
struct ResultsFormatter {

    static func printResults(_ result: ProcessingResult) async {
        print("Diarization Results:")
        print("Audio File: \(result.audioFile)")
        print("Duration: \(String(format: "%.1f", result.durationSeconds))s")
        print("Processing Time: \(String(format: "%.1f", result.processingTimeSeconds))s")
        let rtfx = result.realTimeFactor
        print("Speed Factor (RTFx): \(String(format: "%.2f", rtfx))x")
        print("Detected Speakers: \(result.speakerCount)")
        if let metrics = result.metrics {
            print(
                "DER: \(String(format: "%.1f", metrics.der))% (Miss="
                    + "\(String(format: "%.1f", metrics.missRate))%, FA="
                    + "\(String(format: "%.1f", metrics.falseAlarmRate))%, SE="
                    + "\(String(format: "%.1f", metrics.speakerErrorRate))%, JER="
                    + "\(String(format: "%.1f", metrics.jer))%)"
            )
            if !metrics.speakerMapping.isEmpty {
                print("Speaker Mapping:")
                for (pred, truth) in metrics.speakerMapping.sorted(by: { $0.key < $1.key }) {
                    print("      \(pred) → \(truth)")
                }
            }
        }
        if let timings = result.timings {
            print("")
            printSingleRunTimings(timings, durationSeconds: result.durationSeconds)
        }
    }

    static func saveResults(_ result: ProcessingResult, to file: String) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(result)
        try data.write(to: URL(fileURLWithPath: file))
    }

    private static func printSingleRunTimings(
        _ timings: PipelineTimings,
        durationSeconds: Float
    ) {
        let stages: [(String, TimeInterval)] = [
            ("Model Compilation", timings.modelCompilationSeconds),
            ("Audio Loading", timings.audioLoadingSeconds),
            ("Segmentation", timings.segmentationSeconds),
            ("Embedding Extraction", timings.embeddingExtractionSeconds),
            ("Speaker Clustering", timings.speakerClusteringSeconds),
            ("Post Processing", timings.postProcessingSeconds),
        ]

        let total = timings.totalProcessingSeconds
        let totalAudioMinutes = Double(durationSeconds) / 60.0
        print("⏱️  Pipeline Timing Breakdown")
        let separator = String(repeating: "=", count: 95)
        print(separator)
        print("│ Stage                 │   Time   │ Percentage │ Per Audio Minute │")
        let headerSeparator = "├───────────────────────┼──────────┼────────────┼──────────────────┤"
        print(headerSeparator)

        for (stageName, stageTime) in stages {
            let stageNamePadded = stageName.padding(toLength: 19, withPad: " ", startingAt: 0)
            let timeStr = String(format: "%.3fs", stageTime).padding(
                toLength: 8, withPad: " ", startingAt: 0)
            let percentage = total > 0 ? (stageTime / total) * 100 : 0
            let percentageStr = String(format: "%.1f%%", percentage).padding(
                toLength: 10, withPad: " ", startingAt: 0)
            let perMinute = totalAudioMinutes > 0 ? stageTime / totalAudioMinutes : 0
            let perMinuteStr = String(format: "%.3fs/min", perMinute).padding(
                toLength: 16, withPad: " ", startingAt: 0)

            print("│ \(stageNamePadded) │ \(timeStr) │ \(percentageStr) │ \(perMinuteStr) │")
        }

        let footerSeparator = "└───────────────────────┴──────────┴────────────┴──────────────────┘"
        print(footerSeparator)
        let totalText = String(format: "%.3f", total)
        print("Bottleneck Stage: \(timings.bottleneckStage)  •  Total Processing: \(totalText)s")
    }

    static func saveBenchmarkResults(_ summary: BenchmarkSummary, to file: String) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(summary)
        try data.write(to: URL(fileURLWithPath: file))
    }

    static func formatTime(_ seconds: Float) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%02d:%02d", minutes, remainingSeconds)
    }

    static func printBenchmarkResults(
        _ results: [BenchmarkResult], avgDER: Float, avgJER: Float, dataset: String,
        customThresholds: (der: Float?, jer: Float?, rtf: Float?) = (nil, nil, nil)
    ) -> PerformanceAssessment {
        print("🏆 \(dataset) Benchmark Results")
        let separator = String(repeating: "=", count: 75)
        print("\(separator)")

        // Print table header
        print("│ Meeting ID    │  DER   │  JER   │  RTFx  │ Duration │ Speakers │")
        let headerSep = "├───────────────┼────────┼────────┼────────┼──────────┼──────────┤"
        print("\(headerSep)")

        // Print individual results
        for result in results.sorted(by: { $0.meetingId < $1.meetingId }) {
            let meetingDisplay = String(result.meetingId.prefix(13)).padding(
                toLength: 13, withPad: " ", startingAt: 0)
            let derStr = String(format: "%.1f%%", result.der).padding(
                toLength: 6, withPad: " ", startingAt: 0)
            let jerStr = String(format: "%.1f%%", result.jer).padding(
                toLength: 6, withPad: " ", startingAt: 0)
            let rtfx = result.realTimeFactor
            let rtfxStr = String(format: "%.2fx", rtfx).padding(
                toLength: 6, withPad: " ", startingAt: 0)
            let durationStr = formatTime(result.durationSeconds).padding(
                toLength: 8, withPad: " ", startingAt: 0)
            let speakerStr = String(result.speakerCount).padding(
                toLength: 8, withPad: " ", startingAt: 0)

            print(
                "│ \(meetingDisplay) │ \(derStr) │ \(jerStr) │ \(rtfxStr) │ \(durationStr) │ \(speakerStr) │"
            )
        }

        // Print summary section
        let midSep = "├───────────────┼────────┼────────┼────────┼──────────┼──────────┤"
        print("\(midSep)")

        let avgDerStr = String(format: "%.1f%%", avgDER).padding(
            toLength: 6, withPad: " ", startingAt: 0)
        let avgJerStr = String(format: "%.1f%%", avgJER).padding(
            toLength: 6, withPad: " ", startingAt: 0)
        let avgRtf = results.reduce(0.0) { $0 + $1.realTimeFactor } / Float(results.count)
        let avgRtfx = avgRtf
        let avgRtfxStr = String(format: "%.2fx", avgRtfx).padding(
            toLength: 6, withPad: " ", startingAt: 0)
        let totalDuration = results.reduce(0.0) { $0 + $1.durationSeconds }
        let avgDurationStr = formatTime(totalDuration).padding(
            toLength: 8, withPad: " ", startingAt: 0)
        let totalSpeakers = results.reduce(0) { $0 + $1.speakerCount }
        let avgSpeakers = Float(totalSpeakers) / Float(results.count)
        let avgSpeakerStr = String(format: "%.1f", avgSpeakers).padding(
            toLength: 8, withPad: " ", startingAt: 0)

        print(
            "│ AVERAGE       │ \(avgDerStr) │ \(avgJerStr) │ \(avgRtfxStr) │ \(avgDurationStr) │ \(avgSpeakerStr) │"
        )
        let bottomSep = "└───────────────┴────────┴────────┴────────┴──────────┴──────────┘"
        print("\(bottomSep)")

        // Print threshold comparison metrics if custom thresholds are provided
        if customThresholds.der != nil || customThresholds.jer != nil || customThresholds.rtf != nil {
            print("📊 Accuracy Metrics")
            let metricsHeader = "Metric    Value    Threshold    Status"
            print(metricsHeader)

            // DER threshold check
            if let derThreshold = customThresholds.der {
                let derStatus = avgDER < derThreshold ? "✅" : "❌"
                print(
                    "DER (Diarization Error Rate)    \(String(format: "%.1f", avgDER))%    < \(String(format: "%.1f", derThreshold))%    \(derStatus)"
                )
            }

            // JER threshold check
            if let jerThreshold = customThresholds.jer {
                let jerStatus = avgJER < jerThreshold ? "✅" : "❌"
                print(
                    "JER (Jaccard Error Rate)    \(String(format: "%.1f", avgJER))%    < \(String(format: "%.1f", jerThreshold))%    \(jerStatus)"
                )
            }

            // RTFx threshold check
            if let rtfThreshold = customThresholds.rtf {
                let rtfStatus = avgRtf < rtfThreshold ? "✅" : "❌"
                print(
                    "RTFx (Real-Time Factor)   \(String(format: "%.2f", avgRtf))x    < \(String(format: "%.2f", rtfThreshold))x    \(rtfStatus)"
                )
            }

            // Speaker count (always shown if we have thresholds)
            let groundTruthSpeakers = results.first?.groundTruthSpeakerCount ?? 0
            let speakerStatus = abs(avgSpeakers - Float(groundTruthSpeakers)) < 1.0 ? "✅" : "❌"
            print(
                "Speakers Detected    \(String(format: "%.0f", avgSpeakers))    \(groundTruthSpeakers)    \(speakerStatus)"
            )
        }

        // Print detailed timing breakdown
        printTimingBreakdown(results)

        // Print statistics
        if results.count > 1 {
            let derValues = results.map { $0.der }
            let jerValues = results.map { $0.jer }
            let derStdDev = calculateStandardDeviation(derValues)
            let jerStdDev = calculateStandardDeviation(jerValues)

            print("📊 Statistical Analysis:")
            print(
                "   DER: \(String(format: "%.1f", avgDER))% ± \(String(format: "%.1f", derStdDev))% (min: \(String(format: "%.1f", derValues.min()!))%, max: \(String(format: "%.1f", derValues.max()!))%)"
            )
            print(
                "   JER: \(String(format: "%.1f", avgJER))% ± \(String(format: "%.1f", jerStdDev))% (min: \(String(format: "%.1f", jerValues.min()!))%, max: \(String(format: "%.1f", jerValues.max()!))%)"
            )
            print("   Files Processed: \(results.count)")
            print(
                "   Total Audio: \(formatTime(totalDuration)) (\(String(format: "%.1f", totalDuration/60)) minutes)"
            )
        }

        // Print research comparison
        print("📝 Research Comparison:")
        print("   Your Results:          \(String(format: "%.1f", avgDER))% DER")
        print("   Powerset BCE (2023):   18.5% DER")
        print("   EEND (2019):           25.3% DER")
        print("   x-vector clustering:   28.7% DER")

        if dataset == "AMI-IHM" {
            print("   Note: IHM typically achieves 5-10% lower DER than SDM")
        }

        // Performance assessment (still pass RTF to assess function, but we display RTFx)
        let avgRTF = results.reduce(0.0) { $0 + $1.realTimeFactor } / Float(results.count)
        let assessment = PerformanceAssessment.assess(
            der: avgDER, jer: avgJER, rtf: avgRTF, customThresholds: customThresholds)
        print("\(assessment.description)")

        return assessment
    }

    /// Print detailed timing breakdown for pipeline stages
    static func printTimingBreakdown(_ results: [BenchmarkResult]) {
        guard !results.isEmpty else { return }

        print("⏱️  Pipeline Timing Breakdown")
        let timingSeparator = String(repeating: "=", count: 95)
        print("\(timingSeparator)")

        // Calculate average timings across all results
        let avgTimings = calculateAverageTimings(results)
        let totalAvgTime = avgTimings.totalProcessingSeconds

        // Print timing table header
        print("│ Stage                 │   Time   │ Percentage │ Per Audio Minute │")
        let timingHeaderSep = "├───────────────────────┼──────────┼────────────┼──────────────────┤"
        print("\(timingHeaderSep)")

        // Print each stage
        let stages: [(String, TimeInterval)] = [
            ("Model Compilation", avgTimings.modelCompilationSeconds),
            ("Audio Loading", avgTimings.audioLoadingSeconds),
            ("Segmentation", avgTimings.segmentationSeconds),
            ("Embedding Extraction", avgTimings.embeddingExtractionSeconds),
            ("Speaker Clustering", avgTimings.speakerClusteringSeconds),
            ("Post Processing", avgTimings.postProcessingSeconds),
        ]

        let totalAudioMinutes = results.reduce(0.0) { $0 + Double($1.durationSeconds) } / 60.0

        for (stageName, stageTime) in stages {
            let stageNamePadded = stageName.padding(toLength: 19, withPad: " ", startingAt: 0)
            let timeStr = String(format: "%.3fs", stageTime).padding(
                toLength: 8, withPad: " ", startingAt: 0)
            let percentage = totalAvgTime > 0 ? (stageTime / totalAvgTime) * 100 : 0
            let percentageStr = String(format: "%.1f%%", percentage).padding(
                toLength: 10, withPad: " ", startingAt: 0)
            let perMinute = totalAudioMinutes > 0 ? stageTime / totalAudioMinutes : 0
            let perMinuteStr = String(format: "%.3fs/min", perMinute).padding(
                toLength: 16, withPad: " ", startingAt: 0)

            print("│ \(stageNamePadded) │ \(timeStr) │ \(percentageStr) │ \(perMinuteStr) │")
        }

        // Print total
        let totalSep = "├───────────────────────┼──────────┼────────────┼──────────────────┤"
        print("\(totalSep)")
        let totalTimeStr = String(format: "%.3fs", totalAvgTime).padding(
            toLength: 8, withPad: " ", startingAt: 0)
        let totalPerMinuteStr = String(
            format: "%.3fs/min", totalAudioMinutes > 0 ? totalAvgTime / totalAudioMinutes : 0
        ).padding(toLength: 16, withPad: " ", startingAt: 0)
        print("│ TOTAL                 │ \(totalTimeStr) │ 100.0%     │ \(totalPerMinuteStr) │")

        let timingBottomSep = "└───────────────────────┴──────────┴────────────┴──────────────────┘"
        print("\(timingBottomSep)")

        // Print bottleneck analysis
        let bottleneck = avgTimings.bottleneckStage
        print("🔍 Performance Analysis:")
        print("   Bottleneck Stage: \(bottleneck)")
        print(
            "   Inference Only: \(String(format: "%.3f", avgTimings.totalInferenceSeconds))s (\(String(format: "%.1f", (avgTimings.totalInferenceSeconds / totalAvgTime) * 100))% of total)"
        )
        print(
            "   Setup Overhead: \(String(format: "%.3f", avgTimings.modelCompilationSeconds))s (\(String(format: "%.1f", (avgTimings.modelCompilationSeconds / totalAvgTime) * 100))% of total)"
        )

        // Optimization suggestions
        if avgTimings.segmentationSeconds > avgTimings.embeddingExtractionSeconds * 2 {
            print(
                "💡 Optimization Suggestion: Segmentation is the bottleneck - consider model optimization"
            )
        } else if avgTimings.embeddingExtractionSeconds > avgTimings.segmentationSeconds * 2 {
            print(
                "💡 Optimization Suggestion: Embedding extraction is the bottleneck - consider batch processing"
            )
        }
    }

    /// Calculate average timings across all benchmark results
    static func calculateAverageTimings(_ results: [BenchmarkResult]) -> PipelineTimings {
        let count = Double(results.count)
        guard count > 0 else { return PipelineTimings() }

        let avgModelCompilation =
            results.reduce(0.0) { $0 + $1.timings.modelCompilationSeconds } / count
        let avgAudioLoading = results.reduce(0.0) { $0 + $1.timings.audioLoadingSeconds } / count
        let avgSegmentation = results.reduce(0.0) { $0 + $1.timings.segmentationSeconds } / count
        let avgEmbedding =
            results.reduce(0.0) { $0 + $1.timings.embeddingExtractionSeconds } / count
        let avgClustering = results.reduce(0.0) { $0 + $1.timings.speakerClusteringSeconds } / count
        let avgPostProcessing =
            results.reduce(0.0) { $0 + $1.timings.postProcessingSeconds } / count

        return PipelineTimings(
            modelCompilationSeconds: avgModelCompilation,
            audioLoadingSeconds: avgAudioLoading,
            segmentationSeconds: avgSegmentation,
            embeddingExtractionSeconds: avgEmbedding,
            speakerClusteringSeconds: avgClustering,
            postProcessingSeconds: avgPostProcessing
        )
    }

    static func calculateStandardDeviation(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0.0 }
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Float(values.count - 1)
        return sqrt(variance)
    }
}

#endif
