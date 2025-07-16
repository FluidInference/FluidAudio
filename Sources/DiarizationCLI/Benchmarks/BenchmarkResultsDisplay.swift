#if os(macOS)
import Foundation
import FluidAudio

public struct BenchmarkResultsDisplay {

public static func printBenchmarkResults(
    _ results: [BenchmarkResult], avgDER: Float, avgJER: Float, dataset: String
) {
    print("\n🏆 \(dataset) Benchmark Results")
    let separator = String(repeating: "=", count: 75)
    print("\(separator)")

    // Print table header
    print("│ Meeting ID    │  DER   │  JER   │  RTF   │ Duration │ Speakers │")
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
        let rtfStr = String(format: "%.2fx", result.realTimeFactor).padding(
            toLength: 6, withPad: " ", startingAt: 0)
        let durationStr = OutputFormatter.formatTime(result.durationSeconds).padding(
            toLength: 8, withPad: " ", startingAt: 0)
        let speakerStr = String(result.speakerCount).padding(
            toLength: 8, withPad: " ", startingAt: 0)

        print(
            "│ \(meetingDisplay) │ \(derStr) │ \(jerStr) │ \(rtfStr) │ \(durationStr) │ \(speakerStr) │"
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
    let avgRtfStr = String(format: "%.2fx", avgRtf).padding(
        toLength: 6, withPad: " ", startingAt: 0)
    let totalDuration = results.reduce(0.0) { $0 + $1.durationSeconds }
    let avgDurationStr = OutputFormatter.formatTime(totalDuration).padding(
        toLength: 8, withPad: " ", startingAt: 0)
    let avgSpeakers = results.reduce(0) { $0 + $1.speakerCount } / results.count
    let avgSpeakerStr = String(format: "%.1f", Float(avgSpeakers)).padding(
        toLength: 8, withPad: " ", startingAt: 0)

    print(
        "│ AVERAGE       │ \(avgDerStr) │ \(avgJerStr) │ \(avgRtfStr) │ \(avgDurationStr) │ \(avgSpeakerStr) │"
    )
    let bottomSep = "└───────────────┴────────┴────────┴────────┴──────────┴──────────┘"
    print("\(bottomSep)")

    // Print detailed timing breakdown
    printTimingBreakdown(results)

    // Print statistics
    if results.count > 1 {
        let derValues = results.map { $0.der }
        let jerValues = results.map { $0.jer }
        let derStdDev = OutputFormatter.calculateStandardDeviation(derValues)
        let jerStdDev = OutputFormatter.calculateStandardDeviation(jerValues)

        print("\n📊 Statistical Analysis:")
        print(
            "   DER: \(String(format: "%.1f", avgDER))% ± \(String(format: "%.1f", derStdDev))% (min: \(String(format: "%.1f", derValues.min()!))%, max: \(String(format: "%.1f", derValues.max()!))%)"
        )
        print(
            "   JER: \(String(format: "%.1f", avgJER))% ± \(String(format: "%.1f", jerStdDev))% (min: \(String(format: "%.1f", jerValues.min()!))%, max: \(String(format: "%.1f", jerValues.max()!))%)"
        )
        print("   Files Processed: \(results.count)")
        print(
            "   Total Audio: \(OutputFormatter.formatTime(totalDuration)) (\(String(format: "%.1f", totalDuration/60)) minutes)"
        )
    }

    // Print research comparison
    print("\n📝 Research Comparison:")
    print("   Your Results:          \(String(format: "%.1f", avgDER))% DER")
    print("   Powerset BCE (2023):   18.5% DER")
    print("   EEND (2019):           25.3% DER")
    print("   x-vector clustering:   28.7% DER")

    if dataset == "AMI-IHM" {
        print("   Note: IHM typically achieves 5-10% lower DER than SDM")
    }

    // Performance assessment
    if avgDER < 20.0 {
        print("\n🎉 EXCELLENT: Competitive with state-of-the-art research!")
    } else if avgDER < 30.0 {
        print("\n✅ GOOD: Above research baseline, room for optimization")
    } else if avgDER < 50.0 {
        print("\n⚠️  NEEDS WORK: Significant room for parameter tuning")
    } else {
        print("\n🚨 CRITICAL: Check configuration - results much worse than expected")
    }
}

/// Print detailed timing breakdown for pipeline stages
static func printTimingBreakdown(_ results: [BenchmarkResult]) {
    guard !results.isEmpty else { return }

    print("\n⏱️  Pipeline Timing Breakdown")
    let timingSeparator = String(repeating: "=", count: 95)
    print("\(timingSeparator)")

    // Calculate average timings across all results
    let avgTimings = OutputFormatter.calculateAverageTimings(results)
    let totalAvgTime = avgTimings.totalProcessingSeconds

    // Print timing table header
    print("│ Stage                 │   Time   │ Percentage │ Per Audio Minute │")
    let timingHeaderSep =
        "├───────────────────────┼──────────┼────────────┼──────────────────┤"
    print("\(timingHeaderSep)")

    // Print each stage
    let stages: [(String, TimeInterval)] = [
        ("Model Download", avgTimings.modelDownloadSeconds),
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

    let timingBottomSep =
        "└───────────────────────┴──────────┴────────────┴──────────────────┘"
    print("\(timingBottomSep)")

    // Print bottleneck analysis
    let bottleneck = avgTimings.bottleneckStage
    print("\n🔍 Performance Analysis:")
    print("   Bottleneck Stage: \(bottleneck)")
    print(
        "   Inference Only: \(String(format: "%.3f", avgTimings.totalInferenceSeconds))s (\(String(format: "%.1f", (avgTimings.totalInferenceSeconds / totalAvgTime) * 100))% of total)"
    )
    print(
        "   Setup Overhead: \(String(format: "%.3f", avgTimings.modelDownloadSeconds + avgTimings.modelCompilationSeconds))s (\(String(format: "%.1f", ((avgTimings.modelDownloadSeconds + avgTimings.modelCompilationSeconds) / totalAvgTime) * 100))% of total)"
    )

    // Optimization suggestions
    if avgTimings.modelDownloadSeconds > avgTimings.totalInferenceSeconds {
        print(
            "\n💡 Optimization Suggestion: Model download is dominating execution time - consider model caching"
        )
    } else if avgTimings.segmentationSeconds > avgTimings.embeddingExtractionSeconds * 2 {
        print(
            "\n💡 Optimization Suggestion: Segmentation is the bottleneck - consider model optimization"
        )
    } else if avgTimings.embeddingExtractionSeconds > avgTimings.segmentationSeconds * 2 {
        print(
            "\n💡 Optimization Suggestion: Embedding extraction is the bottleneck - consider batch processing"
        )
    }
}
}
#endif
