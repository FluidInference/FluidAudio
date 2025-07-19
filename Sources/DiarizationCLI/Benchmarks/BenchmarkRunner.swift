#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// AMI benchmark runner with metrics calculation and result formatting
struct BenchmarkRunner {
    
    // MARK: - AMI Benchmark Implementation
    
    static func runAMISDMBenchmark(
        manager: DiarizerManager, config: DiarizerConfig, outputFile: String?, autoDownload: Bool, singleFile: String? = nil, iterations: Int = 1, customThresholds: (der: Float?, jer: Float?, rtf: Float?) = (nil, nil, nil)
    ) async -> PerformanceAssessment {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let amiDirectory = homeDir.appendingPathComponent(
            "FluidAudioDatasets/ami_official/sdm")

        // Check if AMI dataset exists, download if needed
        if !FileManager.default.fileExists(atPath: amiDirectory.path) {
            if autoDownload {
                print("📥 AMI SDM dataset not found - downloading automatically...")
                await DatasetDownloader.downloadAMIDataset(variant: .sdm, force: false, singleFile: singleFile)

                // Check again after download
                if !FileManager.default.fileExists(atPath: amiDirectory.path) {
                    print("❌ Failed to download AMI SDM dataset")
                    return .critical
                }
            } else {
                print("⚠️ AMI SDM dataset not found")
                print("📥 Download options:")
                print("   Option 1: Use --auto-download flag")
                print("   Option 2: Download manually:")
                print("      1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
                print(
                    "      2. Select test meetings: ES2002a, ES2003a, ES2004a, IS1000a, IS1001a")
                print("      3. Download 'Headset mix' (Mix-Headset.wav files)")
                print("      4. Place files in: \(amiDirectory.path)")
                print("   Option 3: Use download command:")
                print("      swift run fluidaudio download --dataset ami-sdm")
                return .critical
            }
        }

        let commonMeetings: [String]
        if let singleFile = singleFile {
            commonMeetings = [singleFile]
            print("📋 Testing single file: \(singleFile)")
        } else {
            commonMeetings = [
                // Core AMI test set - smaller subset for initial benchmarking
                "ES2002a", "ES2003a", "ES2004a", "ES2005a",
                "IS1000a", "IS1001a", "IS1002b",
                "TS3003a", "TS3004a",
            ]
        }

        var benchmarkResults: [BenchmarkResult] = []
        var totalDER: Float = 0.0
        var totalJER: Float = 0.0
        var processedFiles = 0

        print("📊 Running AMI SDM Benchmark")
        print("   Looking for Mix-Headset.wav files in: \(amiDirectory.path)")

        for meetingId in commonMeetings {
            let audioFileName = "\(meetingId).Mix-Headset.wav"
            let audioPath = amiDirectory.appendingPathComponent(audioFileName)

            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                print("   ⏭️ Skipping \(audioFileName) (not found)")
                continue
            }

            print("   🎵 Processing \(audioFileName)...")

            do {
                let audioLoadingStartTime = Date()
                let audioSamples = try await AudioProcessor.loadAudioFile(path: audioPath.path)
                let audioLoadingTime = Date().timeIntervalSince(audioLoadingStartTime)
                let duration = Float(audioSamples.count) / 16000.0

                let startTime = Date()
                let result = try await manager.performCompleteDiarization(
                    audioSamples, sampleRate: 16000)
                let processingTime = Date().timeIntervalSince(startTime)

                // Create complete timing information including audio loading
                let completeTimings = PipelineTimings(
                    modelDownloadSeconds: result.timings.modelDownloadSeconds,
                    modelCompilationSeconds: result.timings.modelCompilationSeconds,
                    audioLoadingSeconds: audioLoadingTime,
                    segmentationSeconds: result.timings.segmentationSeconds,
                    embeddingExtractionSeconds: result.timings.embeddingExtractionSeconds,
                    speakerClusteringSeconds: result.timings.speakerClusteringSeconds,
                    postProcessingSeconds: result.timings.postProcessingSeconds
                )

                // Get ground truth speaker count
                let groundTruthSpeakerCount = AMIParser.getGroundTruthSpeakerCount(for: meetingId)

                // Load ground truth annotations
                let groundTruth = await AMIParser.loadAMIGroundTruth(for: meetingId, duration: duration)

                // Calculate metrics
                let metrics = MetricsCalculator.calculateDiarizationMetrics(
                    predicted: result.segments,
                    groundTruth: groundTruth,
                    totalDuration: duration
                )

                totalDER += metrics.der
                totalJER += metrics.jer
                processedFiles += 1

                let rtf = Float(processingTime) / duration

                print(
                    "     ✅ DER: \(String(format: "%.1f%%", metrics.der)), JER: \(String(format: "%.1f%%", metrics.jer)), RTF: \(String(format: "%.2f", rtf))x"
                )

                benchmarkResults.append(
                    BenchmarkResult(
                        meetingId: meetingId,
                        durationSeconds: duration,
                        processingTimeSeconds: processingTime,
                        realTimeFactor: rtf,
                        der: metrics.der,
                        jer: metrics.jer,
                        segments: result.segments,
                        speakerCount: metrics.mappedSpeakerCount,
                        groundTruthSpeakerCount: groundTruthSpeakerCount,
                        timings: completeTimings
                    ))

            } catch {
                print("     ❌ Failed: \(error)")
            }
        }

        guard processedFiles > 0 else {
            print("❌ No files were processed successfully")
            return .critical
        }

        let avgDER = totalDER / Float(processedFiles)
        let avgJER = totalJER / Float(processedFiles)

        // Print detailed results table
        let assessment = ResultsFormatter.printBenchmarkResults(benchmarkResults, avgDER: avgDER, avgJER: avgJER, dataset: "AMI-SDM", customThresholds: customThresholds)

        // Save results if requested
        if let outputFile = outputFile {
            let summary = BenchmarkSummary(
                dataset: "AMI-SDM",
                averageDER: avgDER,
                averageJER: avgJER,
                processedFiles: processedFiles,
                totalFiles: commonMeetings.count,
                results: benchmarkResults
            )

            do {
                try await ResultsFormatter.saveBenchmarkResults(summary, to: outputFile)
                print("💾 Benchmark results saved to: \(outputFile)")
            } catch {
                print("⚠️ Failed to save results: \(error)")
            }
        }
        
        return assessment
    }

    static func runAMIIHMBenchmark(
        manager: DiarizerManager, config: DiarizerConfig, outputFile: String?, autoDownload: Bool, singleFile: String? = nil, iterations: Int = 1, customThresholds: (der: Float?, jer: Float?, rtf: Float?) = (nil, nil, nil)
    ) async -> PerformanceAssessment {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let amiDirectory = homeDir.appendingPathComponent(
            "FluidAudioDatasets/ami_official/ihm")

        // Check if AMI dataset exists, download if needed
        if !FileManager.default.fileExists(atPath: amiDirectory.path) {
            if autoDownload {
                print("📥 AMI IHM dataset not found - downloading automatically...")
                await DatasetDownloader.downloadAMIDataset(variant: .ihm, force: false, singleFile: singleFile)

                // Check again after download
                if !FileManager.default.fileExists(atPath: amiDirectory.path) {
                    print("❌ Failed to download AMI IHM dataset")
                    return .critical
                }
            } else {
                print("⚠️ AMI IHM dataset not found")
                print("📥 Download options:")
                print("   Option 1: Use --auto-download flag")
                print("   Option 2: Download manually:")
                print("      1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
                print(
                    "      2. Select test meetings: ES2002a, ES2003a, ES2004a, IS1000a, IS1001a")
                print("      3. Download 'Individual headsets' (Headset-0.wav files)")
                print("      4. Place files in: \(amiDirectory.path)")
                print("   Option 3: Use download command:")
                print("      swift run fluidaudio download --dataset ami-ihm")
                return .critical
            }
        }

        let commonMeetings = [
            // Core AMI test set - smaller subset for initial benchmarking
            "ES2002a", "ES2003a", "ES2004a", "ES2005a",
            "IS1000a", "IS1001a", "IS1002b",
            "TS3003a", "TS3004a",
        ]

        var benchmarkResults: [BenchmarkResult] = []
        var totalDER: Float = 0.0
        var totalJER: Float = 0.0
        var processedFiles = 0

        print("📊 Running AMI IHM Benchmark")
        print("   Looking for Headset-0.wav files in: \(amiDirectory.path)")

        for meetingId in commonMeetings {
            let audioFileName = "\(meetingId).Headset-0.wav"
            let audioPath = amiDirectory.appendingPathComponent(audioFileName)

            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                print("   ⏭️ Skipping \(audioFileName) (not found)")
                continue
            }

            print("   🎵 Processing \(audioFileName)...")

            do {
                let audioLoadingStartTime = Date()
                let audioSamples = try await AudioProcessor.loadAudioFile(path: audioPath.path)
                let audioLoadingTime = Date().timeIntervalSince(audioLoadingStartTime)
                let duration = Float(audioSamples.count) / 16000.0

                let startTime = Date()
                let result = try await manager.performCompleteDiarization(
                    audioSamples, sampleRate: 16000)
                let processingTime = Date().timeIntervalSince(startTime)

                // Create complete timing information including audio loading
                let completeTimings = PipelineTimings(
                    modelDownloadSeconds: result.timings.modelDownloadSeconds,
                    modelCompilationSeconds: result.timings.modelCompilationSeconds,
                    audioLoadingSeconds: audioLoadingTime,
                    segmentationSeconds: result.timings.segmentationSeconds,
                    embeddingExtractionSeconds: result.timings.embeddingExtractionSeconds,
                    speakerClusteringSeconds: result.timings.speakerClusteringSeconds,
                    postProcessingSeconds: result.timings.postProcessingSeconds
                )

                // Get ground truth speaker count
                let groundTruthSpeakerCount = AMIParser.getGroundTruthSpeakerCount(for: meetingId)

                // Load ground truth annotations
                let groundTruth = await AMIParser.loadAMIGroundTruth(for: meetingId, duration: duration)

                // Calculate metrics
                let metrics = MetricsCalculator.calculateDiarizationMetrics(
                    predicted: result.segments,
                    groundTruth: groundTruth,
                    totalDuration: duration
                )

                totalDER += metrics.der
                totalJER += metrics.jer
                processedFiles += 1

                let rtf = Float(processingTime) / duration

                print(
                    "     ✅ DER: \(String(format: "%.1f", metrics.der))%, JER: \(String(format: "%.1f", metrics.jer))%, RTF: \(String(format: "%.2f", rtf))x"
                )

                benchmarkResults.append(
                    BenchmarkResult(
                        meetingId: meetingId,
                        durationSeconds: duration,
                        processingTimeSeconds: processingTime,
                        realTimeFactor: rtf,
                        der: metrics.der,
                        jer: metrics.jer,
                        segments: result.segments,
                        speakerCount: metrics.mappedSpeakerCount,
                        groundTruthSpeakerCount: groundTruthSpeakerCount,
                        timings: completeTimings
                    ))

            } catch {
                print("     ❌ Failed: \(error)")
            }
        }

        guard processedFiles > 0 else {
            print("❌ No files were processed successfully")
            return .critical
        }

        let avgDER = totalDER / Float(processedFiles)
        let avgJER = totalJER / Float(processedFiles)

        // Print detailed results table
        let assessment = ResultsFormatter.printBenchmarkResults(benchmarkResults, avgDER: avgDER, avgJER: avgJER, dataset: "AMI-IHM", customThresholds: customThresholds)

        // Save results if requested
        if let outputFile = outputFile {
            let summary = BenchmarkSummary(
                dataset: "AMI-IHM",
                averageDER: avgDER,
                averageJER: avgJER,
                processedFiles: processedFiles,
                totalFiles: commonMeetings.count,
                results: benchmarkResults
            )

            do {
                try await ResultsFormatter.saveBenchmarkResults(summary, to: outputFile)
                print("💾 Benchmark results saved to: \(outputFile)")
            } catch {
                print("⚠️ Failed to save results: \(error)")
            }
        }
        
        return assessment
    }
}

#endif